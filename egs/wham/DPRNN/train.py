import os
import argparse
import json
from pprint import pprint

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from asteroid import DPRNNTasNet
from asteroid.engine.optimizers import make_optimizer
from asteroid.engine.system import System
from asteroid.losses import (
    PITLossWrapper,
    pairwise_neg_sisdr,
    pairwise_mse,
    pairwise_neg_sisdr,
    pairwise_neg_sdsdr,
)

from torch.nn.modules.loss import _Loss


class STFTMSE(_Loss):
    def forward(self, est_targets, targets):
        # assumes sr = 8000
        assert (est_targets.shape[1], targets.shape[1]) == (1, 1)
        win = torch.hann_window(256, device=targets.device)
        loss = (
            torch.stft(est_targets[:, 0], 256, 64, window=win)
            - torch.stft(targets[:, 0], 256, 64, window=win)
        ) ** 2
        m = loss.mean(dim=list(range(1, loss.ndim)))
        return 1000 * m.unsqueeze(1).unsqueeze(1)


pairwise_stftmse = STFTMSE()


def mk_bark_loss(sr):
    import zounds
    class SR8000(zounds.timeseries.samplerate.AudioSampleRate):
        def __init__(self):
            super().__init__(8000, 256, 64)

    scale = zounds.BarkScale(zounds.FrequencyBand(20, int(sr / 2)), 512)
    return zounds.learn.PerceptualLoss(
        scale,
        SR8000(),
        lap=1,
        log_factor=10,
        basis_size=512,
        frequency_weighting=zounds.AWeighting(),
        cosine_similarity=False,
    ).cuda()


#singlesrc_bark_loss = mk_bark_loss(8000)


def peaknorm(t, n):
    return (
        torch.pow(10, torch.tensor(n / 20.0))
        / t.abs().max(dim=-1, keepdims=True)[0]
        * t
    )


class MySystem(System):
    def validation_step(self, batch, batch_nb):
        inputs, targets = batch
        est_targets = self(inputs)
        loss = self.loss_func(est_targets, targets)
        stftmse = pairwise_stftmse(est_targets, targets).mean()
        stftmse_norm = pairwise_stftmse(
            peaknorm(est_targets, -10), peaknorm(targets, -10)
        ).mean()
        sisdr = pairwise_neg_sisdr(est_targets, targets).mean()
        sdsdr = pairwise_neg_sdsdr(est_targets, targets).mean()
        mse = pairwise_mse(est_targets, targets).mean()
        return {
            "val_loss": loss,
            "m_stftmse": stftmse,
            "m_stftmse_norm": stftmse_norm,
            "m_sisdr": sisdr,
            "m_sdsdr": sdsdr,
            "m_mse": mse,
        }

    def validation_epoch_end(self, outputs):
        pprint(
            (
                "Validation results",
                {
                    k: float(torch.stack([x[k] for x in outputs]).mean().item())
                    for k in outputs[0].keys()
                },
            )
        )
        return super().validation_epoch_end(outputs)


# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp_dir", default="exp/tmp", help="Full path to save best validation model"
)


def main(conf):
    import ds

    train_loader, val_loader = ds.getds(False, conf)

    model = DPRNNTasNet(**conf["filterbank"], **conf["masknet"], n_src=1)
    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    # Define scheduler
    scheduler = None
    if conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5)
    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    # loss_func = PITLossWrapper(STFTMSE(), pit_from="pw_mtx")
    # loss_func = PITLossWrapper(singlesrc_mse, pit_from='pw_pt')
    #loss_func = PITLossWrapper(singlesrc_bark_loss, pit_from="pw_pt")
    system = System(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )

    # Define callbacks
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="val_loss", mode="min", save_top_k=5, verbose=1
    )
    early_stopping = False
    if conf["training"]["early_stop"]:
        early_stopping = EarlyStopping(monitor="val_loss", patience=30,
                                       verbose=1)

    # Don't ask GPU if they are not available.
    assert torch.cuda.is_available()
    gpus = -1 if torch.cuda.is_available() else None
    no_train = 0
    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        checkpoint_callback=checkpoint,
        early_stop_callback=early_stopping,
        default_root_dir=exp_dir,
        gpus=gpus,
        # distributed_backend='ddp',
        benchmark=True,
        # precision=16,
        limit_train_batches=2 if no_train else int(50_000/conf["training"]["batch_size"]),
        gradient_clip_val=conf["training"]["gradient_clipping"],
    )
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    # Save best model (next PL version will make this easier)
    best_path = [b for b, v in best_k.items() if v == min(best_k.values())][0]
    state_dict = torch.load(best_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()
    # to_save.update(train_set.get_infos())
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    import yaml
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open(os.environ["CONF"]) as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    main(arg_dic)
