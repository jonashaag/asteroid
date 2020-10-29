import os
import argparse
import json

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    GPUStatsMonitor,
)

from asteroid import DCCRNet
from asteroid.engine import schedulers

from asteroid.engine.optimizers import make_optimizer
from asteroid.engine.system import System
from asteroid.losses import singlesrc_neg_sisdr

from ranger2020 import Ranger

# from warmup_scheduler import GradualWarmupScheduler

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")


if 0:
    from asteroid.filterbanks import make_enc_dec
    from ds import itu_r_468_matrix

    asteroid_stft, _ = make_enc_dec(
        "stft",
        kernel_size=512,
        n_filters=512,
        sample_rate=16000.0,
    )
    asteroid_stft = asteroid_stft.cuda()

    itu_matrix = torch.from_numpy(itu_r_468_matrix(512, 16000))[None, None, :, None]

    def loss_func(est_target, target_wav):
        from asteroid.filterbanks.transforms import take_mag

        est_wav, est_stft = est_target
        target_wav = target_wav.unsqueeze(1)
        assert len(est_wav.shape) == 3, est_wav.shape
        assert len(target_wav.shape) == 3, target_wav.shape
        assert len(est_stft.shape) == 4, est_stft.shape
        target_stft = asteroid_stft(target_wav)
        est_mag = take_mag(est_stft)
        target_mag = take_mag(target_stft)
        over_under_weights = 1 + 4 * (est_mag.abs() < target_mag.abs())
        magmae = (((est_mag - target_mag).abs() * over_under_weights) * itu_r_468_matrix).mean()
        wavmae = ((est_wav - target_wav)).abs().mean()
        return 1.5 * magmae + 1 * wavmae

    def getds(conf):
        import ds

        ds.configure(
            conf["data"]["segment"],
            conf["data"]["real_rirs_dir"],
            conf["data"]["dns_rirs_dir"],
            conf["data"]["dns_noise_dir"],
            conf["data"]["dns_clean_dir"],
        )
        # ds.bench(); return
        train_ds, val_ds = ds.make_ds()
        # train_ds.getitem(166507, 4415633321459280479, log=True); return
        return train_ds, val_ds


else:
    import numpy as np

    class MyDs(torch.utils.data.Dataset):
        def __len__(self):
            return 100_000

        def __getitem__(self, idx):
            rand = np.random.default_rng(idx)
            return (
                rand.uniform(size=(3 * 16000,)).astype("float32"),
                rand.uniform(size=(3 * 16000,)).astype("float32"),
            )

    def loss_func(est_target, target):
        return (est_target[0].squeeze(1) - target).abs().mean()

    def getds(conf):
        return MyDs(), MyDs()


def main(conf):
    val_ds, train_ds = getds(conf)
    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
        pin_memory=True,
    )

    # import tqdm; for _ in tqdm.tqdm(train_loader): pass

    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
        pin_memory=True,
    )

    model = DCCRNet(architecture="DCCRN-CL")
    # model = DCUNet(architecture="DCUNet-16")
    optimizer = Ranger(model.parameters(), **conf["optim"])
    # optimizer = make_optimizer(model.parameters(), optimizer="adam", **conf["optim"])

    # Define scheduler
    scheduler = None
    if conf["training"]["half_lr"]:
        scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }

    if 0:
        scheduler = {
            "scheduler": GradualWarmupScheduler(
                optimizer=optimizer,
                multiplier=1,
                total_epoch=10,
                after_scheduler=scheduler["scheduler"] if scheduler else None,
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }

    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    # loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    # loss_func = lambda est_target, target: singlesrc_neg_sisdr(est_target.squeeze(1), target).mean()
    system = System(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=conf,
    )

    # Define callbacks
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="val_loss", mode="min", save_top_k=5, save_last=True, verbose=True
    )
    early_stopping = False
    if conf["training"]["early_stop"]:
        early_stopping = EarlyStopping(monitor="val_loss", patience=15, verbose=True)
    callbacks = [
        LearningRateMonitor(),
        # GPUStatsMonitor(),
    ]

    # Don't ask GPU if they are not available.
    gpus = -1 if torch.cuda.is_available() else None
    # assert torch.cuda.is_available()
    trainer = pl.Trainer(
        # weights_summary='full',
        callbacks=callbacks,
        max_epochs=conf["training"]["epochs"],
        checkpoint_callback=checkpoint,
        early_stop_callback=early_stopping,
        default_root_dir=exp_dir,
        gpus=gpus,
        benchmark=True,
        # distributed_backend="ddp",
        # val_check_interval=0.34,
        # limit_train_batches=10,
        # limit_val_batches=10,
        gradient_clip_val=conf["training"]["gradient_clipping"],
        # resume_from_checkpoint="/root/asteroid/egs/dns_challenge/dccrn/exp/tmp/checkpoints.ckpt",
    )
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()
    # to_save.update(train_set.get_infos())
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    import yaml
    from pprint import pprint
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open("local/conf.yml") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    pprint(arg_dic)
    main(arg_dic)
