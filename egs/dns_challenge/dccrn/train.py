import os
import argparse
import json

import torch
import tqdm
import numpy as np
import glob
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

torch.manual_seed(42)

# from warmup_scheduler import GradualWarmupScheduler

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")


def sisdr(e, t):
    #return (e-t).abs().mean()
    return singlesrc_neg_sisdr(e.squeeze(1), t.squeeze(1)).mean()


if 0:
    from asteroid.filterbanks import make_enc_dec
    from asteroid.filterbanks.transforms import take_mag
    from ds import itu_r_468_matrix
    from torch.nn.modules.loss import _Loss

    class MyLoss(_Loss):
        def __init__(self):
            super().__init__()
            self.asteroid_stft, _ = make_enc_dec(
                "stft",
                kernel_size=512,
                n_filters=512,
                sample_rate=16000.0,
            )
            self.itu_matrix = torch.from_numpy(itu_r_468_matrix(512, 16000))[None, None, :, None]

        def forward(self, est_wav, target_wav):
            target_wav = target_wav.unsqueeze(1)
            assert len(est_wav.shape) == 3, est_wav.shape
            assert len(target_wav.shape) == 3, target_wav.shape

            target_stft = self.asteroid_stft.to(est_wav.device)(target_wav)
            est_stft = self.asteroid_stft.to(est_wav.device)(est_wav)
            est_mag = take_mag(est_stft)
            target_mag = take_mag(target_stft)
            over_under_weights = 1 + 4 * (est_mag.abs() < target_mag.abs())
            magmae = (((est_mag - target_mag).abs() * over_under_weights) * self.itu_matrix.to(est_wav.device)).mean()
            wavmae = ((est_wav - target_wav)).abs().mean()
            return 1000 * (1 * wavmae + 2 * magmae)

    def getds(conf, **kwargs):
        import ds

        ds.configure(
            conf["data"]["segment"],
            conf["data"]["real_rirs_dir"],
            conf["data"]["dns_rirs_dir"],
            conf["data"]["dns_noise_dir"],
            conf["data"]["dns_clean_dir"],
        )
        # ds.bench(); return
        train_ds, val_ds = ds.make_ds(**kwargs)
        # train_ds.getitem(166507, 4415633321459280479, log=True); return
        return train_ds, val_ds


elif 0:
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

    def getds(conf, **kwargs):
        return MyDs(), MyDs()

else:
    import soundfile as sf

    class MyDs(torch.utils.data.Dataset):
        def __init__(self, files):
            self.files = files

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            return torch.from_numpy(sf.read(self.files[idx][0], dtype="float32")[0]), torch.from_numpy(sf.read(self.files[idx][1], dtype="float32")[0]),

    def getds(conf, target):
        files = glob.glob(conf["data"]["dns_xy_dir"] + "/**/x_*.wav", recursive=True)
        files = [(f, f.replace("x_", "y_")) for f in files]
        ds = MyDs(files)
        r = 0.8
        return random_split(ds, [int(len(ds) * r), len(ds) - int(len(ds) * r)], generator=torch.Generator().manual_seed(42))


class NpyDs(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.files = glob.glob(self.root + "/**/*.npy", recursive=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return tuple(np.load(self.files[idx]))


def main(conf):
    exp_dir = conf["main_args"]["exp_dir"]
    train_ds, val_ds = getds(conf, target="denoise" if "denoise" in exp_dir else "dereverb")

    if 0:
        VAL_FOLDER = os.path.join(exp_dir, "val-cache")
        if not os.path.exists(VAL_FOLDER):
            os.makedirs(VAL_FOLDER)
            val_ds.save_to_dir = VAL_FOLDER
            val_ds.log = "ds" in exp_dir
            val_loader = DataLoader(
                val_ds,
                shuffle=False,
                batch_size=5,
                num_workers=conf["training"]["num_workers"],
                drop_last=True,
                pin_memory=True,
            )
            for batch in tqdm.tqdm(val_loader):
                pass

        #train_ds = NpyDs(VAL_FOLDER)
        val_ds = NpyDs(VAL_FOLDER)

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
        pin_memory=True,
    )

    # for _ in tqdm.tqdm(train_loader): pass

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
    #optimizer = make_optimizer(model.parameters(), optimizer="adam", **conf["optim"])

    # Define scheduler
    scheduler = None
    if 1 and conf["training"]["half_lr"]:
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
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    # loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    # loss_func = lambda est_target, target: singlesrc_neg_sisdr(est_target.squeeze(1), target).mean()
    #loss_func = MyLoss()
    loss_func = sisdr
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
        #checkpoint_dir, save_top_k=None, verbose=True
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
    ckpt = f"{exp_dir}/last.ckpt"
    if 0 and os.path.exists(ckpt):
        print("*** NOTE: Resuming from last.ckpt")
    else:
        print("*** NOTE: Starting from scratch (no checkpoint)")
        ckpt = None
    trainer = pl.Trainer(
        precision=16,
        # weights_summary='full',
        callbacks=callbacks,
        max_epochs=conf["training"]["epochs"],
        checkpoint_callback=checkpoint,
        early_stop_callback=early_stopping,
        default_root_dir=exp_dir,
        gpus=gpus,
        benchmark=True,
        #distributed_backend="ddp",
        #reload_dataloaders_every_epoch=True,
        val_check_interval=0.3,
        #limit_train_batches=0.01,
        #limit_val_batches=0,#.3,
        gradient_clip_val=conf["training"]["gradient_clipping"],
        resume_from_checkpoint=ckpt,
    )
    with torch.autograd.set_detect_anomaly(not (not 0)):
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
    #pprint(arg_dic)
    main(arg_dic)
