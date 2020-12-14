import os
import argparse
import json

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from asteroid import DPRNNTasNet
from asteroid.data.wham_dataset import (
    PlWhamDataModule,
    WhamConfig,
    TrainConfig,
    Experiment,
    OptimizerConfig,
)
from asteroid.engine.optimizers import make_optimizer
from asteroid.engine.system import System
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp_dir", default="exp/tmp", help="Full path to save best validation model"
)


def mmd_dump(obj):
    return obj.Schema().dump(obj)


def make_trainer(exp, model):
    # TODO: this should be a class so that you can easily change parts

    if exp.train_config.n_checkpoints:
        checkpoint_callback = ModelCheckpoint(
            exp.output_dir.joinpath("checkpoints"),
            monitor="val_loss",
            mode="min",
            save_top_k=exp.train_config.n_checkpoints,
            verbose=1,
        )
    else:
        checkpoint_callback = None

    if exp.train_config.early_stop:
        early_stop_callback = EarlyStopping(
            monitor="val_loss", patience=exp.train_config.early_stop_patience, verbose=1
        )
    else:
        early_stop_callback = None

    # TODO
    # Don't ask GPU if they are not available.
    # gpus = -1 if torch.cuda.is_available() else None

    return pl.Trainer(
        max_epochs=exp.train_config.max_epochs,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
        default_root_dir=exp.output_dir,
        distributed_backend="ddp",
        gradient_clip_val=exp.train_config.gradient_clipping,
    )


def make_system(exp, model):
    # TODO: this should be a class so that you can easily change parts
    # TODO: or it should be part of System


    optimizer = make_optimizer(model.parameters(), 
    **mmd_dump(exp.train_config.optimizer_config))

    if exp.train_config.half_lr:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=30)
    else:
        scheduler = None

    # TODO: make loss configurable from conf
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    return System(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        datamodule=PlWhamDataModule(exp),
        scheduler=scheduler,
    )


def main(conf):
    conf["training"]["batch_size"] = "1"
    # TODO: this should fail
    conf["data"]["mode"] = "a"
    exp = Experiment.Schema().load({
        "train_config": {**conf["training"], "optimizer_config": conf["optim"]},
        "dataset_config": conf["data"],
        "output_dir": conf["main_args"]["exp_dir"],
    })
    print(exp)

    # Update number of source values (It depends on the task)
    # TODO: bring back nondefault
    conf["masknet"].update({"n_src": exp.dataset_config.task_info["default_nsrc"]})

    model = DPRNNTasNet(**conf["filterbank"], **conf["masknet"])
    trainer = make_trainer(exp, model)
    system = make_system(exp, model)

    print(mmd_dump(exp))
    #trainer.fit(system)

    # TODO
    # best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    # with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
    #    json.dump(best_k, f, indent=0)

    ## Save best model (next PL version will make this easier)
    # best_path = [b for b, v in best_k.items() if v == min(best_k.values())][0]
    # state_dict = torch.load(best_path)
    # system.load_state_dict(state_dict=state_dict['state_dict'])
    # system.cpu()

    # to_save = system.model.serialize()
    # to_save.update(train_set.get_infos())
    # torch.save(to_save, os.path.join(exp_dir, 'best_model.pth'))


if __name__ == "__main__":
    import yaml
    from pprint import pprint as print
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
    print(arg_dic)
    main(arg_dic)
