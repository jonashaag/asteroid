from pathlib import Path
import yaml
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import asteroid.engine.optimizers
import dataclasses
import asteroid.losses
from asteroid.engine.system import System
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.data.wham_dataset import *
import asteroid.dataclasses


class GenericModule(pl.LightningModule):
    def __init__(self, model, loss, optimizers, schedulers, callbacks=None, metrics=None):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizers = optimizeres
        self.schedulers = schedulers
        self.callbacks = callbacks or []
        self.metrics = metrics or {}

    def configure_optimizers(self):
        return self.optimizers, self.schedulers

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def common_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        metrics = self.compute_metrics(batch, batch_idx, y_hat)
        return {"loss": loss, "metrics": metrics}

    def training_step(self, batch, batch_idx):
        return self.train_result(self.common_step(batch, batch_idx))

    def validation_step(self, batch, batch_idx):
        return self.eval_result(self.common_step(batch, batch_idx))

    def train_result(self, step):
        result = pl.TrainResult(minimize=step["loss"])
        for k, v in step["metrics"]:
            result.log(k, v)
        return result

    def eval_result(self, step):
        result = pl.EvalResult()
        result.log("loss", step["loss"])
        for k, v in step["metrics"]:
            result.log(k, v)
        return result



def make_checkpoint_callback(n_checkpoints):
    if n_checkpoints:
        return ModelCheckpoint(
            exp.output_dir.joinpath("checkpoints"),
            monitor="val_loss",
            mode="min",
            save_top_k=exp.train_config.n_checkpoints,
            verbose=1,
        )
def make_early_stop_callback(early_stop_patience):
    if early_stop_patience:
        return EarlyStopping(
            monitor="val_loss", patience=exp.train_config.early_stop_patience, verbose=1
        )

def make_optimizer(exp, model):
    optimizer_cls = asteroid.engine.optimizers.get(exp.train_config.optimizer)
    return optimizer_cls(**exp.train_config.optimizer_config)


def make_scheduler(exp, optimizer):
    if exp.train_config.half_lr:
        return ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=30)


def make_loss(exp):
    # TODO: implement asteroid.losses.get()
    loss_func = getattr(asteroid.losses, exp.train_config.loss)
    # TODO: PIT configurable
    loss_func = PITLossWrapper(loss_func, pit_from="pw_mtx")
    return loss_func


def rmnone(x):
    return [a for a in x if x is not None]


def make_module(exp, model):
    optimizers = make_optimizers(exp, model)
    schedulers = make_schedulers(exp, optimizer)
    # TODO: implement metrics
    metrics = []
    loss = make_loss(exp)
    return GenericModule(model, loss, optimizers, schedulers, callbacks, metrics)


def make_trainer():
    # TODO
    # Don't ask GPU if they are not available.
    # gpus = -1 if torch.cuda.is_available() else None
    return pl.Trainer(
        checkpoint_callback=make_checkpoint_callback(exp),
        early_stop_callback=make_early_stop_callback(exp),

        distributed_backend="ddp",
        max_epochs=exp.train_config.max_epochs,
        default_root_dir=exp.output_dir,
        gradient_clip_val=exp.train_config.gradient_clipping,
    )


import typer

app = typer.Typer()

# TODO replace this with global registry
datasets = {"asteroid.data.wham": (WhamDataset, WhamConfig)}
models = {"asteroid.model.convtasnet": (ConvTasNet, ConvTasNetConfig)}


def model_config_from_yaml(dct):
    return models[dct["name"]][0], asteroid.dataclasses.unserialize_dataclass(
        models[dct["name"]][1], dct["conf"])


def dataset_config_from_yaml(dct):
    return datasets[dct["name"]][0], asteroid.dataclasses.unserialize_dataclass(
        datasets[dct["name"]][1], dct["conf"])


def exp_config_from_yaml(dct):
    return asteroid.dataclasses.unserialize_dataclass(Experiment, dct)


@app.command("data-mkconfig")
def data_mkconfig(dataset: str):
    # TODO: implement other dataset and override args
    print(yaml.safe_dump({
            "name": dataset,
            "conf": asteroid.dataclasses.serialize_dataclass(datasets[dataset][1]()),
    }))

@app.command("model-mkconfig")
def data_mkconfig(model: str):
    # TODO: implement other dataset and override args
    print(yaml.safe_dump({
            "name": model,
            "conf": asteroid.dataclasses.serialize_dataclass(models[model][1]()),
    }))

@app.command("train-mkexperimentconfig")
def train_mkexperiment(model_config: Path, dataset_config: Path):
    model_cls, model_config = model_config_from_yaml(yaml.safe_load(model_config.open()))
    dataset_cls, dataset_config = dataset_config_from_yaml(yaml.safe_load(dataset_config.open()))
    # TODO: implement other dataset and override args
    print(yaml.safe_dump(asteroid.dataclasses.serialize_dataclass(ExperimentConfig(
        "",
        TrainConfig(),
        dataset_cls, dataset_config,
        model_cls, model_config,
    ))))


@app.command("train")
def train(exp_config: Path):
    exp = exp_config_from_yaml(yaml.safe_load(exp_config.open()))
    print(exp)



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
