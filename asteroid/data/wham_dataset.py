import abc
import torch
import uuid
from torch.utils import data
import json
import os
import numpy as np
import functools
import soundfile as sf
from .wsj0_mix import wsj0_license
import dataclasses
from pathlib import Path
from typing import List, Union, Literal, Generic, TypeVar, Optional, Any, Type
import pytorch_lightning as pl
import asteroid

PathOrStr = Union[Path, str]

EPS = 1e-8

WHAM_TASKS = {
    "enh_single": {
        "mixture": "mix_single",
        "sources": ["s1"],
        "infos": ["noise"],
        "default_nsrc": 1,
        "uses_wham": True,
    },
    "enh_both": {
        "mixture": "mix_both",
        "sources": ["mix_clean"],
        "infos": ["noise"],
        "default_nsrc": 1,
        "uses_wham": True,
    },
    "sep_clean": {
        "mixture": "mix_clean",
        "sources": ["s1", "s2"],
        "infos": [],
        "default_nsrc": 2,
        "uses_wham": False,
    },
    "sep_noisy": {
        "mixture": "mix_both",
        "sources": ["s1", "s2"],
        "infos": ["noise"],
        "default_nsrc": 2,
        "uses_wham": False,
    },
}


def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    # TODO: move this
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


MixMode = Literal["min", "max"]
Task = Literal["enh_single", "enh_both", "sep_clean", "sep_noisy"]
Stage = Literal["fit", "test"]




@dataclasses.dataclass
class TrainConfig:
    batch_size: int = 42
    # TODO: auto generate these from Trainer.__init__?
    # Include more params?
    max_epochs: int = 200
    num_workers: int = 4
    early_stop: bool = True
    early_stop_patience: int = 30
    gradient_clipping: int = 5
    half_lr: bool = True
    optimizer: str = "adam"
    optimizer_config: Optional[dict] = None
    loss: str = "pairwise_neg_sisdr"
    n_checkpoints: int = 5


@dataclasses.dataclass
class DatasetConfig:
    sample_rate: int = 8000
    normalize_audio: bool = False
    segment: float = 4.0
    mode: MixMode = "min"

    @property
    def segment_len(self):
        # TODO: bring back test mode
        if self.segment is not None:
            return int(self.segment * self.sample_rate)
        else:
            return None


@dataclasses.dataclass
class WhamConfig(DatasetConfig):
    # TODO: remove default values here when https://github.com/python/cpython/pull/17322 is merged
    task: Task = "sep_clean"
    train_dir: Path = Path("")
    valid_dir: Path = Path("")

    def __post_init__(self, *args, **kwargs):
        self.task_info = WHAM_TASKS[self.task]


D = TypeVar("D", bound=DatasetConfig)

@dataclasses.dataclass
class ModelConfig:
    pass


M = TypeVar("M", bound=ModelConfig)


class Model(abc.ABC, Generic[M]):
    @abc.abstractmethod
    def __init__(self, config: M):
        pass

class Dataset(abc.ABC, data.Dataset, Generic[D]):
    @abc.abstractmethod
    def __init__(self, config: D, stage: Stage):
        pass


@dataclasses.dataclass
class ExperimentConfig(Generic[D, M]):
    tag: str
    train_config: TrainConfig
    dataset: Type[Dataset[D]]
    dataset_config: D
    model: Type[Model[M]]
    model_config: M

    def __post_init__(self, *args, **kwargs):
        # TODO: use dataclasses.field(...) for tag once https://github.com/python/cpython/pull/17322 is merged
        if not self.tag:
            # TODO: change to run.sh ID generator
            self.tag = uuid.uuid4().hex





class WhamDataset(Dataset[WhamConfig]):
    """Dataset class for WHAM source separation and speech enhancement tasks."""

    dataset_name = "WHAM!"

    def __init__(self, config: WhamConfig, stage: Stage):
        # TODO: bring nach nondefault_nsrc
        # TODO: do away with for_train
        super(WhamDataset, self).__init__()
        self.config = config

        data_dir = (
            self.config.train_dir if self.stage == "fit" else self.config.valid_dir
        )
        mixture_info = json.load(data_dir.joinpath(["mixture"] + ".json").open())
        source_infos = [
            json.load(data_dir.joinpath(source + ".json").open())
            for source in self.config.task_info["sources"]
        ]
        # Handle the case n_src > default_nsrc
        # while len(sources_infos) < self.n_src:
        #    sources_infos.append([None for _ in range(len(self.mix))])

        self.utterances = [
            (n_samples, mix_path, [src_path for src_path, _ in sources])
            for (mix_path, n_samples), *sources in zip(mixture_info, *source_infos)
            if n_samples >= self.config.segment_len
        ]

        drop_len = len(mixture_info) - len(self.utterances)
        dropped_hours = (
            sum(
                n_samples
                for _, n_samples in mixture_info
                if n_samples < self.config.segment_len
            )
            / self.config.sample_rate
            / 3600
        )
        print(
            f"Dropping {drop_len}/{len(mixture_info)} utts ({dropped_hours:.2f} h) that are shorter than {self.config.segment_len} samples"
        )

    @property
    def n_src(self):
        return len(self.utterances[0][2])

    """TODO: this is broken, __add__ shouldn't mutate
    def __add__(self, wham):
        if self.n_src != wham.n_src:
            raise ValueError('Only datasets having the same number of sources'
                             'can be added together. Received '
                             '{} and {}'.format(self.n_src, wham.n_src))
        if self.seg_len != wham.seg_len:
            self.seg_len = min(self.seg_len, wham.seg_len)
            print('Segment length mismatched between the two Dataset'
                  'passed one the smallest to the sum.')
        self.mix = self.mix + wham.mix
        self.sources = [a + b for a, b in zip(self.sources, wham.sources)]
    """

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        """ Gets a mixture/sources pair.
        Returns:
            mixture, vstack([source_arrays])
        """
        # Random start
        # TODO: bring back test mode
        n_samples, mix_path, src_paths = self.utterances[idx]
        rand_start = np.random.randint(0, max(1, n_samples - self.config.segment_len))
        stop = rand_start + self.config.segment_len

        if 0:
            # Load mixture
            mixture, _ = sf.read(mix_path, start=rand_start, stop=stop, dtype="float32")
            # Load sources
            sources = [
                sf.read(src_path, start=rand_start, stop=stop, dtype="float32")[0]
                for src_path in src_paths
            ]
        else:
            mixture = np.zeros((self.config.segment_len,))
            sources = [np.zeros_like(mixture) for _ in src_paths]
        # if src[idx] is None:
        #    # Target is filled with zeros if n_src > default_nsrc
        #    s = np.zeros((seg_len, ))

        mixture_tensor = torch.from_numpy(mixture)
        source_tensors = torch.from_numpy(np.vstack(sources))

        if self.config.normalize_audio:
            m_std = mixture_tensor.std(-1, keepdim=True)
            return (
                normalize_tensor_wav(mixture_tensors, eps=EPS, std=m_std),
                normalize_tensor_wav(source_tensors, eps=EPS, std=m_std),
            )
        else:
            return mixture_tensor, source_tensors

    def get_infos(self):
        """ Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        return {
            "dataset": self.dataset_name,
            "config": self.config,
            "licenses": [wsj0_license]
            + ([wham_noise_license] if self.config.task_info["uses_wham"] else []),
        }

class Experiment: pass

class PlWhamDataModule(pl.LightningDataModule):
    def __init__(self, experiment: Experiment):
        super().__init__()
        self.experiment = experiment

    def train_dataloader(self):
        return DataLoader(
            WhamDataset(self.experiment.dataset_config, "fit"),
            shuffle=True,
            batch_size=self.experiment.train_config.batch_size,
            num_workers=self.experiment.train_config.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            WhamDataset(self.experiment.dataset_config, "test"),
            batch_size=self.experiment.train_config.batch_size,
            num_workers=self.experiment.train_config.num_workers,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            WhamDataset(self.experiment.dataset_config, "test"),
            batch_size=self.experiment.train_config.batch_size,
            num_workers=self.experiment.train_config.num_workers,
            drop_last=True,
        )


wham_noise_license = dict(
    title="The WSJ0 Hipster Ambient Mixtures dataset",
    title_link="http://wham.whisper.ai/",
    author="Whisper.ai",
    author_link="https://whisper.ai/",
    license="CC BY-NC 4.0",
    license_link="https://creativecommons.org/licenses/by-nc/4.0/",
    non_commercial=True,
)




@dataclasses.dataclass
class ConvTasNetConfig:
    # TODO: separate enc/dec, masknn params here?
    n_src: int = 2
    out_chan: Optional[int] = None
    n_blocks: int = 8
    n_repeats: int = 3
    bn_chan: int = 128
    hid_chan: int = 512
    skip_chan: int = 128
    conv_kernel_size: int = 3
    norm_type: str = "gLN"
    mask_act: str = 'relu'
    in_chan: Optional[int] = None
    fb_name: str = 'free'
    kernel_size: int =16
    n_filters: int = 512
    stride: int = 8



class ConvTasNet(Model[ConvTasNetConfig]):
    def __init__(self, config: M):
        pass
