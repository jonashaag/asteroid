import torch
from torch.utils import data
import json
import os
import soundfile as sf
import random


class VBDDataset(data.Dataset):
    dataset_name = "VBD"

    def __init__(self, clean_root, noisy_root, speakers_subset=None, **kwargs):
        self.kwargs = kwargs
        self.clean_root = clean_root
        self.noisy_root = noisy_root
        self.speakers_subset = speakers_subset
        self.clean_files = [(f, f.split("_")[0]) for f in os.listdir(clean_root)]
        self.noisy_files = [(f, f.split("_")[0]) for f in os.listdir(noisy_root)]

        if speakers_subset is not None:
            self.clean_files = [
                (f, s) for f, s in self.clean_files if s in speakers_subset
            ]
            self.noisy_files = [
                (f, s) for f, s in self.noisy_files if s in speakers_subset
            ]

        if set(self.clean_files) != set(self.noisy_files):
            raise FileNotFoundError(
                f"Some utterances are missing from clean or noisy: {set(self.clean_files) ^ set(self.noisy_files)}"
            )

        files_pairs = [
            (os.path.join(clean_root, clean), [os.path.join(noisy_root, noisy)])
            for (clean, _), (noisy, _) in zip(self.clean_files, self.noisy_files)
        ]
        super().__init__(files_pairs, **kwargs)

    def get_speakers_split(self, ratio, random_seed=0):
        speakers = list(set(s for _, s in self.clean_files))
        speakers.sort()
        if random_seed is not None:
            random.seed(random_seed)
        split1_idxs = set(
            random.sample(range(len(speakers)), int(len(speakers) * ratio))
        )
        split1_speakers = [s for s, idx in enumerate(speakers) if idx in split1_idxs]
        split2_speakers = [
            s for s, idx in enumerate(speakers) if idx not in split1_idxs
        ]
        split1 = self.__class__(
            self.clean_root, self.noisy_root, split1_speakers, **self.kwargs
        )
        split2 = self.__class__(
            self.clean_root, self.noisy_root, split2_speakers, **self.kwargs
        )
        return split1, split2

    def get_train_val_split(self):
        # This is just one idea on how to do a train/val split.
        return self.get_speakers_split(ratio=0.9)

    def get_infos(self):
        """ Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        return {
            "dataset": self.dataset_name,
            "task": "enhancement",
            "licenses": [vbd_license],
        }


vbd_license = dict(
    title="Noisy speech database for training speech enhancement algorithms and TTS models",
    title_link="https://datashare.is.ed.ac.uk/handle/10283/2791",
    author="Valentini-Botinhao, Cassia",
    license="CC BY 4.0",
    license_link="https://creativecommons.org/licenses/by/4.0/",
    non_commercial=False,
)
