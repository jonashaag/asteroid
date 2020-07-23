import torch
from torch.utils import data
import json
import os
import soundfile as sf
from .mix_files_dataset import MixFilesDataset


class DNSDataset(MixFilesDataset):
    dataset_name = "DNS"

    def __init__(self, json_dir):
        with open(os.path.join(json_dir, "file_infos.json")) as f:
            mix_infos = json.load(f)

        files_pairs = [
            (utt_info["mix"], [utt_info["clean"], utt_info["noise"]])
            for utt_info in mix_infos
        ]

        super().__init__(files_pairs)

    def get_infos(self):
        """ Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        return {
            "dataset": self.dataset_name,
            "task": "enhancement",
            "licenses": [dns_license],
        }


dns_license = dict(
    title="Deep Noise Suppression (DNS) Challenge",
    title_link="https://github.com/microsoft/DNS-Challenge",
    author="Microsoft",
    author_link="https://www.microsoft.com/fr-fr/",
    license="CC BY 4.0",
    license_link="https://creativecommons.org/licenses/by/4.0/",
    non_commercial=False,
)
