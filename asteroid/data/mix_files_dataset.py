import torch
import soundfile as sf

try:
    import librosa
except ImportError:
    librosa = None


class MixFilesDataset(torch.utils.data.Dataset):
    """Base class for datasets that read soundfile compatible files for each source.

     Args:
        file_pairs (sequence of (mix: str, sources: list of str) tuples): Source files.
        sr (int, optional): Target sample rate. If specified, source files are
            resampled to the given sample rate using `librosa.core.resample`.
        resample_args (dict, optional): Arguments passed to `librosa.core.resample`.
    """

    def __init__(self, file_pairs, sr=None, resample_args=None):
        self.file_pairs = file_pairs
        self.sr = sr
        self.resample_args = resample_args

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        """ Gets a mixture/sources pair.
        Returns:
            mixture, vstack([source_arrays])
        """
        mixture, sources = self.file_pairs[self.wav_ids[idx]]
        mixture = torch.from_numpy(self._load_file(mixture))
        sources = torch.stack(
            [torch.from_numpy(self._load_file(source)) for source in sources]
        )
        return mixture, sources

    def _load_file(self, filename):
        data, sr = sf.read(filename, dtype="float32")
        if self.sr is not None and sr != self.sr:
            if librosa is None:
                raise ImportError("Resampling requires 'librosa'")
            else:
                data = librosa.core.resample(
                    data, sr, self.sr, **(self.resample_args or {})
                )
        return data
