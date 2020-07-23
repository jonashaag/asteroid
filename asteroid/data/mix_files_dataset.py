import torch
import numpy as np
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

    def __init__(self, file_pairs, segment=None, sr=None, resample_args=None):
        if segment is None:
            seg_len = None
        else:
            if sr is None:
                raise ValueError("Must pass 'sr' if 'segment' is not None")
            seg_len = int(segment * sr)
            ok_file_pairs = []
            for mixture, sources in file_pairs:
                if self._quick_get_file_len(mixture) >= seg_len:
                    ok_file_pairs.append((mixture, sources))
            print(f"Dropped {len(file_pairs) - len(ok_file_pairs)}/{len(file_pairs)} utts that are too short")

        self.file_pairs = ok_file_pairs
        self.sr = sr
        self.resample_args = resample_args
        self.seg_len = seg_len

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        """ Gets a mixture/sources pair.
        Returns:
            mixture, vstack([source_arrays])
        """
        mixture, sources = self.file_pairs[idx]
        mixture = self._load_file(mixture)
        sources = [self._load_file(s) for s in sources]
        assert {len(mixture)} == set(map(len, sources)), self.file_pairs[idx]

        if self.seg_len is not None:
            rand_start = np.random.randint(0, max(1, len(mixture) - self.seg_len))
            stop = rand_start + self.seg_len
            mixture = mixture[rand_start:stop]
            sources = [s[rand_start:stop] for s in sources]

        return torch.from_numpy(mixture), torch.stack([torch.from_numpy(s) for s in sources])

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

    def _quick_get_file_len(self, filename):
        with sf.SoundFile(filename) as f:
            return f.frames
