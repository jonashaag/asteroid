import torch
import itertools
import glob
import subprocess
import librosa
import numpy as np
import scipy
import tqdm.contrib.concurrent
import itu_r_468_weighting.filter
import fastalign


def configure(len_s, real_irs_path, dns_irs_path, dns_noise_path, dns_clean_path):
    global len_samples, real_rirs_files, dns_irs_files, dns_noise_files, dns_clean_files

    len_samples = int(len_s * 16000)

    real_rirs_files = glob.glob(real_irs_path + "/**/*.wav", recursive=True)
    assert len(real_rirs_files) > 10000
    dns_irs_files = glob.glob(dns_irs_path + "/**/*.wav", recursive=True)
    assert len(dns_irs_files) > 10000
    dns_noise_files = glob.glob(dns_noise_path + "/**/*.wav", recursive=True)
    assert len(dns_noise_files) > 10000
    dns_clean_files = glob.glob(dns_clean_path + "/**/*.wav", recursive=True)
    assert len(dns_clean_files) > 10000


def deterministic_shuffle(x):
    np.random.default_rng(4242).shuffle(x)


def deterministic_sample(x, n):
    idxs = np.random.default_rng(4242).integers(0, len(x), size=n)
    return [x[i] for i in idxs]


def groupby(it, key):
    bykey = {}
    for x, k in zip(it, tqdm.contrib.concurrent.process_map(key, it, chunksize=1000)):
        bykey.setdefault(k, []).append(x)
    return bykey


def make_split(it, key, splits):
    assert 0.999 <= sum(splits) <= 1.001

    it = it[:]
    it.sort()
    deterministic_shuffle(it)

    bykey = groupby(it, key)
    keys_list = list(bykey.keys())
    keys_list.sort()
    deterministic_shuffle(keys_list)

    splits = [int(len(keys_list) * r) for r in splits[:-1]] + [len(keys_list)]
    for nkeys in splits:
        this_keys = keys_list[:nkeys]
        yield this_keys[:], list(itertools.chain(*[bykey[k] for k in this_keys]))
        keys_list = keys_list[nkeys:]


def grouper(f):
    if "_reader_" in f: return f.split("_reader_")[1].split("_")[0]
    if "AI" in f and "male/" in f: return f.split("male/")[1].split("/")[0]
    if "FULL/" in f: return f.split("FULL/")[1].split("/")[0]
    return str((hash(f) % 313) % 10)


def make_ds():
    (train_spks, train_clean), (val_spks, val_clean) = make_split(
        dns_clean_files, grouper, [0.95, 0.05])
    print(f"Train: # spks: {len(train_spks)}, # elems: {len(train_clean)}")
    print(f"Val: # spks: {len(val_spks)}, # elems: {len(val_clean)}")
    return MyDs(train_clean, deterministic=False), MyDs(val_clean, deterministic=False)


class MyDs(torch.utils.data.Dataset):
    def __init__(self, clean_files, deterministic):
        self.clean_files = clean_files
        self.deterministic = deterministic

    def __len__(self):
        return len(self.clean_files)


    def __getitem__(self, idx):
        return self.getitem(idx)

    def getitem(self, idx, torch_seed=None, log=False):
        if torch_seed is None:
            if self.deterministic:
                torch_seed = idx
            else:
                torch_seed = torch.seed()
        rand = np.random.default_rng(torch_seed)
        while True:
            try:
                clean_f = np_choice(rand, self.clean_files)
                x, _, _, _, y = randmix(rand, clean_f, log=log)
                return x.astype("float32"), y.astype("float32")
            except Exception as err:
                import traceback; traceback.print_stack()
                print(f"Error generating sample for {idx} (det: {self.deterministic}, torch seed: {torch_seed}): {err}")


# -----
# From https://github.com/rclement/yodel/blob/master/yodel/filter.py
import math

class Biquad:
    """
    A biquad filter is a 2-poles/2-zeros filter allowing to perform
    various kind of filtering. Signal attenuation is at a rate of 12 dB
    per octave.
    *Reference:*
        "Cookbook formulae for audio EQ biquad filter coefficients",
        Robert Bristow-Johnson
        (http://www.musicdsp.org/files/Audio-EQ-Cookbook.txt)
    """

    def __init__(self):
        """
        Create an inactive biquad filter with a flat frequency response.
        To make the filter active, use one of the provided methods:
        :py:meth:`low_pass`, :py:meth:`high_pass`, :py:meth:`band_pass`,
        :py:meth:`all_pass`, :py:meth:`notch`, :py:meth:`peak`,
        :py:meth:`low_shelf`, :py:meth:`high_shelf` and :py:meth:`custom`.
        """
        self.reset()

    def reset(self):
        """
        Make the filter inactive with a flat frequency response.
        """
        self._a_coeffs0 = 0.0
        self._a_coeffs1 = 0.0
        self._a_coeffs2 = 0.0
        self._b_coeffs0 = 1.0
        self._b_coeffs1 = 0.0
        self._b_coeffs2 = 0.0
        self._x1 = 0.0
        self._x2 = 0.0
        self._y1 = 0.0
        self._y2 = 0.0

    def low_pass(self, samplerate, cutoff, resonance):
        """
        Make a low-pass filter.
        :param samplerate: sample-rate in Hz
        :param cutoff: cut-off frequency in Hz
        :param resonance: resonance or Q-factor
        """
        self._compute_constants(samplerate, cutoff, resonance)
        self._a_coeffs0 = (1.0 + self._alpha)
        self._a_coeffs1 = (-2.0 * self._cos_w0) / self._a_coeffs0
        self._a_coeffs2 = (1.0 - self._alpha) / self._a_coeffs0
        self._b_coeffs0 = ((1.0 - self._cos_w0) / 2.0) / self._a_coeffs0
        self._b_coeffs1 = (1.0 - self._cos_w0) / self._a_coeffs0
        self._b_coeffs2 = ((1.0 - self._cos_w0) / 2.0) / self._a_coeffs0

    def high_pass(self, samplerate, cutoff, resonance):
        """
        Make a high-pass filter.
        :param samplerate: sample-rate in Hz
        :param cutoff: cut-off frequency in Hz
        :param resonance: resonance or Q-factor
        """
        self._compute_constants(samplerate, cutoff, resonance)
        self._a_coeffs0 = (1.0 + self._alpha)
        self._a_coeffs1 = (-2.0 * self._cos_w0) / self._a_coeffs0
        self._a_coeffs2 = (1.0 - self._alpha) / self._a_coeffs0
        self._b_coeffs0 = ((1.0 + self._cos_w0) / 2.0) / self._a_coeffs0
        self._b_coeffs1 = -(1.0 + self._cos_w0) / self._a_coeffs0
        self._b_coeffs2 = ((1.0 + self._cos_w0) / 2.0) / self._a_coeffs0

    def band_pass(self, samplerate, center, resonance):
        """
        Make a band-pass filter.
        :param samplerate: sample-rate in Hz
        :param center: center frequency in Hz
        :param resonance: resonance or Q-factor
        """
        self._compute_constants(samplerate, center, resonance)
        self._a_coeffs0 = (1.0 + self._alpha)
        self._a_coeffs1 = (-2.0 * self._cos_w0) / self._a_coeffs0
        self._a_coeffs2 = (1.0 - self._alpha) / self._a_coeffs0
        self._b_coeffs0 = (self._alpha) / self._a_coeffs0
        self._b_coeffs1 = 0
        self._b_coeffs2 = (- self._alpha) / self._a_coeffs0

    def all_pass(self, samplerate, center, resonance):
        """
        Make an all-pass filter.
        :param samplerate: sample-rate in Hz
        :param center: center frequency in Hz
        :param resonance: resonance or Q-factor
        """
        self._compute_constants(samplerate, center, resonance)
        self._a_coeffs0 = (1.0 + self._alpha)
        self._a_coeffs1 = (-2.0 * self._cos_w0) / self._a_coeffs0
        self._a_coeffs2 = (1.0 - self._alpha) / self._a_coeffs0
        self._b_coeffs0 = (1.0 - self._alpha) / self._a_coeffs0
        self._b_coeffs1 = (-2.0 * self._cos_w0) / self._a_coeffs0
        self._b_coeffs2 = (1.0 + self._alpha) / self._a_coeffs0

    def notch(self, samplerate, center, resonance):
        """
        Make a notch filter.
        :param samplerate: sample-rate in Hz
        :param center: center frequency in Hz
        :param resonance: resonance or Q-factor
        """
        self._compute_constants(samplerate, center, resonance)
        self._a_coeffs0 = (1.0 + self._alpha)
        self._a_coeffs1 = (-2.0 * self._cos_w0) / self._a_coeffs0
        self._a_coeffs2 = (1.0 - self._alpha) / self._a_coeffs0
        self._b_coeffs0 = (1.0) / self._a_coeffs0
        self._b_coeffs1 = (-2.0 * self._cos_w0) / self._a_coeffs0
        self._b_coeffs2 = (1.0) / self._a_coeffs0

    def peak(self, samplerate, center, resonance, dbgain):
        """
        Make a peak filter.
        :param samplerate: sample-rate in Hz
        :param center: center frequency in Hz
        :param resonance: resonance or Q-factor
        :param dbgain: gain in dB
        """
        self._compute_constants(samplerate, center, resonance, dbgain)
        self._a_coeffs0 = (1.0 + self._alpha / self._a)
        self._a_coeffs1 = (-2.0 * self._cos_w0) / self._a_coeffs0
        self._a_coeffs2 = (1.0 - self._alpha / self._a) / self._a_coeffs0
        self._b_coeffs0 = (1.0 + self._alpha * self._a) / self._a_coeffs0
        self._b_coeffs1 = (-2.0 * self._cos_w0) / self._a_coeffs0
        self._b_coeffs2 = (1.0 - self._alpha * self._a) / self._a_coeffs0

    def low_shelf(self, samplerate, cutoff, resonance, dbgain):
        """
        Make a low-shelf filter.
        :param samplerate: sample-rate in Hz
        :param cutoff: cut-off frequency in Hz
        :param resonance: resonance or Q-factor
        :param dbgain: gain in dB
        """
        self._compute_constants(samplerate, cutoff, resonance, dbgain)
        self._a_coeffs0 = ((self._a + 1) +
                             (self._a - 1) * self._cos_w0 + self._sqrtAlpha)
        self._a_coeffs1 = ((-2.0 * ((self._a - 1) +
                                      (self._a + 1) * self._cos_w0)) /
                             self._a_coeffs0)
        self._a_coeffs2 = (((self._a + 1) +
                              (self._a - 1) * self._cos_w0 - self._sqrtAlpha) /
                             self._a_coeffs0)
        self._b_coeffs0 = ((self._a *
                              ((self._a + 1) -
                               (self._a - 1) * self._cos_w0 +
                               self._sqrtAlpha)) /
                             self._a_coeffs0)
        self._b_coeffs1 = ((2.0 * self._a *
                              ((self._a - 1) - (self._a + 1) * self._cos_w0)) /
                             self._a_coeffs0)
        self._b_coeffs2 = ((self._a *
                              ((self._a + 1) -
                               (self._a - 1) * self._cos_w0 -
                               self._sqrtAlpha)) /
                             self._a_coeffs0)

    def high_shelf(self, samplerate, cutoff, resonance, dbgain):
        """
        Make a high-shelf filter.
        :param samplerate: sample-rate in Hz
        :param cutoff: cut-off frequency in Hz
        :param resonance: resonance or Q-factor
        :param dbgain: gain in dB
        """
        self._compute_constants(samplerate, cutoff, resonance, dbgain)
        self._a_coeffs0 = ((self._a + 1) -
                             (self._a - 1) * self._cos_w0 + self._sqrtAlpha)
        self._a_coeffs1 = ((2.0 *
                              ((self._a - 1) - (self._a + 1) * self._cos_w0)) /
                             self._a_coeffs0)
        self._a_coeffs2 = (((self._a + 1) -
                              (self._a - 1) * self._cos_w0 -
                              self._sqrtAlpha) /
                             self._a_coeffs0)
        self._b_coeffs0 = ((self._a *
                              ((self._a + 1) +
                               (self._a - 1) * self._cos_w0 +
                               self._sqrtAlpha)) /
                             self._a_coeffs0)
        self._b_coeffs1 = ((-2.0 * self._a *
                              ((self._a - 1) + (self._a + 1) * self._cos_w0)) /
                             self._a_coeffs0)
        self._b_coeffs2 = ((self._a *
                              ((self._a + 1) +
                               (self._a - 1) * self._cos_w0 -
                               self._sqrtAlpha)) /
                             self._a_coeffs0)

    def custom(self, a0, a1, a2, b0, b1, b2):
        """
        Make a custom filter.
        :param a0: a[0] coefficient
        :param a1: a[1] coefficient
        :param a2: a[2] coefficient
        :param b0: b[0] coefficient
        :param b1: b[1] coefficient
        :param b2: b[2] coefficient
        """
        self._a_coeffs0 = a0
        self._a_coeffs1 = a1 / a0
        self._a_coeffs2 = a2 / a0
        self._b_coeffs0 = b0 / a0
        self._b_coeffs1 = b1 / a0
        self._b_coeffs2 = b2 / a0

    def process_sample(self, x):
        """
        Filter a single sample and return the filtered sample.
        :param x: input sample
        :rtype: filtered sample
        """
        curr = x
        y = (self._b_coeffs0 * x +
             self._b_coeffs1 * self._x1 +
             self._b_coeffs2 * self._x2 -
             self._a_coeffs1 * self._y1 -
             self._a_coeffs2 * self._y2)
        self._x2 = self._x1
        self._x1 = curr
        self._y2 = self._y1
        self._y1 = y
        return y

    def process(self, x, y):
        """
        Filter an input signal. Can be used for in-place filtering.
        :param x: input buffer
        :param y: output buffer
        """
        num_samples = len(x)
        for n in range(0, num_samples):
            y[n] = self.process_sample(x[n])

    def _compute_constants(self, fs, fc, q, dbgain=0):
        """
        Pre-compute internal mathematical constants
        """
        self._fc = fc
        self._q = q
        self._dbgain = dbgain
        self._fs = fs
        self._a = math.pow(10, (self._dbgain / 40.0))
        self._w0 = 2.0 * math.pi * self._fc / self._fs
        self._cos_w0 = math.cos(self._w0)
        self._sin_w0 = math.sin(self._w0)
        self._alpha = self._sin_w0 / (2.0 * self._q)
        self._sqrtAlpha = 2.0 * math.sqrt(self._a) * self._alpha


import numba

NumbaBiquad = numba.experimental.jitclass([
    (f, numba.float64) for f in [f"_{x}_coeffs{i}" for x in "ab" for i in range(3)] + """
_fc
_q
_dbgain
_fs
_a
_w0
_cos_w0
_sin_w0
_alpha
_sqrtAlpha
_x1
_x2
_y1
_y2""".strip().splitlines()
])(Biquad)
# -----


def itu_r_468_weighted(spec, n_fft, sr):
    return spec * np.array([
        itu_r_468_weighting.filter.r468(f, "1khz", "factor")
        for f in librosa.fft_frequencies(sr, n_fft)])[:, None]


def itu_r_468_weighted_torch(spec, n_fft, sr):
    return spec * torch.tensor([
        itu_r_468_weighting.filter.r468(f, "1khz", "factor")
        for f in librosa.fft_frequencies(sr, n_fft)], device=spec.device)[None, None]


def rand_shelv(rand, sr, min_cutoff, max_cutoff, min_q, max_q, min_g, max_g, t, data):
    f = rand.uniform(min_cutoff, max_cutoff)
    q = rand.uniform(min_q, max_q)
    g = rand.uniform(min_g, max_g)
    filt = NumbaBiquad()
    getattr(filt, t)(sr, f, q, g)
    out = np.zeros_like(data)
    filt.process(data, out)
    return out


def rand_biquad(rand, sr, min_cf, max_cf, min_q, max_q, data):
    filt = NumbaBiquad()
    filt.band_pass(sr, rand.uniform(min_cf, max_cf), rand.uniform(min_q, max_q))
    out = np.zeros_like(data)
    filt.process(data, out)
    return rand.uniform(1, 5) * (out + 1e-10) + data


def rand_lowpass(rand, sr, min_cutoff, max_cutoff, min_q, max_q, data):
    f = rand.uniform(min_cutoff, max_cutoff)
    q = rand.uniform(min_q, max_q)
    filt = NumbaBiquad()
    filt.low_pass(sr, f, q)
    out = np.zeros_like(data)
    filt.process(data, out)
    return out


def rand_pitch_shift(rand, sr, min_speedup, max_speedup, data, fast_ok=False):
    return librosa.resample(
        data, sr, int(rand.uniform(min_speedup, max_speedup) * sr),
        res_type="kaiser_fast" if fast_ok else "kaiser_best"
    )


def rand_eq_pitch(rand, data, always_eq, enable_biquad, enable_lowpass, fast_ok=False, log=False):
    if always_eq or np_proba(rand, 1/2):
        if enable_biquad and np_proba(rand, 1/3):
            if log: print("biqad")
            data = rand_biquad(rand, 16000,0,8000,0.5,1.5,data)
        else:
            if log: print("shelv")
            t = np_choice(rand, ("low_shelf", "high_shelf"))
            data = rand_shelv(
                rand, 16000,
                0,
                4000 if t == "low_shelf" else 8000,  # never attenuate high frequencies
                0.5, 1.5,
                -15 if t == "low_shelf" else 0,  # never attenuate high frequencies
                15, t, data)
    if np_proba(rand, 1/2):
        if log: print("pitch")
        data = rand_pitch_shift(rand, 16000, 0.95, 1.05, data, fast_ok=fast_ok)
    # Never attenuate high frequencies.
    # This is also partly covered by piping through codecs.
    #if np_proba(rand, 3/100):
    #    if log: print("lowpass")
    #    with ti("lowpass"):
    #        data = rand_lowpass(rand, 16000, 4000, 7000, 0.5, 1.5, data)
    return data


def rand_clipping(rand, min_percentile, max_percentile, *data):
    p = rand.uniform(min_percentile, max_percentile)
    for d in data:
        percentile = np.percentile(
            np.abs(d), int(100 * p))
        d = d.copy()
        d[np.where(d > percentile)] = percentile
        d[np.where(d < -percentile)] = -percentile
        yield d


def loadrandir(rand, sr, augment, n, log=False):
    if np_proba(rand, 1/10):
        ir_f = np_choice(rand, dns_irs_files)
    else:
        ir_f = np_choice(rand, real_rirs_files)
    if log: print(ir_f)
    ir, sr_ = librosa.load(ir_f, sr=None, duration=3, dtype="float64")
    assert sr_ == sr, (ir_f, sr_, sr)
    ir = fast_trim_zeros(ir)

    if augment:
        if np_proba(rand, 1/2):
            if log: print("Speedup IR")
            ir = librosa.resample(ir, sr, int(rand.uniform(0.8, 1.2) * sr))

    i = safe_peaknorm(ir)
    ir = np.pad(ir, (int(0.0025 * sr), 0))
    skip = int(np.abs(ir).argmax() - 0.0025 * sr)
    assert skip >= 0, ("skip", skip)
    ir = ir[skip:]

    if augment:
        #if log: print("IR len", len(ir)/sr)
        if np_proba(rand, 1/3):
            if log: print("Decay IR")
            decay_start = int(rand.normal(1, 0.2) * sr)
            decay_len = int(rand.uniform(0.005, 0.1) * sr)
            if len(ir) >= decay_start + decay_len:
                if log: print("Decay IR from/len", decay_start/sr, decay_len/sr)
                decay_slope = rand.integers(3, 10)
                decay = np.exp(-np.linspace(0, decay_slope, decay_len))
                ir = ir[:decay_start+decay_len]
                ir[decay_start:] *= decay
        irs = rand_boost_ir(rand, n=n, ir=ir)
    else:
        irs = [ir for _ in range(n)]
    return irs


def convolve_direct(sr, data, ir):
    direct_path = ir[:int(0.0075 * sr)]
    return scipy.signal.convolve(data, direct_path, "valid")


def rand_boost_ir(rand, n, ir):
    irs = []
    for i in range(n):
        ir2 = ir.copy()
        ir2[np.abs(ir2).argmax()] *= rand.uniform(1, 10)
        irs.append(ir2)
    return irs


def crop_or_pad(rand, target_len, *data):
    assert all(len(d) == len(data[0]) for d in data), list(map(len, data))
    if len(data[0]) > target_len:
        off = rand.integers(0, len(data[0]) - target_len)
        data = [d[off:] for d in data]
    elif len(data[0]) < target_len:
        pad_left = rand.integers(0, target_len - len(data[0]) + 1)
        data = [np.pad(d, (pad_left, 0)) for d in data]
    data = [librosa.util.fix_length(d, target_len) for d in data]
    return data


def rand_codec(rand, sr, *data, log=False):
    CODECS = [
        ("libopus", "ogg", [8, 16, 32, 64]),
        ("libvorbis", "ogg", [45, 64, 80, 96]),
        ("libmp3lame", "mp3", [32, 40, 48, 56, 64, 80, 96]),
#         ("ac3", "ac3", 37, 128),
#         ("wmav2", "wma", 24, 128),
    ]
    encoder, container, bitrates = np_choice(rand, CODECS)
    bitrate = np_choice(rand, bitrates)
    ffmpeg_cmd = (
            "/usr/bin/ffmpeg "
            # Waveform input
            f"-f f32le -ar {sr} -ac 1 -i - "
            # Codec
            f"-c:a {encoder} -b:a {bitrate}k -f {container} - "
            # Decoding + waveform outut
            "| /usr/bin/ffmpeg -ac 1 -i - -c:a pcm_f32le -f f32le -"
        )
    if log: print("Run", ffmpeg_cmd)
    for idx, d in enumerate(data):
        assert "float64" in str(d.dtype)
        try:
            proc = subprocess.run(["sh", "-c", ffmpeg_cmd], input=d.astype("float32").tobytes(), capture_output=True,check=True)
        except subprocess.CalledProcessError as err:
            raise RuntimeError(f"Error in ffmpeg on item {idx}:" + err.stderr.decode("ascii", "ignore")) from err
        out = np.frombuffer(proc.stdout, "float32").astype("float64")
        if encoder == "libopus":
            # Opus will always output 48 kHz
            out = librosa.resample(out, 48_000, sr, res_type="kaiser_fast")
        _, out = align_audio(sr, int(0.2 * sr), 0, d, out)  # comparing less than entire signal is unstable
        assert np.abs(1 - d.nbytes/out.nbytes) < 0.2, ffmpeg_cmd
        yield librosa.util.fix_length(out, len(d))


def align_audio(sr, maxoff_samples, lookahead_samples, target, pred):
    """Align pred with target: If pred is delayed, cut some samples. If target is delayed, pad pred."""
    assert "float64" in str(target.dtype)
    assert "float64" in str(pred.dtype)
    dist = align_dist(sr, maxoff_samples, lookahead_samples, target, pred)
    if dist < 0:
        return dist, np.pad(pred, (-dist, 0))
    else:
        return dist, pred[dist:]


def align_dist(sr, maxoff_samples, lookahead_samples, target, pred):
    if lookahead_samples <= 0:
        lookahead_samples = max(len(target), len(pred))
    return fastalign.fastalign(pred.astype("float32"), target.astype("float32"), maxoff_samples, lookahead_samples)


def np_proba(rand, p):
    return rand.uniform() < p

def np_choice(rand, collection):
    return collection[rand.integers(len(collection))]


def trim_zeros_block(filt, trim='fb', block_size=1024):
    """Trim blocks of zeros"""
    trim = trim.upper()
    first = 0
    if 'F' in trim:
        for i in range(0, len(filt), block_size):
            if np.any(filt[i:i+block_size] != 0.):
                first = i
                break
    last = len(filt)
    if 'B' in trim:
        for i in range(len(filt)-1, block_size - 1, -block_size):
            if np.any(filt[i-block_size:i] != 0.):
                last = i
                break
    return filt[first:last]


def fast_trim_zeros(filt, trim='fb'):
    filt = trim_zeros_block(filt, trim)
    return np.trim_zeros(filt, trim)


def safe_peaknorm(a):
    if np.all(a == 0):
        return a
    else:
        return a / (np.abs(a).max() + 1e-10)


#np.seterr("raise")


def randmix(rand, speech_f, log=False):
    if log: print(speech_f)

    if np_proba(rand, 3/100):
        if log: print("no speech")
        speech = np.zeros((len_samples,))
        speech_f = None
    else:
        speech, sr = librosa.load(speech_f, sr=None, dtype="float64")
        assert sr == 16000, (speech_f, "sr", sr)
        speech = fast_trim_zeros(speech)
        speech = safe_peaknorm(speech)

    if len(fast_trim_zeros(speech)) and np_proba(rand, 1/5):
        if log: print("no noise")
        noise = np.zeros_like(speech)
    else:
        noise_f = np_choice(rand, dns_noise_files)
        if log: print(noise_f)
        noise, sr = librosa.load(noise_f, sr=None, dtype="float64")
        assert sr == 16000, (noise_f, "sr", sr)
        noise = safe_peaknorm(noise)

        if np_proba(rand, 4/5):  # performance optim
            noise = rand_eq_pitch(rand, noise, always_eq=False, enable_biquad=False, enable_lowpass=True, log=log)

    if np_proba(rand, 4/5):
        if np_proba(rand, 3/5):
            if log: print("IR on speech and noise")
            speech_ir, noise_ir = loadrandir(rand, 16000, augment=True, n=2, log=log)
            speech_x = scipy.signal.convolve(speech, speech_ir, "valid")
            speech_y = convolve_direct(16000, speech, speech_ir)[-len(speech_x):]
            noise_x = scipy.signal.convolve(noise, noise_ir, "valid")
        else:
            if log: print("IR on speech only")
            speech_ir, = loadrandir(rand, 16000, augment=True, n=1, log=log)
            speech_x = scipy.signal.convolve(speech, speech_ir, "valid")
            speech_y = convolve_direct(16000, speech, speech_ir)[-len(speech_x):]
            noise_x = noise
    else:
        if log: print("EQ on speech")
        speech_y = speech_x = rand_eq_pitch(rand, speech, always_eq=True, enable_biquad=True, enable_lowpass=True, fast_ok=True, log=log)
        noise_x = noise

    speech_x, speech_y = crop_or_pad(rand, len_samples, speech_x, speech_y)
    noise_x, = crop_or_pad(rand, len_samples, noise_x)

    if np_proba(rand, 1/20):
        if log: print("clipping")
        speech_x, noise_x, = rand_clipping(rand, 0.99, 1, speech_x, noise_x)


    assert "float64" in str(speech_x.dtype)
    assert "float64" in str(noise_x.dtype)
    speech_x_magspec = np.abs(librosa.stft(speech_x + 1e-20, 512))
    noise_x_magspec = np.abs(librosa.stft(noise_x + 1e-20, 512))
    sw = itu_r_468_weighted(speech_x_magspec, 512, 16000)
    nw = itu_r_468_weighted(noise_x_magspec, 512, 16000)
    r = (
        librosa.db_to_amplitude(
            librosa.amplitude_to_db(sw).min() - librosa.amplitude_to_db(nw).min())
        * rand.uniform(0.01, 0.05))

    if log: print("r", r, noise_x.min())
    noise_xr = (r + 1e-10) * (noise_x + 1e-10)
    mix = speech_x + noise_xr

    # Codec is by far the slowest, can increase to 1/5 if CPU fast enough
    if speech_f is not None and speech_f.endswith(".wav") and np_proba(rand, 1/7):
        mix, speech_x, noise_x, speech_y = rand_codec(rand, 16000, mix, speech_x, noise_x, speech_y, log=log)

    mix, speech_x, noise_x, speech_y = crop_or_pad(rand, len_samples, mix, speech_x, noise_x, speech_y)

    librosa.util.valid_audio(mix)
    librosa.util.valid_audio(speech_y)

    return map(safe_peaknorm, [mix, speech, speech_x, noise_x, speech_y])


def bench1():
    rand = np.random.default_rng()
    *_, = randmix(rand, np_choice(rand, dns_clean_files))

def bench():
    import cProfile
    for _ in range(3):
        bench1()
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(50):
        bench1()
    pr.create_stats()
    import pstats
    pstats.Stats(pr).strip_dirs().sort_stats("cumtime").print_stats(10)
