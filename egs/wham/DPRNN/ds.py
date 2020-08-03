from torch.utils.data import DataLoader, Dataset
import os, glob, ujson, random, librosa, scipy.signal, pyloudnorm, numpy as np, pickle, tqdm, tqdm.contrib.concurrent, pandas as pd
import hashlib

IR_DUR = 0.5
IR_DUR = 1


def ds_hash(*xs):
    s = ujson.dumps([
        deterministic_sample(x, min(10_000, int(len(x)/10)))
        for x in xs
    ])
    return hashlib.md5(s.encode("utf8")).hexdigest()


def safe_int(x):
    assert float(int(x)) == x
    return int(x)


class StaticRandomSubsetDataset:
    def __init__(self, ds, r=None, n=None):
        self.ds = ds
        self.r = r
        self.n = n
        self.idxs = deterministic_sample(range(len(ds)), n or int(len(ds) * r))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return self.ds[self.idxs[idx]]


def deterministic_shuffle(x):
    random.Random(4242).shuffle(x)

def deterministic_sample(x, n):
    return random.Random(4242).sample(x, n)


def loadir_mono(f, sr):
    assert sr == 8000
    dat = librosa.core.load("/home/jo/dev/audio-experiments-data/" + f.replace("wav48", "wav8"), sr=sr, mono=False, duration=IR_DUR)[0]
    assert dat.ndim == 1, f
    return pyloudnorm.normalize.peak(dat, -10)

def loadir(f, c, sr):
    dat = librosa.core.load("/home/jo/dev/audio-experiments-data/" + f, sr=sr, mono=False, duration=IR_DUR)[0]
    if dat.ndim == 1:
        dat = [dat]
    dat = dat[c]
    return pyloudnorm.normalize.peak(dat, -10)


VCTK_ROOT = "/home/jo/dev/audio-experiments-data/VCTK-8kHz"


class MyDataset(Dataset):
    def __init__(self, sr, segment, vctk_files, irs):
        self.sr = sr
        self.segment = segment

        self.vctk_files = vctk_files[:]
        self.irs = irs[:]
        deterministic_shuffle(self.vctk_files)
        deterministic_shuffle(self.irs)

        self.items = []
        skipctr = 0
        if segment:
            self.is_test = False
            self.chunk_len = int(segment * self.sr)
            self.src_chunk_len = self.chunk_len + len(irs[0])
            for vctk_idx, (f, s, e) in enumerate(self.vctk_files):
                poses = [
                    start + self.src_chunk_len
                    for start in range(s, e+1, int(self.chunk_len / 2))
                ]
                if len(poses) == 1 and poses[0][1] < self.src_chunk_len:
                    # skip very short samples
                    skipctr += 1
                    continue
                #for ir_idx in range(len(irs)):
                #    for pos, chunk_e in poses:
                #        self.items.append((vctk_idx, ir_idx, pos, chunk_e))
                for pos, chunk_e in poses:
                    self.items.append((vctk_idx, pos, chunk_e))
        else:
            self.is_test = True
            for vctk_idx, (f, s, e) in enumerate(self.vctk_files):
                if e - s < 1 * sr + len(irs[0]):
                    # skip very short samples
                    skipctr += 1
                    continue
                #for ir_idx in range(len(irs)):
                #    self.items.append((vctk_idx, ir_idx, s, e))
                self.items.append((vctk_idx, s, e))
        print(("skipped", skipctr, "of", len(self.vctk_files)))

    def __len__(self):
        # * 10: just a hack to increase max ds size.
        return len(self.items) * 10# * len(self.irs)

    def __getitem__(self, idx):
        # self.items too large / shuffle too slow
        #vctk_idx, ir_idx, start, end = self.items[idx]

        # shuffle too slow
        #ir_idx = idx % len(self.irs)
        #vctk_idx, start, end = self.items[safe_int((idx - ir_idx) / len(self.irs))]

        vctk_idx, start, end = self.items[int(idx / 10)]
        ir_idx = np.random.randint(0, len(self.irs))

        fname, _, _ = self.vctk_files[vctk_idx]
        fname = f'{VCTK_ROOT}/{fname.split("_")[0]}/{fname}'
        audio, _ = librosa.core.load(fname, sr=self.sr, res_type="kaiser_fast")
        audio = audio[start:end]
        audio = pyloudnorm.normalize.peak(audio, -10)
        ir = self.irs[ir_idx]
        if not self.is_test and len(audio) < self.src_chunk_len:
            audio = librosa.util.fix_length(audio, self.src_chunk_len)
        clean = scipy.signal.convolve(
            audio,
            librosa.util.fix_length(ir[: ir.argmax() + int(0.05 * self.sr)], len(ir)),
            "valid",
        )[1:]
        noisy = scipy.signal.convolve(audio, ir, "valid")[1:]
        assert clean.shape == noisy.shape
        if not self.is_test:
            sources = np.vstack((librosa.util.fix_length(clean, self.chunk_len),))
            noisy = librosa.util.fix_length(noisy, self.chunk_len)
        else:
            assert clean.shape == noisy.shape
            sources = clean[None]
        return noisy, sources


def getvctk():
    try:
        with open("/tmp/vctk.pkl", "rb") as f:
            vctk_subset = pickle.load(f)
    except Exception as e:
        print(e)
        vctk_subset = [
            os.path.basename(f)
            for f in sorted(glob.glob(f"{VCTK_ROOT}/**/*.wav", recursive=True))
        ]
        with open("/tmp/vctk.pkl", "wb") as f:
            pickle.dump(vctk_subset, f)
    vctk_subset = set(vctk_subset)
    print(("VCTK subset #", len(vctk_subset), VCTK_ROOT))
    vctk_files = [
        (f, s, e)
        for f, s, e in ujson.load(open(f"{VCTK_ROOT}/silences.json"))
        if f in vctk_subset
    ]
    vctk_people = list(set(f.split("_")[0] for f, _, _ in vctk_files))
    deterministic_shuffle(vctk_people)
    # vctk_people = vctk_people[:int(len(vctk_people)/2)]
    # TODO: fix test distribution == val distribution
    # test_people  = vctk_people[                             :1*int(0.2 * len(vctk_people))]
    test_people = []
    val_people = vctk_people[
        0 * int(0.1 * len(vctk_people)) : 2 * int(0.1 * len(vctk_people))
    ]
    train_people = vctk_people[2 * int(0.1 * len(vctk_people)) :]
    print(("People #", len(test_people), len(val_people), len(train_people)))
    return vctk_files, train_people, val_people, test_people


def getirs(sr):
    if 1:
        irs = pd.read_csv("/home/jo/dev/audio-experiments/pred-all-autogluon.csv")
        irs = sorted([
            (f, c) for f, c in irs[irs["pred_autogluon"] > 0.6][["file", "chan"]].values
            if "wav48" in f
        ])
    else:
        irs = sorted([
            (f, c) for f, c , k in ujson.load(open("/home/jo/dev/audio-experiments/small-manual.json"))
            if k == "yes"
        ])
    #irs = tqdm.contrib.concurrent.thread_map(lambda x: loadir(*x, sr), irs)
    irs = tqdm.contrib.concurrent.thread_map(lambda x: loadir_mono(f"{x[0][:-4]}_{x[1]}.wav", sr), irs)
    # TODO: find better way to deal with these (late argmax) (0.45 - was 1770)
    irs = [ir for ir in irs if ir.argmax() < 0.45 * IR_DUR * sr]
    deterministic_shuffle(irs)
    # test_irs  = irs[                     :1*int(0.2 * len(irs))]
    test_irs = []
    val_irs = irs[0 * int(0.1 * len(irs)) : (3 if len(irs) > 2_000 else 2) * int(0.1 * len(irs))]
    train_irs = irs[(3 if len(irs) > 2_000 else 2) * int(0.1 * len(irs)) :]
    print(("IRs #", len(test_irs), len(val_irs), len(train_irs)))
    return train_irs, val_irs, test_irs


def getds(for_test, conf):
    vctk_files, train_people, val_people, test_people = getvctk()
    train_irs, val_irs, test_irs = getirs(conf["data"]["sample_rate"])

    if for_test:
        # Test set
        # todo: proper subset
        test_set = StaticRandomSubsetDataset(
            MyDataset(
                conf["data"]["sample_rate"], None,
        [
            (f, s, e) for f, s, e in vctk_files if f.split("_")[0] in val_people
        ]

                val_irs,
            ),
            n=1000,
        )
        print("DS hash", ds_hash([test_set.ds.items, test_set.idxs]))
        return test_set

    train_set = MyDataset(
        conf["data"]["sample_rate"],
        conf["data"]["segment"],
        [
            (f, s, e) for f, s, e in vctk_files if f.split("_")[0] in train_people
        ]
        train_irs,
    )
    val_set = StaticRandomSubsetDataset(
        MyDataset(
            conf["data"]["sample_rate"],
            conf["data"]["segment"],
            val_people,
        [
            (f, s, e) for f, s, e in vctk_files if f.split("_")[0] in val_people
        ]
            val_irs,
        ),
        #1.0,
        #0.005,
        n=10_000,
    )
    print(("DS len", len(train_set), len(val_set)))
    import time
    print("ds hashes", ds_hash(train_set.items), ds_hash(val_set.ds.items, val_set.idxs))

    # with open("/tmp/42", "wb") as f:
    #    def sample(l, n):
    #       idxs = random.sample(range(len(l)), n)
    #       return [l[idx] for idx in idxs]
    #    pickle.dump((sample(train_set, 10), sample(val_set, 10)), f)
    # exit(0)

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
        pin_memory=True,
    )
    # val_loader = DataLoader(val_set, shuffle=False,
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
        pin_memory=True,
    )

    return train_loader, val_loader
