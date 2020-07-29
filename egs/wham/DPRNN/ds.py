from torch.utils.data import DataLoader
import os, glob, ujson, random, librosa, scipy.signal, pyloudnorm, numpy as np, pickle, tqdm, tqdm.contrib.concurrent
import hashlib


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


def loadir(f, sr):
    dat = librosa.core.load(f, sr=sr, mono=False, duration=0.5)[0]
    assert len(dat.shape) == 1, f
    return pyloudnorm.normalize.peak(dat, -10)


VCTK_ROOT = "/root/VCTK-8kHz"


class MyDataset:
    def __init__(self, sr, segment, people, irs, vctk_files):
        self.sr = sr
        self.vctk_files = [
            (f, s, e) for f, s, e in vctk_files if f.split("_")[0] in people
        ]
        self.irs = irs[:]
        deterministic_shuffle(self.vctk_files)
        deterministic_shuffle(self.irs)
        self.items = []
        if segment:
            self.is_test = False
            self.chunk_len = int(segment * self.sr)
            self.src_chunk_len = self.chunk_len + len(irs[0])
            skipctr = 0
            for vctk_idx, (f, s, e) in enumerate(self.vctk_files):
                if e - s < self.src_chunk_len:
                    # skip very short samples
                    skipctr += 1
                    continue
                poses = []
                pos = s
                while 1:
                    chunk_e = pos + self.src_chunk_len
                    poses.append((pos, chunk_e))
                    if chunk_e > e:
                        break
                    pos += int(self.chunk_len / 2)
                for ir_idx in range(len(irs)):
                    for pos, chunk_e in poses:
                        self.items.append((vctk_idx, ir_idx, pos, chunk_e))
            print(("skipped", skipctr, "of", len(self.vctk_files)))
        else:
            self.is_test = True
            for vctk_idx, (f, s, e) in enumerate(self.vctk_files):
                for ir_idx in range(len(irs)):
                    self.items.append((vctk_idx, ir_idx, s, e))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        vctk_idx, ir_idx, start, end = self.items[idx]

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
    irs = tqdm.contrib.concurrent.thread_map(lambda x: loadir(x, sr), sorted(glob.glob("/root/8kHz/*")))
    # TODO: find better way to deal with these (late argmax)
    irs = [ir for ir in irs if ir.argmax() < 1770]
    deterministic_shuffle(irs)
    # test_irs  = irs[                     :1*int(0.2 * len(irs))]
    test_irs = []
    val_irs = irs[0 * int(0.1 * len(irs)) : 2 * int(0.1 * len(irs))]
    train_irs = irs[2 * int(0.1 * len(irs)) :]
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
                conf["data"]["sample_rate"], None, val_people, val_irs, vctk_files
            ),
            0.01,
        )
        print("DS hash", hashlib.md5(ujson.dumps(test_set.items).encode("utf8")).hexdigest())
        return test_set

    train_set = MyDataset(
        conf["data"]["sample_rate"],
        conf["data"]["segment"],
        train_people,
        train_irs,
        vctk_files,
    )
    val_set = StaticRandomSubsetDataset(
        MyDataset(
            conf["data"]["sample_rate"],
            conf["data"]["segment"],
            val_people,
            val_irs,
            vctk_files,
        ),
        #0.005,
        n=5_000,
    )
    print(("DS len", len(train_set), len(val_set)))
    print("DS hashes", hashlib.md5(ujson.dumps(train_set.items).encode("utf8")).hexdigest(),
                       hashlib.md5(ujson.dumps([val_set.ds.items, val_set.idxs]).encode("utf8")).hexdigest())

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
