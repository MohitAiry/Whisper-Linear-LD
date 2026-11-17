"""Step 1c data: LD sets on continuous audio, frame labels at 25 Hz (40 ms), crops for training.

Frame k of a session covers [0.04 k, 0.04 (k + 1)) s. Its target is the language whose reference segments
cover the frame centre; 0 (non-speech) when none does; -1 (ignored) when two languages overlap there.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
# Data root: set LD_DATA to use another location. Expected layout (see README "Data"):
#   $LD_DATA/audio/<set>/<session>.wav   16 kHz mono
#   $LD_DATA/rttm/<set>/<session><suffix>.rttm   reference language RTTMs
#   $LD_DATA/features/<set>/<session>.npy   Whisper block-28 features (scripts/ld_extract_whisper.py)
DATA = Path(os.environ.get("LD_DATA", ROOT / "data"))
FEATS = DATA / "features"
SPLITS = ROOT / "data/splits"
FPS = 25
HOP = 1.0 / FPS

# name: (wav dir, reference RTTM dir, reference file suffix, session-name filter)
R, W = DATA / "rttm", DATA / "audio"
SETS = {
    "ldsim_train": (W / "ldsim_train", R / "ldsim_train", "", None),  # LD_simulated train (sim_train_0001..)
    "ldsim_test": (W / "ldsim_test", R / "ldsim_test", "", None),  # LD_simulated test
    "ldsim_complex_test": (W / "ldsim_complex_test", R / "ldsim_complex_test", "", None),  # LD_simulated_complex test
    "displace": (W / "displace", R / "displace", "_LANGUAGE", None),  # DISPLACE dev audio + language labels
    "family_ld": (W / "family_ld", R / "family_ld", "", None),  # family_ld audio + RTTMs
    "pilot": (W / "pilot", R / "pilot", "", None),  # our synthetic pilot set (all splits in one folder)
}
# class 0 = non-speech; the rest are every language in the training sets (LD_sim's 10 + pilot's 5 overlap)
LANGS = ["EN", "HI", "KN", "ML", "TA", "TE", "AS", "BN", "MR", "OD"]
ALIAS = {"HN": "HI", "KA": "KN"}
LID = {l: i + 1 for i, l in enumerate(LANGS)}
LID.update({a: LID[l] for a, l in ALIAS.items()})
NCLS = len(LANGS) + 1


def read_rttm(path):
    out = []
    for line in open(path):
        x = line.split()
        if len(x) >= 8 and x[0] in ("SPEAKER", "LANGUAGE"):
            out.append((float(x[3]), float(x[3]) + float(x[4]), x[7]))
    return sorted(out)


def ref_path(set_name, sid):
    _, rdir, suffix, _ = SETS[set_name]
    return Path(rdir) / f"{sid}{suffix}.rttm"


def frame_targets(segs, n, ids=LID):
    """-> int64 (n,): class per 40 ms frame centre (0 none, -1 two languages or a label outside ids)."""
    centre = (np.arange(n) + 0.5) * HOP
    y = np.zeros(n, np.int64)
    hits = np.zeros(n, np.int64)
    for s, e, l in segs:
        i0, i1 = np.searchsorted(centre, s), np.searchsorted(centre, e)
        if i1 <= i0:
            continue
        c = ids.get(l, -1)
        prev = y[i0:i1]
        clash = (hits[i0:i1] > 0) & (prev != c)
        y[i0:i1] = np.where(clash, -1, np.where(hits[i0:i1] > 0, prev, c))
        hits[i0:i1] += 1
    return y


def feat_dir(feat_root, set_name):
    return Path(feat_root) / set_name


def sessions(feat_root, set_name, split=None):
    """Session ids with extracted features. split: for pilot the name's split field; for ldsim_train
    'train' = first 350 sessions (E0's training split), 'val' = the other 100; for family_ld 'dev' / 'test'
    = the protocol-p1 split by family (data/splits/family_ld_p1.json)."""
    d = feat_dir(feat_root, set_name)
    sids = sorted(p.stem for p in d.glob("*.npy") if not p.name.endswith(".tmp.npy"))
    if set_name == "pilot" and split:
        sids = [s for s in sids if s.split("_")[1] == split]
    if set_name == "ldsim_train" and split:
        sids = [s for s in sids if (int(s.split("_")[-1]) <= 350) == (split == "train")]  # sim_train_0001..0350
    if set_name == "family_ld" and split:
        keep = set(json.loads((SPLITS / "family_ld_p1.json").read_text())[split])
        sids = [s for s in sids if s in keep]
    return sids


class FrameCrops(torch.utils.data.Dataset):
    """Random crops of `crop` frames from whole sessions (features float16 (T, D) .npy, memory-mapped)."""

    def __init__(self, feat_root, entries, crop=500, epoch_size=2000, seed=0):
        # entries: list of (set_name, sid); sampling weight proportional to session length
        self.items = []
        for set_name, sid in entries:
            f = feat_dir(feat_root, set_name) / f"{sid}.npy"
            n = np.load(f, mmap_mode="r").shape[0]
            y = frame_targets(read_rttm(ref_path(set_name, sid)), n)
            self.items.append((str(f), y))
        lens = np.array([len(y) for _, y in self.items], float)
        self.p = lens / lens.sum()
        self.crop, self.epoch_size, self.seed, self.epoch = crop, epoch_size, seed, 0

    def set_epoch(self, epoch):  # call before each epoch (workers copy the dataset when an epoch starts)
        self.epoch = epoch

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, self.epoch, i))
        k = rng.choice(len(self.items), p=self.p)
        f, y = self.items[k]
        x = np.load(f, mmap_mode="r")
        n = len(y)
        s = int(rng.integers(0, max(1, n - self.crop + 1)))
        xs = np.asarray(x[s:s + self.crop], dtype=np.float32)
        ys = y[s:s + self.crop]
        if len(ys) < self.crop:  # short session: pad, padded frames ignored
            pad = self.crop - len(ys)
            xs = np.pad(xs, ((0, pad), (0, 0)))
            ys = np.pad(ys, (0, pad), constant_values=-1)
        return torch.from_numpy(xs), torch.from_numpy(ys)


def class_counts(dataset):
    c = np.zeros(NCLS)
    for _, y in dataset.items:
        v = y[y >= 0]
        c += np.bincount(v, minlength=NCLS)
    return c


def write_rttm(path, sid, segs):
    with open(path, "w") as f:
        for s, e, l in segs:
            f.write(f"SPEAKER {sid} 1 {s:.3f} {e - s:.3f} <NA> <NA> {l} <NA> <NA>\n")


def load_json(p):
    return json.loads(Path(p).read_text())
