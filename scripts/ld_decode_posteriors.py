"""Decode per-frame posteriors with the p1.1 decoder rule: (lambda, tau) chosen on family_ld dev only.

    # precomputed log-posteriors, e.g. the Whisper LID baseline: <dir>/<set>/<session>.npy, (T, 11) at 25 Hz
    # posterior ensemble of trained checkpoints (mean of probabilities), on the cached Whisper L28 features
    python -I scripts/ld_decode_posteriors.py --ckpt runs/ld_p11_F1linear_s{0..4}/sel.pt --name F1linear_ens5

Grid lambda in {0, 2, 4, 8, 16, 32}, tau in {0, .02, .05, .1}; frame DER on family_ld dev as in ld_frame_eval.py.
Writes runs/ensemble/<name>/{family_ld_test,family_ld_dev,displace}/<session>_pred.rttm and decoder.json.
DISPLACE and family_ld test are decoded with the chosen setting only.
"""
import argparse
import itertools
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from ldiar.frame_data import FEATS, SPLITS, LANGS, feat_dir, ref_path, sessions, write_rttm  # noqa: E402
from ldiar.frame_model import FrameTagger, path_to_segments, session_languages, viterbi, viterbi_multi  # noqa: E402
from ld_frame_eval import PENALTIES, SHARES, frame_der, ref_targets  # noqa: E402

NAMES = ["NS"] + LANGS
SETS = {"family_ld_dev": ("family_ld", "dev"), "family_ld_test": ("family_ld", "test"), "displace": ("displace", None)}


def grid_task(args):
    lp, y, share = args
    return share, [frame_der(y, h) for h in viterbi_multi(lp, PENALTIES, session_languages(lp, share))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--post", help="directory of <set>/<session>.npy log-posteriors")
    p.add_argument("--ckpt", nargs="+", help="checkpoints to ensemble (mean of probabilities)")
    p.add_argument("--feat-root", default=str(FEATS))
    p.add_argument("--name", required=True)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    assert bool(a.post) != bool(a.ckpt), "give --post or --ckpt"
    pool = mp.get_context("fork").Pool(a.workers)
    torch.set_num_threads(8)
    models = []
    for c in a.ckpt or []:
        ck = torch.load(c, map_location="cpu", weights_only=False)
        g = ck["cfg"]
        m = FrameTagger(g["d_in"], g["ncls"], g["d"], g["layers"], g["dropout"]).eval()
        m.load_state_dict(ck["model"])
        models.append(m)

    def logpost(set_name, sid):
        if a.post:
            return np.load(Path(a.post) / set_name / f"{sid}.npy").astype(np.float32)
        x = torch.from_numpy(np.load(feat_dir(a.feat_root, set_name) / f"{sid}.npy").astype(np.float32))
        with torch.no_grad():
            pr = sum(m.session_logprobs(x).exp() for m in models) / len(models)
        return torch.log(pr.clamp_min(1e-12)).numpy().astype(np.float32)

    def ids(set_name, split):
        if a.post:  # sessions that have posteriors (restricted to the split)
            have = {f.stem for f in (Path(a.post) / set_name).glob("*.npy")}
            allowed = set(sessions(a.feat_root, set_name, split)) if Path(a.feat_root, set_name).is_dir() else None
            out = sorted(s for s in have if allowed is None or s in allowed)
            if split and allowed is None:
                keep = set(json.loads((SPLITS / "family_ld_p1.json").read_text())[split])
                out = [s for s in out if s in keep]
            return out
        return sessions(a.feat_root, set_name, split)

    out = ROOT / "runs/ensemble" / a.name
    dev = {sid: logpost("family_ld", sid) for sid in ids("family_ld", "dev")}
    devy = {sid: ref_targets(ref_path("family_ld", sid), len(lp)) for sid, lp in dev.items()}
    tot = {(pen, sh): [0, 0] for pen, sh in itertools.product(PENALTIES, SHARES)}
    for sh, res in pool.imap_unordered(grid_task, [(dev[s], devy[s], sh) for s in dev for sh in SHARES]):
        for pen, (e, n) in zip(PENALTIES, res):
            tot[(pen, sh)][0] += e
            tot[(pen, sh)][1] += n
    grid = {f"{pen}_{sh}": 100 * e / max(n, 1) for (pen, sh), (e, n) in tot.items()}
    best = min(grid, key=grid.get)
    pen, sh = float(best.split("_")[0]), float(best.split("_")[1])
    print(f"{a.name}: {len(dev)} dev files; chosen penalty {pen:g}, min_share {sh:g}, family_ld dev frame DER {grid[best]:.2f}", flush=True)
    test_der = {}
    for ev, (set_name, split) in SETS.items():
        od = out / ev
        od.mkdir(parents=True, exist_ok=True)
        e = n = 0
        for sid in ids(set_name, split):
            lp = dev[sid] if ev == "family_ld_dev" else logpost(set_name, sid)
            path = viterbi(lp, pen, session_languages(lp, sh))
            write_rttm(od / f"{sid}_pred.rttm", sid, path_to_segments(path, NAMES))
            de, dn = frame_der(ref_targets(ref_path(set_name, sid), len(lp)), path)
            e, n = e + de, n + dn
        test_der[ev] = round(100 * e / max(n, 1), 2)
        print(f"  {ev}: frame DER {test_der[ev]}", flush=True)
    (out / "decoder.json").write_text(json.dumps(dict(source=a.post or a.ckpt, chosen=dict(penalty=pen, min_share=sh, dev_der=grid[best]),
                                                     grid=grid, test_frame_der=test_der), indent=1))


if __name__ == "__main__":
    main()
