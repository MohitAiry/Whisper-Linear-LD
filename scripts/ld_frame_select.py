"""Protocol p1.1: choose the training epoch of a frame tagger on the real family_ld dev split.

    python -I scripts/ld_frame_select.py --run-dir runs/ld_p11_F0_s0

Candidates: <run>/ckpt/epNN.pt (written by ld_frame_train.py --save-epochs) and <run>/avg.pt (protocol p1's
checkpoint). For every candidate: log-probabilities on family_ld dev (8 files, 6 families), the full decoder
grid (switch penalty x session language pruning, as in ld_frame_eval.py), and frame DER on dev. The
(candidate, decoder) pair with the lowest dev frame DER is chosen, and the candidate is copied to <run>/sel.pt.
Test sets are not read here. ld_frame_eval.py --protocol p11 then evaluates sel.pt.

For the analysis only (P1.2, not used for the choice), each candidate also gets pilot dev and LD_sim val
statistics: unweighted NLL on speech and non-speech frames, and frame language accuracy.
Writes <run>/selection.json last.
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ldiar.frame_data import FEATS, SPLITS, feat_dir, frame_targets, read_rttm, ref_path, sessions  # noqa: E402
from ldiar.frame_model import FrameTagger, session_languages, viterbi_multi  # noqa: E402
from ld_frame_eval import PENALTIES, SHARES, frame_der, ref_targets  # noqa: E402

TUNE = ("family_ld", "dev")
ANALYSIS = [("pilot", "dev"), ("ldsim_train", "val")]


def decode_task(args):
    """-> (share, [(errors, ref frames) per penalty]) for one session."""
    lp, y, share = args
    paths = viterbi_multi(lp, PENALTIES, session_languages(lp, share))
    return share, [frame_der(y, h) for h in paths]


def load(feat_root, name, split, dev):
    out = []
    for sid in sessions(feat_root, name, split):
        x = torch.from_numpy(np.load(feat_dir(feat_root, name) / f"{sid}.npy")).to(dev, torch.float32)
        out.append((sid, x))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--feat-root", default=str(FEATS))
    p.add_argument("--workers", type=int, default=5)
    a = p.parse_args()
    pool = mp.get_context("fork").Pool(a.workers)  # before CUDA is initialised
    run = Path(a.run_dir)
    cands = sorted((run / "ckpt").glob("ep*.pt")) + [run / "avg.pt"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tune = load(a.feat_root, *TUNE, dev)
    tune_y = {sid: ref_targets(ref_path(TUNE[0], sid), len(x)) for sid, x in tune}
    ana = {f"{n}:{s}": [(x, torch.from_numpy(frame_targets(read_rttm(ref_path(n, sid)), len(x))).to(dev))
                        for sid, x in load(a.feat_root, n, s, dev)] for n, s in ANALYSIS}

    rows = []
    for c in cands:
        ck = torch.load(c, map_location=dev, weights_only=False)
        g = ck["cfg"]
        model = FrameTagger(g["d_in"], g["ncls"], g["d"], g["layers"], g["dropout"]).to(dev).eval()
        model.load_state_dict(ck["model"])
        with torch.no_grad():
            lps = {sid: model.session_logprobs(x).cpu().numpy() for sid, x in tune}
            stats = {}
            for k, items in ana.items():
                nll_sp = nll_ns = 0.0
                n_sp = n_ns = ok = n_both = 0
                for x, y in items:
                    lp = model.session_logprobs(x)
                    m = y >= 0
                    lp, yy = lp[m], y[m]
                    nll = -lp.gather(1, yy[:, None])[:, 0]
                    ns = yy == 0
                    nll_sp += nll[~ns].sum().item(); n_sp += int((~ns).sum())
                    nll_ns += nll[ns].sum().item(); n_ns += int(ns.sum())
                    am = lp.argmax(1)
                    both = (~ns) & (am > 0)
                    ok += int((am[both] == yy[both]).sum()); n_both += int(both.sum())
                stats[k] = dict(nll_speech=nll_sp / max(n_sp, 1), nll_nonspeech=nll_ns / max(n_ns, 1) if n_ns else None,
                                lang_acc=ok / max(n_both, 1))
        tasks = [(lps[sid], tune_y[sid], share) for sid in lps for share in SHARES]
        tot = {(pen, share): [0, 0] for pen in PENALTIES for share in SHARES}
        for share, res in pool.imap_unordered(decode_task, tasks):
            for pen, (e, n) in zip(PENALTIES, res):
                tot[(pen, share)][0] += e
                tot[(pen, share)][1] += n
        grid = {f"{pen}_{share}": 100 * e / max(n, 1) for (pen, share), (e, n) in tot.items()}
        key = min(grid, key=grid.get)
        pen, share = key.split("_")
        rows.append(dict(candidate=c.name, epoch=ck["epoch"], dev_der=grid[key], penalty=int(pen),
                         min_share=float(share), grid=grid, analysis=stats))
        print(f"{c.name:8s} epoch {str(ck['epoch']):20s} family_ld dev frame DER {grid[key]:6.2f} "
              f"(penalty {pen}, share {share}) | " + " ".join(
                  f"{k} lang {v['lang_acc']:.3f} nll_sp {v['nll_speech']:.2f}" for k, v in stats.items()), flush=True)
    pool.close()
    best = min(rows, key=lambda r: r["dev_der"])
    ck = torch.load(run / ("ckpt" if best["candidate"] != "avg.pt" else "") / best["candidate"], map_location="cpu",
                    weights_only=False)
    torch.save(dict(ck, selected_from=best["candidate"]), run / "sel.pt")
    print(f"chosen: {best['candidate']} (epoch {best['epoch']}), family_ld dev frame DER {best['dev_der']:.2f}")
    (run / "selection.json").write_text(json.dumps(dict(protocol="p1.1", tune="family_ld:dev", chosen={
        k: best[k] for k in ("candidate", "epoch", "dev_der", "penalty", "min_share")}, candidates=rows), indent=1))


if __name__ == "__main__":
    main()
