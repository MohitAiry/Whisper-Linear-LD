"""Step 1c evaluation: decode a frame tagger into language RTTMs.

    python -I scripts/ld_frame_eval.py --run-dir runs/ld_F0_s0

1. Log-probabilities for the validation sets (LD_sim sessions 351-450, pilot dev) and the test sets.
2. Decoder settings chosen on validation only, by frame-level DER (collar 0, two-language frames ignored,
   optimal one-to-one label mapping per session): switch penalty x session language pruning (a language is
   kept if it takes >= min_share of the session's speech frames by argmax).
3. Test RTTMs in <run>/eval/<dataset>/rttms/<session>_pred.rttm (the layout of ld_eval.slurm, so dscore and
   ld_probe2_real.py read them unchanged) and <run>/eval/decoder.json. dscore scoring is done by
   slurm_scripts/ld_frame_eval.sh. DISPLACE dev stays report-only: nothing is chosen on it.

--protocol p1 (2026-11-03, the reportable protocol): checkpoint avg.pt (mean of the last 5 epochs' weights,
no selection), decoder settings chosen on the real family_ld dev split only (6 families, 8 files,
data/splits/family_ld_p1.json), and family_ld is reported on its 19-file test split ("family_ld_test";
the dev split is reported as "family_ld_dev", labelled dev). Output in <run>/eval_p1/.
--protocol p0 reproduces the original evaluation (best.pt, synthetic validation, <run>/eval/).
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ldiar.frame_data import FEATS, SPLITS, LANGS, LID, NCLS, feat_dir, frame_targets, read_rttm, ref_path, sessions, \
    write_rttm  # noqa: E402
from ldiar.frame_model import FrameTagger, path_to_segments, session_languages, viterbi  # noqa: E402

PROTOCOLS = {
    "p0": dict(ckpt="best.pt", out="eval", tune=[("ldsim_train", "val"), ("pilot", "dev")],
               test={"ld_simulated": ("ldsim_test", None), "ld_simulated_complex": ("ldsim_complex_test", None),
                     "displace": ("displace", None), "family_ld": ("family_ld", None),
                     "pilot_test": ("pilot", "test")}),
    "p1": dict(ckpt="avg.pt", out="eval_p1", tune=[("family_ld", "dev")],
               test={"family_ld_test": ("family_ld", "test"), "displace": ("displace", None),
                     "ld_simulated": ("ldsim_test", None), "ld_simulated_complex": ("ldsim_complex_test", None),
                     "pilot_test": ("pilot", "test"), "family_ld_dev": ("family_ld", "dev")}),
}
# p1.1 (2026-11-04): as p1, but the checkpoint is sel.pt, the training epoch chosen on family_ld dev by
# scripts/ld_frame_select.py (candidates ckpt/epNN.pt + avg.pt). The decoder is re-tuned here on the same dev
# split, which gives the decoder of the joint (epoch, decoder) choice. family_ld_dev is then optimistic.
PROTOCOLS["p11"] = dict(PROTOCOLS["p1"], ckpt="sel.pt", out="eval_p11")
PENALTIES = [0, 2, 4, 8, 16, 32]
SHARES = [0, 0.02, 0.05, 0.1]
NAMES = ["NS"] + LANGS


def frame_der(y, h):
    """-> (errors, reference speech frames) with an optimal one-to-one mapping of hyp to ref labels."""
    m = y >= 0
    y, h = y[m], h[m]
    ref_l, hyp_l = np.unique(y[y > 0]), np.unique(h[h > 0])
    n_ref = int((y > 0).sum())
    if n_ref == 0:
        return int((h > 0).sum()), 0
    ov = np.array([[np.sum((y == r) & (h == q)) for q in hyp_l] for r in ref_l]).reshape(len(ref_l), len(hyp_l))
    matched = ov[linear_sum_assignment(-ov)].sum() if ov.size else 0
    # miss + false alarm + confusion = (ref speech) + (hyp speech beyond ref speech) - matched, in frames:
    both = int(((y > 0) & (h > 0)).sum())
    miss = n_ref - both
    fa = int(((y == 0) & (h > 0)).sum())
    conf = both - matched
    return miss + fa + conf, n_ref


def ref_targets(path, n):
    """Frame targets for scoring: unknown labels (DISPLACE's anonymous L1/L2/L3) get their own ids."""
    segs = read_rttm(path)
    ids = dict(LID)
    for _, _, l in segs:
        ids.setdefault(l, NCLS + len(ids))
    return frame_targets(segs, n, ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--feat-root", default=str(FEATS))
    p.add_argument("--protocol", choices=sorted(PROTOCOLS), default="p1")
    a = p.parse_args()
    P = PROTOCOLS[a.protocol]
    VAL, TEST = P["tune"], P["test"]
    run = Path(a.run_dir)
    out = run / P["out"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(run / P["ckpt"], map_location=dev, weights_only=False)
    c = ck["cfg"]
    model = FrameTagger(c["d_in"], c["ncls"], c["d"], c["layers"], c["dropout"]).to(dev).eval()
    model.load_state_dict(ck["model"])

    def logprobs(set_name, split):
        out = {}
        for sid in sessions(a.feat_root, set_name, split):
            x = torch.from_numpy(np.load(feat_dir(a.feat_root, set_name) / f"{sid}.npy")).to(dev, torch.float32)
            out[sid] = model.session_logprobs(x).cpu().numpy()
        return out

    # 1-2. decoder settings on validation
    val = {}
    for name, split in VAL:
        lps = logprobs(name, split)
        val[f"{name}:{split}"] = [(lp, ref_targets(ref_path(name, sid), len(lp))) for sid, lp in lps.items()]
    grid = {}
    for pen, share in itertools.product(PENALTIES, SHARES):
        ders = {}
        for k, items in val.items():
            e = n = 0
            for lp, y in items:
                h = viterbi(lp, pen, session_languages(lp, share))
                de, dn = frame_der(y, h)
                e, n = e + de, n + dn
            ders[k] = 100 * e / max(n, 1)
        grid[f"{pen}_{share}"] = dict(penalty=pen, min_share=share, mean=float(np.mean(list(ders.values()))), **ders)
        print(f"penalty {pen:2d} share {share:.2f}: " + " ".join(f"{k} {v:.2f}" for k, v in ders.items()), flush=True)
    best = min(grid.values(), key=lambda r: r["mean"])
    print(f"chosen on validation: penalty {best['penalty']} min_share {best['min_share']} (frame DER {best['mean']:.2f})")

    # 3. test sets
    test_der = {}
    for ev, (set_name, split) in TEST.items():
        od = out / ev / "rttms"
        od.mkdir(parents=True, exist_ok=True)
        e = n = 0
        for sid, lp in logprobs(set_name, split).items():
            path = viterbi(lp, best["penalty"], session_languages(lp, best["min_share"]))
            write_rttm(od / f"{sid}_pred.rttm", sid, path_to_segments(path, NAMES))
            de, dn = frame_der(ref_targets(ref_path(set_name, sid), len(lp)), path)
            e, n = e + de, n + dn
        test_der[ev] = round(100 * e / max(n, 1), 2)
        print(f"{ev}: frame DER {test_der[ev]:.2f}", flush=True)
    (out / "decoder.json").write_text(json.dumps(dict(protocol=a.protocol, ckpt=P["ckpt"], chosen=best, grid=grid,
                                                     test_frame_der=test_der, epoch=ck["epoch"]), indent=1))


if __name__ == "__main__":
    main()
