"""Step 1c training: frame-level language tagger (ldiar/frame_model.py) on cached encoder features.

    python -I scripts/ld_frame_train.py --run-dir runs/ld_F0_s0 --seed 0 --train ldsim_train:train
    python -I scripts/ld_frame_train.py --run-dir runs/ld_F1_s0 --seed 0 --train ldsim_train:train pilot:train

--train entries are <set>[:<split>] (see ldiar.frame_data.sessions). Validation (model selection) is fixed
for every run: LD_sim train sessions 351-450 and the pilot dev split; the test sets (family_ld, DISPLACE,
LD_sim test, pilot test) are never read here. Selection = lowest mean validation loss over the two sets.
Writes args.json, train_log.jsonl, best.pt and done.json (last).

Protocol p1 (2026-11-03): the system checkpoint is avg.pt, the mean of the weights after each of the last
--avg-last epochs of a fixed --epochs schedule. No checkpoint is selected on any validation set; best.pt
(protocol p0: lowest mean validation loss) is still written for comparison. Validation is computed every
--val-every epochs for the learning curves only.

Protocol p1.1 (2026-11-04): --save-epochs writes ckpt/epNN.pt after each listed epoch (candidates for
scripts/ld_frame_select.py, which picks the epoch on family_ld dev). Saving does not change training.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ldiar.frame_data import FEATS, SPLITS, NCLS, FrameCrops, class_counts, feat_dir, frame_targets, read_rttm, ref_path, \
    sessions  # noqa: E402
from ldiar.frame_model import FrameTagger  # noqa: E402

VAL = [("ldsim_train", "val"), ("pilot", "dev")]


def entries(spec, feat_root):
    out = []
    for e in spec:
        name, _, split = e.partition(":")
        out += [(name, s) for s in sessions(feat_root, name, split or None)]
    return out


def load_val(feat_root):
    val = {}
    for name, split in VAL:
        items = []
        for sid in sessions(feat_root, name, split):
            x = np.load(feat_dir(feat_root, name) / f"{sid}.npy")
            items.append((torch.from_numpy(x), torch.from_numpy(frame_targets(read_rttm(ref_path(name, sid)), len(x)))))
        val[f"{name}:{split}"] = items
    return val


@torch.no_grad()
def evaluate(model, val, weight, dev):
    model.eval()
    res = {}
    for k, items in val.items():
        tot, n, conf = 0.0, 0, np.zeros((NCLS, NCLS))
        for x, y in items:
            lp = model.session_logprobs(x.to(dev, torch.float32))
            yy = y.to(dev)
            m = yy >= 0
            tot += F.nll_loss(lp[m], yy[m], weight=weight, reduction="sum").item()
            n += weight[yy[m]].sum().item()
            conf += np.bincount((yy[m] * NCLS + lp[m].argmax(1)).cpu().numpy(), minlength=NCLS * NCLS).reshape(NCLS, NCLS)
        rec = conf.diagonal() / np.maximum(conf.sum(1), 1)
        present = conf.sum(1) > 0
        sp = conf[1:, 1:]
        res[k] = dict(loss=tot / max(n, 1), bacc=float(rec[present].mean()),
                      speech_lang_acc=float(sp.diagonal().sum() / max(sp.sum(), 1)),
                      speech_recall=float(conf[1:, 1:].sum() / max(conf[1:].sum(), 1)),
                      nonspeech_recall=float(rec[0]))
    model.train()
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--feat-root", default=str(FEATS))
    p.add_argument("--train", nargs="+", default=["ldsim_train:train"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--epoch-size", type=int, default=2000, help="crops per epoch")
    p.add_argument("--crop", type=int, default=500, help="frames (20 s)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--avg-last", type=int, default=5, help="p1: average the weights of the last K epochs")
    p.add_argument("--val-every", type=int, default=1, help="validation (logging, p0 best.pt) every N epochs")
    p.add_argument("--save-epochs", default="", help="p1.1: comma list of epochs to save as ckpt/epNN.pt")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    run = Path(a.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(json.dumps(vars(a), indent=1))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr = entries(a.train, a.feat_root)
    ds = FrameCrops(a.feat_root, tr, crop=a.crop, epoch_size=a.epoch_size, seed=a.seed)
    cnt = class_counts(ds)
    w = np.where(cnt > 0, (cnt.sum() / np.maximum(cnt, 1)) ** 0.5, 0.0)  # sqrt inverse frequency
    w = w / (w[cnt > 0] * cnt[cnt > 0]).sum() * cnt.sum()  # mean weight per frame = 1
    weight = torch.tensor(w, dtype=torch.float32, device=dev)
    print(f"train: {len(tr)} sessions; frames per class {cnt.astype(int).tolist()}; weights {np.round(w, 2).tolist()}",
          flush=True)
    val = load_val(a.feat_root)

    d_in = np.load(feat_dir(a.feat_root, tr[0][0]) / f"{tr[0][1]}.npy", mmap_mode="r").shape[1]
    model = FrameTagger(d_in, NCLS, a.d, a.layers, a.dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    steps = a.epochs * math.ceil(a.epoch_size / a.batch)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    best, t0 = float("inf"), time.time()
    save_eps = {int(e) for e in a.save_epochs.split(",") if e}
    cfg = dict(d_in=d_in, ncls=NCLS, d=a.d, layers=a.layers, dropout=a.dropout)
    avg, n_avg = None, 0
    log = open(run / "train_log.jsonl", "w")
    for ep in range(1, a.epochs + 1):
        ds.set_epoch(ep)
        loader = torch.utils.data.DataLoader(ds, batch_size=a.batch, num_workers=a.workers, shuffle=False,
                                             drop_last=True, pin_memory=dev.type == "cuda")
        tl, nb = 0.0, 0
        for x, y in loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits = model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, NCLS), y.reshape(-1), weight=weight, ignore_index=-1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tl, nb = tl + loss.item(), nb + 1
        if ep in save_eps:  # p1.1 candidates
            (run / "ckpt").mkdir(exist_ok=True)
            torch.save(dict(model=model.state_dict(), cfg=cfg, epoch=ep), run / "ckpt" / f"ep{ep:02d}.pt")
        if ep > a.epochs - a.avg_last:  # p1: running mean of the end-of-epoch weights
            sd = {k: t.detach().float().clone() for k, t in model.state_dict().items()}
            n_avg += 1
            avg = sd if avg is None else {k: avg[k] + (sd[k] - avg[k]) / n_avg for k in avg}
        if ep % a.val_every and ep != a.epochs:
            print(f"ep {ep:2d} train {tl / max(nb, 1):.3f} ({time.time() - t0:.0f}s)", flush=True)
            continue
        v = evaluate(model, val, weight, dev)
        vloss = float(np.mean([r["loss"] for r in v.values()]))
        rec = dict(epoch=ep, train_loss=tl / max(nb, 1), val_loss=vloss, val=v, seconds=round(time.time() - t0))
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"ep {ep:2d} train {tl / max(nb, 1):.3f} val {vloss:.3f} "
              + " ".join(f"{k} bacc {r['bacc']:.3f} lang {r['speech_lang_acc']:.3f}" for k, r in v.items())
              + f" ({time.time() - t0:.0f}s)", flush=True)
        if vloss < best:
            best = vloss
            torch.save(dict(model=model.state_dict(), cfg=dict(d_in=d_in, ncls=NCLS, d=a.d, layers=a.layers,
                                                               dropout=a.dropout), epoch=ep, val=v),
                       run / "best.pt")
    torch.save(dict(model=avg, cfg=cfg, epoch=f"avg of epochs {a.epochs - n_avg + 1}-{a.epochs}"), run / "avg.pt")
    (run / "done.json").write_text(json.dumps(dict(best_val_loss=best, avg_epochs=n_avg,
                                                   seconds=round(time.time() - t0))))


if __name__ == "__main__":
    main()
