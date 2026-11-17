"""Step 1c front end: one hidden layer of the frozen Whisper-large-v3 encoder on continuous audio (no VAD).

    python -I scripts/ld_extract_whisper.py --out data/features [--sets pilot family_ld ...] [--layer 28]

Picked by the Step 1b probe study (runs/ld_probe_fe): layer 28 had the best language probes on family_ld
(0.952) and pilot->family_ld transfer (0.911), with voice identity down to 0.41 (espeak, E0's front end:
0.69 / 0.59 / 0.90). Whisper sees 30 s windows (its training length) with a 25 s hop; each window keeps
its central frames (2.5 s margins, except at the session ends), so every frame has >= 2.5 s of context
on both sides. The 50 Hz output is mean-pooled in pairs to 25 Hz (frame k = [0.04 k, 0.04 k + 0.04) s)
and saved as float16 (T, 1280) <out>/<set>/<session>.npy (~230 MB per hour). Resumable per session;
<out>/<set>/done.json is written last.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ldiar.frame_data import FEATS, SETS  # noqa: E402

SR = 16000
REPOS = {"whisper": "openai/whisper-large-v3",
         "espeak": "facebook/wav2vec2-lv-60-espeak-cv-ft"}  # E0's front end, for the front-end ablation
WIN, HOP, MARGIN = 1500, 1250, 125  # in 50 Hz encoder frames: 30 s window, 25 s hop, 2.5 s margin


def windows(n50):
    """-> list of (window start frame, keep start, keep end), keep ranges tile [0, n50)."""
    out, w = [], 0
    while True:
        s = HOP * w
        k0 = 0 if w == 0 else s + MARGIN
        last = s + WIN >= n50
        k1 = n50 if last else s + WIN - MARGIN
        out.append((s, k0, k1))
        if last:
            return out
        w += 1


class Sessions(torch.utils.data.Dataset):
    def __init__(self, wavs, fe):
        self.wavs, self.fe = wavs, fe
        self.whisper = fe.__class__.__name__.startswith("Whisper")

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, i):
        wav, sr = sf.read(str(self.wavs[i]), dtype="float32", always_2d=True)
        wav = wav.mean(1)
        assert sr == SR, (self.wavs[i], sr)
        n50 = math.ceil(len(wav) / 320)
        ws = windows(n50)
        key = "input_features" if self.whisper else "input_values"
        mels = [self.fe(wav[s * 320:(s + WIN) * 320], sampling_rate=SR, return_tensors="pt")[key][0]
                for s, _, _ in ws]
        return i, n50, ws, mels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(FEATS))
    p.add_argument("--sets", nargs="+", default=list(SETS))
    p.add_argument("--encoder", choices=sorted(REPOS), default="whisper")
    p.add_argument("--layer", type=int, default=28,
                   help="hidden_states index (whisper 28 = output of block 28; espeak 24 = E0's last layer)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--limit", type=int, default=None, help="smoke test: first N sessions per set")
    a = p.parse_args()

    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, WhisperFeatureExtractor, WhisperModel
    REPO = REPOS[a.encoder]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    if a.encoder == "whisper":
        fe = WhisperFeatureExtractor.from_pretrained(REPO)
        enc = WhisperModel.from_pretrained(REPO, torch_dtype=dtype).encoder
        enc.layers = enc.layers[:a.layer]  # blocks after the wanted one are never run
        grab = {}
        enc.layers[-1].register_forward_hook(
            lambda m, i, o: grab.__setitem__("h", o[0] if isinstance(o, tuple) else o))
        enc = enc.to(dev).eval()
        D = enc.config.d_model

        def run(x):
            enc(x)
            return grab["h"]
    else:  # hidden_states[layer] exactly as E0 reads it (the last one includes the final layer norm)
        fe = Wav2Vec2FeatureExtractor.from_pretrained(REPO)
        enc = Wav2Vec2ForCTC.from_pretrained(REPO, torch_dtype=dtype).wav2vec2.to(dev).eval()
        D = enc.config.hidden_size

        def run(x):
            return enc(x, output_hidden_states=True).hidden_states[a.layer]

    for set_name in a.sets:
        wav_dir = Path(SETS[set_name][0])
        od = Path(a.out) / set_name
        od.mkdir(parents=True, exist_ok=True)
        wavs = sorted(wav_dir.glob("*.wav"))[:a.limit]
        todo = [w for w in wavs if not (od / f"{w.stem}.npy").exists()]
        t0, secs = time.time(), 0.0
        loader = torch.utils.data.DataLoader(Sessions(todo, fe), batch_size=None, num_workers=a.workers,
                                             prefetch_factor=2 if a.workers else None)
        with torch.no_grad():
            for i, n50, ws, mels in loader:
                h50 = torch.empty((n50, D), dtype=torch.float32, device=dev)
                j = 0
                while j < len(ws):  # batches of equal-length windows (only the last one may be shorter)
                    k = j + 1
                    while k < len(ws) and k - j < a.batch and mels[k].shape == mels[j].shape:
                        k += 1
                    H = run(torch.stack(mels[j:k]).to(dev, dtype)).float()
                    for b, (s, k0, k1) in enumerate(ws[j:k]):
                        need = k1 - s
                        h = H[b] if H.shape[1] >= need else torch.cat([H[b], H[b, -1:].expand(need - H.shape[1], -1)])
                        h50[k0:k1] = h[k0 - s:k1 - s]  # w2v2 gives ~1 frame fewer than ceil(n/320): repeat the last
                    j = k
                if n50 % 2:
                    h50 = torch.cat([h50, h50[-1:]])
                h25 = h50.view(-1, 2, h50.shape[1]).mean(1)
                tmp = od / f"{todo[i].stem}.tmp.npy"
                np.save(tmp, h25.to(torch.float16).cpu().numpy())
                tmp.rename(od / f"{todo[i].stem}.npy")  # complete files only
                secs += n50 / 50
                print(f"  {set_name} {todo[i].stem}: {n50 / 50:.0f}s, {len(ws)} windows ({time.time() - t0:.0f}s)",
                      flush=True)
        if a.limit is None:
            (od / "done.json").write_text(json.dumps(dict(sessions=len(wavs), extracted_now=len(todo),
                                                          hours_now=round(secs / 3600, 2), repo=REPO, encoder=a.encoder,
                                                          layer=a.layer, fps=25,
                                                          seconds=round(time.time() - t0, 1))))
        print(f"=== {set_name}: {len(todo)} sessions, {secs / 3600:.2f} h in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
