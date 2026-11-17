"""Whisper + linear-probe language diarization: audio -> RTTM.

    python infer.py --audio a.wav [b.wav | some_dir ...] --out out_dir
                    [--ckpt checkpoints/linear_p11_s4/sel.pt ...]   (default: 5 linear-probe seeds, posterior ensemble)
                    [--penalty 32] [--min-share 0.1] [--save-features] [--save-posteriors]

Pipeline (identical to the code that produced the reported results):
  1. audio, 16 kHz mono (other rates are resampled; all training/test audio was 16 kHz)
  2. frozen Whisper-large-v3 encoder (openai/whisper-large-v3, revision 06f233fe06e710322aca913c1bc4249a0d71fce1),
     output of transformer block 28 of 32 (blocks 29-32 are not run), on 30 s windows with a 25 s hop; each window
     keeps its central frames (2.5 s margins) -> 50 Hz x 1280; bf16 on GPU (as for training), fp32 on CPU
  3. mean of frame pairs -> 25 Hz (40 ms) x 1280, rounded to float16 (the stored training features were float16)
  4. frame head (default: linear probe, LayerNorm + Linear 1280->11; BiLSTM tagger checkpoints also load): per-frame
     log-probabilities over {non-speech, EN, HI, KN, ML, TA, TE, AS, BN, MR, OD}; sessions in 120 s chunks with 10 s
     context on each side. With several checkpoints, the per-frame probabilities are averaged (ensemble).
  5. session language pruning (languages with >= min_share of the argmax speech frames are allowed) and Viterbi
     with a switch penalty (nats per change of state)
  6. runs of equal non-zero labels -> RTTM lines "SPEAKER <file id> 1 <start> <dur> <NA> <NA> <LANG> <NA> <NA>"
Default decoder settings are the ones every seed chose on family_ld dev (penalty 32, min_share 0.1).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ldiar.frame_data import LANGS, write_rttm  # noqa: E402
from ldiar.frame_model import FrameTagger, path_to_segments, session_languages, viterbi  # noqa: E402

SR = 16000
REPO, REVISION, LAYER = "openai/whisper-large-v3", "06f233fe06e710322aca913c1bc4249a0d71fce1", 28
WIN, HOP, MARGIN = 1500, 1250, 125  # 50 Hz encoder frames: 30 s window, 25 s hop, 2.5 s margin
NAMES = ["NS"] + LANGS


def windows(n50):
    """-> list of (window start frame, keep start, keep end); the keep ranges tile [0, n50)."""
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


class WhisperL28:
    """Frozen Whisper-large-v3 encoder truncated after block 28, returning 25 Hz float16-rounded features."""

    def __init__(self, device, batch=16):
        from transformers import WhisperFeatureExtractor, WhisperModel
        self.dev, self.batch = device, batch
        self.dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        self.fe = WhisperFeatureExtractor.from_pretrained(REPO, revision=REVISION)
        enc = WhisperModel.from_pretrained(REPO, revision=REVISION, torch_dtype=self.dtype).encoder
        enc.layers = enc.layers[:LAYER]
        self.grab = {}
        enc.layers[-1].register_forward_hook(
            lambda m, i, o: self.grab.__setitem__("h", o[0] if isinstance(o, tuple) else o))
        self.enc = enc.to(device).eval()
        self.D = enc.config.d_model

    @torch.no_grad()
    def __call__(self, wav):
        n50 = math.ceil(len(wav) / 320)
        ws = windows(n50)
        mels = [self.fe(wav[s * 320:(s + WIN) * 320], sampling_rate=SR, return_tensors="pt")["input_features"][0]
                for s, _, _ in ws]
        h50 = torch.empty((n50, self.D), dtype=torch.float32, device=self.dev)
        j = 0
        while j < len(ws):  # batches of equal-length windows, exactly as in scripts/ld_extract_whisper.py
            k = j + 1
            while k < len(ws) and k - j < self.batch and mels[k].shape == mels[j].shape:
                k += 1
            self.enc(torch.stack(mels[j:k]).to(self.dev, self.dtype))
            H = self.grab["h"].float()
            for b, (s, k0, k1) in enumerate(ws[j:k]):
                need = k1 - s
                h = H[b] if H.shape[1] >= need else torch.cat([H[b], H[b, -1:].expand(need - H.shape[1], -1)])
                h50[k0:k1] = h[k0 - s:k1 - s]
            j = k
        if n50 % 2:
            h50 = torch.cat([h50, h50[-1:]])
        h25 = h50.view(-1, 2, h50.shape[1]).mean(1)
        return h25.to(torch.float16).cpu().numpy()


def load_audio(path):
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(1)
    if sr != SR:
        import torchaudio
        wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, SR).numpy()
        print(f"  note: {path.name} resampled {sr} -> {SR} Hz (the model was trained on 16 kHz audio)")
    return wav


def load_tagger(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    c = ck["cfg"]
    m = FrameTagger(c["d_in"], c["ncls"], c["d"], c["layers"], c["dropout"]).to(device).eval()
    m.load_state_dict(ck["model"])
    return m, ck.get("epoch")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio", nargs="+", required=True, help="wav files and/or directories of wav files")
    p.add_argument("--out", required=True)
    p.add_argument("--ckpt", nargs="+", default=[str(HERE / f"checkpoints/linear_p11_s{s}/sel.pt") for s in range(5)],
                   help="default: the 5 linear-probe checkpoints (checkpoints/linear_p11_s{0..4}/sel.pt)")
    p.add_argument("--penalty", type=float, default=32.0, help="Viterbi switch penalty (nats)")
    p.add_argument("--min-share", type=float, default=0.1, help="session language pruning threshold")
    p.add_argument("--features", default=None,
                   help="directory of precomputed 25 Hz features <file id>.npy (skips Whisper)")
    p.add_argument("--save-features", action="store_true")
    p.add_argument("--save-posteriors", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    dev = torch.device(a.device)
    files = []
    for x in map(Path, a.audio):
        files += sorted(x.glob("*.wav")) if x.is_dir() else [x]
    out = Path(a.out)
    (out / "rttm").mkdir(parents=True, exist_ok=True)
    taggers = [load_tagger(c, dev) for c in a.ckpt]
    print(f"{len(files)} files; {len(taggers)} checkpoint(s): " + ", ".join(f"{c} (epoch {e})" for c, (_, e) in zip(a.ckpt, taggers))
          + f"; penalty {a.penalty}, min_share {a.min_share}; device {dev}")
    whisper = None
    for f in files:
        fid = f.stem
        feat_file = Path(a.features) / f"{fid}.npy" if a.features else None
        if feat_file is not None and feat_file.exists():
            x = np.load(feat_file)
        else:
            whisper = whisper or WhisperL28(dev)
            x = whisper(load_audio(f))
            if a.save_features:
                (out / "features").mkdir(exist_ok=True)
                np.save(out / "features" / f"{fid}.npy", x)
        xt = torch.from_numpy(x).to(dev, torch.float32)
        with torch.no_grad():
            if len(taggers) == 1:
                lp = taggers[0][0].session_logprobs(xt).cpu().numpy()
            else:
                prob = sum(m.session_logprobs(xt).exp() for m, _ in taggers) / len(taggers)
                lp = torch.log(prob.clamp_min(1e-12)).cpu().numpy().astype(np.float32)
        allowed = session_languages(lp, a.min_share)
        path = viterbi(lp, a.penalty, allowed)
        segs = path_to_segments(path, NAMES)
        write_rttm(out / "rttm" / f"{fid}_pred.rttm", fid, segs)
        if a.save_posteriors:
            (out / "posteriors").mkdir(exist_ok=True)
            np.save(out / "posteriors" / f"{fid}.npy", lp.astype(np.float16))
        langs = [NAMES[i] for i in np.flatnonzero(allowed) if i > 0]
        print(f"  {fid}: {len(x) * 0.04:.0f} s, {len(segs)} segments, allowed languages {langs}", flush=True)
    (out / "run.json").write_text(json.dumps(dict(ckpts=a.ckpt, penalty=a.penalty, min_share=a.min_share,
                                                  whisper=f"{REPO}@{REVISION} block {LAYER}", files=len(files)), indent=1))


if __name__ == "__main__":
    main()
