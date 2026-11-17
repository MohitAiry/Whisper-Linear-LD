# whisper-linear-ld: language diarization with a frozen Whisper encoder and a linear probe

Frame-level **language diarization** (LD) for Indian code-switched speech: given a recording, output *which language
is spoken when*, as an RTTM file. It supports 10 languages: `EN HI KN ML TA TE AS BN MR OD`.

The system is deliberately small:

```
audio 16 kHz ─► frozen Whisper-large-v3 encoder, block 28 of 32 ─► 25 Hz features (1280-d)
          ─► linear probe (LayerNorm + Linear 1280→11, 16.7k parameters), 5-seed posterior ensemble
          ─► session language pruning (τ = 0.1) ─► Viterbi with a switch penalty (λ = 32 nats) ─► RTTM
```

**Zero-shot on DISPLACE:** no DISPLACE data is used for training, model selection or decoder tuning.

---

## 1. Results

DER %, scored with the bundled dscore. Each cell is **lenient / strict**:
- **lenient:** collar 0.25 s, overlapping speech excluded;
- **strict:** no collar, overlapping speech scored. This is the official DISPLACE setting.

| System | family_ld test (19 files) | DISPLACE dev (35 files, zero-shot) |
|---|---|---|
| **Linear probe, 5-seed ensemble** (`infer.py` default) | **6.9 / 10.2** | **14.4 / 26.8** |
| Linear probe, single seed (mean of 5) | 7.2 / 10.5 | 14.8 / 27.2 |
| Whisper's own language ID, 3 s windows, same decoder (no training) | 18.3 / 21.8 | 24.7 / 35.4 |
| One language label per session (oracle speech regions) | 43.2 / 43.4 | 33.4 / 38.1 |

**95% confidence intervals** (bootstrap over test sessions), strict DER of the ensemble:
- family_ld test: 6.7–14.1;
- DISPLACE: 24.1–30.2.

The trained probe beats Whisper's own language ID by −11.3 strict DER on family_ld (CI −17.2 to −6.3) and by −8.2 on DISPLACE (CI −10.8 to −6.1). Full statistics are in `results/confidence_final.json`.

**Per seed** (`checkpoints/linear_p11_s<seed>/`). Every seed chose the same decoder, λ = 32 and τ = 0.1, on family_ld dev.

| Seed | Selected epoch | family_ld dev frame DER | family_ld test | DISPLACE |
|---|---|---|---|---|
| 0 | 2 | 8.85 | 6.83 / 10.20 | 14.28 / 26.74 |
| 1 | 2 | 8.76 | 6.88 / 10.25 | 14.69 / 27.11 |
| 2 | 3 | 8.75 | 7.21 / 10.60 | 14.15 / 26.63 |
| 3 | 5 | 8.95 | 7.82 / 11.26 | 16.42 / 28.72 |
| 4 | 2 | 8.83 | 6.69 / 10.07 | 14.45 / 26.94 |

**Context.** The DISPLACE 2024 winner reports 28.2 on the *2024* dev set after in-domain fine-tuning. Our DISPLACE set is the dev set we hold (the 2023 release), so the numbers are not directly comparable.

---

## 2. Quick start: audio → RTTM

```bash
pip install -r requirements.txt          # Python 3.10; a CUDA GPU is recommended (CPU works, slowly)
python infer.py --audio my_dir_or_files.wav --out out/
bash score.sh REF_RTTM_DIR out/rttm      # prints DER_ign, DER, DER_c0, JER_c0
```

**First run:** `openai/whisper-large-v3` is downloaded at the pinned revision `06f233fe06e710322aca913c1bc4249a0d71fce1`, about 3 GB.

**Output:**
- one RTTM per input file, `out/rttm/<file id>_pred.rttm`;
- the line format is `SPEAKER <file id> 1 <start> <dur> <NA> <NA> <LANG> <NA> <NA>`;
- non-speech is not written.

**Options:**
- `--ckpt a.pt b.pt …`: use other checkpoints instead of the 5-seed ensemble;
- `--penalty`, `--min-share`: change the decoder settings;
- `--save-features`, `--save-posteriors`: keep the intermediate outputs;
- `--features DIR`: reuse saved features and skip Whisper.

**Input:** 16 kHz mono is expected; other sample rates are resampled.

**Scoring** uses the optimal one-to-one label mapping, so anonymous reference labels (e.g. DISPLACE's `L1`/`L2`) work. `score.sh` pairs files by the file ID inside the RTTM. Reference files may be named `<id>.rttm` or `<id>_*.rttm`.

---

## 3. How it works

| Step | Input → output | Time resolution |
|---|---|---|
| Whisper-large-v3 encoder, truncated after block 28 (frozen). Windows of 30 s with a 25 s hop; each window keeps its central frames (2.5 s margins) | waveform → (T₅₀, 1280) | 50 Hz (20 ms) |
| Mean of frame pairs, rounded to float16 | → (T, 1280) | 25 Hz (40 ms) |
| Linear probe: LayerNorm + Linear(1280 → 11), log-softmax | → (T, 11) log-probabilities over {non-speech, 10 languages} | 25 Hz |
| Ensemble: mean of the 5 seeds' probabilities | → (T, 11) | 25 Hz |
| Session language pruning: keep languages with ≥ τ of the argmax speech frames | → allowed set | session |
| Viterbi over {non-speech, allowed languages}, λ nats per change of state | → one label per frame | 25 Hz |
| Runs of equal language labels | → RTTM segments | 40 ms grid |

**Why a linear probe and not a sequence model?** We compared BiLSTM, per-frame MLP and linear heads with the same data, selection and decoder.
- **Frame-wise heads were best:** a BiLSTM was +2.2 strict DER worse than a per-frame MLP on family_ld and +5.7 worse on DISPLACE. The linear probe matched the MLP with a fraction of its parameters and lower seed variance, so it is the default.
- **Temporal context hurts:** it lets a model learn the splice cues of simulated training data, which don't exist in real speech.
- **The probe is stable:** its dev-set curve is flat over epochs, while the BiLSTM's DER rises with longer training.

**Error profile on DISPLACE (strict, ensemble):**
- missed overlapping speech: 6.3;
- false alarms in silence: 6.3;
- language confusion: 14.2, mostly short word-level switches.

---

## 4. Re-training and re-evaluating

### 4.1 Data

The audio and reference RTTMs are **not** in this repository. Put them under `data/`, or anywhere else and set `LD_DATA=/path`:

```
$LD_DATA/audio/<set>/<session>.wav          16 kHz mono
$LD_DATA/rttm/<set>/<session><suffix>.rttm  reference language RTTMs (suffix "_LANGUAGE" for displace, else none)
$LD_DATA/features/<set>/<session>.npy       written by scripts/ld_extract_whisper.py
```

| `<set>` | Used for | Content |
|---|---|---|
| `ldsim_train` | training (sessions `sim_train_0001`–`0350`); `0351`–`0450` are validation | LD_simulated train: spliced monolingual clips, 10 languages |
| `pilot` | training (`pilot_train_*`); `pilot_dev_*` validation; `pilot_test_*` test | our synthetic code-switched conversations (5 languages, 8.06 h) |
| `family_ld` | **decoder and epoch selection** (dev split) and **test** (test split) | real EN/KN/ML family conversations, 27 files; split in `data/splits/family_ld_p1.json` |
| `displace` | test only | DISPLACE dev audio + language labels |
| `ldsim_test`, `ldsim_complex_test` | optional extra tests | LD_simulated test sets |

**Pilot set:** shared separately as the `omni_gen_data` folder. To link it into this layout, run `bash scripts/link_pilot.sh /path/to/omni_gen_data`.

**Session IDs:** the exact IDs of every training, selection and test set are in `data/splits/sessions_used.json`.

### 4.2 Steps

```bash
export LD_DATA=/path/to/data            # optional; default ./data
export PYTHONPATH=$PWD

# 1. Whisper block-28 features (GPU; ~230 MB per hour of audio, float16). Resumable.
python -I scripts/ld_extract_whisper.py --sets ldsim_train pilot family_ld displace

# 2. one seed: train 30 epochs -> p1.1 selection on family_ld dev -> decode test sets -> dscore (CPU is enough)
bash scripts/train_linear.sh 0          # -> runs/linear_s0/{sel.pt,selection.json,eval_p11/summary.csv}

# 3. the 5-seed posterior ensemble, with the decoder chosen again on family_ld dev
for s in 1 2 3 4; do bash scripts/train_linear.sh $s; done
python -I scripts/ld_decode_posteriors.py --ckpt runs/linear_s{0..4}/sel.pt --name linear_ens5
bash scripts/score_run.sh runs/ensemble/linear_ens5
```

**Training:**
- each epoch is 2,000 random crops of 500 frames (20 s), batch 32, AdamW (lr 1e-3, weight decay 0.01, gradient clipping at 5) and class-weighted cross-entropy;
- the data is LD_simulated sessions 1–350 plus the pilot's train split;
- 13 epoch checkpoints are kept (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30), plus the average of the last 5.

**Selection ("p1.1"):** every candidate checkpoint × decoder grid (λ ∈ {0, 2, 4, 8, 16, 32}, τ ∈ {0, .02, .05, .1}) is scored by frame DER on **family_ld dev only**. The best pair gives `sel.pt` and `selection.json`.

**What is never used for choices:** the test sets (family_ld test, DISPLACE) play no part in training, epoch selection or decoder tuning.

**To compare with our runs,** see `checkpoints/linear_p11_s*/`:
- `args.json`: training arguments;
- `train_log.jsonl`: the training curve;
- `selection.json`: the score of every candidate;
- `decoder.json`: the decoder grid;
- `summary.csv`: dscore results.

---

## 5. Repository contents

| Path | What it is |
|---|---|
| `infer.py` | stand-alone inference: audio → Whisper features → probe ensemble → pruning + Viterbi → RTTM |
| `score.sh`, `dscore/` | scoring; dscore from github.com/nryant/dscore (commit 824f126, BSD-2-Clause, `dscore/LICENSE`). One-line change: `dtype=int` instead of `np.int64` in `scorelib/metrics.py` |
| `checkpoints/linear_p11_s{0..4}/` | the 5 selected probes (`sel.pt`, 68 KB each) with their training and selection logs |
| `ldiar/frame_data.py` | data sets, 40 ms frame labels, training crops; data root from `LD_DATA` |
| `ldiar/frame_model.py` | `FrameTagger` (`layers=-1` is the linear probe), chunked inference, Viterbi, language pruning, segments |
| `scripts/ld_extract_whisper.py` | feature extraction |
| `scripts/ld_frame_train.py` | probe training |
| `scripts/ld_frame_select.py` | p1.1 selection |
| `scripts/ld_frame_eval.py` | decoding and frame DER on the test sets (`--protocol p11`) |
| `scripts/ld_decode_posteriors.py` | posterior ensemble of several checkpoints, decoder re-tuned on family_ld dev |
| `scripts/train_linear.sh` | the full recipe for one seed |
| `scripts/score_run.sh` | dscore for all test sets of a run |
| `scripts/link_pilot.sh` | links the shared pilot data set into the expected layout |
| `data/splits/` | family_ld dev/test split; session IDs of every set |
| `results/` | bootstrap CIs and paired tests (`confidence_final.json`), plus the head and training-data ablations. These use our own frame-exact scorer, which agrees with dscore to within ~0.1 DER |

**Verification:** this repository's code was checked against the original pipeline (2026-11-16):
- **evaluation:** re-evaluating seed 0 gives the same dscore numbers on all six test sets;
- **inference:** `infer.py` writes byte-identical RTTMs for all 35 DISPLACE files;
- **training:** a short training run loads all 454 training sessions.
