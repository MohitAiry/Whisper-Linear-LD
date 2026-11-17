#!/bin/bash
# Train, select and evaluate one seed of the Whisper + linear-probe system (the recipe of checkpoints/linear_p11_s*).
#   bash scripts/train_linear.sh <seed> [run_dir]          (default run_dir: runs/linear_s<seed>)
# Needs Whisper block-28 features for ldsim_train, pilot, family_ld and displace (scripts/ld_extract_whisper.py).
# CPU is enough (the probe has 16.7k parameters); set PYTHON to choose the interpreter.
#   1. train 30 epochs of 2,000 random 20 s crops on LD_simulated train (sessions 1-350) + pilot train, keeping
#      13 epoch checkpoints  2. p1.1 selection: checkpoint x decoder grid on family_ld dev frame DER -> sel.pt
#   3. decode the test sets with the chosen decoder  4. dscore (scripts/score_run.sh)
set -eo pipefail
cd "$(dirname "$(realpath "$0")")/.."
export PYTHONPATH=$PWD OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
PY=${PYTHON:-python}; seed=${1:?seed}; run=${2:-runs/linear_s$seed}
[ -f $run/done.json ] || { rm -rf $run; $PY -I scripts/ld_frame_train.py --run-dir $run --seed $seed \
    --train ldsim_train:train pilot:train --layers -1 --workers 2 --epochs 30 --avg-last 5 --val-every 5 \
    --save-epochs 1,2,3,4,5,6,8,10,12,15,20,25,30; }
[ -f $run/selection.json ] || $PY -I scripts/ld_frame_select.py --run-dir $run --workers 3
[ -f $run/eval_p11/decoder.json ] || $PY -I scripts/ld_frame_eval.py --run-dir $run --protocol p11
bash scripts/score_run.sh $run/eval_p11
echo "=== $run done"
