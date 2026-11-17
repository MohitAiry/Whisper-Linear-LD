#!/bin/bash
# dscore for the RTTMs written by scripts/ld_frame_eval.py or ld_decode_posteriors.py.
#   bash scripts/score_run.sh <eval_dir>        e.g. runs/linear_s0/eval_p11 or runs/ensemble/linear_ens5
# Scores each <eval_dir>/<test set>[/rttms] against $LD_DATA/rttm/<set> (default data/rttm); writes summary.csv.
set -eo pipefail
H=$(realpath "$1"); ROOT=$(dirname "$(realpath "$0")")/..; DATA=${LD_DATA:-$ROOT/data}
declare -A REF=([family_ld_test]=family_ld [family_ld_dev]=family_ld [family_ld]=family_ld [displace]=displace
                [ld_simulated]=ldsim_test [ld_simulated_complex]=ldsim_complex_test [pilot_test]=pilot)
echo "dataset,files,DER_ign,DER,DER_c0,JER_c0" > "$H/summary.csv"
for name in family_ld_test family_ld_dev family_ld displace ld_simulated ld_simulated_complex pilot_test; do
    d=$H/$name; [ -d $d/rttms ] && d=$d/rttms; [ -d $d ] || continue
    echo "$name,$(bash $ROOT/score.sh "$DATA/rttm/${REF[$name]}" "$d" | tail -1)" >> "$H/summary.csv"
done
cat "$H/summary.csv"
