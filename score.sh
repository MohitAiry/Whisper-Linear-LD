#!/bin/bash
# Score hypothesis RTTMs against reference RTTMs with the bundled dscore (github.com/nryant/dscore @ 824f126).
#   bash score.sh REF_DIR HYP_DIR
# Prints the four settings used in our results:
#   DER_ign = collar 0.25 s, overlapped speech excluded   (our primary table setting)
#   DER     = collar 0.25 s, overlapped speech scored
#   DER_c0  = no collar, overlapped speech scored          (official DISPLACE setting; use this for DISPLACE)
#   JER_c0  = Jaccard error rate, no collar
# Only reference files that have a hypothesis are scored. Needs python with numpy, scipy, intervaltree, tabulate.
set -e
REF=$(realpath "$1"); HYP=$(realpath "$2"); D=$(dirname "$(realpath "$0")")/dscore
PY=${PYTHON:-python}
refs=(); hyps=()
for h in "$HYP"/*.rttm; do
    id=$(head -1 "$h" | awk '{print $2}'); r=$(ls "$REF"/"$id".rttm "$REF"/"$id"_*.rttm 2>/dev/null | head -1)
    [ -n "$r" ] && { refs+=("$r"); hyps+=("$h"); } || echo "no reference for $id (skipped)" >&2
done
cd "$D"
run() { $PY score.py -r "${refs[@]}" -s "${hyps[@]}" "$@" 2>/dev/null | grep "OVERALL"; }
ign=$(run --collar 0.25 --ignore_overlaps | awk '{print $4}')
der=$(run --collar 0.25 | awk '{print $4}')
c0=$(run --collar 0 | awk '{print $4","$5}')
echo "files,DER_ign,DER,DER_c0,JER_c0"
echo "${#hyps[@]},$ign,$der,$c0"
