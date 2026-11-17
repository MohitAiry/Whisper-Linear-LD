#!/bin/bash
# Link the shared pilot data set (omni_gen_data/{train,dev,test}/{wav,rttm_language}) into the layout the code
# expects: $LD_DATA/audio/pilot/<session>.wav and $LD_DATA/rttm/pilot/<session>.rttm (symlinks, no copies).
#   bash scripts/link_pilot.sh /path/to/omni_gen_data
set -eo pipefail
SRC=$(realpath "${1:?path to omni_gen_data}"); DATA=${LD_DATA:-$(dirname "$(realpath "$0")")/../data}
mkdir -p "$DATA/audio/pilot" "$DATA/rttm/pilot"
for sp in train dev test; do
    for f in "$SRC/$sp/wav"/*.wav; do ln -sf "$f" "$DATA/audio/pilot/"; done
    for f in "$SRC/$sp/rttm_language"/*.rttm; do ln -sf "$f" "$DATA/rttm/pilot/"; done
done
echo "pilot: $(ls "$DATA/audio/pilot" | wc -l) wavs, $(ls "$DATA/rttm/pilot" | wc -l) RTTMs in $DATA"
