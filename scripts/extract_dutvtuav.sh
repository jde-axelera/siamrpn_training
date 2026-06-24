#!/bin/bash
# Extract all DUT-VTUAV zips into train/, renaming ir/ -> infrared/ and ir.txt -> groundtruth.txt
# Deletes each zip after extraction to save space.

DSET_DIR="/data/siamrpn_training/data/dut_vtuav"
TRAIN_DIR="${DSET_DIR}/train"
mkdir -p "${TRAIN_DIR}"

for ZIP in "${DSET_DIR}"/train_*.zip; do
    [ -f "$ZIP" ] || continue
    echo "[$(date +%H:%M:%S)] Extracting $(basename $ZIP) ..."
    unzip -q "$ZIP" -d "${TRAIN_DIR}"

    echo "[$(date +%H:%M:%S)] Renaming ir/ -> infrared/ and ir.txt -> groundtruth.txt ..."
    find "${TRAIN_DIR}" -mindepth 2 -maxdepth 2 -name "ir" -type d | while read IR_DIR; do
        mv "$IR_DIR" "$(dirname $IR_DIR)/infrared"
    done
    find "${TRAIN_DIR}" -mindepth 2 -maxdepth 2 -name "ir.txt" | while read IR_TXT; do
        mv "$IR_TXT" "$(dirname $IR_TXT)/groundtruth.txt"
    done

    echo "[$(date +%H:%M:%S)] Removing $ZIP ..."
    rm "$ZIP"

    echo "[$(date +%H:%M:%S)] Disk free: $(df -h /data | awk NR==2{print })"
    echo "---"
done

echo "Done. Sequences extracted:"
ls "${TRAIN_DIR}" | wc -l
