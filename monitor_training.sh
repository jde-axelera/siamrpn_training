#!/usr/bin/env bash
# =============================================================================
#  monitor_training.sh — auto-evaluate SiamRPN++ every 25 epochs
#  Fixes: (1) escaped [ in grep patterns, (2) space before epoch in summary
#         lines → use \[ * pattern, (3) catch-up on missed milestones
# =============================================================================
WORK_DIR="/home/ubuntu/siamrpn_training"
PYSOT_DIR="${WORK_DIR}/pysot"
EXPORT_DIR="${WORK_DIR}/exported"
EVAL_DIR="${WORK_DIR}/eval_results"
EVAL_INTERVAL=25
POLL_SECS=30
MAX_SEQS=20
MAX_FRAMES=150

PYTHON="${HOME}/miniconda3/envs/pysot/bin/python"
EXPORT_SCRIPT="${WORK_DIR}/export_onnx.py"
EVAL_SCRIPT="${WORK_DIR}/eval_onnx.py"
CONFIG="${PYSOT_DIR}/experiments/siamrpn_r50_alldatasets/config.yaml"
BEST_CKPT="${PYSOT_DIR}/snapshot/all_datasets/best_model.pth"

mkdir -p "${EVAL_DIR}"

RED='\033[0;31m'; GRN='\033[0;32m'; YEL='\033[0;33m'
CYN='\033[0;36m'; WHT='\033[1;37m'; RST='\033[0m'

log() { echo -e "[$(date '+%H:%M:%S')] $*"; }

find_latest_log() {
    ls -t "${PYSOT_DIR}/logs/all_datasets/"*.log 2>/dev/null | head -1
}

# Summary lines look like: "Epoch [ 91/500]  train=..."
# Step lines look like:    "Epoch[92] step[150/312]"
# We only want completed-epoch numbers from summary lines.
# Pattern: "Epoch [" (with space) then optional space then digits then "/"
current_epoch_from_log() {
    grep -oP 'Epoch \[ *\K[0-9]+(?=/)' "$1" 2>/dev/null | tail -1
}

val_loss_for_epoch() {
    # $2 = epoch number; match "Epoch [ 75/..." allowing optional space
    grep -P "Epoch \[ *$2/" "$1" 2>/dev/null \
        | grep -oP 'val=\K[0-9]+\.[0-9]+' | tail -1
}

train_loss_for_epoch() {
    grep -P "Epoch \[ *$2/" "$1" 2>/dev/null \
        | grep -oP 'train=\K[0-9]+\.[0-9]+' | tail -1
}

epoch_finished() {
    # Returns 0 (true) if the summary line for epoch $2 is in log $1
    grep -qP "Epoch \[ *$2/" "$1" 2>/dev/null
}

print_trend_table() {
    echo ""
    echo -e "${WHT}╔══════════════════════════════════════════════════════════════════╗${RST}"
    echo -e "${WHT}║        SiamRPN++ Training Progress — IoU Trend (25-ep)          ║${RST}"
    echo -e "${WHT}╠════════╦══════════╦═══════════════╦══════════╦════════════╦═════╣${RST}"
    printf "${WHT}║ %6s ║ %8s ║ %13s ║ %8s ║ %10s ║ %-4s║${RST}\n" \
           "Epoch" "Val Loss" "Mean IoU" "Succ@0.5" "AUC" "Bar"
    echo -e "${WHT}╠════════╬══════════╬═══════════════╬══════════╬════════════╬═════╣${RST}"

    for jf in $(ls "${EVAL_DIR}"/epoch_*.json 2>/dev/null | sort -V); do
        ep=$(${PYTHON} -c "import json; d=json.load(open('${jf}')); print(d.get('epoch',0))" 2>/dev/null)
        iou=$(${PYTHON} -c "import json; d=json.load(open('${jf}')); print(d['overall'].get('mean_iou',0))" 2>/dev/null)
        suc=$(${PYTHON} -c "import json; d=json.load(open('${jf}')); print(d['overall'].get('success_rate@0.5',0))" 2>/dev/null)
        auc=$(${PYTHON} -c "import json; d=json.load(open('${jf}')); print(d['overall'].get('auc',0))" 2>/dev/null)
        vl=$(${PYTHON} -c "import json; d=json.load(open('${jf}')); print(d.get('val_loss','?'))" 2>/dev/null)
        tl=$(${PYTHON} -c "import json; d=json.load(open('${jf}')); print(d.get('train_loss','?'))" 2>/dev/null)
        iou_f=$(echo "$iou" | awk '{printf "%.4f", $1}')
        if awk "BEGIN{exit !($iou >= 0.50)}"; then col="${GRN}"
        elif awk "BEGIN{exit !($iou >= 0.30)}"; then col="${YEL}"
        else col="${RED}"; fi
        bar_len=$(awk "BEGIN{n=int($iou*20); if(n<1)n=0; print n}")
        bar=$(${PYTHON} -c "print('█'*${bar_len} or '·')")
        printf "${col}║ %6s ║ %8s ║ %13s ║ %8s ║ %10s ║ %-4s║${RST}\n" \
               "$ep" "$vl" "$iou_f" \
               "$(awk "BEGIN{printf \"%.1f%%\", $suc*100}")" \
               "$(echo "$auc" | awk '{printf "%.4f", $1}')" \
               "${bar}"
    done
    echo -e "${WHT}╚════════╩══════════╩═══════════════╩══════════╩════════════╩═════╝${RST}"
    echo ""
}

run_eval() {
    local epoch="$1" logfile="$2"
    local eval_json="${EVAL_DIR}/epoch_$(printf '%04d' ${epoch}).json"
    if [ -f "${eval_json}" ]; then
        log "Epoch ${epoch}: already evaluated — skipping."; return
    fi

    log "${CYN}━━━ Epoch ${epoch}: exporting ONNX from best_model.pth ━━━${RST}"
    if [ ! -f "${BEST_CKPT}" ]; then
        log "${RED}best_model.pth not found — skipping${RST}"; return
    fi

    ${PYTHON} "${EXPORT_SCRIPT}" \
        --cfg "${CONFIG}" --ckpt "${BEST_CKPT}" --out "${EXPORT_DIR}" 2>&1 \
        | grep -v 'TracerWarning\|UserWarning\|device_discovery'

    [ $? -ne 0 ] && { log "${RED}ONNX export failed${RST}"; return; }

    log "Running IoU eval on all test sequences (epoch ${epoch})..."
    local vl tl
    vl=$(val_loss_for_epoch "${logfile}" "${epoch}")
    tl=$(train_loss_for_epoch "${logfile}" "${epoch}")

    ${PYTHON} "${EVAL_SCRIPT}" \
        --work_dir "${WORK_DIR}" --onnx_dir "${EXPORT_DIR}" \
        --out_json "${eval_json}" --epoch "${epoch}" \
        --max_seqs "${MAX_SEQS}" --max_frames "${MAX_FRAMES}" 2>&1 \
        | grep -v 'device_discovery\|UserWarning'

    # Stamp val/train loss into JSON
    if [ -f "${eval_json}" ]; then
        ${PYTHON} - <<PYEOF
import json
d = json.load(open('${eval_json}'))
d['val_loss']   = '${vl}'
d['train_loss'] = '${tl}'
json.dump(d, open('${eval_json}', 'w'), indent=2)
PYEOF
        log "${GRN}Epoch ${epoch} eval saved → ${eval_json}${RST}"
    else
        log "${RED}Eval JSON not created — check eval_onnx.py output${RST}"
    fi
}

# ── main ──────────────────────────────────────────────────────────────────────
log "Monitor started — eval every ${EVAL_INTERVAL} epochs"
log "Epoch-48 baseline already evaluated. Catching up on any missed milestones..."
echo ""

# Ep48 was already evaluated manually; start catch-up from there
last_evaluated=48

# Initial catch-up: check all missed milestones right now
logfile=$(find_latest_log)
if [ -n "${logfile}" ]; then
    cur=$(current_epoch_from_log "${logfile}")
    if [ -n "${cur}" ]; then
        # Iterate every missed multiple of EVAL_INTERVAL between last_evaluated+1 and cur
        next=$(( (last_evaluated / EVAL_INTERVAL + 1) * EVAL_INTERVAL ))
        while [ "${next}" -le "${cur}" ]; do
            if epoch_finished "${logfile}" "${next}"; then
                log "Catch-up: running eval for epoch ${next}"
                run_eval "${next}" "${logfile}"
                last_evaluated="${next}"
            fi
            next=$(( next + EVAL_INTERVAL ))
        done
    fi
fi

print_trend_table

# ── polling loop ──────────────────────────────────────────────────────────────
poll_count=0
while true; do
    sleep "${POLL_SECS}"
    logfile=$(find_latest_log)
    [ -z "${logfile}" ] && { log "No log found — waiting..."; continue; }

    cur=$(current_epoch_from_log "${logfile}")
    [ -z "${cur}" ] && continue

    # Check every missed milestone since last_evaluated
    next=$(( (last_evaluated / EVAL_INTERVAL + 1) * EVAL_INTERVAL ))
    while [ "${next}" -le "${cur}" ]; do
        if epoch_finished "${logfile}" "${next}"; then
            run_eval "${next}" "${logfile}"
            last_evaluated="${next}"
            print_trend_table
        fi
        next=$(( next + EVAL_INTERVAL ))
    done

    # Live status every ~5 min
    poll_count=$(( (poll_count + 1) % 10 ))
    if [ "${poll_count}" -eq 0 ]; then
        vl=$(val_loss_for_epoch "${logfile}" "${cur}")
        tl=$(train_loss_for_epoch "${logfile}" "${cur}")
        nxt=$(( (last_evaluated / EVAL_INTERVAL + 1) * EVAL_INTERVAL ))
        log "[live] epoch=${cur}  train=${tl:-?}  val=${vl:-?}  next_eval=ep${nxt}"
    fi
done
