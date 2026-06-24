#!/bin/bash
set -e
cd /data/siamrpn_training
PY=/data/miniconda3/envs/pysot/bin/python
TORCHRUN=/data/miniconda3/envs/pysot/bin/torchrun
CFG=pysot/experiments/siamrpn_r50_alldatasets/config.yaml
PRETRAINED=pretrained/sot_resnet50.pth
mkdir -p logs

echo ========================================
echo  PHASE 1: Single-GPU overfit
echo ========================================
$PY overfit_test.py     --cfg $CFG --pretrained $PRETRAINED     --samples 32 --epochs 80 --lr 0.005     --out overfit_1gpu.csv     2>&1 | tee logs/overfit_1gpu.log

echo ========================================
echo  PHASE 2: 4-GPU DDP overfit
echo ========================================
$TORCHRUN --nproc_per_node=4 overfit_test.py     --cfg $CFG --pretrained $PRETRAINED     --samples 32 --epochs 80 --lr 0.005     --out overfit_4gpu.csv     2>&1 | tee logs/overfit_4gpu.log

echo ========================================
echo  COMPARISON
echo ========================================
$PY overfit_test.py --compare overfit_1gpu.csv overfit_4gpu.csv
