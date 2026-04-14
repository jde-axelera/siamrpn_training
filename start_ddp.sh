#!/bin/bash
cd /data/siamrpn_training
mkdir -p logs
LOGFILE=logs/train_ddp_20260414_144020.log
exec /data/miniconda3/envs/pysot/bin/torchrun     --nproc_per_node=4     train_siamrpn_aws.py     --cfg pysot/experiments/siamrpn_r50_alldatasets/config.yaml     --pretrained pretrained/sot_resnet50.pth     2>&1 | tee "$LOGFILE"
