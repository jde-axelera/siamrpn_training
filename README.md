# SiamRPN++ Multi-Dataset IR Tracking — Training Pipeline

End-to-end training pipeline for finetuning **SiamRPN++** on infrared aerial
imagery using five public datasets. One script handles everything: environment
setup, data download, annotation conversion, training (500 epochs, multi-GPU),
best-model saving, and ONNX export (opset 17).

---

## Quick Start

```bash
# 1. Clone this repo onto your AWS instance
git clone <this-repo> && cd <this-repo>

# 2. (Optional) Fill in Google Drive IDs for MassMIND at the top of the script
#    MASSMIND_IMAGES_GDRIVE_ID="..."
#    MASSMIND_MASKS_GDRIVE_ID="..."

# 3. Run
chmod +x run_aws_training.sh
./run_aws_training.sh
```

The script is fully idempotent — re-running it skips steps that are already done.

---

## What the Script Does (12 Steps)

| Step | Action |
|------|--------|
| 1 | Install Miniconda if missing |
| 2 | Create conda env `pysot` (Python 3.10) |
| 3 | Clone PySOT, install PyTorch (CUDA 11.8) + all dependencies |
| 4 | Patch PySOT for NumPy 1.24+ and device-agnostic CUDA calls |
| 5a | Download **Anti-UAV410** (~9.4 GB) via gdown |
| 5b | Download **MSRS** via git clone + LFS |
| 5c | Download **MassMIND** via gdown (requires Drive IDs) |
| 5d | Check for **PFTrack/VT-MOT** (manual download, prints instructions) |
| 5e | Check for **MVSS-Baseline** (manual download, prints instructions) |
| 6 | Download `sot_resnet50.pth` pretrained backbone |
| 7 | Convert all dataset annotations → PySOT JSON format |
| 8 | Write training config YAML (all 5 datasets, 500 epochs) |
| 9 | Write training Python script |
| 10 | Write ONNX export Python script |
| 11 | **Run training** — logs to file + TensorBoard |
| 12 | **Export best checkpoint** → `template_encoder.onnx` + `tracker.onnx` |

---

## Datasets

| Dataset | Modality | Size | Annotations | Download |
|---------|----------|------|-------------|----------|
| [Anti-UAV410](https://github.com/HwangBo94/Anti-UAV410) | IR thermal | 410 sequences | SOT bounding boxes | Auto (gdown) |
| [MSRS](https://github.com/Linfeng-Tang/MSRS) | Paired IR + visible | 1,444 image pairs | Semantic segmentation → pseudo-sequences | Auto (git clone) |
| [PFTrack / VT-MOT](https://github.com/wqw123wqw/PFTrack) | RGB + IR | 582 sequences, 401K frames | MOT `gt.txt` → SOT per-object | Manual (Baidu Cloud) |
| [MassMIND](https://github.com/uml-marine-robotics/MassMIND) | LWIR maritime | 2,916 images | Instance segmentation masks → bboxes | Auto (gdown, needs IDs) |
| [MVSS-Baseline](https://github.com/jiwei0921/MVSS-Baseline) | RGB + thermal video | Multiple sequences | Semantic masks per frame → bboxes | Manual (request access) |

### Sampling Weights

Datasets are combined with weighted random sampling each epoch:

```
Anti-UAV410  ×3.0   ← primary IR tracking dataset
VT-MOT       ×2.0   ← large multi-object IR tracking
MVSS         ×1.5   ← thermal video sequences
MSRS         ×1.0
MassMIND     ×1.0
```

Weights are configurable in the `DATASET` section of the generated config YAML.

---

## Manual Downloads

### PFTrack / VT-MOT
1. Go to: `https://pan.baidu.com/s/1C8rXxVmxg6jAB7Xs7E45zw`
   Password: `chcw`
2. Extract to `~/siamrpn_training/data/vtmot/`
3. Expected structure:
```
vtmot/
  train/
    <sequence>/
      infrared/  *.jpg
      visible/   *.jpg
      gt/gt.txt
  test/
    <sequence>/
      ...
```

### MVSS-Baseline
1. Request access at: `https://github.com/jiwei0921/MVSS-Baseline`
2. Extract to `~/siamrpn_training/data/mvss/`
3. Expected structure:
```
mvss/
  sequences/
    <sequence>/
      thermal/  *.png
      labels/   *.png   (semantic mask, one file per frame)
```

### MassMIND Google Drive IDs
Find the file IDs at `https://github.com/uml-marine-robotics/MassMIND` and set
them at the top of `run_aws_training.sh`:
```bash
MASSMIND_IMAGES_GDRIVE_ID="<id from Google Drive share link>"
MASSMIND_MASKS_GDRIVE_ID="<id from Google Drive share link>"
```

---

## Configurable Variables

Edit the top of `run_aws_training.sh` before running:

```bash
WORK_DIR="${HOME}/siamrpn_training"   # where everything is stored
EPOCHS=500                            # training epochs
BATCH_SIZE=32                         # per-GPU batch size
NUM_WORKERS=8                         # dataloader workers
VIDEOS_PER_EPOCH=10000                # samples drawn per epoch
BASE_LR=0.005                         # initial learning rate
BACKBONE_TRAIN_EPOCH=10               # when to unfreeze backbone layers
```

---

## Recommended AWS Instance

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 1× V100 (16 GB) | 4× A100 (40 GB) |
| CPU | 8 vCPU | 32 vCPU |
| RAM | 64 GB | 128 GB |
| Storage | 100 GB SSD | 500 GB SSD |
| AMI | Deep Learning AMI (Ubuntu 22.04) | same |

On a single A100 (40 GB, batch 32): ~3–4 min/epoch → 500 epochs ≈ 26–32 hrs.

---

## Monitoring Training

```bash
# Live log tail
tail -f ~/siamrpn_training/pysot/logs/all_datasets/training_*.log

# TensorBoard (forward port 6006 via SSH first)
tensorboard --logdir ~/siamrpn_training/pysot/logs/all_datasets --port 6006

# GPU usage
watch -n 2 nvidia-smi
```

---

## Output Files

```
~/siamrpn_training/
├── pysot/
│   ├── snapshot/all_datasets/
│   │   ├── best_model.pth          ← lowest val-loss checkpoint
│   │   ├── checkpoint_e10.pth
│   │   ├── checkpoint_e20.pth
│   │   └── ...
│   └── logs/all_datasets/
│       ├── training_<timestamp>.log
│       └── events.out.tfevents.*   ← TensorBoard
├── exported/
│   ├── template_encoder.onnx       ← run once per target init  (input: 1×3×127×127)
│   └── tracker.onnx                ← run per frame             (inputs: zf_0/1/2 + 1×3×255×255)
└── pretrained/
    └── sot_resnet50.pth
```

### ONNX Model Interfaces

**`template_encoder.onnx`** — called once when the tracker is initialised on a target:
```
input  : template  (1, 3, 127, 127)   — cropped target patch
outputs: zf_0, zf_1, zf_2            — multi-scale neck features
```

**`tracker.onnx`** — called every frame:
```
inputs : zf_0, zf_1, zf_2, search (1, 3, 255, 255)
outputs: cls  (1, 10, 25, 25)   — classification scores (5 anchors × 2 classes)
         loc  (1, 20, 25, 25)   — location deltas       (5 anchors × 4 coords)
```

---

## Resume Training

If training is interrupted, resume from the latest checkpoint:

```bash
python ~/siamrpn_training/train_siamrpn_aws.py \
    --cfg    ~/siamrpn_training/pysot/experiments/siamrpn_r50_alldatasets/config.yaml \
    --resume ~/siamrpn_training/pysot/snapshot/all_datasets/checkpoint_e120.pth
```

---

## Manual ONNX Export

```bash
python ~/siamrpn_training/export_onnx.py \
    --cfg   ~/siamrpn_training/pysot/experiments/siamrpn_r50_alldatasets/config.yaml \
    --ckpt  ~/siamrpn_training/pysot/snapshot/all_datasets/best_model.pth \
    --out   ~/siamrpn_training/exported \
    --opset 17
```

---

## Annotation Format (PySOT JSON)

All datasets are converted to this format before training:

```json
{
  "sequence_name": {
    "0": {
      "000001": [x1, y1, x2, y2],
      "000002": [x1, y1, x2, y2]
    }
  }
}
```

- Outer key: sequence name
- `"0"`: track ID (always 0 for SOT)
- Inner key: zero-padded frame index (6 digits)
- Value: bounding box as `[x1, y1, x2, y2]` in pixel coordinates

---

## Model Comparison: Fine-tuned vs Official Pretrained

Evaluation on `ir_crop.mp4` (5966 frames, IR aerial, −90° rotation applied).
Both models initialised with the same bounding box at frame 0.
Scores are the SiamRPN++ foreground confidence after cosine-window penalty.

> **Verdict: mixed result — fine-tuning improved failure recovery but degraded stability.**

### Score statistics

| Metric | `best_model.pth` (fine-tuned) | `siamrpn_r50_l234_dwxcorr` (official) |
|---|---|---|
| Mean score | 0.625 | **0.701** |
| Median score | 0.693 | **0.897** |
| Std (volatility) | 0.282 | **0.362** |
| Frames ≥ 0.7 | 49.0 % | **70.2 %** |
| Frames ≥ 0.9 | 15.9 % | **49.1 %** |
| **Frames < 0.10 (total loss)** | **9.3 %** | 16.4 % |
| **Frames < 0.20 (low conf)** | **11.5 %** | 18.4 % |
| Score volatility (Δ/frame) | 0.094 | **0.040** |
| BBox centre drift median (px/frame) | 8.66 | **2.08** |
| Failure runs ≥ 5 frames | **21** | 26 |

### Per-segment breakdown (1000-frame chunks)

| Frames | Fine-tuned | Official | Winner |
|---|---|---|---|
| 0 – 1000 | 0.765 | **0.939** | official |
| 1000 – 2000 | 0.720 | **0.923** | official |
| 2000 – 3000 | 0.661 | **0.916** | official |
| 3000 – 4000 | 0.615 | **0.793** | official |
| **4000 – 5000** | **0.693** | 0.394 | **fine-tuned** |
| 5000 – 5966 | **0.285** | 0.224 | fine-tuned |

### Visual comparison

**Frame 100 — early tracking (both models on-target)**

![Early tracking](docs/comparison_early_tracking_f100.jpg)

**Frame 2500 — mid-video (official leading by a wide margin)**

![Mid video](docs/comparison_mid_video_f2500.jpg)

**Frame 3950 — official model collapses; fine-tuned model still active**

![Official failure](docs/comparison_official_failure_f3950.jpg)

**Frame 4400 — fine-tuned recovers; official still lost**

![Best recovery](docs/comparison_best_recovery_f4400.jpg)

### Interpretation

The official PySOT model was trained on millions of **RGB** frames (VID, YoutubeVOS,
COCO, ImageNetDet) and learned a stable, generalised tracking prior. It outperforms
the fine-tuned model on **4 out of 6 segments** and is **4× more spatially stable**
(2.1 vs 8.7 px/frame median drift) and **2.4× less score-erratic**.

Fine-tuning on IR data improved recovery in the hard late segment (frames 4000+),
where the RGB-trained model completely loses the target. However it degraded
everything else. Likely causes:

1. **Training data is too homogeneous** — single video or similar scenes → model
   memorises scene features rather than generalising to IR appearance.
2. **Insufficient regularisation** — the fine-tuned RPN over-fits the IR domain
   and loses the stable RGB prior without gaining a robust IR one.
3. **Early stopping at epoch 123** was correct by val-loss, but val-loss does not
   measure spatial stability.

**Next steps to beat the official model on all segments:**
- More diverse IR training sequences (multiple scenes, altitudes, weather)
- Data augmentation: random brightness/contrast for thermal, simulated noise
- Lower LR fine-tuning of the full model (backbone + neck + head jointly, LR ≈ 1e-4)
- Evaluate on a held-out IR sequence, not the same video used during training

---

## References

- **SiamRPN++**: Li et al., CVPR 2019 — [paper](https://arxiv.org/abs/1812.11703)
- **PySOT**: [github.com/STVIR/pysot](https://github.com/STVIR/pysot)
- **Anti-UAV410**: [github.com/HwangBo94/Anti-UAV410](https://github.com/HwangBo94/Anti-UAV410)
- **MSRS**: [github.com/Linfeng-Tang/MSRS](https://github.com/Linfeng-Tang/MSRS)
- **PFTrack**: [github.com/wqw123wqw/PFTrack](https://github.com/wqw123wqw/PFTrack)
- **MassMIND**: [github.com/uml-marine-robotics/MassMIND](https://github.com/uml-marine-robotics/MassMIND)
- **MVSS-Baseline**: [github.com/jiwei0921/MVSS-Baseline](https://github.com/jiwei0921/MVSS-Baseline)
