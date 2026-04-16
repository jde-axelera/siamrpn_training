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

## Debugging the Tracker

### Debug script

```bash
cd /data/siamrpn_training
python debug_siamrpn.py \
    --cfg   pysot/experiments/siamrpn_r50_alldatasets/config.yaml \
    --ckpt  pysot/snapshot/all_datasets/best_model.pth \
    --video ir_crop.mp4 \
    --csv   ir_crop_best_model_rot90ccw_results.csv \
    --out   debug_best_model.pdf \
    --init_box 339 148 391 232
```

Generates an 11-page PDF. See `debug_siamrpn.py`.

### What each page shows

| Page | Content |
|------|---------|
| 1 | Overview: score, bbox W/H, area, centre drift, rolling score volatility over all 5966 frames. Red lines mark the 8 debug frames. |
| 2–9 | Per-frame analysis (4×3 grid) at frames 100, 750, 1650, 3250, 3900, 4050, 4800, 5100 |
| 10 | Anchor shapes: 5 aspect ratios (0.33, 0.5, 1.0, 2.0, 3.0) visualised on 255×255 search canvas + full 25×25 centre grid |
| 11 | Diagnosis table: per-frame metrics with colour-coded anomaly detection |

**Per-frame page layout (4 rows × 3 columns):**

| Row | Col 0 | Col 1 | Col 2 |
|-----|-------|-------|-------|
| 0 | Full rotated frame — green=bbox, yellow=search area | Template crop (127×127, frozen at frame 0) | Search crop (255×255) with decoded best bbox |
| 1 | **CLS heatmap** — correlation response (fg prob, max over 5 anchors). Sharp peak = model localising. Diffuse = lost. Red `+` = best anchor. | Penalized score map (after scale/ratio penalty + cosine window) | Top-20 candidate bboxes ranked by penalized score |
| 2 | **dw heatmap** — log width scale at each search location. Uniformly red = model predicts large boxes everywhere. | **dh heatmap** — same for height | W/H distribution of top-100 decoded candidates vs current state (dashed lines) |
| 3 | Neck feature map scale-0 (layer2 backbone, mean over 256ch) | Neck scale-1 (layer3) | Neck scale-2 (layer4) |

### Key debug images

**Score and bbox trajectory:**

![Overview](docs/debug_overview.jpg)

**Frame 100 — good tracking (sharp correlation peak):**

![Frame 100](docs/debug_frame_100.jpg)

**Frame 750 — first score drop (diffuse correlation response):**

![Frame 750](docs/debug_frame_750.jpg)

**Frame 3900 — recovery (sharp peak re-emerges):**

![Frame 3900](docs/debug_frame_3900.jpg)

**Frame 5100 — total loss (flat response map):**

![Frame 5100](docs/debug_frame_5100.jpg)

**Anchor shapes and grid:**

![Anchors](docs/debug_anchors.jpg)

### Debug video

`debug_annotated.mp4` — per-frame 3-panel video showing:
- Left: full frame with bbox (green), search area (yellow), score bar, frame counter
- Centre: search crop with correlation heatmap overlaid + best anchor marker
- Right: dw/dh regression heatmaps side-by-side

```bash
python debug_video.py \
    --cfg   pysot/experiments/siamrpn_r50_alldatasets/config.yaml \
    --ckpt  pysot/snapshot/all_datasets/best_model.pth \
    --video ir_crop.mp4 \
    --out   debug_annotated.mp4 \
    --init_box 339 148 391 232
```

---

## Findings from Debug Analysis

### 1 — Diffuse correlation response causes score drops

The CLS heatmap (Pages 3, 4, 5 of debug PDF) shows a **flat, spatially uniform** response when score < 0.4. The model has no confident spatial peak — it cannot distinguish the target from the IR background. This happens when the search patch no longer contains the target's original appearance (drift accumulation, or abrupt target motion pushing it to the edge of the search area).

### 2 — Consistently positive dw/dh drives bbox growth

At **every** debug frame, the dw and dh regression maps are predominantly positive (warm colours). This means the MultiRPN head predicts `exp(dw) > 1.0` almost everywhere in the search image — systematically larger than the anchor size. After the smooth state update (`width = size*(1−lr) + pred_w*lr`), this adds a few pixels to bbox W/H every frame. Over 5966 frames this compounds to the 5.8× area growth observed.

**Root cause hypothesis:** training data objects (Anti-UAV410 drones) are larger relative to the search crop than this ground vehicle target. The RPN biases toward predicting large boxes because that minimised training loss.

### 3 — Search area growth is a positive feedback loop

`s_x ∝ sqrt(w × h)`. As the bbox grows → s_x grows → the search crop covers more of the image → the model sees more background → correlation response degrades → score drops → bbox grows faster (low-penalty large predictions accepted). Pages 2–4 of the debug PDF show `s_x` growing from ~160 at frame 100 to >400 at frame 4000.

### 4 — Neck features degrade at wrong scale

When `s_x` is 3× the correct value the target occupies only ~9% of the search crop instead of the expected ~50%. The neck features (Row 3 of each debug page) show increasingly diffuse, background-dominated activations as `s_x` grows. The correlation with the (unchanged) template becomes meaningless.

### 5 — Template is stale after long drift

The template is frozen at frame 0. IR targets change appearance over a 4-minute sequence (altitude, aspect angle, thermal emission). By frame 3000+ the template appearance no longer matches the current target. Dynamic template update would recover this.

---

## Recommendations

### Immediate fixes (no retraining required)

#### R1 — EMA template update with size gate

Update the template features slowly when tracking is confident and bbox hasn't grown too large. Implemented in `run_video_inference_updated.py`.

```python
UPDATE_THRESH  = 0.75   # score gate
MAX_SIZE_RATIO = 1.5    # don't update if bbox > 1.5× initial
EMA_ALPHA      = 0.015  # blend rate (very slow)
FROZEN_S_Z     = s_z_at_init  # always crop at init scale, never inflated scale

size_ratio = max(self.size[0]/init_w, self.size[1]/init_h)
if score[best_idx] > UPDATE_THRESH and size_ratio < MAX_SIZE_RATIO:
    new_z = get_subwindow(img, center, EXEMPLAR_SIZE, FROZEN_S_Z, avg)
    new_zf = model.neck(model.backbone(new_z))
    model.zf = [(1-EMA_ALPHA)*o + EMA_ALPHA*n for o, n in zip(model.zf, new_zf)]
```

**Critical:** use the frozen initial `s_z` (not the current inflated one) so the crop scale stays consistent with the template.

#### R2 — Hard clamp on search area growth

Cap `s_x` to `max_s_x_factor × s_z_init` to break the feedback loop:

```python
MAX_SX_FACTOR = 2.5
s_x = min(s_x, MAX_SX_FACTOR * s_z_init)
```

#### R3 — Increase scale penalty

Raise `PENALTY_K` from 0.05 to 0.15–0.20 in `config.yaml`. This more aggressively penalises RPN candidates that are very different in size from the current state, reducing the rate of bbox growth per frame.

#### R4 — Score-gated bbox size update

Only accept bbox size changes when score is high:

```python
SIZE_UPDATE_THRESH = 0.55
if score[best_idx] >= SIZE_UPDATE_THRESH:
    width  = size[0] * (1 - lr) + pred_w * lr
    height = size[1] * (1 - lr) + pred_h * lr
else:
    width, height = size[0], size[1]  # freeze size, only move centre
```

### Training improvements (require retraining)

| Issue | Fix |
|-------|-----|
| Model biased toward large bbox predictions | Add bbox area regularisation loss; balance training data by target/search area ratio |
| Template gets stale | Add online fine-tuning samples to training (UpdateNet approach) |
| Single-video IR data | Add more diverse IR sequences at different altitudes, speeds, backgrounds |
| RGB-domain bias | Ensure all training data is thermal-only (Anti-UAV410 is mixed IR quality) |
| Early stopping at epoch 123 | Train with spatial stability metric (not just cls+loc loss) as early-stop criterion |

### Architecture improvements

| Improvement | Description |
|-------------|-------------|
| **Dual template** | Keep `zf_init` (frozen) + `zf_dynamic` (EMA updated). Run RPN head on both, merge scores with `w=0.6/0.4`. Prevents complete drift even if dynamic template degrades. |
| **Template gallery** | Ring buffer of 5 high-confidence templates; each frame select the one giving highest score on current search. Handles appearance variation without drift risk. |
| **Search scale adaptation** | Predict the likely new scale from `loc` output before expanding `s_x`; constrain `s_x` to be consistent with the predicted `dw/dh`. |

---

## References

- **SiamRPN++**: Li et al., CVPR 2019 — [paper](https://arxiv.org/abs/1812.11703)
- **PySOT**: [github.com/STVIR/pysot](https://github.com/STVIR/pysot)
- **Anti-UAV410**: [github.com/HwangBo94/Anti-UAV410](https://github.com/HwangBo94/Anti-UAV410)
- **MSRS**: [github.com/Linfeng-Tang/MSRS](https://github.com/Linfeng-Tang/MSRS)
- **PFTrack**: [github.com/wqw123wqw/PFTrack](https://github.com/wqw123wqw/PFTrack)
- **MassMIND**: [github.com/uml-marine-robotics/MassMIND](https://github.com/uml-marine-robotics/MassMIND)
- **MVSS-Baseline**: [github.com/jiwei0921/MVSS-Baseline](https://github.com/jiwei0921/MVSS-Baseline)
