# SiamRPN++ IR/Thermal Tracking — Fine-Tuning & Deployment Pipeline

Fine-tunes **SiamRPN++** (ResNet-50 backbone, FPN neck, multi-scale RPN head) on 9 IR/thermal datasets covering UAV tracking, maritime, aerial wildlife, and road scenes. Outputs two ONNX models for edge deployment on the [Axelera Voyager](https://www.axelera.ai) platform.

Built on [PySOT](https://github.com/STVIR/pysot). Designed to run on AWS GPU instances (CUDA 11.8).

---

## Table of Contents

1. [Background: From Detection to Tracking](#1-background-from-detection-to-tracking)
2. [Architecture: How SiamRPN++ Works](#2-architecture-how-siamrpn-works)
3. [Training: How It Differs from YOLO](#3-training-how-it-differs-from-yolo)
4. [Quick Start](#4-quick-start)
5. [Datasets](#5-datasets)
6. [Training Details](#6-training-details)
7. [ONNX Export](#7-onnx-export)
8. [Evaluation](#8-evaluation)
9. [Demo Videos](#9-demo-videos)
10. [Inference Hyperparameters](#10-inference-hyperparameters)
11. [Directory Layout](#11-directory-layout)
12. [Operations & Advanced](#12-operations--advanced)

---

## 1. Background: From Detection to Tracking

If you are familiar with YOLO-style detection, the key mental shift is this:

> **Detection** asks *"what objects are in this image?"*
> **Tracking** asks *"where did this specific object go in the next frame?"*

A YOLO model is trained to recognise categories (car, person, drone). At inference it scans a single image and outputs all bounding boxes by class. It has no memory — each frame is processed in isolation.

A **visual tracker** does something fundamentally different. You give it one reference image of a target — called the **template** — and it continuously localises that same instance across subsequent video frames, regardless of what category it belongs to. It never needs to know if the target is a drone, a bird, or a boat. It just asks: *"does this patch of pixels look like my template?"*

This distinction matters for IR/thermal deployment:
- IR imagery lacks the colour and texture cues that category-based detectors rely on.
- You often want to track a specific intruder drone, not detect all drones in the scene.
- Trackers are more robust to domain shift because they match appearance, not a learned class prototype.

---

## 2. Architecture: How SiamRPN++ Works

SiamRPN++ is a **Siamese network** — two branches sharing the same backbone weights that process two inputs simultaneously.

```
Template (127×127 px)          Search Region (255×255 px)
        │                                │
   ┌────▼────┐                      ┌────▼────┐
   │Backbone │  ResNet-50 + FPN     │Backbone │  (shared weights)
   └────┬────┘                      └────┬────┘
        │  zf (3 scales)                 │  xf (3 scales)
        │                                │
        └──────────┬─────────────────────┘
                   │
        ┌──────────▼──────────┐
        │  DW-XCorr per scale │  Depth-wise cross-correlation
        └──────────┬──────────┘
                   │  response maps
        ┌──────────▼──────────┐
        │     RPN Head        │  cls + loc branches
        └──────────┬──────────┘
                   │
        ┌──────────┴──────────┐
        │  cls (1,10,25,25)   │  ← 5 anchors × 2 (fg/bg) × 25×25 grid
        │  loc (1,20,25,25)   │  ← 5 anchors × 4 (dx,dy,dw,dh) × 25×25 grid
        └─────────────────────┘
```

### Key components

**Siamese backbone (shared weights)**
Both the template and search crops pass through the same ResNet-50. Because the weights are shared, both branches learn the same feature representation — similarity in feature space corresponds to visual similarity in image space.

**FPN neck (multi-scale)**
A Feature Pyramid Network aggregates features from layers 2, 3, and 4 of ResNet-50, giving three feature scales `zf_0 / zf_1 / zf_2`. This helps handle targets at different scales.

**Depth-wise cross-correlation (DW-XCorr)**
This is the core of the Siamese approach. The template features `zf` act as a *learned sliding filter* that is convolved over the search features `xf`. Positions in the resulting response map with high activation indicate where the search region looks most similar to the template.

In YOLO terms: instead of a fixed classifier head reading from backbone features, SiamRPN++ uses the *template itself* as a dynamic query kernel. The backbone learns to produce feature maps where cross-correlation naturally peaks at the target location.

**RPN head (same concept as YOLO)**
A region proposal network identical in spirit to what Faster-RCNN and YOLOv2+ use — a 25×25 spatial grid, 5 anchor aspect ratios per location (0.33, 0.5, 1.0, 2.0, 3.0), each predicting:
- `cls`: is this anchor on the target (fg) or background (bg)?
- `loc`: dx, dy, dw, dh offsets to refine the anchor to the exact box.

Total anchors per frame: 25 × 25 × 5 = **3,125** — all evaluated in a single forward pass.

### At inference

1. **Initialisation (once per target):** Crop the target from the first frame → run through the template encoder → cache `zf_0 / zf_1 / zf_2`.
2. **Every frame:** Crop a 255×255 search region centred on the last known position → run through the tracker model with the cached `zf` → decode the highest-scoring anchor → update position.

This is why export produces **two separate ONNX models** — the template encoder runs once, the tracker runs every frame.

---

## 3. Training: How It Differs from YOLO

| Aspect | YOLO training | SiamRPN++ training |
|--------|--------------|-------------------|
| Input | Single image per sample | (template crop, search crop) pair from the same video |
| Label | Class ID + bbox per object | Binary fg/bg + bbox for the *one* tracked object |
| Loss | Multi-class CE + box regression | Binary CE (focal) + smooth-L1 (same as RPN) |
| Dataset | Images with annotations | Video sequences with per-frame bbox tracks |
| Category awareness | Yes — 80 COCO classes | None — similarity-based, category-agnostic |
| Epoch structure | One pass through the dataset | Fixed budget of N random (template, search) pairs |
| Batch item | One image | One (z, x) pair sampled from a random sequence |

### Why pairs, not full frames?

The Siamese network must learn what *similarity* looks like, not what categories look like. To create a training signal, you need two crops of the same object:
- **Template** `z`: a 127×127 crop centred on the target with 0.5× context padding at frame `t`.
- **Search region** `x`: a 255×255 crop centred near the same target at frame `t + δ` (δ = 1–100 frames), with the target potentially shifted, scaled, or partially occluded.

The model must output high classification scores and accurate offsets at the anchor overlapping the target in `x`, and low scores everywhere else. All other anchors are background.

### Loss

```
L_total = L_cls + λ · L_loc

L_cls = Binary cross-entropy (weighted: ignore anchors with IoU 0.3–0.6, 
                                         positive: IoU > 0.6, 
                                         negative: IoU < 0.3)
L_loc = Smooth-L1 on (dx, dy, dw, dh) for positive anchors only
λ     = 1.2   (balances classification vs. localisation)
```

---

## 4. Quick Start

```bash
# 1. Clone this repo
git clone https://github.com/jde-axelera/siamrpn_training.git
cd siamrpn_training

# 2. Clone PySOT alongside
git clone https://github.com/STVIR/pysot.git

# 3. Run the full pipeline (downloads data, installs deps, trains, exports)
chmod +x run_aws_training.sh
./run_aws_training.sh
```

On AWS instances with a small root partition, redirect everything to a data volume:

```bash
./run_aws_training.sh --install-dir=/data
```

**Verify setup first** (smoke test — 1 epoch, no downloads):

```bash
./run_aws_training.sh --smoke-test
```

**Resume from a checkpoint:**

```bash
./run_aws_training.sh --resume pysot/snapshot/all_datasets/checkpoint_epoch050.pth
```

### Pipeline scripts

| Step | Script | Purpose |
|------|--------|---------|
| Full pipeline | `run_aws_training.sh` | Orchestrates all steps end-to-end |
| Training | `train_siamrpn_aws.py` | Multi-GPU fine-tuning with SGDR + early stopping |
| Export | `export_onnx.py` | Converts best checkpoint → two ONNX models |
| Evaluate | `eval_onnx.py` | IoU / AUC metrics on test splits |
| Monitor | `monitor_training.sh` | Auto-evaluates every 25 epochs during training |
| Demo | `run_onnx_tracker.py` | Annotated MP4 with GT + predicted boxes |
| GT demo | `make_test_demo.py` | Ground-truth-only demo video from test splits |
| Report | `generate_report.py` | Multi-page PDF training report |

---

## 5. Datasets

Nine IR/thermal datasets are used. All annotations are automatically converted to a unified PySOT JSON format during the pipeline setup step.

### Dataset inventory

| # | Dataset | Size | Domain | Source |
|---|---------|------|--------|--------|
| 1 | **Anti-UAV410** | 410 seqs, 9.4 GB | IR UAV tracking | Google Drive (`gdown`) |
| 2 | **Anti-UAV300** | IR video sequences | IR UAV tracking | Google Drive (`gdown`) |
| 3 | **MSRS** | 1,444 image pairs | Paired IR/visible road | GitHub (`git clone` + LFS) |
| 4 | **VT-MOT / PFTrack** | 582 seqs, 401K frames | RGB+IR multi-object | Baidu Cloud (code: `chcw`) |
| 5 | **MassMIND** | 2,916 LWIR images | Maritime LWIR | Google Drive (`gdown`) |
| 6 | **MVSS-Baseline** | Video sequences | RGB-thermal video seg | Google Drive (author approval) |
| 7 | **DUT-VTUAV** | 500 seqs, 1.7M frames | RGB+Thermal UAV | Google Drive (rclone — see §12) |
| 8 | **BIRDSAI** | 48 TIR sequences | Aerial wildlife thermal | LILA (`wget`) |
| 9 | **HIT-UAV** | 2,898 IR images | High-altitude IR detection | Kaggle CLI |

Datasets missing their annotation JSON (VT-MOT, MVSS, AntiUAV300, BIRDSAI, HIT-UAV) are automatically excluded at training time and the remaining sampling weights are renormalised.

### Unified annotation format

All datasets are converted to a single JSON structure regardless of their original format (MOT `gt.txt`, YOLO normalised coords, segmentation masks, MP4 video):

```json
{
  "sequence_name": {
    "0": {
      "000001": [x1, y1, x2, y2],
      "000002": [x1, y1, x2, y2],
      ...
    }
  }
}
```

The outer `"0"` key represents the object track ID (always 0 for single-object tracking). Frame keys are zero-padded 6-digit integers. Coordinates are pixel-space `[x1, y1, x2, y2]`.

### Dataset annotation samples

Sample IR frames with ground-truth bounding boxes from each training dataset (4 sequences × 4 frames — full report at [`report/datasets_annotation_check.pdf`](report/datasets_annotation_check.pdf)):

**Anti-UAV 410** — IR UAV targets (tiny, high-altitude)
![Anti-UAV 410](docs/ds_strip_Anti_UAV_410.jpg)

**MSRS** — Paired IR/visible road scenes (vehicles, pedestrians)
![MSRS](docs/ds_strip_MSRS.jpg)

**MassMIND** — Maritime LWIR (vessels)
![MassMIND](docs/ds_strip_MassMIND.jpg)

**DUT-VTUAV** — RGB+Thermal UAV (multi-category: vehicles, people, animals)
![DUT-VTUAV](docs/ds_strip_DUT_VTUAV.jpg)

---

## 6. Training Details

### System requirements

| Component | Requirement |
|-----------|-------------|
| OS | Ubuntu 20.04+ |
| GPU | CUDA 11.8-compatible (AWS `g4dn` or `p3` recommended) |
| Storage | ≥ 300 GB on data partition (datasets + checkpoints) |

Python packages are installed automatically. Key dependencies: `torch + torchvision` (CUDA 11.8), `opencv-python`, `yacs`, `onnx`, `onnxruntime`, `gdown`, `tensorboard`.

### Run training directly

After the pipeline has set up the environment and data:

```bash
conda activate pysot
python train_siamrpn_aws.py \
    --cfg  pysot/experiments/siamrpn_r50_alldatasets/config.yaml \
    --pretrained pretrained/sot_resnet50.pth
```

### Siamese pair sampling

Each epoch draws **10,000 (template, search) pairs** stochastically:

1. A dataset is chosen according to its `sample_prob` (see table below).
2. A random sequence is drawn from that dataset.
3. A valid frame pair `(t, t + δ)` is sampled — δ ∈ [1, 100] frames.
4. Template (127×127) and search (255×255) crops are extracted with random augmentation.

At batch size 32 this gives **≈ 312 gradient steps per epoch**, independent of total dataset size. This is intentional — it keeps each epoch a fixed, predictable wall-clock duration and lets you control total training cost via epoch count, not dataset size.

#### Dataset sampling probabilities

Probabilities are non-proportional — smaller, task-critical IR-specific datasets are upweighted to ensure the model sees balanced examples across object scales and thermal conditions:

| Dataset | Sequences | Config weight | `sample_prob`* | Rationale |
|---------|-----------|--------------|----------------|-----------|
| AntiUAV410 | 200 | 3.0 | 0.400 | Primary target — IR drone tracking, highest priority |
| DUT-VTUAV | 225 | 2.5 | 0.333 | Large-scale multi-category UAV, good diversity |
| MSRS | 541 | 1.0 | 0.133 | Paired IR/visible — ground-level object variety |
| MassMIND | 1,801 | 1.0 | 0.133 | Maritime domain — downweighted despite large size |
| VT-MOT | 582 | 2.0 | — | Active if annotation present; IR multi-object diversity |
| MVSS | — | 1.5 | — | Active if annotation present |
| Anti-UAV300 | — | 2.5 | — | Active if annotation present; same domain as AntiUAV410 |
| BIRDSAI | 48 | 1.5 | — | Active if annotation present; aerial wildlife thermal |
| HIT-UAV | 2,898 | 1.0 | — | Active if annotation present; high-altitude detection |

\* `sample_prob` shown for the 4 currently active datasets. Weights renormalise automatically over whatever datasets have annotation files present.

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Epochs | 500 | |
| Batch size | 32 | |
| Data workers | 8 | |
| Base LR | 0.005 | |
| Videos per epoch | 10,000 pairs | Fixed budget |
| Backbone freeze | Epochs 0–10 | Only neck + RPN head trained initially |
| Warmup | 5 epochs | Linear ramp from ~0 → base LR |
| LR schedule | SGDR | Cosine annealing, restarts at epochs 50 / 150 / 350 |
| Plateau rescue | ReduceLROnPlateau | ×0.3 after 15 stagnant epochs, floor 1e-7 |
| Early stopping | Patience 50, Δ = 1e-4 | Relative validation loss |
| Checkpoint frequency | Every 10 epochs | Rolling window of 2 + best model |
| Gradient clip | Norm = 10 | |

### Learning rate schedule

```
Epochs  0 –  5    Linear warm-up → base_lr (0.005)
Epochs  5 – end   SGDR cosine annealing  (T₀=50, T_mult=2)
                    restarts at: 50 → 150 → 350 → ...
Plateau detected  ReduceLROnPlateau  ×0.3, floor 1e-7
```

The backbone freeze for the first 10 epochs follows the same logic as transfer learning in YOLO — let the task-specific head (neck + RPN) adapt to the new domain first, then fine-tune the backbone at a lower effective learning rate via `BACKBONE.LAYERS_LR`.

### Monitor during training

In a separate terminal, the monitor script polls logs every 30 seconds and auto-evaluates every 25 epochs:

```bash
./monitor_training.sh &
```

TensorBoard:

```bash
tensorboard --logdir pysot/logs/all_datasets
```

---

## 7. ONNX Export

The model is split into two ONNX files to enable efficient edge deployment — the template is encoded once per target, only the tracker runs every frame.

```bash
python export_onnx.py \
    --cfg  pysot/experiments/siamrpn_r50_alldatasets/config.yaml \
    --ckpt pysot/snapshot/all_datasets/best_model.pth \
    --out  exported/
```

| File | Input shape | Output | When to run |
|------|-------------|--------|-------------|
| `template_encoder.onnx` | `(1, 3, 127, 127)` | `zf_0`, `zf_1`, `zf_2` | Once per target at initialisation |
| `tracker.onnx` | `zf_0/1/2` + `(1, 3, 255, 255)` | `cls (1,10,25,25)`, `loc (1,20,25,25)` | Every frame |

Exported at **opset 11** using the TorchScript-based exporter (`dynamo=False`) to correctly handle the depth-wise cross-correlation's dynamic group convolution. The script automatically runs the ONNX checker and an ONNX Runtime validation pass after export.

---

## 8. Evaluation

```bash
python eval_onnx.py \
    --work_dir /home/ubuntu/siamrpn_training \
    --onnx_dir /home/ubuntu/siamrpn_training/exported \
    --out_json eval_results/epoch_150.json \
    --epoch    150 \
    --max_seqs 20 \
    --max_frames 150
```

Evaluates on 4 test datasets: Anti-UAV410, MSRS, MassMIND, DUT-VTUAV.

**Metrics (per sequence and per dataset):**

| Metric | Description |
|--------|-------------|
| Mean IoU | Average Intersection over Union across frames |
| Success rate | Fraction of frames with IoU ≥ 0.5 |
| AUC | Area under the success curve at IoU thresholds 0.0–1.0 |

Results written to `eval_results/epoch_NNNN.json`.

---

## 9. Demo Videos

**ONNX tracker demo** — renders GT (green) vs. predicted (yellow dashed) boxes with per-frame IoU overlay:

```bash
python run_onnx_tracker.py \
    --work_dir /home/ubuntu/siamrpn_training \
    --seqs_per_dataset 3 \
    --max_frames_per_seq 200 \
    --fps 15 --width 640 --height 480
```

Output: `demo/onnx_tracker_demo.mp4`

**Ground-truth demo** — useful for inspecting annotation quality before or during training:

```bash
python make_test_demo.py \
    --work_dir /home/ubuntu/siamrpn_training \
    --seqs_per_dataset 4 \
    --max_frames_per_seq 120
```

Output: `demo/test_gt_demo.mp4`

---

## 10. Inference Hyperparameters

These parameters control how the tracker decodes the RPN output at inference time and are not trained — they are post-processing choices tuned on a validation set.

| Parameter | Value | What it does |
|-----------|-------|-------------|
| `EXEMPLAR_SIZE` | 127 | Template crop size fed to the encoder |
| `SEARCH_SIZE` | 255 | Search region crop size fed to the tracker |
| `OUTPUT_SIZE` | 25 | Spatial grid size of the RPN output (25×25) |
| `STRIDE` | 8 | Backbone output stride — each grid cell = 8 px in the search image |
| `CONTEXT_AMOUNT` | 0.5 | Padding fraction when cropping (half the object's mean dimension added on each side) |
| `ANCHOR_RATIOS` | [0.33, 0.5, 1.0, 2.0, 3.0] | Aspect ratios — covers tall, square, and wide targets |
| `ANCHOR_SCALES` | [8] | Single scale; multi-scale handled by FPN at the feature level |
| `PENALTY_K` | 0.04 | Penalises large jumps in target size/ratio between frames |
| `WINDOW_INF` | 0.40 | Cosine window weight — suppresses responses far from image centre |
| `LR` | 0.30 | Smooth update rate for bounding box size (prevents jitter) |
| `SCORE_THRESH` | 0.20 | Minimum foreground score to accept a prediction |

**Total anchors per frame:** 25 × 25 × 5 = 3,125. In practice only a handful of anchors near the target will score above `SCORE_THRESH`.

---

## 11. Directory Layout

```
siamrpn_training/
├── run_aws_training.sh           # Master pipeline script
├── train_siamrpn_aws.py          # Training engine
├── export_onnx.py                # ONNX export
├── eval_onnx.py                  # Evaluation
├── run_onnx_tracker.py           # Visual demo
├── make_test_demo.py             # GT demo video
├── generate_report.py            # PDF training report
├── monitor_training.sh           # Live monitoring
├── docs/                         # README assets (annotation strips, samples)
│
├── data/                         # gitignored — created by pipeline
│   ├── anti_uav410/              #   train/, val/, *_pysot.json
│   ├── anti_uav300/
│   ├── msrs/
│   ├── vtmot/
│   ├── massmind/
│   ├── mvss/
│   ├── dut_vtuav/
│   ├── birdsai/
│   └── hit_uav/
│
├── pysot/                        # gitignored — clone separately
│   ├── snapshot/all_datasets/
│   │   ├── best_model.pth
│   │   └── checkpoint_epoch***.pth
│   ├── logs/all_datasets/
│   └── experiments/siamrpn_r50_alldatasets/config.yaml
│
├── pretrained/                   # gitignored
│   └── sot_resnet50.pth
├── exported/                     # gitignored
│   ├── template_encoder.onnx
│   └── tracker.onnx
├── eval_results/                 # gitignored
│   └── epoch_****.json
├── demo/                         # gitignored
│   └── *.mp4
└── report/                       # gitignored
    └── SiamRPN_IR_Training_Report.pdf
```

---

## 12. Operations & Advanced

### Running on a limited root partition

AWS instances often have a small root volume (8–30 GB) that fills up quickly once PyTorch, datasets, and checkpoints accumulate. Use `--install-dir` to redirect everything to a larger data volume:

```bash
./run_aws_training.sh --install-dir=/data
```

This sets automatically:
- Miniconda → `/data/miniconda3`
- Work dir → `/data/siamrpn_training`
- `PIP_CACHE_DIR` → `/data/pip_cache`
- `CONDA_PKGS_DIRS` → `/data/conda_pkgs`
- `TORCH_HOME` → `/data/torch_cache`

### Downloading DUT-VTUAV via rclone

DUT-VTUAV (~200 GB) exceeds gdown's Google Drive quota. Use rclone to download directly to the server at full network speed.

**Step 1 — authenticate on your local machine (has a browser):**

```bash
rclone authorize "drive"
```

Sign in, allow access, then copy the JSON token from the terminal.

**Step 2 — configure rclone on the server:**

```bash
rclone config
# → n (new remote), name: gdrive, storage: drive
# → blank for client_id, client_secret, root_folder_id
# → scope: 1 (full access)
# → n for auto config (headless server)
# → paste the JSON token from Step 1
```

**Step 3 — download train split only:**

```bash
rclone copy \
  --drive-root-folder-id 1GwYNPcrkUM-gVDAObxNqERi_2Db7okjP \
  --include "train_*/**" \
  --progress --transfers 8 \
  gdrive: /data/dut_vtuav_raw/
```

Monitor progress: `watch -n 10 "du -sh /data/dut_vtuav_raw/"`

### DUT-VTUAV annotation notes

DUT-VTUAV is recorded at **10 fps** but annotated at **1 fps**. Annotation line `i` maps to frame `i × 10`. The conversion in `run_aws_training.sh` accounts for this automatically.

| Split | Sequences | Annotated frames |
|-------|-----------|-----------------|
| Train | 225 | 52,640 |
| Test  | 25 | 5,738 |

Sample IR frames across object categories:

| Excavator | Car | Pedestrian |
|-----------|-----|-----------|
| ![excavator](docs/sample_01_excavator_003.jpg) | ![car](docs/sample_02_car_045.jpg) | ![pedestrian](docs/sample_03_pedestrian_049.jpg) |
| ![bus](docs/sample_04_bus_002.jpg) | ![animal](docs/sample_05_animal_002.jpg) | ![bike](docs/sample_06_bike_002.jpg) |

Tracking strip — `excavator_003` from start to end (15,580 frames):

![tracking strip](docs/track_strip_excavator.jpg)

### Known fixes

| Date | Issue | Fix |
|------|-------|-----|
| 2026-04-13 | `gdown --fuzzy` removed in gdown 6.x | Removed flag from all 7 calls |
| 2026-04-13 | Root partition fills on small AWS volumes | `--install-dir` flag redirects all writes |
| 2026-04-14 | `DataParallel` breaks pysot dict-based `forward` in PyTorch 2.x | Disabled `DataParallel`; single-GPU training |
| 2026-04-14 | `model_builder.py` calls `next(self.parameters()).device` inside DataParallel replica | Changed to `data['template'].device` |
| 2026-04-14 | DUT-VTUAV annotations mapped to wrong frames | Frame key = `i × 10` (1 fps annos, 10 fps video) |
| 2026-04-14 | MSRS annotations used full-image placeholder `[0,0,640,480]` | Extract largest connected component from segmentation labels |
| 2026-04-14 | MSRS frame finder always loaded files[0] / files[1] | Use `seq_idx + frame_id − 1` offset |

---

## Related

- [PySOT](https://github.com/STVIR/pysot) — base tracking framework
- [SiamRPN++ (Li et al., CVPR 2019)](https://arxiv.org/abs/1812.11703) — tracker architecture
- [Axelera Voyager SDK](https://www.axelera.ai) — target deployment platform
