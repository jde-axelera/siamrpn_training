# SiamRPN++ IR/Thermal Tracking — Fine-Tuning & Deployment Pipeline

Fine-tunes **SiamRPN++** (ResNet-50 backbone, FPN neck, multi-scale RPN head) on 10 IR/thermal UAV-tracking datasets and exports the result to two ONNX models ready for edge deployment.

Built on top of [PySOT](https://github.com/STVIR/pysot). Designed to run on AWS GPU instances (CUDA 11.8).

---

## Overview

| Step | Script | What it does |
|------|--------|-------------|
| Full pipeline | `run_aws_training.sh` | Orchestrates all steps end-to-end |
| Training | `train_siamrpn_aws.py` | Multi-GPU fine-tuning with SGDR + early stopping |
| Export | `export_onnx.py` | Converts best checkpoint → two ONNX models |
| Evaluate | `eval_onnx.py` | IoU / AUC metrics on test splits |
| Monitor | `monitor_training.sh` | Auto-evaluates every 25 epochs during training |
| Demo | `run_onnx_tracker.py` | Annotated MP4 with GT + predicted boxes |
| GT demo | `make_test_demo.py` | Ground-truth-only demo video from test splits |
| Report | `generate_report.py` | Multi-page PDF training report |

---

## Quick Start

```bash
# 1. Clone this repo
git clone https://github.com/jde-axelera/siamrpn_training.git
cd siamrpn_training

# 2. Clone PySOT alongside
git clone https://github.com/STVIR/pysot.git

# 3. Create data directory
mkdir data

# 4. Run the full pipeline
chmod +x run_aws_training.sh
./run_aws_training.sh
```

Verify the setup first with a smoke test (1 epoch, no downloads):
```bash
./run_aws_training.sh --smoke-test
```

Resume training from a checkpoint:
```bash
./run_aws_training.sh --resume pysot/snapshot/all_datasets/checkpoint_epoch050.pth
```

---

## Datasets

The pipeline trains on 10 IR/thermal datasets covering UAV tracking, maritime, aerial wildlife, and road scenes.

| # | Dataset | Size | Domain | Source |
|---|---------|------|--------|--------|
| 1 | **Anti-UAV410** | 410 seqs, 9.4 GB | IR UAV tracking | Google Drive (`gdown`) |
| 2 | **Anti-UAV300** | IR video sequences | IR UAV tracking | Google Drive (`gdown`) |
| 3 | **MSRS** | 1,444 image pairs | Paired IR/visible road | GitHub (`git clone` + LFS) |
| 4 | **VT-MOT / PFTrack** | 582 seqs, 401K frames | RGB+IR multi-object | Baidu Cloud (code: `chcw`) |
| 5 | **MassMIND** | 2,916 LWIR images | Maritime LWIR | Google Drive (`gdown`) |
| 6 | **MVSS-Baseline** | Video sequences | RGB-thermal video seg | Google Drive (author approval) |
| 7 | **DUT-VTUAV** | 500 seqs, 1.7M frames | RGB+Thermal UAV | Google Drive (`gdown --folder`) |
| 8 | **DUT-Anti-UAV** | IR drone sequences | IR drone tracking | Google Drive (`gdown`) |
| 9 | **BIRDSAI** | 48 TIR sequences | Aerial wildlife thermal | LILA (`wget`) |
| 10 | **HIT-UAV** | 2,898 IR images | High-altitude IR detection | Kaggle CLI |

All dataset annotations are automatically converted to a unified **PySOT JSON format**:
```json
{ "seq_name": { "0": { "000001": [x1, y1, x2, y2], ... } } }
```
Conversions handle: MOT `gt.txt` → per-object SOT tracks, instance segmentation masks → bounding boxes, YOLO normalised coords → pixel coordinates, MP4 video → frame extraction.

---

## Requirements

**System:** Ubuntu 20.04+, CUDA 11.8-compatible GPU (AWS `g4dn` or `p3` recommended)

**Python packages** (installed automatically by the pipeline):
```
torch + torchvision   (CUDA 11.8)
cython<3.0
opencv-python
yacs
tqdm
pyyaml
matplotlib
colorama
tensorboard
tensorboardX
scipy
gdown
onnx
onnxruntime
```

**Optional:**
```
kaggle      — HIT-UAV download
git-lfs     — MSRS download
reportlab   — PDF report generation
```

---

---

## Training Methodology

### Siamese Pair Sampling

SiamRPN++ is trained on pairs of image crops rather than full frames. Each training sample consists of:

- **Template** `z` — a 127×127 px crop centred on the target at frame `t`, providing the reference appearance.
- **Search region** `x` — a 255×255 px crop centred near the target at frame `t + δ` (a few frames later), within which the model must localise the object.

The backbone encodes both crops, a depth-wise cross-correlation (DW-XCorr) is applied between the template features and the search features, and the RPN head outputs a classification map and a localisation map. The loss is a combination of cross-entropy (foreground/background) and smooth-L1 (bounding-box regression) over all anchor positions.

---

### Epoch Length and Dataset Sampling

Training does **not** loop once through every sequence per epoch. Instead, each epoch draws a fixed budget of **10,000 (template, search) pairs** sampled stochastically from the combined dataset:

1. A dataset is chosen according to its `sample_prob` (see table below).
2. A sequence is sampled uniformly from that dataset.
3. A valid frame pair `(t, t + δ)` is sampled from that sequence.
4. Template and search crops are extracted and augmented.

This is repeated 10,000 times. At batch size 32 this yields **≈ 312 gradient steps per epoch**, giving a predictable and tunable training cadence independent of total dataset size.

#### Dataset sampling probabilities

Probabilities are **not proportional to dataset size** — smaller, task-critical datasets are upweighted so the model receives balanced exposure across object categories and IR conditions:

| Dataset | Sequences | `sample_prob` | Role |
|---------|-----------|--------------|------|
| AntiUAV410 | 200 | 0.316 | Primary IR UAV target — highest weight |
| DUT-VTUAV | 225 | 0.263 | Large-scale RGB+Thermal UAV scenes |
| DUT-Anti-UAV | 20 | 0.211 | IR drone close-up sequences |
| MSRS | 541 | 0.105 | Paired IR/visible road scenes |
| MassMIND | 1,801 | 0.105 | Maritime LWIR — largest dataset, downweighted |

Datasets missing their annotation JSON (VT-MOT, MVSS, AntiUAV300, BIRDSAI, HIT-UAV) are automatically excluded and the remaining probabilities are renormalised to sum to 1.

---

### Learning Rate Schedule

```
Epochs 0 – 5    Linear warm-up from ~0 → base_lr (0.005)
Epochs 5 – end  SGDR cosine annealing  T₀ = 50, T_mult = 2
                  → restarts at epochs 50, 150, 350, …
Plateau rescue  ReduceLROnPlateau  ×0.3 after 15 stagnant epochs, floor 1e-7
```

The backbone is frozen for the first 10 epochs (`BACKBONE_TRAIN_EPOCH = 10`); only the neck and RPN head are trained. After epoch 10 the backbone is unfrozen and trained at a reduced learning-rate scale (`BACKBONE.LAYERS_LR`).

---

### Early Stopping and Checkpointing

| Setting | Value |
|---------|-------|
| Patience | 50 epochs |
| Min improvement Δ | 1 × 10⁻⁴ (relative val loss) |
| Checkpoint frequency | Every 10 epochs |
| Checkpoint retention | Rolling window of 2 + best model |
| Gradient clip | Norm = 10 |

The best checkpoint is saved to `pysot/snapshot/all_datasets/best_model.pth` whenever validation loss improves.


---

## ONNX Export

```bash
python export_onnx.py \
    --cfg pysot/experiments/siamrpn_r50_alldatasets/config.yaml \
    --ckpt pysot/snapshot/all_datasets/best_model.pth \
    --out  exported/
```

Produces two models:

| File | Input | Output | Run when |
|------|-------|--------|----------|
| `template_encoder.onnx` | `(1, 3, 127, 127)` | `zf_0`, `zf_1`, `zf_2` | Once per target at init |
| `tracker.onnx` | `zf_0/1/2` + `(1, 3, 255, 255)` | `cls (1,10,25,25)`, `loc (1,20,25,25)` | Every frame |

Exported with **opset 17**. The script runs ONNX checker + ONNX Runtime validation automatically.

---

## Evaluation

```bash
python eval_onnx.py \
    --work_dir /home/ubuntu/siamrpn_training \
    --onnx_dir /home/ubuntu/siamrpn_training/exported \
    --out_json eval_results/epoch_150.json \
    --epoch    150 \
    --max_seqs 20 \
    --max_frames 150
```

Evaluates on 5 test datasets: Anti-UAV410, Anti-UAV300, MSRS, DUT-Anti-UAV, MassMIND.

**Metrics per sequence and per dataset:**
- Mean IoU
- Success rate at IoU ≥ 0.5
- AUC (area under success curve, thresholds 0.0–1.0)

Results written to `eval_results/epoch_NNNN.json`.

---

## Demo Videos

**ONNX tracker demo** — renders GT (green) vs. predicted (yellow dashed) boxes with per-frame IoU:
```bash
python run_onnx_tracker.py \
    --work_dir /home/ubuntu/siamrpn_training \
    --seqs_per_dataset 3 \
    --max_frames_per_seq 200 \
    --fps 15 --width 640 --height 480
```
Output: `demo/onnx_tracker_demo.mp4`

**Ground-truth demo** — useful for inspecting annotation quality before training:
```bash
python make_test_demo.py \
    --work_dir /home/ubuntu/siamrpn_training \
    --seqs_per_dataset 4 \
    --max_frames_per_seq 120
```
Output: `demo/test_gt_demo.mp4`

---

## Training Report

```bash
python generate_report.py --work_dir /home/ubuntu/siamrpn_training
```

Generates `report/SiamRPN_IR_Training_Report.pdf` containing: annotated dataset samples, learning curves, LR schedule plot, architecture description, convergence statistics.

---

## Tracker Hyperparameters (inference)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `EXEMPLAR_SIZE` | 127 | Template crop size (px) |
| `SEARCH_SIZE` | 255 | Search region crop size (px) |
| `OUTPUT_SIZE` | 25 | RPN output spatial resolution |
| `STRIDE` | 8 | Backbone output stride |
| `CONTEXT_AMOUNT` | 0.5 | Context padding fraction |
| `ANCHOR_RATIOS` | [0.33, 0.5, 1.0, 2.0, 3.0] | 5 aspect ratios |
| `ANCHOR_SCALES` | [8] | 1 scale → 3,125 anchors total |
| `PENALTY_K` | 0.04 | Size/ratio change penalty |
| `WINDOW_INF` | 0.40 | Cosine window influence weight |
| `LR` | 0.30 | Bounding box smooth-update rate |
| `SCORE_THRESH` | 0.20 | Minimum fg score to accept prediction |

---

## Directory Layout

```
siamrpn_training/
├── run_aws_training.sh       # Master pipeline script
├── train_siamrpn_aws.py      # Training engine
├── export_onnx.py            # ONNX export
├── eval_onnx.py              # Evaluation
├── run_onnx_tracker.py       # Visual demo
├── make_test_demo.py         # GT demo video
├── generate_report.py        # PDF report
├── monitor_training.sh       # Live monitoring
│
├── data/                     # gitignored — created by pipeline
│   ├── anti_uav410/          #   train/, val/, *_pysot.json
│   ├── anti_uav300/
│   ├── msrs/
│   ├── vtmot/
│   ├── massmind/
│   ├── mvss/
│   ├── dut_vtuav/
│   ├── dut_anti_uav/
│   ├── birdsai/
│   └── hit_uav/
│
├── pysot/                    # gitignored — clone separately
│   ├── snapshot/all_datasets/
│   │   ├── best_model.pth
│   │   └── checkpoint_epoch***.pth
│   ├── logs/all_datasets/
│   └── experiments/siamrpn_r50_alldatasets/config.yaml
│
├── pretrained/               # gitignored
│   └── sot_resnet50.pth
├── exported/                 # gitignored
│   ├── template_encoder.onnx
│   └── tracker.onnx
├── eval_results/             # gitignored
│   └── epoch_****.json
├── demo/                     # gitignored
│   └── *.mp4
└── report/                   # gitignored
    └── SiamRPN_IR_Training_Report.pdf
```

---

## Running on a Limited Root Partition

AWS instances often have a small root volume (8–30 GB) that fills up quickly once
PyTorch, datasets, and checkpoints are installed. Use `--install-dir` to redirect
**everything** — Miniconda, conda packages, pip cache, datasets, checkpoints, and
exported models — to a larger data volume:

```bash
./run_aws_training.sh --install-dir=/data
```

This sets the following automatically:
- Miniconda installed to `/data/miniconda3`
- `WORK_DIR` = `/data/siamrpn_training`
- `PIP_CACHE_DIR` = `/data/pip_cache`
- `CONDA_PKGS_DIRS` = `/data/conda_pkgs`
- `TORCH_HOME` = `/data/torch_cache`

You can also set `INSTALL_DIR` as an environment variable before running:

```bash
export INSTALL_DIR=/data
./run_aws_training.sh
```


## Related

- [PySOT](https://github.com/STVIR/pysot) — base tracking framework
- [SiamRPN++ (Li et al., CVPR 2019)](https://arxiv.org/abs/1812.11703) — tracker architecture
- [Axelera Voyager SDK](https://www.axelera.ai) — target deployment platform


---

## Downloading DUT-VTUAV Dataset

The DUT-VTUAV dataset is hosted on Google Drive and is too large for `gdown` due to quota limits. Use `rclone` to download it directly to the machine at full network speed.

### 1. Install rclone

```bash
curl https://rclone.org/install.sh | sudo bash
```

### 2. Authenticate with Google Drive

On a machine with a browser (your laptop), run:

```bash
rclone authorize "drive"
```

This opens a browser — sign in with your Google account, allow access, then copy the JSON token that appears in the terminal.

### 3. Configure rclone on the server

On the server (headless), run:

```bash
rclone config
```

Answer the prompts:
```
n                  # New remote
gdrive             # Name
drive              # Storage type: Google Drive
                   # client_id → Enter (blank)
                   # client_secret → Enter (blank)
1                  # scope: full drive access
                   # root_folder_id → Enter (blank)
                   # service_account_file → Enter (blank)
n                  # Edit advanced config? No
n                  # Use auto config? No  (headless server)
```

Paste the JSON token from Step 2 when prompted, then:
```
y                  # Confirm
q                  # Quit config
```

### 4. Download train split only

```bash
rclone copy --drive-root-folder-id 1GwYNPcrkUM-gVDAObxNqERi_2Db7okjP --include "train_*/**" --progress --transfers 8 gdrive: /data/dut_vtuav_raw/
```

- `--drive-root-folder-id` — pins rclone to the DUT-VTUAV shared folder directly
- `--include "train_*/**"` — downloads only `train_*` folders, skipping test splits
- `--transfers 8` — 8 parallel file downloads (increase on fast connections)

To download everything including test splits:

```bash
rclone copy --drive-root-folder-id 1GwYNPcrkUM-gVDAObxNqERi_2Db7okjP --progress --transfers 8 gdrive: /data/dut_vtuav_raw/
```

### 5. Monitor download progress

```bash
watch -n 10 "du -sh /data/dut_vtuav_raw/"
```

---

## DUT-VTUAV Dataset Notes

### Annotation Format

The DUT-VTUAV dataset stores video at **10 fps** but ground-truth annotations at **1 fps** — a consistent 10:1 ratio across all sequences. Annotation line `i` maps to frame `i × 10`. The conversion in `run_aws_training.sh` accounts for this.

| Sequence | Video frames | Anno lines | Ratio |
|----------|-------------|-----------|-------|
| animal_002 | 6,249 | 625 | 10× |
| bike_002 | 2,390 | 239 | 10× |
| excavator_003 | 15,590 | 1,559 | 10× |

### Train / Test Split

Downloaded sequences (250 total) are split **90 / 10**:

| Split | Sequences | Annotated frames |
|-------|-----------|-----------------|
| Train | 225 | 52,640 |
| Test  | 25 | 5,738 |

### Sample Annotations

Six IR frames with ground-truth bounding boxes (green) across object categories:

| Excavator | Car | Pedestrian |
|-----------|-----|-----------|
| ![excavator](docs/sample_01_excavator_003.jpg) | ![car](docs/sample_02_car_045.jpg) | ![pedestrian](docs/sample_03_pedestrian_049.jpg) |
| ![bus](docs/sample_04_bus_002.jpg) | ![animal](docs/sample_05_animal_002.jpg) | ![bike](docs/sample_06_bike_002.jpg) |

### Single-Sequence Tracking Strip

Three frames from `excavator_003` (start · mid · end) showing the bounding box tracking the object across 15,580 frames:

![tracking strip](docs/track_strip_excavator.jpg)

---

## Known Fixes

| Date | Issue | Fix |
|------|-------|-----|
| 2026-04-13 | `gdown --fuzzy` removed in gdown 6.x | Removed flag from all 7 calls |
| 2026-04-13 | Root partition fills on small AWS volumes | `--install-dir` flag redirects all writes to data partition |
| 2026-04-14 | `DataParallel` breaks pysot dict-based `forward` in PyTorch 2.x | Disabled `DataParallel`; single-GPU training |
| 2026-04-14 | `model_builder.py` calls `next(self.parameters()).device` inside DataParallel replica | Changed to `data['template'].device` |
| 2026-04-14 | DUT-VTUAV annotations mapped to wrong frames | Frame key = `i × 10` not `i + 1` (1 fps annos, 10 fps video) |
