# SiamRPN++ IR Training — 444-Epoch Run

Reference documentation for the successful training run completed on **2026-04-23**, which produced the IR tracking model currently used downstream.

| Field | Value |
|---|---|
| Final epoch | 444 |
| Best val loss | 0.318 |
| Checkpoint | `pysot/snapshot/all_datasets_ir_siamese/best_model.pth` |
| Hardware | 8× NVIDIA A100-SXM4-40GB |
| Launcher | `torchrun --nproc_per_node=8` (DDP) |
| Backbone init | `pretrained/sot_resnet50.pth` |
| Config | `configs/config_ir_siamese.yaml` |
| Training script | `train_siamrpn_aws.py` |

---

## 1. Environment

```bash
# On the AWS box
ssh -i "<jaydeep-axelera.pem>" ubuntu@<current-ip>     # IP changes on stop/start
cd /home/ubuntu/data/siamrpn_training
source /opt/conda/etc/profile.d/conda.sh
conda activate siamrpn
```

The `siamrpn` conda env is at `/opt/conda/envs/siamrpn`. Persistent tmux session: `tmux attach -t siamrpn_train`.

Working layout on the server:

```
/home/ubuntu/data/siamrpn_training/
├── train_siamrpn_aws.py
├── configs/config_ir_siamese.yaml
├── pretrained/sot_resnet50.pth          # backbone init
├── data/                                 # ~247 GB, see §3
└── pysot/
    ├── snapshot/all_datasets_ir_siamese/  # checkpoints written here
    └── logs/all_datasets_ir_siamese/      # TensorBoard logs
```

---

## 2. Datasets

The config registers 7 datasets, but **only two carried the 2026-04-23 run** because annotation JSONs for the others were missing on the server instance. Datasets without their `*_pysot.json` are skipped silently at load time.

### 2.1 Active in the 444-epoch run

| Dataset | Size | Weight | Bboxes | % of samples |
|---|---|---|---|---|
| Anti-UAV410 | 19 GB | 3.0 | 208,832 | **55.5%** |
| DUT-VTUAV  | 222 GB | 2.5 | 52,640 | **44.5%** |

### 2.2 Registered but unused (no annotations on the server)

`MVSS`, `ANTIUAV300`, `BIRDSAI`, `HITUAV` — kept in the config for forward compatibility. `MSRS` and `MASSMIND` are commented out (no motion signal). `DUT-Anti-UAV` is excluded entirely (it's RGB, not IR).

### 2.3 Download

| Dataset | Source |
|---|---|
| Anti-UAV410 | https://anti-uav.github.io/ (request access from the organizers) |
| DUT-VTUAV   | https://github.com/zhang-pengyu/DUT-VTUAV |
| Anti-UAV300 | https://anti-uav.github.io/ (older release) |
| BIRDSAI     | https://sites.google.com/view/elizabethbondi/dataset |
| HIT-UAV     | https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset |
| MVSS        | https://github.com/lmyybh/MVSS |
| MSRS        | https://github.com/Linfeng-Tang/MSRS |
| MassMIND    | https://github.com/uml-marine-robotics/MassMIND |

To copy from the AWS server instead of re-downloading:

```bash
scp -r -i "<key>" ubuntu@<ip>:/home/ubuntu/data/siamrpn_training/data/anti_uav410/ ./data/
scp -r -i "<key>" ubuntu@<ip>:/home/ubuntu/data/siamrpn_training/data/dut_vtuav/   ./data/
```

### 2.4 Expected on-disk layout

```
data/
├── anti_uav410/
│   ├── train/<seq_id>/000001.jpg ...      # IR frames
│   ├── train/<seq_id>/IR_label.json       # source annotations
│   ├── train_pysot.json                   # converted (see §2.5)
│   ├── val/...
│   └── val_pysot.json
├── dut_vtuav/
│   ├── train/<seq_id>/000001.jpg ...
│   ├── train_pysot.json
│   ├── test/...
│   └── test_pysot.json
└── <other_datasets>/...
```

### 2.5 Convert annotations to PySOT format

Each dataset must be converted to the **PySOT SubDataset JSON** schema:

```json
{
  "<sequence_name>": {
    "0": {                              // track id (single-target SOT)
      "000001": [x1, y1, x2, y2],       // bbox in absolute pixel coords
      "000002": [x1, y1, x2, y2],
      ...
    }
  }
}
```

For Anti-UAV410, use the included converter:

```bash
python convert_antiuav410.py \
    --data_root /home/ubuntu/data/siamrpn_training/data/anti_uav410 \
    --splits train val
```

This reads each sequence's `IR_label.json` (`gt_rect` in `[x,y,w,h]`, `exist` mask), drops invisible/invalid frames, and writes `train_pysot.json` and `val_pysot.json` at the dataset root.

For any other dataset, write a similar converter that:
1. Lists sequences under `<root>/<split>/`.
2. For each frame with a valid bbox, emits `{frame_str: [x1, y1, x2, y2]}`.
3. Wraps as `{seq_name: {"0": frames_dict}}`.

The training script picks up `<root>/<split>_pysot.json` automatically per the path in the cfg.

---

## 3. Configuration

The full cfg is at [config_ir_siamese.yaml](config_ir_siamese.yaml). Key hyperparameters used in the 444-epoch run:

### Model

| Block | Setting |
|---|---|
| Backbone | ResNet-50, layers 2/3/4 trainable, init from `sot_resnet50.pth` |
| Backbone freeze | First **10 epochs** frozen, then fine-tuned at `0.1×` head LR |
| Neck (`ADJUST`) | `AdjustAllLayer`, `[512,1024,2048] → [256,256,256]` |
| Head | `MultiRPN`, weighted fusion over 3 levels |
| Anchors | `STRIDE=8`, `SCALES=[8]`, `RATIOS=[0.37, 0.56, 0.79, 1.11, 2.26]` (KMeans-5 over 288k bboxes) |
| Input sizes | EXEMPLAR 127, SEARCH 255, OUTPUT 25 |

### Optimization

| Setting | Value |
|---|---|
| Epochs | 500 (early-stop disabled, converged on epoch 444) |
| Batch size (per GPU) | 32 |
| Grad accumulation | 4 → effective batch 128/GPU → 1024 across 8 GPUs |
| Optimizer | SGD, momentum 0.9, weight decay 1e-4 |
| LR warmup | step, 5 epochs, 1e-4 → 5e-3 |
| LR main | log decay, 5e-3 → 1e-5 |
| Grad clip | norm 10.0 |
| Loss weights | CLS 1.0, LOC 1.2 |
| Sampling | 10,000 (template, search) pairs per epoch, weighted across datasets |

### Pair sampling

- `FRAME_RANGE` controls max temporal distance between template and search frames (50 for Anti-UAV410 / DUT-VTUAV).
- Each dataset's `WEIGHT` biases the weighted sampler (higher = more frequent draws).
- `NEG: 0.4` — 40% of pairs are negatives (template and search from different sequences) to learn distractor rejection.
- `GRAY: 0.5` — half the pairs run through BGR→GRAY→BGR replication, randomizing the 1ch↔3ch path.

---

## 4. Augmentation pipeline

Augmentation runs **every batch** inside the dataset's `__getitem__`. There is no CLI flag — strength is controlled entirely via cfg. Pipeline order:

```
load frame  →  _get_center_crop  →  _get_bbox  →  Augmentation.__call__
                (custom)            (PySOT)       (PySOT, 5 stages)
```

### 4.1 Pre-crop: `_get_center_crop`

Located at [train_siamrpn_aws.py:89](train_siamrpn_aws.py#L89). Crops a **square** patch centered on the annotation using PySOT's exemplar window `s_z = √((w+ctx)(h+ctx))` with `ctx = 0.5(w+h)`. Pads with the image mean at boundaries, then resizes to `output_size` (254 px for template branch, 510 px for search branch).

This step exists because PySOT's `Augmentation` assumes the target is already at image center. Without it, rectangular full frames are cropped on the wrong region — a bug that previously cost a 140-epoch run with loss 0.26 and IoU 0.000 on its own training data.

### 4.2 PySOT `Augmentation.__call__`

Located at [pysot/pysot/datasets/augmentation.py:117](../../pysot/pysot/datasets/augmentation.py#L117). Five stages, fixed order:

1. **Gray** (`if cfg.DATASET.GRAY > rand()`): BGR → grayscale → 3ch replication.
2. **Shift + Scale** (always): jitter the crop box by `±SHIFT` pixels and scale by `(1 ± SCALE)` per axis, clamped to image bounds. The bbox is rewritten into the new frame.
3. **Color** (`if COLOR > rand()`): RGB PCA jitter from a fixed eigenvector matrix.
4. **Blur** (`if BLUR > rand()`): random motion-blur kernel, size 5–45, mixed horizontal/vertical.
5. **Flip** (`if FLIP > rand()`): horizontal flip + bbox mirror.

Final output: jittered image at 127×127 (template) or 255×255 (search) + bbox in the final frame coords.

### 4.3 Augmentation parameters used in the 444-epoch run

| Branch | SHIFT (px) | SCALE | BLUR | FLIP | COLOR |
|---|---|---|---|---|---|
| TEMPLATE | 4  | 0.10 | 0.2 | 0.5 | **0.0** (off) |
| SEARCH   | 64 | 0.50 | 0.3 | 0.5 | **0.0** (off) |

Plus the global probabilities:

| | Value |
|---|---|
| `GRAY` | 0.5 |
| `NEG`  | 0.4 |

**IR-specific choices:**
- Color PCA jitter is fully disabled — IR is not RGB.
- Gray at 0.5 makes the network robust to both 1ch and 3ch IR sources.
- Search SHIFT (64 px) ≫ TEMPLATE SHIFT (4 px) — standard SiamRPN++; the template should stay tight on the target, the search should learn translation invariance.

### 4.4 Known dead config knob

`SCALE_ONE_SIDED: True` appears in the cfg but is **not read anywhere** in `train_siamrpn_aws.py` or `pysot/`. The actual scale jitter is symmetric `(1 ± SCALE)`, so targets get enlarged ~50% of the time during the 444-epoch run despite the cfg claiming otherwise. If "shrink-only" behavior is needed, patch [`Augmentation._shift_scale_aug`](../../pysot/pysot/datasets/augmentation.py#L70).

### 4.5 Verify augmentation before launch

```bash
python debug_dataflow.py        # 2-minute pre-flight: visualizes training pairs
```

Skipping this once cost a full 140-epoch run. **Never skip it.** Visualised pairs catch preprocessing bugs that would otherwise be invisible until epoch 100+.

---

## 5. Running the training

### 5.1 Smoke test (1 epoch, 64 samples)

Validates the pipeline end-to-end without committing to a real run. Disables early stopping and checkpoint rotation.

```bash
python train_siamrpn_aws.py \
    --cfg configs/config_ir_siamese.yaml \
    --pretrained pretrained/sot_resnet50.pth \
    --smoke-test
```

### 5.2 Full training (single GPU)

```bash
python train_siamrpn_aws.py \
    --cfg configs/config_ir_siamese.yaml \
    --pretrained pretrained/sot_resnet50.pth
```

### 5.3 Full training — 8× A100 DDP (the 444-epoch run)

This is the exact command from [launch_train.sh](launch_train.sh):

```bash
torchrun --nproc_per_node=8 train_siamrpn_aws.py \
    --cfg configs/config_ir_siamese.yaml \
    --pretrained pretrained/sot_resnet50.pth \
    --no_early_stop \
    2>&1 | tee train_8gpu_$(date +%Y%m%d_%H%M%S).log
```

`--no_early_stop` is required to reach epoch 444 — without it, the default patience=50 will trip earlier.

### 5.4 Resume from checkpoint

```bash
torchrun --nproc_per_node=8 train_siamrpn_aws.py \
    --cfg configs/config_ir_siamese.yaml \
    --resume pysot/snapshot/all_datasets_ir_siamese/checkpoint_ep0XXX.pth
```

### 5.5 CLI flags reference

| Flag | Effect |
|---|---|
| `--cfg PATH` | Required. YAML config path. |
| `--pretrained PATH` | Backbone init checkpoint. Overrides `cfg.TRAIN.PRETRAINED`. |
| `--resume PATH` | Full checkpoint to resume from (model + optimizer + epoch). |
| `--seed N` | RNG seed (default 42). Per-rank offset is added in DDP. |
| `--no_early_stop` | Run all `cfg.TRAIN.EPOCH` epochs regardless of stagnation. |
| `--no_plateau` | Skip ReduceLROnPlateau rescue; use fixed log decay only. |
| `--smoke-test` | 1 epoch, 64-sample slice. |

---

## 6. Outputs

After training, the snapshot dir contains:

```
pysot/snapshot/all_datasets_ir_siamese/
├── best_model.pth              # lowest val loss (the 444-epoch run: 0.318)
├── checkpoint_ep0010.pth       # rotated every 10 epochs
├── checkpoint_ep0020.pth
├── ...
└── checkpoint_ep0440.pth
```

TensorBoard logs at `pysot/logs/all_datasets_ir_siamese/` — start with `tensorboard --logdir pysot/logs/`.

---

## 7. Pre-flight checklist (run before every training session)

These three rules come from a session that produced a model with loss 0.26 and IoU=0.000 on training data — caught only because the user manually tested mid-training.

1. **Visualize training pairs** — `python debug_dataflow.py`. 2 minutes. Catches preprocessing bugs immediately.
2. **Test on training data at ~20 epochs** — `python test_on_training_data.py`. If IoU=0.000, stop. Loss alone tells you nothing about tracking quality.
3. **Low loss ≠ correct training** — a model can memorize spurious correlations and still produce a healthy-looking loss curve. Correct training starts at loss ≈ 1.7 and decreases. A run that starts at 0.26 is suspicious, not impressive.

---

## 8. Reproducing the 444-epoch model from scratch

1. Provision an 8× A100 box, install the `siamrpn` conda env.
2. Copy `data/anti_uav410/` and `data/dut_vtuav/` to `/home/ubuntu/data/siamrpn_training/data/` (~241 GB combined).
3. Run `convert_antiuav410.py` and DUT-VTUAV's converter to produce `*_pysot.json`.
4. Place `sot_resnet50.pth` at `pretrained/`.
5. Run `python debug_dataflow.py` and inspect the dumped pairs.
6. Run `python train_siamrpn_aws.py --smoke-test ...` and confirm a clean 1-epoch loop.
7. Launch the full 8-GPU command from §5.3 inside a tmux session.
8. Monitor TensorBoard; expect best val loss around epoch ~440 at ~0.318.
