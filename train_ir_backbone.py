#!/usr/bin/env python3
"""
train_ir_backbone.py
Stage 1: IR-domain backbone fine-tuning for SiamRPN++.

Combines all available IR detection/tracking image data into a unified
binary detection task (background vs. object) and fine-tunes the ResNet-50
backbone so it extracts discriminative IR features before SiamRPN++ neck/RPN
training (Stage 2).

Data sources
------------
  - HIT-UAV       : 2 866 IR images, YOLO bbox labels, 5 classes
  - Anti-UAV410   : 200 IR sequences  → sampled frames + tracking bboxes
  - DUT-VTUAV     : 225 sequences     → sampled frames + tracking bboxes
  - DUT-AntiUAV   : Anti-UAV-Tracking → sampled frames + tracking bboxes
  - MassMIND      : IR images         → sampled frames + tracking bboxes

All labels are collapsed to a single "object" class (class index 1) so the
backbone learns IR object-vs-background discrimination without requiring
cross-dataset label harmonisation.

Architecture
------------
  torchvision FasterRCNN + ResNet-50-FPN backbone (2 classes: bg + object)
  Backbone initialised from sot_resnet50.pth (PySOT pretrained weights)

Training schedule
-----------------
  Phase 1 (epochs  1 –  5): backbone frozen  — train FPN + RPN + head only
  Phase 2 (epochs  6 – 20): unfreeze layer3/4 — lr 5e-4
  Phase 3 (epochs 21 – 30): unfreeze all layers — lr 1e-4

Output
------
  <out>  (default: pretrained/ir_backbone.pth)
  Pure backbone state-dict (keys: conv1/bn1/layer1-4).
  Load with:  model.backbone.load_state_dict(torch.load(out), strict=True)

Usage
-----
  python train_ir_backbone.py \\
      --data_root /data/siamrpn_training/data \\
      --pretrained /data/siamrpn_training/pretrained/sot_resnet50.pth \\
      --out  /data/siamrpn_training/pretrained/ir_backbone.pth \\
      --epochs 30 --batch 16 --workers 8 --sample_stride 10
"""
import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from PIL import Image
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torchvision.transforms.v2 as T

# ── reproducibility ───────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ─────────────────────────────────────────────────────────────────────────────
#  Data loaders — each returns a flat list of (img_path_str, boxes_xyxy)
#  boxes_xyxy: numpy (N,4) in absolute pixel [x1,y1,x2,y2], may be empty (N=0)
# ─────────────────────────────────────────────────────────────────────────────

def load_hituav(data_root: str) -> list:
    """HIT-UAV YOLO format.  Combines train + val + test splits."""
    base  = Path(data_root) / 'hit_uav'
    items = []
    for split in ('train', 'val', 'test'):
        img_dir = base / 'images' / split
        lbl_dir = base / 'labels' / split
        if not img_dir.exists():
            continue
        for img_p in sorted(img_dir.glob('*.jpg')):
            lbl_p = lbl_dir / img_p.with_suffix('.txt').name
            boxes = []
            if lbl_p.exists():
                try:
                    img_tmp = Image.open(img_p)
                    iw, ih  = img_tmp.size
                    img_tmp.close()
                except Exception:
                    continue
                for line in lbl_p.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id = int(parts[0])
                    if cls_id == 4:          # DontCare
                        continue
                    cx, cy, w, h = map(float, parts[1:])
                    x1 = (cx - w / 2) * iw
                    y1 = (cy - h / 2) * ih
                    x2 = (cx + w / 2) * iw
                    y2 = (cy + h / 2) * ih
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
            items.append((str(img_p), np.array(boxes, dtype=np.float32).reshape(-1, 4)))
    print(f"  HIT-UAV      : {len(items):6d} images")
    return items


def _load_sot_json(json_path: str, img_root: str, sample_stride: int = 10) -> list:
    """
    Generic loader for PySOT-format JSONs used in this repo.

    JSON structure:
      { video_name: { "0": { frame_str: [x1,y1,x2,y2], ... } } }
    """
    if not Path(json_path).exists():
        return []
    data  = json.load(open(json_path))
    root  = Path(img_root)
    items = []
    for vid_name, vid_data in data.items():
        frames_dict = vid_data.get('0', vid_data)  # handle both levels
        if not isinstance(frames_dict, dict):
            continue
        frame_keys = sorted(frames_dict.keys())
        sampled    = frame_keys[::sample_stride]
        for fk in sampled:
            bbox = frames_dict[fk]
            if not bbox or (isinstance(bbox, list) and len(bbox) == 0):
                continue
            # Find image file (try .jpg, .png)
            for ext in ('.jpg', '.png', '.jpeg'):
                img_p = root / vid_name / (fk + ext)
                if img_p.exists():
                    break
            else:
                continue
            # bbox: [x1,y1,x2,y2] absolute coords
            b = np.array(bbox, dtype=np.float32).reshape(-1)
            if len(b) < 4:
                continue
            x1, y1, x2, y2 = b[:4]
            if x2 > x1 and y2 > y1:
                boxes = np.array([[x1, y1, x2, y2]], dtype=np.float32)
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)
            items.append((str(img_p), boxes))
    return items


def load_anti_uav410(data_root: str, sample_stride: int = 10) -> list:
    base  = Path(data_root) / 'anti_uav410'
    items = _load_sot_json(str(base / 'train_pysot.json'),
                           str(base / 'train'), sample_stride)
    print(f"  Anti-UAV410  : {len(items):6d} sampled frames (stride {sample_stride})")
    return items


def load_dut_vtuav(data_root: str, sample_stride: int = 10) -> list:
    """DUT-VTUAV sequences store IR frames under {seq}/infrared/."""
    base      = Path(data_root) / 'dut_vtuav'
    json_path = base / 'train_pysot.json'
    if not json_path.exists():
        print(f"  DUT-VTUAV    :      0 (no JSON)")
        return []
    data  = json.load(open(json_path))
    root  = base / 'train'
    items = []
    for vid_name, vid_data in data.items():
        frames_dict = vid_data.get('0', vid_data)
        if not isinstance(frames_dict, dict):
            continue
        frame_keys = sorted(frames_dict.keys())
        for fk in frame_keys[::sample_stride]:
            bbox = frames_dict[fk]
            if not bbox:
                continue
            # IR frames live under {seq}/infrared/
            img_p = root / vid_name / 'infrared' / (fk + '.jpg')
            if not img_p.exists():
                continue
            b = np.array(bbox, dtype=np.float32).reshape(-1)
            if len(b) < 4:
                continue
            x1, y1, x2, y2 = b[:4]
            boxes = np.array([[x1, y1, x2, y2]], dtype=np.float32) \
                    if x2 > x1 and y2 > y1 else np.zeros((0, 4), dtype=np.float32)
            items.append((str(img_p), boxes))
    print(f"  DUT-VTUAV    : {len(items):6d} sampled frames (stride {sample_stride})")
    return items


def load_massmind(data_root: str, sample_stride: int = 10) -> list:
    """
    MassMIND: flat image structure.
    JSON video name = 'massmind_{stem}' → image at images/Images/{stem}.png
    Each 'sequence' is a single image with multiple bbox annotations.
    """
    base   = Path(data_root) / 'massmind'
    json_p = base / 'train_pysot.json'
    if not json_p.exists():
        print(f"  MassMIND     :      0 (no JSON)")
        return []
    data     = json.load(open(json_p))
    img_root = base / 'images' / 'Images'
    items    = []
    vid_names = list(data.keys())[::sample_stride]   # stride over sequences
    for vid_name in vid_names:
        vid_outer = data[vid_name]
        # vid_name = 'massmind_{stem}'  →  image stem
        stem  = vid_name.replace('massmind_', '')
        img_p = img_root / (stem + '.png')
        if not img_p.exists():
            for ext in ('.jpg', '.jpeg'):
                img_p = img_root / (stem + ext)
                if img_p.exists():
                    break
            else:
                continue
        # Collect all bboxes for this image from all frames in the sequence
        frames_dict = vid_outer.get('0', vid_outer) if isinstance(vid_outer, dict) else {}
        all_boxes = []
        for fk, bbox in frames_dict.items():
            if not bbox:
                continue
            b = np.array(bbox, dtype=np.float32).reshape(-1)
            if len(b) >= 4:
                x1, y1, x2, y2 = b[:4]
                if x2 > x1 and y2 > y1:
                    all_boxes.append([x1, y1, x2, y2])
        boxes = np.array(all_boxes, dtype=np.float32).reshape(-1, 4) \
                if all_boxes else np.zeros((0, 4), dtype=np.float32)
        items.append((str(img_p), boxes))
    print(f"  MassMIND     : {len(items):6d} images (stride {sample_stride} over sequences)")
    return items


def load_dut_anti_uav(data_root: str, sample_stride: int = 10) -> list:
    """
    DUT-AntiUAV: images at images/Anti-UAV-Tracking-V0/{video}/{frame}.jpg
    Frame keys are 5-digit zero-padded (00001, not 000001).
    """
    base   = Path(data_root) / 'dut_anti_uav'
    json_p = base / 'train_pysot.json'
    if not json_p.exists():
        print(f"  DUT-AntiUAV  :      0 (no JSON)")
        return []
    data  = json.load(open(json_p))
    root  = base / 'images' / 'Anti-UAV-Tracking-V0'
    items = []
    for vid_name, vid_outer in data.items():
        frames_dict = vid_outer.get('0', vid_outer) if isinstance(vid_outer, dict) else {}
        if not isinstance(frames_dict, dict):
            continue
        frame_keys = sorted(frames_dict.keys())
        for fk in frame_keys[::sample_stride]:
            bbox = frames_dict[fk]
            if not bbox:
                continue
            # JSON keys may be 6-digit (000001) but files may be 5-digit (00001)
            fk_variants = [fk, str(int(fk)).zfill(5), str(int(fk)).zfill(4)]
            found_p = None
            for fk_v in fk_variants:
                for ext in ('.jpg', '.png', '.jpeg'):
                    candidate = root / vid_name / (fk_v + ext)
                    if candidate.exists():
                        found_p = candidate
                        break
                if found_p:
                    break
            if found_p is None:
                continue
            b = np.array(bbox, dtype=np.float32).reshape(-1)
            if len(b) < 4:
                continue
            x1, y1, x2, y2 = b[:4]
            boxes = np.array([[x1, y1, x2, y2]], dtype=np.float32) \
                    if x2 > x1 and y2 > y1 else np.zeros((0, 4), dtype=np.float32)
            items.append((str(found_p), boxes))
    print(f"  DUT-AntiUAV  : {len(items):6d} sampled frames (stride {sample_stride})")
    return items


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────────────────

class IRDetectionDataset(Dataset):
    """
    Unified IR detection dataset.
    All objects are labelled as class 1 (generic target).
    Images loaded as RGB (FasterRCNN requirement).
    """
    def __init__(self, samples: list, augment: bool = True):
        # Filter out samples where image doesn't exist
        self.samples  = [(p, b) for p, b in samples if Path(p).exists()]
        self.augment  = augment
        self._build_transforms()

    def _build_transforms(self):
        aug = []
        if self.augment:
            aug += [
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.3, contrast=0.3),
                T.RandomGrayscale(p=0.1),
            ]
        aug.append(T.ToDtype(torch.float32, scale=True))
        self.transforms = T.Compose(aug)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, boxes_np = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img_t = T.ToImage()(img)

        n = len(boxes_np)
        if n > 0:
            # Clamp boxes to image bounds
            w, h = img.size
            boxes_np = boxes_np.copy()
            boxes_np[:, [0, 2]] = boxes_np[:, [0, 2]].clip(0, w)
            boxes_np[:, [1, 3]] = boxes_np[:, [1, 3]].clip(0, h)
            # Remove degenerate boxes
            keep = (boxes_np[:, 2] - boxes_np[:, 0] > 1) & \
                   (boxes_np[:, 3] - boxes_np[:, 1] > 1)
            boxes_np = boxes_np[keep]

        if len(boxes_np) == 0:
            # No valid boxes — return image with empty targets
            # FasterRCNN can handle this during training (skips loss)
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.int64)
        else:
            boxes_t  = torch.as_tensor(boxes_np, dtype=torch.float32)
            labels_t = torch.ones(len(boxes_np), dtype=torch.int64)

        target = {
            'boxes':    boxes_t,
            'labels':   labels_t,
            'image_id': torch.tensor([idx]),
        }

        img_t, target = self.transforms(img_t, target)
        return img_t, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ─────────────────────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(pretrained_backbone: str, num_classes: int = 2) -> FasterRCNN:
    """
    FasterRCNN with ResNet-50-FPN backbone.
    Backbone initialised from sot_resnet50.pth if path exists.
    """
    backbone = resnet_fpn_backbone(
        backbone_name='resnet50',
        weights=None,
        trainable_layers=5,
    )

    if pretrained_backbone and Path(pretrained_backbone).exists():
        pretrained_sd = torch.load(pretrained_backbone, map_location='cpu')
        model_sd      = backbone.body.state_dict()
        # PySOT's ResNet50 uses 3×3 conv in downsample layers while torchvision
        # uses 1×1. Load only layers whose shapes match exactly.
        compatible = {k: v for k, v in pretrained_sd.items()
                      if k in model_sd and model_sd[k].shape == v.shape}
        skipped    = [k for k in pretrained_sd
                      if k in model_sd and model_sd[k].shape != pretrained_sd[k].shape]
        missing    = [k for k in model_sd if k not in pretrained_sd
                      and not k.endswith('num_batches_tracked')]
        model_sd.update(compatible)
        backbone.body.load_state_dict(model_sd)
        print(f"  Backbone loaded from {Path(pretrained_backbone).name}: "
              f"{len(compatible)} layers matched, "
              f"{len(skipped)} shape-mismatched (kept torchvision init), "
              f"{len(missing)} missing")
        if skipped:
            print(f"  Skipped (shape mismatch): {skipped}")
    else:
        print("  WARNING: pretrained backbone not found — using random init")

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        # Smaller anchors for small UAV/person targets
        rpn_anchor_generator=torchvision.models.detection.rpn.AnchorGenerator(
            sizes=((8,), (16,), (32,), (64,), (128,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        ),
        min_size=320, max_size=640,
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def _backbone_body(model):
    """Return the ResNet body regardless of DataParallel wrapping."""
    m = model.module if isinstance(model, nn.DataParallel) else model
    return m.backbone.body


def set_phase(model, phase: int, args):
    """
    Phase 1: backbone frozen — train FPN + head only.
    Phase 2: unfreeze backbone layer3 + layer4.
    Phase 3: unfreeze entire backbone.
    """
    body = _backbone_body(model)
    m    = model.module if isinstance(model, nn.DataParallel) else model

    # Freeze/unfreeze backbone body
    unfreeze_prefixes = {
        1: [],
        2: ['layer3', 'layer4'],
        3: ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4'],
    }[phase]

    for name, param in body.named_parameters():
        prefix = name.split('.')[0]
        param.requires_grad = prefix in unfreeze_prefixes

    # FPN and detection head always trainable
    for param in m.backbone.fpn.parameters():
        param.requires_grad = True
    for param in m.rpn.parameters():
        param.requires_grad = True
    for param in m.roi_heads.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Phase {phase}: {trainable/1e6:.1f}M trainable params")


def get_optimizer(model, phase: int, base_lr: float):
    lrs = {1: base_lr, 2: base_lr * 0.5, 3: base_lr * 0.1}
    lr  = lrs[phase]
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4), lr


def train_one_epoch(model, loader, optimizer, device, epoch, log_every=50):
    model.train()
    total_loss = 0.0
    t0 = time.perf_counter()
    for i, (images, targets) in enumerate(loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses    = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += losses.item()
        if (i + 1) % log_every == 0:
            elapsed = time.perf_counter() - t0
            print(f"    e{epoch:02d}  step {i+1:4d}/{len(loader):4d}  "
                  f"loss={losses.item():.4f}  "
                  f"cls={loss_dict.get('loss_classifier', torch.tensor(0)).item():.3f}  "
                  f"box={loss_dict.get('loss_box_reg', torch.tensor(0)).item():.3f}  "
                  f"rpn_cls={loss_dict.get('loss_rpn_box_reg', torch.tensor(0)).item():.3f}  "
                  f"t={elapsed:.0f}s")
            t0 = time.perf_counter()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.train()   # keep in train mode to get losses; eval() disables loss
    total_loss = 0.0
    for images, targets in loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        total_loss += sum(loss_dict.values()).item()
    return total_loss / max(len(loader), 1)


def save_backbone(model, out_path: str):
    """Extract and save just the backbone (ResNet body) state dict."""
    body = _backbone_body(model)
    sd   = body.state_dict()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, out_path)
    print(f"  Backbone saved → {out_path}  ({Path(out_path).stat().st_size/1e6:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root',   default='/data/siamrpn_training/data')
    ap.add_argument('--pretrained',  default='/data/siamrpn_training/pretrained/sot_resnet50.pth')
    ap.add_argument('--out',         default='/data/siamrpn_training/pretrained/ir_backbone.pth')
    ap.add_argument('--epochs',      type=int,   default=30)
    ap.add_argument('--batch',       type=int,   default=16)
    ap.add_argument('--workers',     type=int,   default=8)
    ap.add_argument('--sample_stride', type=int, default=10,
                    help='Sample 1 in N frames from SOT sequences')
    ap.add_argument('--val_frac',    type=float, default=0.1)
    ap.add_argument('--base_lr',     type=float, default=1e-3)
    ap.add_argument('--resume',      default=None,
                    help='Resume from checkpoint path')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpus = torch.cuda.device_count()
    print(f"Device: {device}  GPUs: {n_gpus}")
    print(f"Epochs: {args.epochs}  Batch/GPU: {args.batch}  "
          f"Effective batch: {args.batch * max(n_gpus,1)}")

    # ── collect data ──────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    all_samples = []
    all_samples += load_hituav(args.data_root)
    all_samples += load_anti_uav410(args.data_root, args.sample_stride)
    all_samples += load_dut_vtuav(args.data_root, args.sample_stride)
    all_samples += load_massmind(args.data_root, args.sample_stride)
    all_samples += load_dut_anti_uav(args.data_root, args.sample_stride)
    print(f"Total samples before split: {len(all_samples)}")

    random.shuffle(all_samples)
    n_val  = max(1, int(len(all_samples) * args.val_frac))
    n_train = len(all_samples) - n_val
    train_samples = all_samples[:n_train]
    val_samples   = all_samples[n_train:]
    print(f"Train: {len(train_samples)}  Val: {len(val_samples)}")

    train_ds = IRDetectionDataset(train_samples, augment=True)
    val_ds   = IRDetectionDataset(val_samples,   augment=False)
    print(f"Train (valid images): {len(train_ds)}  Val: {len(val_ds)}")

    eff_batch = args.batch * max(n_gpus, 1)
    train_loader = DataLoader(train_ds, batch_size=eff_batch, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_fn,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=eff_batch, shuffle=False,
                              num_workers=args.workers, collate_fn=collate_fn,
                              pin_memory=True)

    # ── build model ───────────────────────────────────────────────────────────
    print("\nBuilding model...")
    model = build_model(args.pretrained, num_classes=2)
    if n_gpus > 1:
        model = nn.DataParallel(model)
    model.to(device)

    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    # Phase boundaries
    phase_schedule = {
        range(1,   6): (1, args.base_lr),
        range(6,  21): (2, args.base_lr * 0.5),
        range(21, args.epochs + 1): (3, args.base_lr * 0.1),
    }

    best_val_loss = float('inf')
    ckpt_dir = Path(args.out).parent / 'backbone_ckpts'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    current_phase = None
    optimizer = None
    scheduler = None

    print("\nStarting training...")
    for epoch in range(start_epoch, args.epochs + 1):

        # Determine current phase
        new_phase = None
        for rng, (ph, lr) in phase_schedule.items():
            if epoch in rng:
                new_phase = (ph, lr)
                break
        if new_phase is None:
            new_phase = (3, args.base_lr * 0.1)

        ph, lr = new_phase
        if ph != current_phase:
            current_phase = ph
            print(f"\n── Phase {ph} (epoch {epoch}) ──")
            set_phase(model, ph, args)
            optimizer = torch.optim.SGD(
                [p for p in model.parameters() if p.requires_grad],
                lr=lr, momentum=0.9, weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=10, eta_min=lr * 0.01
            )

        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss   = evaluate(model, val_loader, device)
        scheduler.step()

        elapsed = time.perf_counter() - t0
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  t={elapsed:.0f}s")

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            ckpt_path = ckpt_dir / f'backbone_e{epoch:03d}.pth'
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'val_loss': val_loss}, str(ckpt_path))
            save_backbone(model, str(ckpt_dir / f'backbone_body_e{epoch:03d}.pth'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_backbone(model, args.out)
            print(f"  ✓ Best val_loss={best_val_loss:.4f} → saved {args.out}")

    # Final save
    save_backbone(model, args.out)
    print(f"\nDone. Best val_loss={best_val_loss:.4f}")
    print(f"Backbone weights: {args.out}")
    print(f"\nTo use in PySOT SiamRPN++ training:")
    print(f"  cfg.BACKBONE.PRETRAINED = '{args.out}'")


if __name__ == '__main__':
    main()
