#!/usr/bin/env python3
"""
train_stage1_detection.py — Stage 1: Multi-class detection fine-tuning

Initialises backbone from Stage 0 SimCLR checkpoint and trains a 4-class
FasterRCNN on HIT-UAV (person / car / bicycle / vehicle).

Training phases:
  Phase 1  (epochs  1–5 ): freeze backbone, warm up detection head
  Phase 2  (epochs  6–20): unfreeze layer3 + layer4
  Phase 3  (epochs 21–30): unfreeze full backbone

Launch:
  torchrun --nproc_per_node=4 --master_port=29501 train_stage1_detection.py

Output:
  pretrained/ir_detector_backbone.pth
"""

import os, math, random, argparse
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import torchvision.models as tvm
import torchvision.transforms.functional as TF
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# ── Paths ─────────────────────────────────────────────────────────────────────

HIT_UAV_ROOT  = Path('/data/siamrpn_training/data/hit_uav')
SIMCLR_CKPT   = Path('/data/siamrpn_training/pretrained/ir_simclr_backbone.pth')
SIAM_CKPT     = Path('/data/siamrpn_training/pretrained/sot_resnet50.pth')   # fallback
OUT_DIR       = Path('/data/siamrpn_training/pretrained')
LOG_PATH      = '/data/siamrpn_training/stage1_detection.log'

# HIT-UAV class mapping: YOLO id → FasterRCNN label (1-indexed, 0=bg)
# YOLO: 0=person 1=car 2=bicycle 3=vehicle 4=DontCare (skip)
YOLO_TO_LABEL = {0: 1, 1: 2, 2: 3, 3: 4}   # {yolo_id: detector_label}
NUM_CLASSES   = 5                             # bg + 4 road-object classes

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ── DDP ───────────────────────────────────────────────────────────────────────

def setup_ddp():
    if 'LOCAL_RANK' not in os.environ:
        return 0, 1, False
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    return local_rank, world_size, True


# ── Dataset ───────────────────────────────────────────────────────────────────

class HitUAVDataset(Dataset):
    """
    HIT-UAV in YOLO format.  Returns FasterRCNN-compatible targets:
      {'boxes': FloatTensor(N,4) xyxy, 'labels': Int64Tensor(N)}
    """
    def __init__(self, split='train', augment=True):
        self.augment = augment
        self.items = []   # list of (img_path, [(label, x1, y1, x2, y2), ...])

        img_dir = HIT_UAV_ROOT / 'images' / split
        lbl_dir = HIT_UAV_ROOT / 'labels' / split
        if not img_dir.exists():
            return

        for img_p in sorted(img_dir.glob('*.jpg')):
            lbl_p = lbl_dir / img_p.with_suffix('.txt').name
            if not lbl_p.exists():
                continue
            try:
                w, h = Image.open(img_p).size
            except Exception:
                continue

            boxes = []
            for ln in lbl_p.read_text().splitlines():
                parts = ln.strip().split()
                if len(parts) != 5:
                    continue
                yolo_cls = int(parts[0])
                if yolo_cls not in YOLO_TO_LABEL:
                    continue              # skip DontCare (class 4)
                label = YOLO_TO_LABEL[yolo_cls]
                cx, cy, bw, bh = map(float, parts[1:])
                x1, y1 = (cx - bw/2)*w, (cy - bh/2)*h
                x2, y2 = (cx + bw/2)*w, (cy + bh/2)*h
                if x2 - x1 > 4 and y2 - y1 > 4:
                    boxes.append((label, x1, y1, x2, y2))

            if boxes:
                self.items.append((str(img_p), boxes))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, ann = self.items[idx]
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((256, 256, 3), dtype=np.uint8)
        h, w = img.shape[:2]

        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                img = img[:, ::-1].copy()
                ann = [(lbl, w - x2, y1, w - x1, y2)
                       for (lbl, x1, y1, x2, y2) in ann]
            # Random brightness
            if random.random() > 0.5:
                f = random.uniform(0.8, 1.2)
                img = np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        rgb = (rgb - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float()

        boxes  = torch.tensor([[x1, y1, x2, y2] for (_, x1, y1, x2, y2) in ann], dtype=torch.float32)
        labels = torch.tensor([lbl for (lbl, *_) in ann], dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels}
        return tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


# ── Backbone ──────────────────────────────────────────────────────────────────

def build_detector(simclr_ckpt_path, fallback_path):
    """
    Build FasterRCNN with ResNet-50 backbone.
    Loads from Stage 0 SimCLR checkpoint if available, else falls back to
    sot_resnet50.pth (same loading logic as the original binary training).
    """
    base = tvm.resnet50(weights=None)

    if Path(simclr_ckpt_path).exists():
        # SimCLR backbone state dict (just the backbone Sequential, not projection head)
        # Keys look like: '0.weight' (conv1), '1.weight' (bn1), etc.
        # We need to remap to resnet50 keys.
        sd = torch.load(simclr_ckpt_path, map_location='cpu')
        # SimCLR backbone is nn.Sequential(conv1,bn1,relu,maxpool,layer1,...,avgpool)
        # Indices: 0=conv1,1=bn1,2=relu,3=maxpool,4=layer1,5=layer2,6=layer3,7=layer4,8=avgpool
        seq_to_resnet = {
            '0.': 'conv1.', '1.': 'bn1.',
            '4.': 'layer1.', '5.': 'layer2.',
            '6.': 'layer3.', '7.': 'layer4.',
        }
        remapped = {}
        for k, v in sd.items():
            for seq_pfx, res_pfx in seq_to_resnet.items():
                if k.startswith(seq_pfx):
                    remapped[res_pfx + k[len(seq_pfx):]] = v
                    break
        msd = base.state_dict()
        compat = {k: v for k, v in remapped.items() if k in msd and msd[k].shape == v.shape}
        msd.update(compat)
        base.load_state_dict(msd)
        n_loaded = len(compat)
        print(f"  Loaded {n_loaded} layers from SimCLR backbone: {simclr_ckpt_path}")

    elif Path(fallback_path).exists():
        sd  = torch.load(fallback_path, map_location='cpu')
        msd = base.state_dict()
        compat = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
        msd.update(compat)
        base.load_state_dict(msd)
        print(f"  Loaded {len(compat)} layers from fallback: {fallback_path}")
    else:
        print("  WARNING: no backbone checkpoint found — using random init")

    # Build FasterRCNN with the custom backbone
    backbone_body = nn.Sequential(
        base.conv1, base.bn1, base.relu, base.maxpool,
        base.layer1, base.layer2, base.layer3, base.layer4,
    )
    backbone_body.out_channels = 2048

    anchor_gen = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )

    model = FasterRCNN(
        backbone=backbone_body,
        num_classes=NUM_CLASSES,
        rpn_anchor_generator=anchor_gen,
        box_detections_per_img=100,
        min_size=256,
        max_size=800,
    )
    return model


def _backbone_body(model, ddp):
    m = model.module if ddp else model
    return m.backbone


def freeze_backbone(model, ddp):
    bb = _backbone_body(model, ddp)
    for p in bb.parameters():
        p.requires_grad_(False)

def unfreeze_layers(model, ddp, layer_names):
    bb = _backbone_body(model, ddp)
    for name, module in bb.named_modules():
        for lname in layer_names:
            if lname in name:
                for p in module.parameters():
                    p.requires_grad_(True)

def unfreeze_all(model, ddp):
    bb = _backbone_body(model, ddp)
    for p in bb.parameters():
        p.requires_grad_(True)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, rank, world_size, epoch):
    model.train()
    total_loss, n_steps = 0., 0

    for step, (images, targets) in enumerate(loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 5.0)
        optimizer.step()

        total_loss += loss.item()
        n_steps += 1

    avg = total_loss / max(n_steps, 1)
    if world_size > 1:
        t = torch.tensor(avg, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        avg = t.item()
    return avg


@torch.no_grad()
def val_loss(model, loader, device, world_size):
    model.train()   # keep BN in train mode for stable stats
    total, n = 0., 0
    for images, targets in loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        total += sum(loss_dict.values()).item()
        n += 1
    avg = total / max(n, 1)
    if world_size > 1:
        t = torch.tensor(avg, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        avg = t.item()
    return avg


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    rank, world_size, ddp = setup_ddp()
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    def log(msg):
        if rank == 0:
            print(msg, flush=True)
            with open(LOG_PATH, 'a') as f:
                f.write(msg + '\n')

    log(f"=== Stage 1: 4-class Detection  epochs={args.epochs}  batch={args.batch}  world={world_size} ===")

    # Datasets
    train_ds = HitUAVDataset('train', augment=True)
    val_ds   = HitUAVDataset('val',   augment=False)
    # Combine test into train if val is empty
    if len(val_ds) == 0:
        val_ds = HitUAVDataset('test', augment=False)

    if ddp:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = val_sampler = None

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              sampler=train_sampler, shuffle=(train_sampler is None),
                              num_workers=args.workers, collate_fn=collate_fn,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch,
                              sampler=val_sampler, shuffle=False,
                              num_workers=args.workers, collate_fn=collate_fn,
                              pin_memory=True)

    log(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")

    # Model
    model = build_detector(SIMCLR_CKPT, SIAM_CKPT).to(device)
    if ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Phase schedule
    # Phase 1: freeze backbone, high LR on head
    # Phase 2: unfreeze layer3+layer4, moderate LR
    # Phase 3: unfreeze all, low LR
    PHASE1_END = 5
    PHASE2_END = 20

    freeze_backbone(model, ddp)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=1e-4)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val  = float('inf')
    ckpt_path = OUT_DIR / 'ir_detector_checkpoint.pth'
    start_epoch = 1

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu')
        m = model.module if ddp else model
        m.load_state_dict(ckpt['model'])
        start_epoch = ckpt['epoch'] + 1
        best_val    = ckpt.get('best_val', float('inf'))
        log(f"Resumed from epoch {ckpt['epoch']}")

    log(f"  Classes: bg(0) person(1) car(2) bicycle(3) vehicle(4)")

    for epoch in range(start_epoch, args.epochs + 1):
        if ddp and train_sampler:
            train_sampler.set_epoch(epoch)

        # Phase transitions
        if epoch == PHASE1_END + 1:
            log("  → Phase 2: unfreeze layer3 + layer4")
            unfreeze_layers(model, ddp, ['6.', '7.', 'layer3', 'layer4'])
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=5e-4, weight_decay=1e-4)

        if epoch == PHASE2_END + 1:
            log("  → Phase 3: unfreeze full backbone")
            unfreeze_all(model, ddp)
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=1e-4, weight_decay=1e-4)

        train_l = train_one_epoch(model, train_loader, optimizer, device, rank, world_size, epoch)
        val_l   = val_loss(model, val_loader, device, world_size)

        phase = 1 if epoch <= PHASE1_END else (2 if epoch <= PHASE2_END else 3)
        log(f"Epoch {epoch:3d}/{args.epochs}  phase={phase}  train={train_l:.4f}  val={val_l:.4f}")

        if rank == 0:
            m = model.module if ddp else model
            torch.save({'epoch': epoch, 'model': m.state_dict(),
                        'best_val': best_val}, ckpt_path)

            if val_l < best_val:
                best_val = val_l
                # Save backbone only
                bb = m.backbone
                torch.save(bb.state_dict(), OUT_DIR / 'ir_detector_backbone.pth')
                log(f"  *** Val {best_val:.4f} — backbone saved ***")

    if rank == 0:
        log(f"Stage 1 complete. Best val_loss: {best_val:.4f}")
        log(f"Backbone → {OUT_DIR}/ir_detector_backbone.pth")

    if ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',  type=int, default=30)
    p.add_argument('--batch',   type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    main(args)
