#!/usr/bin/env python3
"""
train_stage0_simclr.py — Stage 0: SimCLR self-supervised pre-training on IR data

Learns IR-specific features without any labels using NT-Xent contrastive loss.
All RGB data is excluded; only confirmed-grayscale IR images are used.

Datasets (~1.04M IR images):
  hit_uav      — 2,866   images  (UAV-view road objects)
  anti_uav410  — 438,397 images  (IR sequences, UAV targets)
  dut_vtuav    — 596,348 images  (infrared/ subfolders only — rgb/ excluded)
  massmind     — 2,916   images
  msrs         — 1,083   images  (train/ir/ only)

Launch:
  torchrun --nproc_per_node=4 --master_port=29500 train_stage0_simclr.py [--epochs 100] [--batch 64]

Output:
  pretrained/ir_simclr_backbone.pth  — ResNet-50 backbone weights only (no projection head)
"""

import os, math, random, argparse, glob
from pathlib import Path

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast

import torchvision.models as tvm

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_ROOT = Path('/data/siamrpn_training/data')
OUT_DIR   = Path('/data/siamrpn_training/pretrained')
LOG_PATH  = '/data/siamrpn_training/stage0_simclr.log'

# Only infrared paths — rgb subfolders explicitly excluded
IR_GLOBS = [
    str(DATA_ROOT / 'hit_uav/images/**/*.jpg'),
    str(DATA_ROOT / 'anti_uav410/**/*.jpg'),
    str(DATA_ROOT / 'dut_vtuav/**/infrared/*.jpg'),   # infrared only, not rgb
    str(DATA_ROOT / 'massmind/images/**/*.png'),
    str(DATA_ROOT / 'msrs/train/ir/*.png'),
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32)


# ── DDP setup ─────────────────────────────────────────────────────────────────

def setup_ddp():
    if 'LOCAL_RANK' not in os.environ:
        return 0, 1, False
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    return local_rank, world_size, True


# ── Augmentations ─────────────────────────────────────────────────────────────

class IRAugment:
    """
    Thermal-appropriate augmentations. No colour jitter (images are grayscale).
    Two independent augmented views of the same IR patch.
    """
    def __init__(self, size=224):
        self.size = size

    def _augment(self, img_bgr):
        h, w = img_bgr.shape[:2]

        # 1. Random resized crop
        scale = random.uniform(0.2, 1.0)
        ratio = random.uniform(0.75, 1.33)
        crop_h = int(h * math.sqrt(scale / ratio))
        crop_w = int(w * math.sqrt(scale * ratio))
        crop_h = max(16, min(crop_h, h))
        crop_w = max(16, min(crop_w, w))
        y0 = random.randint(0, h - crop_h)
        x0 = random.randint(0, w - crop_w)
        img = img_bgr[y0:y0+crop_h, x0:x0+crop_w]
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

        # 2. Random horizontal flip
        if random.random() > 0.5:
            img = img[:, ::-1].copy()

        # 3. Random Gaussian blur (emissivity smearing)
        if random.random() > 0.5:
            sigma = random.uniform(0.5, 2.0)
            k = int(2 * math.ceil(2 * sigma) + 1)
            img = cv2.GaussianBlur(img, (k, k), sigma)

        # 4. Random brightness jitter (thermal offset ±15%)
        if random.random() > 0.5:
            factor = random.uniform(0.85, 1.15)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        # 5. Additive thermal noise (sensor noise)
        if random.random() > 0.5:
            noise = np.random.randn(*img.shape).astype(np.float32) * random.uniform(1, 8)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 6. To float tensor, normalize
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(rgb.transpose(2, 0, 1)).float()

    def __call__(self, img_bgr):
        return self._augment(img_bgr), self._augment(img_bgr)


# ── Dataset ───────────────────────────────────────────────────────────────────

class IRSimCLRDataset(Dataset):
    def __init__(self, size=224):
        self.aug = IRAugment(size)
        self.paths = []
        for pattern in IR_GLOBS:
            found = glob.glob(pattern, recursive=True)
            self.paths.extend(found)
        random.shuffle(self.paths)
        print(f"[SimCLR Dataset] {len(self.paths):,} IR images loaded")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        if img is None:
            # Fallback to a random valid image
            img = cv2.imread(self.paths[0])
        try:
            v1, v2 = self.aug(img)
        except Exception:
            v1 = torch.zeros(3, 224, 224)
            v2 = torch.zeros(3, 224, 224)
        return v1, v2


# ── Model ─────────────────────────────────────────────────────────────────────

class SimCLRModel(nn.Module):
    """ResNet-50 backbone + 2-layer MLP projection head."""
    def __init__(self, proj_dim=128):
        super().__init__()
        base = tvm.resnet50(weights=None)
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            base.avgpool,
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim, affine=False),
        )

    def forward(self, x):
        h = self.backbone(x).flatten(1)   # (B, 2048)
        z = self.proj(h)                  # (B, proj_dim)
        return F.normalize(z, dim=1)

    def backbone_state(self):
        """Return just the backbone weights (what downstream stages need)."""
        return self.backbone.state_dict()


# ── NT-Xent loss ──────────────────────────────────────────────────────────────

def nt_xent_loss(z1, z2, temperature=0.07):
    """
    z1, z2: (N, D) L2-normalised projection vectors.
    Positive pair: (z1[i], z2[i]).  All other 2N-2 pairs are negatives.
    """
    N = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)        # (2N, D)
    sim = torch.mm(z, z.t()) / temperature # (2N, 2N)

    # Mask self-similarity
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))

    # Positive index for each anchor
    labels = torch.cat([torch.arange(N, 2*N), torch.arange(N)]).to(z.device)
    loss = F.cross_entropy(sim, labels)
    return loss


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    rank, world_size, ddp = setup_ddp()
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    def log(msg):
        if rank == 0:
            print(msg, flush=True)
            with open(LOG_PATH, 'a') as f:
                f.write(msg + '\n')

    log(f"=== Stage 0: SimCLR  epochs={args.epochs}  batch_per_gpu={args.batch}  world={world_size} ===")

    # Dataset
    ds = IRSimCLRDataset(size=args.size)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
    loader  = DataLoader(ds, batch_size=args.batch, sampler=sampler,
                         num_workers=args.workers, pin_memory=True,
                         drop_last=True, shuffle=(sampler is None))

    # Model
    model = SimCLRModel(proj_dim=128).to(device)
    if ddp:
        model = DDP(model, device_ids=[rank])

    # Optimizer — cosine LR schedule, linear warmup
    effective_batch = args.batch * world_size
    base_lr = 0.3 * effective_batch / 256
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr,
                                momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = GradScaler()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start_epoch = 1

    # Resume
    ckpt_path = OUT_DIR / 'ir_simclr_checkpoint.pth'
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu')
        m = model.module if ddp else model
        m.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        log(f"Resumed from epoch {ckpt['epoch']}, loss={ckpt['loss']:.4f}")

    best_loss = float('inf')

    for epoch in range(start_epoch, args.epochs + 1):
        if ddp:
            sampler.set_epoch(epoch)

        model.train()
        total_loss, n_steps = 0., 0

        for step, (v1, v2) in enumerate(loader):
            v1, v2 = v1.to(device, non_blocking=True), v2.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast():
                z1 = model(v1)
                z2 = model(v2)
                loss = nt_xent_loss(z1, z2, temperature=args.temperature)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_steps += 1

            if rank == 0 and step % 200 == 0:
                log(f"  epoch {epoch:3d}  step {step:5d}/{len(loader)}  loss={loss.item():.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()
        avg_loss = total_loss / max(n_steps, 1)

        if ddp:
            t = torch.tensor(avg_loss, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            avg_loss = t.item()

        log(f"Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if rank == 0:
            m = model.module if ddp else model
            ckpt = {
                'epoch': epoch,
                'model': m.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': avg_loss,
            }
            torch.save(ckpt, ckpt_path)

            if avg_loss < best_loss:
                best_loss = avg_loss
                # Save backbone only — no projection head
                torch.save(m.backbone_state(), OUT_DIR / 'ir_simclr_backbone.pth')
                log(f"  *** New best loss {best_loss:.4f} — backbone saved ***")

            if epoch % 10 == 0:
                torch.save(ckpt, OUT_DIR / f'ir_simclr_ep{epoch:03d}.pth')

    if rank == 0:
        log(f"Stage 0 complete. Best loss: {best_loss:.4f}")
        log(f"Backbone → {OUT_DIR}/ir_simclr_backbone.pth")

    if ddp:
        dist.destroy_process_group()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',      type=int,   default=100)
    p.add_argument('--batch',       type=int,   default=64,   help='per-GPU batch size')
    p.add_argument('--workers',     type=int,   default=6)
    p.add_argument('--size',        type=int,   default=224,  help='input crop size')
    p.add_argument('--temperature', type=float, default=0.07)
    args = p.parse_args()
    train(args)
