#!/usr/bin/env python3
"""
train_stage2_siamese.py — Stage 2: Siamese tracking fine-tuning

Trains the backbone with a video-pair InfoNCE (NT-Xent) objective:
  template crop (frame t)  ←→  search crop (frame t+δ, same object)

For each pair the backbone should produce features where the template
and the target region in the search are closer to each other than to
any other sequence's features in the same batch.

Loss applied at all three SiamRPN++ scales (L2, L3, L4) with equal weight.

Dataset: DUT-VTUAV infrared sequences (596K frames, road objects)
         — car, bus, truck, pedestrian, tricycle, bicycle

Launch:
  torchrun --nproc_per_node=4 --master_port=29502 train_stage2_siamese.py

Output:
  pretrained/ir_siamese_backbone.pth
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

VTUAV_TRAIN   = Path('/data/siamrpn_training/data/dut_vtuav/train')
DETECTOR_CKPT = Path('/data/siamrpn_training/pretrained/ir_detector_backbone.pth')
SIMCLR_CKPT   = Path('/data/siamrpn_training/pretrained/ir_simclr_backbone.pth')
SIAM_CKPT     = Path('/data/siamrpn_training/pretrained/sot_resnet50.pth')
OUT_DIR       = Path('/data/siamrpn_training/pretrained')
LOG_PATH      = '/data/siamrpn_training/stage2_siamese.log'

TEMPLATE_SIZE = 127
SEARCH_SIZE   = 255
MAX_GAP       = 50    # max frame gap between template and search
MIN_BOX_PX    = 8     # ignore boxes smaller than this (pixels)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32)


# ── DDP ───────────────────────────────────────────────────────────────────────

def setup_ddp():
    if 'LOCAL_RANK' not in os.environ:
        return 0, 1, False
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    return local_rank, world_size, True


# ── Cropping helpers ──────────────────────────────────────────────────────────

def crop_and_resize(img, cx, cy, crop_sz, out_sz):
    """
    Crop a square of side crop_sz centred at (cx, cy), resize to out_sz.
    Pads with mean value if crop extends beyond image borders.
    """
    h, w = img.shape[:2]
    avg_ch = img.mean(axis=(0, 1))

    half = crop_sz / 2
    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half

    # Padding amounts
    pad_l = max(0, -x1);  pad_t = max(0, -y1)
    pad_r = max(0, x2-w); pad_b = max(0, y2-h)

    x1c = int(round(x1 + pad_l));  y1c = int(round(y1 + pad_t))
    x2c = int(round(x2 + pad_l));  y2c = int(round(y2 + pad_t))

    if pad_l+pad_r+pad_t+pad_b > 0:
        pl, pr = int(math.ceil(pad_l)), int(math.ceil(pad_r))
        pt, pb = int(math.ceil(pad_t)), int(math.ceil(pad_b))
        img = cv2.copyMakeBorder(img, pt, pb, pl, pr,
                                 cv2.BORDER_CONSTANT, value=avg_ch.tolist())

    crop = img[y1c:y2c, x1c:x2c]
    if crop.size == 0:
        crop = np.full((out_sz, out_sz, 3), avg_ch, dtype=np.uint8)
    crop = cv2.resize(crop, (out_sz, out_sz), interpolation=cv2.INTER_LINEAR)
    return crop


def get_template_crop(img, bbox_xywh):
    """127×127 exemplar with SiamRPN++ context padding."""
    x, y, w, h = bbox_xywh
    cx, cy = x + w/2, y + h/2
    context = 0.5 * (w + h)
    crop_sz = math.sqrt((w + context) * (h + context))
    return crop_and_resize(img, cx, cy, crop_sz, TEMPLATE_SIZE)


def get_search_crop(img, bbox_xywh, jitter_frac=0.2):
    """255×255 search region centred near target with slight jitter."""
    x, y, w, h = bbox_xywh
    cx, cy = x + w/2, y + h/2
    cx += w * random.uniform(-jitter_frac, jitter_frac)
    cy += h * random.uniform(-jitter_frac, jitter_frac)
    context = 0.5 * (w + h)
    crop_sz = math.sqrt((w + context) * (h + context)) * (SEARCH_SIZE / TEMPLATE_SIZE)
    return crop_and_resize(img, cx, cy, crop_sz, SEARCH_SIZE)


def to_tensor(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb.transpose(2, 0, 1)).float()


# ── Dataset ───────────────────────────────────────────────────────────────────

class VTUAVSiameseDataset(Dataset):
    """
    Samples (template_crop, search_crop) pairs from DUT-VTUAV infrared sequences.
    Only uses sequences that have at least 2 valid annotated frames.
    RGB subfolders are never touched.
    """
    def __init__(self, root=VTUAV_TRAIN, max_gap=MAX_GAP, augment=True):
        self.max_gap = max_gap
        self.augment = augment
        self.sequences = []   # list of (seq_dir, frames_list, gt_list)
        self._load_sequences(root)

    def _load_sequences(self, root):
        for seq_dir in sorted(Path(root).iterdir()):
            gt_path  = seq_dir / 'groundtruth.txt'
            ir_dir   = seq_dir / 'infrared'
            if not gt_path.exists() or not ir_dir.exists():
                continue

            # Parse groundtruth (x y w h, space or comma separated)
            gts = []
            for line in gt_path.read_text().strip().splitlines():
                parts = line.strip().replace(',', ' ').split()
                if len(parts) >= 4:
                    x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                    gts.append((x, y, w, h) if w >= MIN_BOX_PX and h >= MIN_BOX_PX else None)
                else:
                    gts.append(None)

            frames = sorted(ir_dir.glob('*.jpg'))
            n = min(len(frames), len(gts))
            gts    = gts[:n]
            frames = frames[:n]

            # Need at least 2 valid frames
            valid = [i for i, g in enumerate(gts) if g is not None]
            if len(valid) >= 2:
                self.sequences.append((seq_dir.name, frames, gts, valid))

        print(f"[Stage2 Dataset] {len(self.sequences)} sequences from DUT-VTUAV infrared")
        total_frames = sum(len(f) for _, f, _, _ in self.sequences)
        print(f"                 {total_frames:,} total frames")

    def __len__(self):
        # Virtual epoch length: 1000 pairs per sequence
        return len(self.sequences) * 1000

    def __getitem__(self, idx):
        seq_idx = idx % len(self.sequences)
        _, frames, gts, valid_idx = self.sequences[seq_idx]

        # Sample template frame
        t_pos = random.choice(valid_idx[:-1])
        # Sample search frame (forward in time, within max_gap)
        candidates = [i for i in valid_idx if t_pos < i <= t_pos + self.max_gap]
        if not candidates:
            # Fallback: any other valid frame
            candidates = [i for i in valid_idx if i != t_pos]
        s_pos = random.choice(candidates)

        t_frame = cv2.imread(str(frames[t_pos]))
        s_frame = cv2.imread(str(frames[s_pos]))
        if t_frame is None or s_frame is None:
            # Return zeros — will contribute zero loss
            z = torch.zeros(3, TEMPLATE_SIZE, TEMPLATE_SIZE)
            x = torch.zeros(3, SEARCH_SIZE, SEARCH_SIZE)
            return z, x

        t_bbox = gts[t_pos]
        s_bbox = gts[s_pos]

        t_crop = get_template_crop(t_frame, t_bbox)
        s_crop = get_search_crop(s_frame, s_bbox)

        if self.augment:
            if random.random() > 0.5:
                t_crop = t_crop[:, ::-1].copy()
                s_crop = s_crop[:, ::-1].copy()
            if random.random() > 0.5:
                f = random.uniform(0.85, 1.15)
                t_crop = np.clip(t_crop.astype(np.float32)*f, 0, 255).astype(np.uint8)
                s_crop = np.clip(s_crop.astype(np.float32)*f, 0, 255).astype(np.uint8)

        return to_tensor(t_crop), to_tensor(s_crop)


# ── Backbone ──────────────────────────────────────────────────────────────────

class MultiLayerBackbone(nn.Module):
    """ResNet-50 returning (L2, L3, L4) feature maps — same as SiamRPN++ usage."""
    def __init__(self):
        super().__init__()
        base = tvm.resnet50(weights=None)
        self.stem   = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward(self, x):
        x  = self.stem(x)
        x  = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        return l2, l3, l4

    def state_dict_backbone(self):
        """Return flat state dict compatible with other backbone savers."""
        sd = {}
        for prefix, module in [
            ('conv1', self.stem[0]), ('bn1', self.stem[1]),
            ('layer1', self.layer1), ('layer2', self.layer2),
            ('layer3', self.layer3), ('layer4', self.layer4),
        ]:
            for k, v in module.state_dict().items():
                sd[f'{prefix}.{k}'] = v
        return sd


def load_backbone(model):
    """Load weights: prefer Stage 1 detector backbone, fall back to SimCLR, then sot."""
    for path, label in [
        (DETECTOR_CKPT, 'Stage1 detector backbone'),
        (SIMCLR_CKPT,   'Stage0 SimCLR backbone'),
        (SIAM_CKPT,     'sot_resnet50 fallback'),
    ]:
        if not Path(path).exists():
            continue
        sd = torch.load(path, map_location='cpu')
        # Attempt to match keys (may need remapping for SimCLR sequential format)
        # Build a flat ResNet50-style dict from whatever format it's in
        target_sd = {}
        target_sd.update(model.layer1.state_dict())   # probe for key format
        # Try direct load (detector backbone format)
        own_sd = {}
        for attr, module in [
            ('stem', model.stem), ('layer1', model.layer1),
            ('layer2', model.layer2), ('layer3', model.layer3),
            ('layer4', model.layer4),
        ]:
            for k in module.state_dict():
                full_k = f'{attr}.{k}'
                # Also try resnet-style key
                res_k  = full_k.replace('stem.0.', 'conv1.').replace('stem.1.', 'bn1.')
                for src_k in [full_k, res_k, k]:
                    if src_k in sd and sd[src_k].shape == module.state_dict()[k].shape:
                        own_sd[full_k] = sd[src_k]
                        break

        if own_sd:
            missing = model.load_state_dict(
                {**model.state_dict(), **own_sd}, strict=False)
            print(f"  Loaded {len(own_sd)} params from {label}")
            return
    print("  WARNING: no checkpoint found — using random init")


# ── InfoNCE loss ──────────────────────────────────────────────────────────────

def nt_xent(z_template, z_search, temperature=0.07):
    """
    z_template, z_search: (B, D) L2-normalised vectors.
    Positive pair: z_template[i] ↔ z_search[i].
    All other B-1 search entries are negatives for each template.
    """
    B = z_template.shape[0]
    z = torch.cat([z_template, z_search], dim=0)    # (2B, D)
    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.t()) / temperature          # (2B, 2B)
    mask = torch.eye(2*B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))
    labels = torch.cat([torch.arange(B, 2*B), torch.arange(B)]).to(z.device)
    return F.cross_entropy(sim, labels)


def pool_center(feat, window=2):
    """
    Pool the central window×window region of a feature map.
    This corresponds to the target region (search crop is centred on target).
    Returns (B, C).
    """
    _, C, H, W = feat.shape
    ch, cw = H // 2, W // 2
    h0 = max(0, ch - window // 2)
    w0 = max(0, cw - window // 2)
    region = feat[:, :, h0:h0+window, w0:w0+window]
    return region.mean(dim=[2, 3])    # (B, C)


def siamese_loss(bb, templates, searches, temperature=0.07):
    """
    Compute InfoNCE at L2, L3, L4.
    template: (B, 3, 127, 127)  search: (B, 3, 255, 255)
    """
    t_l2, t_l3, t_l4 = bb(templates)
    s_l2, s_l3, s_l4 = bb(searches)

    loss = 0.
    for t_feat, s_feat, win in [
        (t_l2, s_l2, 4),   # L2: stride 8  → ~16×16 template, ~32×32 search
        (t_l3, s_l3, 2),   # L3: stride 16 → ~8×8  template, ~16×16 search
        (t_l4, s_l4, 2),   # L4: stride 32 → ~4×4  template, ~8×8  search
    ]:
        z_t = F.normalize(t_feat.mean(dim=[2, 3]), dim=1)   # GAP template
        z_s = F.normalize(pool_center(s_feat, win), dim=1)  # centre-pool search
        loss = loss + nt_xent(z_t, z_s, temperature)

    return loss / 3.0


# ── Training loop ─────────────────────────────────────────────────────────────

def main(args):
    rank, world_size, ddp = setup_ddp()
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    def log(msg):
        if rank == 0:
            print(msg, flush=True)
            with open(LOG_PATH, 'a') as f:
                f.write(msg + '\n')

    log(f"=== Stage 2: Siamese  epochs={args.epochs}  batch={args.batch}  world={world_size} ===")

    # Dataset
    ds = VTUAVSiameseDataset(augment=True)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
    loader  = DataLoader(ds, batch_size=args.batch, sampler=sampler,
                         shuffle=(sampler is None), num_workers=args.workers,
                         pin_memory=True, drop_last=True)

    # Backbone
    bb = MultiLayerBackbone().to(device)
    if rank == 0:
        load_backbone(bb)
    if ddp:
        # Broadcast loaded weights from rank 0 to all ranks
        for p in bb.parameters():
            dist.broadcast(p.data, src=0)
        bb = DDP(bb, device_ids=[rank], find_unused_parameters=False)

    # Very conservative LR — backbone is already well-trained, just aligning it
    optimizer = torch.optim.AdamW(bb.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)
    scaler = GradScaler()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')
    start_epoch = 1

    ckpt_path = OUT_DIR / 'ir_siamese_checkpoint.pth'
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu')
        m = bb.module if ddp else bb
        m.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_loss   = ckpt.get('best_loss', float('inf'))
        log(f"Resumed from epoch {ckpt['epoch']}")

    for epoch in range(start_epoch, args.epochs + 1):
        if ddp and sampler:
            sampler.set_epoch(epoch)

        bb.train()
        total_loss, n_steps = 0., 0

        for step, (templates, searches) in enumerate(loader):
            templates = templates.to(device, non_blocking=True)
            searches  = searches.to(device,  non_blocking=True)

            optimizer.zero_grad()
            with autocast():
                loss = siamese_loss(bb, templates, searches, args.temperature)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(bb.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_steps += 1

            if rank == 0 and step % 200 == 0:
                log(f"  epoch {epoch:3d}  step {step:5d}/{len(loader)}"
                    f"  loss={loss.item():.4f}  lr={scheduler.get_last_lr()[0]:.1e}")

        scheduler.step()
        avg_loss = total_loss / max(n_steps, 1)

        if ddp:
            t = torch.tensor(avg_loss, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            avg_loss = t.item()

        log(f"Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.1e}")

        if rank == 0:
            m = bb.module if ddp else bb
            torch.save({'epoch': epoch, 'model': m.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_loss': best_loss}, ckpt_path)

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(m.state_dict_backbone(), OUT_DIR / 'ir_siamese_backbone.pth')
                log(f"  *** Best loss {best_loss:.4f} — backbone saved ***")

    if rank == 0:
        log(f"Stage 2 complete. Best loss: {best_loss:.4f}")
        log(f"Backbone → {OUT_DIR}/ir_siamese_backbone.pth")

    if ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',      type=int,   default=20)
    p.add_argument('--batch',       type=int,   default=32,  help='per-GPU batch size')
    p.add_argument('--workers',     type=int,   default=6)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--temperature', type=float, default=0.07)
    args = p.parse_args()
    main(args)
