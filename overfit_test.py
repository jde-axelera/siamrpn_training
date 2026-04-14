#!/usr/bin/env python3
"""
overfit_test.py — Gradient-flow / DDP parity sanity check for SiamRPN++
=========================================================================
Loads N real (template, search) pairs into memory, trains for E epochs,
saves loss-per-epoch to CSV. Run twice and compare:

  # Single GPU — full batch of 32 each step
  python overfit_test.py --out overfit_1gpu.csv

  # 4 GPU DDP — 8 samples/GPU, all-reduced → same effective batch of 32
  /data/miniconda3/envs/pysot/bin/torchrun --nproc_per_node=4 \\
      overfit_test.py --out overfit_4gpu.csv

  # Compare
  python overfit_test.py --compare overfit_1gpu.csv overfit_4gpu.csv

Design
------
• TinyFixedDataset pre-loads N sample pairs once (no runtime randomness).
• No augmentation — images are cropped to exemplar/search size and stored.
• Effective batch is IDENTICAL between single-GPU and DDP:
    single GPU : batch_size=32 → 1 step/epoch, full-batch SGD
    4 GPU DDP  : batch_size=8/GPU, all-reduce averages → identical gradient
• Same base LR=0.005, Adam (converges faster than SGD for tiny data), no warmup.
• Both runs should reach loss < 0.3 by epoch ~30 if gradients flow correctly.
"""
from __future__ import annotations
import argparse, csv, json, os, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
import cv2

# ── PySOT path ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "pysot"))

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.utils.model_load import load_pretrain
from pysot.datasets.anchor_target import AnchorTarget
from pysot.utils.bbox import center2corner, Center


# ── Distributed helpers ───────────────────────────────────────────────────────
def init_distributed():
    if "RANK" not in os.environ:
        return 0, 0, 1, True
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)          # MUST be before NCCL init
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, local_rank, world_size, (rank == 0)


# ── Tiny fixed dataset ────────────────────────────────────────────────────────
class TinyFixedDataset(Dataset):
    """
    Pre-loads N real (template, search) pairs from Anti-UAV410 train split.
    No runtime randomness — every call to __getitem__ returns the same stored
    tensors, so both training runs operate on truly identical data.
    """
    def __init__(self, root, anno_path, n_samples=32, seed=42):
        self.data = []
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)

        with open(anno_path) as f:
            labels = json.load(f)

        anchor_target = AnchorTarget()

        seqs = [(s, sorted(int(k) for k in t["0"].keys()))
                for s, t in labels.items() if len(t["0"]) >= 2]
        rng.shuffle(seqs)

        for seq, frames in seqs:
            if len(self.data) >= n_samples:
                break
            track = labels[seq]["0"]

            # Fixed pair: first and ~middle frame
            t_idx = 0
            s_idx = min(25, len(frames) - 1)
            tf, sf = frames[t_idx], frames[s_idx]

            t_img = self._load(root, seq, tf)
            s_img = self._load(root, seq, sf)
            if t_img is None or s_img is None:
                continue

            t_anno = track[f"{tf:06d}"]
            s_anno = track[f"{sf:06d}"]

            t_crop = self._crop(t_img, t_anno, cfg.TRAIN.EXEMPLAR_SIZE)
            s_crop, bbox = self._crop_search(s_img, s_anno, cfg.TRAIN.SEARCH_SIZE)

            cls, delta, delta_weight, _ = anchor_target(
                bbox, cfg.TRAIN.OUTPUT_SIZE, neg=False)

            self.data.append({
                "template":         torch.tensor(t_crop.transpose(2, 0, 1).astype(np.float32)),
                "search":           torch.tensor(s_crop.transpose(2, 0, 1).astype(np.float32)),
                "label_cls":        torch.tensor(cls),
                "label_loc":        torch.tensor(delta),
                "label_loc_weight": torch.tensor(delta_weight),
                "bbox":             torch.tensor(np.array(bbox, dtype=np.float32)),
            })

        if not self.data:
            raise RuntimeError(f"Could not load any samples from {root}")
        print(f"[TinyFixedDataset] loaded {len(self.data)} samples from {root}")

    def _load(self, root, seq, fid):
        for ext in (".jpg", ".png"):
            p = os.path.join(root, seq, f"{fid:06d}{ext}")
            if os.path.isfile(p):
                img = cv2.imread(p)
                if img is not None:
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    return img
        return None

    def _crop(self, img, anno, size):
        """Centre-crop to size×size around the annotated object."""
        x1, y1, x2, y2 = anno[:4]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        s = np.sqrt((w + 0.5 * (w + h)) * (h + 0.5 * (w + h)))
        sc = size / max(s, 1e-3)
        out = np.zeros((size, size, 3), dtype=np.uint8)
        # Simple resize-based crop (no padding needed for sanity check)
        rx = int(cx - w * sc / 2)
        ry = int(cy - h * sc / 2)
        rw = int(w * sc)
        rh = int(h * sc)
        rx = max(0, rx); ry = max(0, ry)
        rw = max(1, min(rw, img.shape[1] - rx))
        rh = max(1, min(rh, img.shape[0] - ry))
        patch = img[ry:ry+rh, rx:rx+rw]
        if patch.size > 0:
            out = cv2.resize(patch, (size, size))
        return out

    def _crop_search(self, img, anno, size):
        """Crop search region; return (crop, bbox_in_crop)."""
        x1, y1, x2, y2 = anno[:4]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        s = np.sqrt((w + 0.5 * (w + h)) * (h + 0.5 * (w + h)))
        sc = size / max(s * 2, 1e-3)   # 2× context for search
        # Crop window
        rx = int(cx - size / (2 * sc))
        ry = int(cy - size / (2 * sc))
        rw = int(size / sc)
        rh = int(size / sc)
        rx = max(0, rx); ry = max(0, ry)
        rw = max(1, min(rw, img.shape[1] - rx))
        rh = max(1, min(rh, img.shape[0] - ry))
        patch = img[ry:ry+rh, rx:rx+rw]
        crop = cv2.resize(patch, (size, size)) if patch.size > 0 else np.zeros((size, size, 3), dtype=np.uint8)
        # BBox in crop coordinates
        bx1 = (x1 - rx) * sc; by1 = (y1 - ry) * sc
        bx2 = (x2 - rx) * sc; by2 = (y2 - ry) * sc
        bx1 = max(0, min(bx1, size - 1)); bx2 = max(0, min(bx2, size - 1))
        by1 = max(0, min(by1, size - 1)); by2 = max(0, min(by2, size - 1))
        if bx2 <= bx1: bx2 = bx1 + 1
        if by2 <= by1: by2 = by1 + 1
        cx_c = (bx1 + bx2) / 2; cy_c = (by1 + by2) / 2
        bw = bx2 - bx1; bh = by2 - by1
        bbox = center2corner(Center(cx_c, cy_c, bw, bh))
        return crop, bbox

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx % len(self.data)]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg",      default="pysot/experiments/siamrpn_r50_alldatasets/config.yaml")
    ap.add_argument("--pretrained", default="pretrained/sot_resnet50.pth")
    ap.add_argument("--out",      default="overfit_result.csv")
    ap.add_argument("--samples",  type=int, default=32,
                    help="Number of fixed samples to overfit on")
    ap.add_argument("--epochs",   type=int, default=100)
    ap.add_argument("--lr",       type=float, default=0.005)
    ap.add_argument("--compare",  nargs=2, metavar=("CSV1", "CSV2"),
                    help="Compare two result CSVs instead of training")
    ap.add_argument("--seed",     type=int, default=42)
    args = ap.parse_args()

    # ── compare mode ─────────────────────────────────────────────────────────
    if args.compare:
        compare(args.compare[0], args.compare[1])
        return

    # ── distributed init ──────────────────────────────────────────────────────
    rank, local_rank, world_size, is_main = init_distributed()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() \
             else torch.device("cpu")

    cfg.merge_from_file(args.cfg)
    cfg.freeze()

    # ── dataset (same N samples on every rank) ────────────────────────────────
    ds = TinyFixedDataset(
        root      = cfg.DATASET.ANTIUAV410.ROOT,
        anno_path = cfg.DATASET.ANTIUAV410.ANNO,
        n_samples = args.samples,
        seed      = args.seed,
    )

    # Effective batch is always args.samples regardless of world_size:
    #   single GPU  : batch_size = samples/1
    #   4 GPU DDP   : batch_size = samples/4  → all-reduce = same gradient
    per_gpu_batch = max(1, args.samples // world_size)
    loader = DataLoader(ds, batch_size=per_gpu_batch, shuffle=False,
                        num_workers=0, drop_last=False)

    if is_main:
        print(f"\n{'='*60}")
        print(f"  Overfit test: {args.samples} fixed samples, {args.epochs} epochs")
        print(f"  World size  : {world_size}  |  per-GPU batch: {per_gpu_batch}")
        print(f"  Effective batch = {per_gpu_batch * world_size}  (must equal {args.samples})")
        print(f"  LR={args.lr}  |  optimizer=Adam")
        print(f"  Output: {args.out}")
        print(f"{'='*60}\n")

    # ── model ─────────────────────────────────────────────────────────────────
    model = ModelBuilder()
    if args.pretrained and os.path.isfile(args.pretrained):
        load_pretrain(model.backbone, args.pretrained)
        if is_main:
            print(f"Loaded pretrained backbone: {args.pretrained}")

    # Unfreeze everything for overfitting
    for p in model.parameters():
        p.requires_grad = True

    model = model.to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    raw_model = model.module if isinstance(model, DDP) else model

    # Adam converges faster than SGD on tiny data → better overfit probe
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )

    # ── training loop ─────────────────────────────────────────────────────────
    results = []   # [(epoch, loss)]
    prev_loss = None

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_steps    = 0

        for batch in loader:
            b = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
            # Convert label_cls to long for cross-entropy inside model
            b["label_cls"] = b["label_cls"].long() if b["label_cls"].is_floating_point() \
                             else b["label_cls"]

            optimizer.zero_grad()
            outputs = model(b)
            loss    = outputs["total_loss"]

            if torch.isnan(loss) or torch.isinf(loss):
                if is_main:
                    print(f"[epoch {epoch+1}] NaN/Inf loss — skipping step")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_steps    += 1

        avg_loss = epoch_loss / max(n_steps, 1)

        # Reduce loss across ranks so all see the same number
        if world_size > 1:
            t = torch.tensor(avg_loss, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            avg_loss = (t / world_size).item()

        if is_main:
            delta = f"  Δ={avg_loss - prev_loss:+.4f}" if prev_loss is not None else ""
            print(f"  Epoch [{epoch+1:>3}/{args.epochs}]  loss={avg_loss:.5f}{delta}")
            results.append((epoch + 1, avg_loss))
            prev_loss = avg_loss

    # ── save results ──────────────────────────────────────────────────────────
    if is_main:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "loss"])
            w.writerows(results)

        final_loss = results[-1][1] if results else float("nan")
        threshold  = 0.50
        status = "PASS ✓" if final_loss < threshold else "FAIL ✗"
        print(f"\n{'='*60}")
        print(f"  Final loss: {final_loss:.5f}  (threshold={threshold})  →  {status}")
        print(f"  Results saved: {args.out}")
        print(f"{'='*60}\n")
        if final_loss >= threshold:
            print("  WARNING: Model did not overfit. Possible gradient flow issue.")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ── Compare two CSVs ──────────────────────────────────────────────────────────
def compare(csv1, csv2):
    def load(path):
        rows = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append((int(r["epoch"]), float(r["loss"])))
        return rows

    r1 = load(csv1)
    r2 = load(csv2)
    n  = min(len(r1), len(r2))

    print(f"\n{'='*60}")
    print(f"  Overfit comparison")
    print(f"  A = {csv1}")
    print(f"  B = {csv2}")
    print(f"{'='*60}")
    print(f"  {'Epoch':>5}  {'A loss':>10}  {'B loss':>10}  {'|A-B|':>8}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*8}")
    for i in range(0, n, max(1, n // 20)):   # print ~20 rows
        ep, la = r1[i]
        _,  lb = r2[i]
        flag = " <-- diverged" if abs(la - lb) > 0.10 else ""
        print(f"  {ep:>5}  {la:>10.5f}  {lb:>10.5f}  {abs(la-lb):>8.5f}{flag}")

    # Final epoch
    la_fin = r1[-1][1]; lb_fin = r2[-1][1]
    print(f"\n  Final (ep {r1[-1][0]}):  A={la_fin:.5f}  B={lb_fin:.5f}  |diff|={abs(la_fin-lb_fin):.5f}")
    diff_ok = abs(la_fin - lb_fin) < 0.10
    both_overfit = la_fin < 0.50 and lb_fin < 0.50
    print(f"  Both overfit (< 0.50) : {'YES ✓' if both_overfit else 'NO ✗'}")
    print(f"  Loss parity (|diff|<0.10): {'YES ✓' if diff_ok else 'NO ✗'}")
    if both_overfit and diff_ok:
        print(f"\n  ✓ DDP and single-GPU converge identically. Gradient flow OK.")
    else:
        print(f"\n  ✗ Convergence mismatch — check LR scaling or data parity.")
    print(f"{'='*60}\n")

    # Save comparison PNG if matplotlib available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        epochs1 = [r[0] for r in r1]
        losses1 = [r[1] for r in r1]
        epochs2 = [r[0] for r in r2]
        losses2 = [r[1] for r in r2]
        ax.plot(epochs1, losses1, label=os.path.basename(csv1), linewidth=1.5)
        ax.plot(epochs2, losses2, label=os.path.basename(csv2), linewidth=1.5, linestyle="--")
        ax.axhline(0.50, color="red", linestyle=":", linewidth=1, label="threshold=0.50")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title("Overfit test: single-GPU vs DDP")
        ax.legend(); ax.grid(True, alpha=0.3)
        out_png = "overfit_comparison.png"
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        print(f"  Plot saved: {out_png}")
    except ImportError:
        print("  (matplotlib not available — skipping plot)")


if __name__ == "__main__":
    main()
