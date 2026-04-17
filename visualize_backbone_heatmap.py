#!/usr/bin/env python3
"""
visualize_backbone_heatmap.py

Extracts layer4 feature activations from the ResNet-50 backbone and overlays
them as heatmaps on ir_crop.mp4 frames.

Runs the backbone in two modes side-by-side:
  Left  : original sot_resnet50.pth  (ImageNet-pretrained, RGB domain)
  Right : ir_backbone.pth            (fine-tuned on IR detection data)

The heatmap shows the mean activation magnitude across all 2048 channels of
layer4 (the deepest feature map).  High activation = backbone considers the
region "interesting".  After IR fine-tuning, we expect higher and tighter
activation over the UAV target and less diffuse background response.

Output
------
  docs/heatmap_comparison_f{NNNN}.jpg  — side-by-side comparison frames
  ir_crop_heatmap.mp4                  — full annotated video (ir_backbone only)

Usage
-----
  python visualize_backbone_heatmap.py \\
      --video    ir_crop.mp4 \\
      --original pretrained/sot_resnet50.pth \\
      --finetuned pretrained/ir_backbone.pth \\
      --rotate  -90 \\
      --out     ir_crop_heatmap.mp4
"""
import argparse, os, sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYSOT_DIR  = os.path.join(SCRIPT_DIR, 'pysot')
if os.path.isdir(PYSOT_DIR):
    sys.path.insert(0, PYSOT_DIR)


# ── backbone loader ───────────────────────────────────────────────────────────

class BackboneExtractor(nn.Module):
    """ResNet-50 up to layer4, returns spatial feature map (B, 2048, H, W)."""
    def __init__(self, weights_path: str):
        super().__init__()
        base = tvm.resnet50(weights=None)
        # Load weights (shape-compatible only, like train_ir_backbone.py)
        if weights_path and Path(weights_path).exists():
            sd       = torch.load(weights_path, map_location='cpu')
            model_sd = base.state_dict()
            compat   = {k: v for k, v in sd.items()
                        if k in model_sd and model_sd[k].shape == v.shape}
            model_sd.update(compat)
            base.load_state_dict(model_sd)
            print(f"  Loaded {Path(weights_path).name}: {len(compat)} layers")
        else:
            print(f"  WARNING: {weights_path} not found — random init")

        # Keep stem + layer1-4, drop avgpool + fc
        self.stem   = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x          # (1, 2048, H/32, W/32)


def load_backbone(path: str, device) -> BackboneExtractor:
    m = BackboneExtractor(path).to(device)
    m.eval()
    return m


# ── helpers ───────────────────────────────────────────────────────────────────

def rotate_frame(frame, deg):
    if deg == -90: return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if deg ==  90: return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180: return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def frame_to_tensor(frame_bgr, device):
    """BGR uint8 → normalised RGB float32 tensor (1, 3, H, W)."""
    rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb  = (rgb - mean) / std
    t    = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return t


@torch.no_grad()
def get_heatmap(backbone: BackboneExtractor, frame_bgr, device):
    """
    Returns a heatmap (H×W float32 in [0,1]) for the given frame.
    Heatmap = mean activation across 2048 channels of layer4,
    resized to original frame dimensions.
    """
    h, w   = frame_bgr.shape[:2]
    t      = frame_to_tensor(frame_bgr, device)
    feats  = backbone(t)                                # (1, 2048, fH, fW)
    hmap   = feats[0].mean(dim=0).cpu().numpy()         # (fH, fW)
    # Normalise to [0, 1]
    hmap   = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
    hmap   = cv2.resize(hmap, (w, h), interpolation=cv2.INTER_CUBIC)
    hmap   = np.clip(hmap, 0, 1)
    return hmap


def overlay_heatmap(frame_bgr, hmap, alpha=0.55):
    """Blend jet-colourmap heatmap onto frame."""
    hmap_u8  = (hmap * 255).astype(np.uint8)
    hmap_col = cv2.applyColorMap(hmap_u8, cv2.COLORMAP_JET)
    blended  = cv2.addWeighted(hmap_col, alpha, frame_bgr, 1 - alpha, 0)
    return blended


def label_bar(frame, text, bg=(30, 30, 30), fg=(220, 220, 220)):
    h, w = frame.shape[:2]
    bar  = np.full((24, w, 3), bg, dtype=np.uint8)
    cv2.putText(bar, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, fg, 1, cv2.LINE_AA)
    return np.vstack([bar, frame])


# ── main ──────────────────────────────────────────────────────────────────────

def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    print("\nLoading backbones...")
    bb_orig = load_backbone(args.original,  device)
    bb_ft   = load_backbone(args.finetuned, device)

    cap   = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = raw_h if abs(args.rotate) == 90 else raw_w
    out_h = raw_w if abs(args.rotate) == 90 else raw_h
    print(f"Video  : {args.video}  {raw_w}×{raw_h}  {fps:.1f}fps  {total} frames")
    print(f"Rotate : {args.rotate:+d}°  output {out_w}×{out_h}")

    # Output video (fine-tuned backbone heatmap only)
    writer = cv2.VideoWriter(
        args.out, cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (out_w * 2, out_h + 24)   # side-by-side + label bar
    )

    # Comparison frames
    docs_dir = Path(os.path.dirname(args.video)) / 'docs'
    docs_dir.mkdir(exist_ok=True)
    key_frames = {100, 500, 1000, 2000, 3000, 4000, 5000, 5500}

    print(f"\nProcessing {total} frames...")
    for fi in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        frr = rotate_frame(frame, args.rotate)

        hmap_orig = get_heatmap(bb_orig, frr, device)
        hmap_ft   = get_heatmap(bb_ft,   frr, device)

        vis_orig = overlay_heatmap(frr.copy(), hmap_orig)
        vis_ft   = overlay_heatmap(frr.copy(), hmap_ft)

        # HUD
        cv2.putText(vis_orig, f'f:{fi:5d}', (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(vis_ft,   f'f:{fi:5d}', (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # Label bars
        vis_orig = label_bar(vis_orig, 'Original sot_resnet50 (ImageNet/RGB)')
        vis_ft   = label_bar(vis_ft,   'IR fine-tuned backbone')

        combined = np.hstack([vis_orig, vis_ft])
        writer.write(combined)

        if fi in key_frames:
            out_p = docs_dir / f'heatmap_comparison_f{fi:04d}.jpg'
            cv2.imwrite(str(out_p), combined, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"  Saved comparison frame: {out_p.name}")

        if fi % 500 == 0:
            print(f"  frame {fi:5d}/{total}")

    cap.release()
    writer.release()
    print(f"\nSaved : {args.out}")
    print(f"Frames: {total}")
    print(f"Comparison images: docs/heatmap_comparison_f*.jpg")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--video',     default='ir_crop.mp4')
    ap.add_argument('--original',  default='pretrained/sot_resnet50.pth')
    ap.add_argument('--finetuned', default='pretrained/ir_backbone.pth')
    ap.add_argument('--rotate',    type=int, default=-90, choices=[0, 90, -90, 180])
    ap.add_argument('--out',       default='ir_crop_heatmap.mp4')
    run(ap.parse_args())
