#!/usr/bin/env python3
"""
run_pretrained_backbone_inference.py
Run SiamRPN++ inference with ONLY the backbone pretrained on ImageNet.
Neck and RPN head are randomly initialized — this shows the pre-training baseline.
"""
import argparse, os, sys, csv
import cv2
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYSOT_DIR  = os.path.join(SCRIPT_DIR, 'pysot')
if os.path.isdir(PYSOT_DIR):
    sys.path.insert(0, PYSOT_DIR)

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.siamrpn_tracker import SiamRPNTracker


def load_backbone_only(ckpt_path, model):
    """Load backbone-only checkpoint into model.backbone with strict=False."""
    raw = torch.load(ckpt_path, map_location='cpu')
    # raw keys: conv1.weight, bn1.weight, layer1.0.conv1.weight, ...
    # model.backbone expects the same keys directly
    missing, unexpected = model.backbone.load_state_dict(raw, strict=False)
    print(f"  Backbone load: {len(missing)} missing, {len(unexpected)} unexpected keys")
    if missing:
        print(f"  Missing (first 5): {missing[:5]}")
    return model


def build_tracker_pretrained_bb(cfg_path, bb_ckpt):
    cfg.merge_from_file(cfg_path)
    cfg.CUDA = torch.cuda.is_available()
    device = 'cuda' if cfg.CUDA else 'cpu'

    model = ModelBuilder()
    model = load_backbone_only(bb_ckpt, model)
    model.eval()
    if cfg.CUDA:
        model = model.cuda()

    tracker = SiamRPNTracker(model)
    print(f"Backbone ckpt : {bb_ckpt}")
    print(f"Device        : {device}")
    print(f"  NOTE: Neck and RPN head are RANDOMLY INITIALIZED")
    return tracker


def rotate_frame(frame, rotate):
    if rotate == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == -90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotate == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def transform_box(x1, y1, x2, y2, rotate, raw_w, raw_h):
    if rotate == -90:
        nx1 = y1
        ny1 = raw_w - 1 - x2
        nx2 = y2
        ny2 = raw_w - 1 - x1
    elif rotate == 90:
        nx1 = raw_h - 1 - y2
        ny1 = x1
        nx2 = raw_h - 1 - y1
        ny2 = x2
    elif rotate == 180:
        nx1 = raw_w - 1 - x2
        ny1 = raw_h - 1 - y2
        nx2 = raw_w - 1 - x1
        ny2 = raw_h - 1 - y1
    else:
        return x1, y1, x2, y2
    return nx1, ny1, nx2, ny2


def draw_box(frame, bbox, score, color=(0, 230, 255), thickness=2):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    label = f"score:{score:.3f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    lx, ly = max(x1, 0), max(y1 - 6, th + 4)
    cv2.rectangle(frame, (lx, ly - th - 4), (lx + tw + 4, ly + 2), (0,0,0), -1)
    cv2.putText(frame, label, (lx + 2, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return frame


def run(args):
    tracker = build_tracker_pretrained_bb(args.cfg, args.bb_ckpt)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {args.video}")

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rotate = args.rotate

    if abs(rotate) == 90:
        out_w, out_h = height, width
    else:
        out_w, out_h = width, height

    print(f"Video         : {args.video}  ({width}×{height}  {fps:.1f}fps  {total} frames)")
    print(f"Rotation      : {rotate:+d}°  → output {out_w}×{out_h}")

    base      = os.path.splitext(os.path.basename(args.video))[0]
    out_video = args.out or f"{base}_pretrained_bb.mp4"
    out_csv   = out_video.replace('.mp4', '_results.csv')

    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_h))
    csv_rows = []

    x1, y1, x2, y2 = args.init_box
    frame_idx = 0
    init_done = False

    while True:
        ret, raw = cap.read()
        if not ret:
            break
        frame = rotate_frame(raw, rotate)

        if frame_idx == args.start_frame:
            if rotate:
                ix1, iy1, ix2, iy2 = transform_box(x1, y1, x2, y2, rotate, width, height)
            else:
                ix1, iy1, ix2, iy2 = x1, y1, x2, y2
            tracker.init(frame, [ix1, iy1, ix2 - ix1, iy2 - iy1])
            init_done = True
            print(f"  Initialized at frame {frame_idx}:  box [{ix1},{iy1},{ix2},{iy2}]")
            bx1, bx2, by1, by2 = ix1, ix2, iy1, iy2
            score = 1.0
        elif init_done:
            out = tracker.track(frame)
            bx, by, bw, bh = out['bbox']
            bx1, by1 = int(bx), int(by)
            bx2, by2 = int(bx + bw), int(by + bh)
            score = out['best_score']

        if init_done:
            draw_box(frame, [bx1, by1, bx2, by2], score)
            csv_rows.append([frame_idx, bx1, by1, bx2, by2, score])

        writer.write(frame)
        if frame_idx % 50 == 0:
            print(f"  frame {frame_idx:4d}/{total}  score={score:.4f}  box=[{bx1},{by1},{bx2},{by2}]")
        frame_idx += 1

    cap.release()
    writer.release()

    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame','x1','y1','x2','y2','score'])
        w.writerows(csv_rows)

    print(f"\nSaved video → {out_video}")
    print(f"Saved CSV   → {out_csv}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg',      required=True,  help='PySOT config.yaml')
    ap.add_argument('--bb_ckpt',  required=True,  help='Backbone-only .pth (e.g. sot_resnet50.pth)')
    ap.add_argument('--video',    required=True)
    ap.add_argument('--init_box', type=int, nargs=4, metavar=('X1','Y1','X2','Y2'), required=True)
    ap.add_argument('--rotate',   type=int, default=0, choices=[0, 90, -90, 180])
    ap.add_argument('--out',      default=None)
    ap.add_argument('--start_frame', type=int, default=0)
    ap.add_argument('--score_thresh', type=float, default=0.20)
    args = ap.parse_args()
    run(args)
