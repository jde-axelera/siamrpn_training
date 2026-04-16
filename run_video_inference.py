#!/usr/bin/env python3
"""
run_video_inference.py  —  SiamRPN++ tracking inference on a video file
=======================================================================
Uses PySOT's SiamRPNTracker (PyTorch, NOT ONNX) with the exact same
preprocessing and decoding as the training pipeline.

Usage
-----
  # On AWS — with the trained IR model:
  python run_video_inference.py \
      --cfg    pysot/experiments/siamrpn_r50_alldatasets/config.yaml \
      --ckpt   pysot/snapshot/all_datasets/best_model_last_train_loss_0.26.pth \
      --video  ir_crop.mp4 \
      --init_box 339 148 391 232

  # Optional flags:
  #   --out          out.mp4            output video path (default: <video>_tracked.mp4)
  #   --start_frame  0                  which frame to init on (0-based)
  #   --show                            display frames in real-time (needs X11/display)
  #   --score_thresh 0.20               minimum score to draw box (default 0.20)
"""
import argparse
import os
import sys
import time
import csv

import cv2
import numpy as np
import torch

# ── locate pysot ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYSOT_DIR  = os.path.join(SCRIPT_DIR, 'pysot')
if os.path.isdir(PYSOT_DIR):
    sys.path.insert(0, PYSOT_DIR)

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.siamrpn_tracker import SiamRPNTracker


# ── helpers ───────────────────────────────────────────────────────────────────
def load_checkpoint(ckpt_path):
    sd = torch.load(ckpt_path, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    return sd


def build_tracker(cfg_path, ckpt_path):
    cfg.merge_from_file(cfg_path)
    cfg.CUDA = torch.cuda.is_available()
    device = 'cuda' if cfg.CUDA else 'cpu'

    model = ModelBuilder()
    model.load_state_dict(load_checkpoint(ckpt_path))
    model.eval()
    if cfg.CUDA:
        model = model.cuda()

    tracker = SiamRPNTracker(model)
    print(f"Model loaded  : {ckpt_path}")
    print(f"Device        : {device}")
    return tracker


def rotate_frame(frame, rotate):
    """Rotate frame by 90 CW (+90) or 90 CCW (-90)."""
    if rotate == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == -90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotate == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def transform_box(x1, y1, x2, y2, rotate, raw_w, raw_h):
    """Transform init box coords from raw frame space to rotated frame space."""
    if rotate == 90:
        # CW: new(x,y) = (raw_h-1-raw_y, raw_x)  →  new_w=raw_h, new_h=raw_w
        nx1 = raw_h - 1 - y2
        ny1 = x1
        nx2 = raw_h - 1 - y1
        ny2 = x2
    elif rotate == -90:
        # CCW: new(x,y) = (raw_y, raw_w-1-raw_x)  →  new_w=raw_h, new_h=raw_w
        nx1 = y1
        ny1 = raw_w - 1 - x2
        nx2 = y2
        ny2 = raw_w - 1 - x1
    elif rotate == 180:
        nx1 = raw_w - 1 - x2
        ny1 = raw_h - 1 - y2
        nx2 = raw_w - 1 - x1
        ny2 = raw_h - 1 - y1
    else:
        return x1, y1, x2, y2
    return nx1, ny1, nx2, ny2


def draw_box(frame, bbox, score, state_color=(0, 230, 255), thickness=2):
    """Draw tracking box and score on frame.  bbox = [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), state_color, thickness)
    label = f"score: {score:.3f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    lx = max(x1, 0)
    ly = max(y1 - 6, th + 4)
    cv2.rectangle(frame, (lx, ly - th - 4), (lx + tw + 4, ly + 2), (0, 0, 0), -1)
    cv2.putText(frame, label, (lx + 2, ly), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, state_color, 1, cv2.LINE_AA)
    return frame


# ── main ──────────────────────────────────────────────────────────────────────
def run(args):
    tracker = build_tracker(args.cfg, args.ckpt)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {args.video}")

    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # ── rotation setup ────────────────────────────────────────────────────────
    rotate = args.rotate
    if rotate:
        out_w = height if abs(rotate) == 90 else width
        out_h = width  if abs(rotate) == 90 else height
        print(f"Video         : {args.video}")
        print(f"Size          : {width}×{height}  FPS={fps:.2f}  frames={total}")
        print(f"Rotation      : {rotate:+d}°  →  output {out_w}×{out_h}")
    else:
        out_w, out_h = width, height
        print(f"Video         : {args.video}")
        print(f"Size          : {width}×{height}  FPS={fps:.2f}  frames={total}")

    # ── output paths ──────────────────────────────────────────────────────────
    base = os.path.splitext(os.path.basename(args.video))[0]
    out_video = args.out or f"{base}_tracked_pytorch.mp4"
    out_csv   = out_video.replace('.mp4', '_results.csv')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video, fourcc, fps, (out_w, out_h))

    # ── seek to start frame ───────────────────────────────────────────────────
    start = args.start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ret, frame = cap.read()
    if not ret:
        sys.exit(f"Could not read start frame {start}")
    if rotate:
        frame = rotate_frame(frame, rotate)

    # ── initialise tracker ────────────────────────────────────────────────────
    if args.init_box:
        x1, y1, x2, y2 = args.init_box
        if rotate:
            x1, y1, x2, y2 = transform_box(x1, y1, x2, y2, rotate, width, height)
            print(f"Init box (raw): {args.init_box}")
    else:
        roi = cv2.selectROI("Select target — press ENTER to confirm", frame, False, False)
        cv2.destroyAllWindows()
        x1, y1 = int(roi[0]), int(roi[1])
        x2, y2 = x1 + int(roi[2]), y1 + int(roi[3])

    print(f"Init box      : [{x1}, {y1}, {x2}, {y2}]  (x1,y1,x2,y2 in tracked frame)")

    # SiamRPNTracker.init() expects (img_bgr, [x, y, w, h])
    bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
    tracker.init(frame, bbox_xywh)

    # Draw init box on the init frame and write it
    init_frame = frame.copy()
    draw_box(init_frame, [x1, y1, x2, y2], 1.0, state_color=(0, 255, 0))
    writer.write(init_frame)

    # ── tracking loop ─────────────────────────────────────────────────────────
    results = []
    frame_idx = start + 1
    t_total_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if rotate:
            frame = rotate_frame(frame, rotate)

        t0 = time.perf_counter()
        out = tracker.track(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        bx, by, bw, bh = out['bbox']
        score = float(out['best_score'])
        bbox_xyxy = [bx, by, bx + bw, by + bh]

        results.append({
            'frame':  frame_idx,
            'x1':     round(bx,  2),
            'y1':     round(by,  2),
            'w':      round(bw,  2),
            'h':      round(bh,  2),
            'score':  round(score, 4),
            'ms':     round(elapsed_ms, 1),
        })

        # Choose colour by score threshold
        color = (0, 230, 255) if score >= args.score_thresh else (80, 80, 220)
        vis = frame.copy()
        draw_box(vis, bbox_xyxy, score, state_color=color)
        writer.write(vis)

        if args.show:
            cv2.imshow('SiamRPN++ Tracking', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quit by user.")
                break

        if frame_idx % 50 == 0:
            fps_live = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0
            print(f"  frame {frame_idx:5d}/{total}  score={score:.3f}  {fps_live:.1f} fps")

        frame_idx += 1

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    # ── save CSV ──────────────────────────────────────────────────────────────
    with open(out_csv, 'w', newline='') as f:
        writer_csv = csv.DictWriter(f, fieldnames=['frame', 'x1', 'y1', 'w', 'h', 'score', 'ms'])
        writer_csv.writeheader()
        writer_csv.writerows(results)

    # ── summary ───────────────────────────────────────────────────────────────
    total_sec = time.perf_counter() - t_total_start
    tracked = len(results)
    if tracked:
        scores = [r['score'] for r in results]
        mean_ms = sum(r['ms'] for r in results) / tracked
        print()
        print(f"Done. {tracked} frames tracked in {total_sec:.1f}s  ({tracked/total_sec:.1f} fps avg)")
        print(f"Score  : mean={sum(scores)/tracked:.3f}  min={min(scores):.3f}  max={max(scores):.3f}")
        print(f"Latency: mean={mean_ms:.1f}ms/frame")
        print(f"Output video : {out_video}")
        print(f"Output CSV   : {out_csv}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='SiamRPN++ video inference (PyTorch)')
    ap.add_argument('--cfg',         required=True,
                    help='Path to config.yaml (e.g. pysot/experiments/.../config.yaml)')
    ap.add_argument('--ckpt',        required=True,
                    help='Path to .pth checkpoint')
    ap.add_argument('--video',       required=True,
                    help='Input video file')
    ap.add_argument('--init_box',    type=int, nargs=4, metavar=('X1','Y1','X2','Y2'),
                    default=None,
                    help='Initial bounding box in x1 y1 x2 y2 format (skip GUI if given)')
    ap.add_argument('--out',         default=None,
                    help='Output video path (default: <video>_tracked_pytorch.mp4)')
    ap.add_argument('--start_frame', type=int, default=0,
                    help='Frame index to initialise tracker on (0-based, default: 0)')
    ap.add_argument('--score_thresh',type=float, default=0.20,
                    help='Score below this draws red box (default: 0.20)')
    ap.add_argument('--rotate',      type=int, default=0, choices=[0, 90, -90, 180],
                    help='Rotate each frame before tracking: 90=CW, -90=CCW (default: 0)')
    ap.add_argument('--show',        action='store_true',
                    help='Display frames in real-time (requires display)')
    args = ap.parse_args()
    run(args)
