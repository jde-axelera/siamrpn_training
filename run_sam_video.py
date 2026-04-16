#!/usr/bin/env python3
"""
run_sam_video.py
Headless SAMv2/v2.1 video segmentation using muggled_sam.

Runs SAM's video predictor on ir_crop.mp4 with a -90° rotation,
using the known init bounding box as the first-frame prompt.
Outputs an annotated MP4 with the segmentation mask overlaid.

Usage:
  python run_sam_video.py \
      --model  /data/muggled_sam/model_weights/sam2.1_hiera_large.pt \
      --video  /data/siamrpn_training/ir_crop.mp4 \
      --out    /data/siamrpn_training/ir_crop_sam_seg.mp4 \
      --rotate -90 \
      --init_box 339 148 391 232

The init_box is in the original (pre-rotation) frame coordinates [x1 y1 x2 y2].
"""
import argparse, os, sys
from collections import deque
from time import perf_counter

import cv2
import numpy as np
import torch

# Allow running from anywhere
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MUGGLED_DIR = '/data/muggled_sam'
if os.path.isdir(MUGGLED_DIR):
    sys.path.insert(0, MUGGLED_DIR)
elif os.path.isdir(os.path.join(SCRIPT_DIR, 'muggled_sam')):
    sys.path.insert(0, SCRIPT_DIR)

from muggled_sam.make_sam import make_sam_from_state_dict


# ── helpers ───────────────────────────────────────────────────────────────────
def rotate_frame(frame, deg):
    if deg == -90: return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if deg ==  90: return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180: return cv2.rotate(frame, cv2.ROTATE_180)
    return frame

def transform_box(x1, y1, x2, y2, rotate, raw_w, raw_h):
    """Transform bbox from original frame coords to rotated frame coords."""
    if rotate == -90:
        return y1, raw_w - 1 - x2, y2, raw_w - 1 - x1
    if rotate ==  90:
        return raw_h - 1 - y2, x1, raw_h - 1 - y1, x2
    if rotate == 180:
        return raw_w - 1 - x2, raw_h - 1 - y2, raw_w - 1 - x1, raw_h - 1 - y1
    return x1, y1, x2, y2

def overlay_mask(frame, mask_uint8, colour=(0, 200, 100), alpha=0.45):
    """Blend a binary mask (H×W uint8 0/255) as a coloured overlay."""
    overlay = frame.copy()
    overlay[mask_uint8 > 0] = colour
    out = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    # Draw mask contour
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, colour, 2)
    return out

def draw_box(frame, x1, y1, x2, y2, label='', colour=(0, 230, 255)):
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 0, 0), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return frame


# ── main ─────────────────────────────────────────────────────────────────────
def run(args):
    rotate  = args.rotate
    x1o, y1o, x2o, y2o = args.init_box   # original frame coords

    # ── load model ─────────────────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype  = torch.bfloat16 if device == 'cuda' else torch.float32
    print(f"Device : {device}  dtype={dtype}")
    print(f"Model  : {args.model}")
    model_config, sammodel = make_sam_from_state_dict(args.model)
    assert sammodel.name in ('samv2', 'samv3'), \
        f"Only SAMv2/v3 support video tracking, got: {sammodel.name}"
    sammodel.to(device=device, dtype=dtype)
    print(f"Loaded : {sammodel.name}")

    # ── open video ──────────────────────────────────────────────────────────
    cap    = cv2.VideoCapture(args.video)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    raw_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w  = raw_h if abs(rotate) == 90 else raw_w
    out_h  = raw_w if abs(rotate) == 90 else raw_h
    print(f"Video  : {args.video}  {raw_w}×{raw_h}  {fps:.1f}fps  {total} frames")
    print(f"Rotate : {rotate:+d}°  output {out_w}×{out_h}")

    # ── transform init box to rotated coordinates ───────────────────────────
    ix1, iy1, ix2, iy2 = transform_box(x1o, y1o, x2o, y2o, rotate, raw_w, raw_h)
    # Normalise to [0,1] in rotated frame
    box_x1n = ix1 / out_w
    box_y1n = iy1 / out_h
    box_x2n = ix2 / out_w
    box_y2n = iy2 / out_h
    boxes_norm = [[(box_x1n, box_y1n), (box_x2n, box_y2n)]]
    print(f"Init box (rotated px): [{ix1},{iy1},{ix2},{iy2}]")
    print(f"Init box (normalised): [({box_x1n:.3f},{box_y1n:.3f}),({box_x2n:.3f},{box_y2n:.3f})]")

    # ── encode first frame and initialise tracker ───────────────────────────
    ok, frame0 = cap.read()
    assert ok, "Could not read first frame"
    frame0_rot = rotate_frame(frame0, rotate)

    imgenc_cfg = {"max_side_length": None, "use_square_sizing": True}
    print("Encoding frame 0 and initialising mask...")
    t0 = perf_counter()
    init_enc, _, _ = sammodel.encode_image(frame0_rot, **imgenc_cfg)
    init_mask, init_mem, init_ptr = sammodel.initialize_video_masking(
        init_enc, boxes_norm, fg_xy_norm_list=[], bg_xy_norm_list=[],
        mask_index_select=None
    )
    print(f"Init done in {(perf_counter()-t0)*1000:.0f} ms")

    # Memory deques — same sizes as in the simple_examples
    prompt_mems = deque([init_mem])
    prompt_ptrs = deque([init_ptr])
    prev_mems   = deque([], maxlen=6)
    prev_ptrs   = deque([], maxlen=15)

    # ── set up output video ─────────────────────────────────────────────────
    out_path = args.out
    writer   = cv2.VideoWriter(out_path,
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps, (out_w, out_h))

    # Draw first frame with init box
    disp0 = frame0_rot.copy()
    disp0 = draw_box(disp0, ix1, iy1, ix2, iy2, 'init box')
    # Show init mask on frame 0
    init_mask_up = torch.nn.functional.interpolate(
        init_mask[:, 0:1, :, :], size=(out_h, out_w),
        mode='bilinear', align_corners=False)
    init_mask_np = ((init_mask_up > 0.0).byte() * 255).cpu().numpy().squeeze()
    disp0 = overlay_mask(disp0, init_mask_np)
    cv2.putText(disp0, 'frame:0  SAM init', (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    writer.write(disp0)

    # ── process remaining frames ────────────────────────────────────────────
    is_cuda = (device == 'cuda')
    encode_times = []
    track_times  = []
    obj_scores   = []

    for fi in range(1, total):
        ok, frame = cap.read()
        if not ok:
            break
        frr = rotate_frame(frame, rotate)

        t1 = perf_counter()
        enc, _, _ = sammodel.encode_image(frr, **imgenc_cfg)
        if is_cuda:
            torch.cuda.synchronize()
        t2 = perf_counter()

        obj_score, best_idx, mask_preds, mem_enc, obj_ptr = sammodel.step_video_masking(
            enc, prompt_mems, prompt_ptrs, prev_mems, prev_ptrs
        )
        if is_cuda:
            torch.cuda.synchronize()
        t3 = perf_counter()

        obj_score = float(obj_score)   # tensor → python float
        encode_times.append(t2 - t1)
        track_times.append(t3 - t2)
        obj_scores.append(obj_score)

        # Update memory
        if obj_score >= 0:
            prev_mems.appendleft(mem_enc)
            prev_ptrs.appendleft(obj_ptr)

        # Upsample mask to frame size
        mask_up = torch.nn.functional.interpolate(
            mask_preds[:, best_idx:best_idx+1, :, :],
            size=(out_h, out_w),
            mode='bilinear', align_corners=False,
        )
        mask_np = ((mask_up > 0.0).byte() * 255).cpu().numpy().squeeze()

        # Render
        disp = overlay_mask(frr, mask_np)

        # Bounding box from mask
        if mask_np.any():
            ys, xs = np.where(mask_np > 0)
            bx1, by1, bx2, by2 = xs.min(), ys.min(), xs.max(), ys.max()
            draw_box(disp, int(bx1), int(by1), int(bx2), int(by2),
                     f's:{obj_score:.2f}', (0, 230, 255))

        hud = f'f:{fi:5d}  obj_score:{obj_score:+.3f}  enc:{(t2-t1)*1000:.0f}ms  trk:{(t3-t2)*1000:.0f}ms'
        cv2.putText(disp, hud, (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        writer.write(disp)

        if fi % 500 == 0:
            print(f"  frame {fi:5d}/{total}  obj_score={obj_score:+.3f}  "
                  f"enc={np.mean(encode_times[-100:])*1000:.0f}ms  "
                  f"trk={np.mean(track_times[-100:])*1000:.0f}ms")

    cap.release()
    writer.release()

    # Summary
    print(f"\nSaved  : {out_path}")
    print(f"Frames : {len(obj_scores)+1}")
    print(f"Obj score — mean: {np.mean(obj_scores):.3f}  "
          f"min: {np.min(obj_scores):.3f}  "
          f"pos(≥0): {sum(s>=0 for s in obj_scores)/len(obj_scores)*100:.1f}%")
    print(f"Encode time — mean: {np.mean(encode_times)*1000:.0f} ms/frame")
    print(f"Track time  — mean: {np.mean(track_times)*1000:.0f} ms/frame")
    total_ms = (sum(encode_times) + sum(track_times)) * 1000
    print(f"Total processing : {total_ms/1000:.0f} s  ({total/(total_ms/1000):.1f} fps)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',    required=True, help='path to SAMv2/v3 .pt weights')
    ap.add_argument('--video',    required=True, help='input video path')
    ap.add_argument('--out',      required=True, help='output annotated video path')
    ap.add_argument('--rotate',   type=int, default=0, choices=[0, 90, -90, 180])
    ap.add_argument('--init_box', type=int, nargs=4,
                    metavar=('X1', 'Y1', 'X2', 'Y2'),
                    required=True,
                    help='init bounding box in ORIGINAL (pre-rotation) frame coords')
    run(ap.parse_args())
