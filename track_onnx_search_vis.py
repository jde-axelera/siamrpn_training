#!/usr/bin/env python3
"""
track_onnx_search_vis.py
Run SiamRPN++ ONNX tracker on a video and render two overlays per frame:
  • Tracked target bbox  (yellow, solid)
  • Search area bbox     (cyan, dashed)  — the s_x × s_x region fed to the tracker

Usage:
  python track_onnx_search_vis.py \
      --enc   exported/template_encoder.onnx \
      --trk   exported/tracker.onnx \
      --video ir_crop.mp4 \
      --init_box 339 148 391 232 \
      --out   ir_crop_onnx_search_vis.mp4
"""
import argparse, csv, os, sys
import numpy as np
import cv2

# ── SiamRPN++ constants (must match training) ─────────────────────────────────
EXEMPLAR_SIZE  = 127
SEARCH_SIZE    = 255
OUTPUT_SIZE    = 25
STRIDE         = 8
CONTEXT_AMOUNT = 0.5
ANCHOR_RATIOS  = [0.33, 0.5, 1.0, 2.0, 3.0]
ANCHOR_SCALES  = [8]
PENALTY_K           = 0.04
WINDOW_INF          = 0.40
LR                  = 0.30
SCORE_THRESH        = 0.20
SIZE_UPDATE_THRESH  = 0.50   # only update w/h when score >= this
MAX_SCALE_PER_FRAME = 1.05   # bbox can grow/shrink at most 5% per frame

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

COLOUR_PRED   = (0, 220, 255)    # yellow  — tracked bbox
COLOUR_SEARCH = (255, 180, 0)    # cyan    — search area
COLOUR_INIT   = (50, 255, 50)    # green   — init box (frame 0 only)


# ── anchor + cosine window ────────────────────────────────────────────────────
def _generate_anchors():
    """
    Exact match of PySOT's SiamRPNTracker.generate_anchor.

    cx/cy are displacements from the centre of the search image (NOT absolute
    pixel positions).  The centre anchor (i=j=12) has cx=cy=0.

    Anchor sizes use PySOT's floor convention:
      ws_pre = int(sqrt(stride² / ratio))   ← floor, not round
      hs_pre = int(ws_pre * ratio)           ← floor, uses pre-scale ws
      w = ws_pre * scale,  h = hs_pre * scale
    """
    ori = -(OUTPUT_SIZE // 2) * STRIDE        # -96  (displacement of j=0 from centre)
    anchors = []
    for ratio in ANCHOR_RATIOS:
        for scale in ANCHOR_SCALES:
            ws_pre = int(np.sqrt(STRIDE ** 2 / ratio))   # floor
            hs_pre = int(ws_pre * ratio)                  # floor, pre-scale ws
            w = float(ws_pre * scale)
            h = float(hs_pre * scale)
            for i in range(OUTPUT_SIZE):    # H (row)
                for j in range(OUTPUT_SIZE): # W (col)
                    cx = ori + j * STRIDE   # displacement from search centre
                    cy = ori + i * STRIDE
                    anchors.append([cx, cy, w, h])
    return np.array(anchors, dtype=np.float32)

ANCHORS = _generate_anchors()

# Do NOT normalise — PySOT uses raw hanning outer-product values (peak ≈ 1).
# Normalising collapses window influence to near-zero.
_cos = np.outer(np.hanning(OUTPUT_SIZE), np.hanning(OUTPUT_SIZE))
WINDOW = np.tile(_cos.flatten(), len(ANCHOR_RATIOS) * len(ANCHOR_SCALES))


# ── image helpers ─────────────────────────────────────────────────────────────
def get_subwindow(img, cx, cy, model_sz, original_sz, avg_chans):
    im_h, im_w = img.shape[:2]
    c  = (original_sz - 1) / 2.0
    x1, y1 = round(cx - c), round(cy - c)
    x2, y2 = round(cx + c), round(cy + c)
    lp = max(0, -x1);  tp = max(0, -y1)
    rp = max(0, x2 - im_w + 1);  bp = max(0, y2 - im_h + 1)
    patch = img[max(0,y1):min(im_h-1,y2)+1, max(0,x1):min(im_w-1,x2)+1].copy()
    if any(p > 0 for p in [lp, tp, rp, bp]):
        patch = cv2.copyMakeBorder(patch, tp, bp, lp, rp,
                                   cv2.BORDER_CONSTANT, value=avg_chans.tolist())
    if patch.shape[:2] != (model_sz, model_sz):
        patch = cv2.resize(patch, (model_sz, model_sz))
    return patch


def preprocess(img_bgr):
    img = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)


# ── decode ONNX outputs ───────────────────────────────────────────────────────
def _softmax2(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def _sz(w, h):
    pad = (w + h) * 0.5
    return np.sqrt((w + pad) * (h + pad))

def _change(r):
    return np.maximum(r, 1.0 / r)

def decode(cls_raw, loc_raw):
    s  = cls_raw.transpose(1, 2, 3, 0).reshape(2, -1).T
    fg = _softmax2(s)[:, 1]
    d  = loc_raw.transpose(1, 2, 3, 0).reshape(4, -1).T
    pcx = d[:, 0] * ANCHORS[:, 2] + ANCHORS[:, 0]
    pcy = d[:, 1] * ANCHORS[:, 3] + ANCHORS[:, 1]
    pw  = np.exp(d[:, 2]) * ANCHORS[:, 2]
    ph  = np.exp(d[:, 3]) * ANCHORS[:, 3]
    return fg, np.stack([pcx, pcy, pw, ph], axis=1)


# ── drawing helpers ───────────────────────────────────────────────────────────
def draw_solid_rect(img, x1, y1, x2, y2, colour, thickness=2):
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colour, thickness)


def draw_dashed_rect(img, x1, y1, x2, y2, colour, thickness=2, dash=10, gap=5):
    pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    for k in range(4):
        p1, p2 = pts[k], pts[k + 1]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        dist = max(1, int(np.hypot(dx, dy)))
        for s in range(0, dist, dash + gap):
            e = min(s + dash, dist)
            pa = (int(p1[0] + dx * s / dist), int(p1[1] + dy * s / dist))
            pb = (int(p1[0] + dx * e / dist), int(p1[1] + dy * e / dist))
            cv2.line(img, pa, pb, colour, thickness)


def put_label(img, text, x, y, colour, bg=True):
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    if bg:
        cv2.rectangle(img, (int(x), int(y) - th - 4),
                      (int(x) + tw + 4, int(y) + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (int(x) + 2, int(y)), font, scale, colour, thick, cv2.LINE_AA)


# ── tracker ───────────────────────────────────────────────────────────────────
class SiamRPNONNX:
    def __init__(self, enc_path, trk_path):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if "CUDAExecutionProvider" in ort.get_available_providers()
                     else ["CPUExecutionProvider"])
        self.enc = ort.InferenceSession(enc_path, opts, providers=providers)
        self.trk = ort.InferenceSession(trk_path, opts, providers=providers)
        self.zf = None
        self.cx = self.cy = self.w = self.h = None
        self.avg_chans = None

    def init(self, img, x1, y1, x2, y2):
        self.cx = (x1 + x2) / 2.0
        self.cy = (y1 + y2) / 2.0
        self.w  = float(x2 - x1)
        self.h  = float(y2 - y1)
        self.avg_chans = img.mean(axis=(0, 1)).astype(np.float32)
        ctx = CONTEXT_AMOUNT * (self.w + self.h)
        s_z = np.sqrt((self.w + ctx) * (self.h + ctx))
        patch = get_subwindow(img, self.cx, self.cy,
                              EXEMPLAR_SIZE, round(s_z), self.avg_chans)
        self.zf = self.enc.run(None, {"template": preprocess(patch)})
        # return search box for init frame (use template size)
        half = s_z / 2.0
        return (self.cx - half, self.cy - half,
                self.cx + half, self.cy + half)

    def track(self, img):
        """Returns (pred_x1,y1,x2,y2, score, search_x1,y1,x2,y2)."""
        im_h, im_w = img.shape[:2]
        ctx = CONTEXT_AMOUNT * (self.w + self.h)
        s_z = np.sqrt((self.w + ctx) * (self.h + ctx))
        s_x = s_z * SEARCH_SIZE / EXEMPLAR_SIZE
        scale_x = SEARCH_SIZE / s_x

        # Search area in original image (before any clamp)
        half_sx = s_x / 2.0
        sa_x1 = self.cx - half_sx
        sa_y1 = self.cy - half_sx
        sa_x2 = self.cx + half_sx
        sa_y2 = self.cy + half_sx

        patch = get_subwindow(img, self.cx, self.cy,
                              SEARCH_SIZE, round(s_x), self.avg_chans)
        feed = {"zf_0": self.zf[0], "zf_1": self.zf[1],
                "zf_2": self.zf[2], "search": preprocess(patch)}
        cls_raw, loc_raw = self.trk.run(None, feed)

        fg, pred_bboxes = decode(cls_raw, loc_raw)

        tws = self.w * scale_x
        ths = self.h * scale_x
        s_c = _change(_sz(pred_bboxes[:, 2], pred_bboxes[:, 3]) / _sz(tws, ths))
        r_c = _change((pred_bboxes[:, 2] / pred_bboxes[:, 3]) / (tws / ths))
        penalty = np.exp(-(r_c * s_c - 1) * PENALTY_K)
        pscore  = (1 - WINDOW_INF) * penalty * fg + WINDOW_INF * WINDOW

        best  = int(np.argmax(pscore))
        score = float(fg[best])

        if score < SCORE_THRESH:
            new_cx, new_cy = self.cx, self.cy
            new_w,  new_h  = self.w,  self.h
        else:
            lr = float(penalty[best]) * score * LR
            # Anchor cx/cy are displacements from the search-image centre, so
            # pred_cx / scale_x is already the displacement in original-image
            # pixels.  Add directly to the current tracker centre.
            pcx = self.cx + pred_bboxes[best, 0] / scale_x
            pcy = self.cy + pred_bboxes[best, 1] / scale_x
            pw  = pred_bboxes[best, 2] / scale_x
            ph  = pred_bboxes[best, 3] / scale_x
            new_cx = self.cx * (1 - lr) + pcx * lr
            new_cy = self.cy * (1 - lr) + pcy * lr

            # Only update size when confident enough; clamp per-frame change
            if score >= SIZE_UPDATE_THRESH:
                new_w = self.w * (1 - lr) + pw * lr
                new_h = self.h * (1 - lr) + ph * lr
                # Hard cap: bbox can't change more than MAX_SCALE_PER_FRAME per frame
                new_w = float(np.clip(new_w, self.w / MAX_SCALE_PER_FRAME,
                                             self.w * MAX_SCALE_PER_FRAME))
                new_h = float(np.clip(new_h, self.h / MAX_SCALE_PER_FRAME,
                                             self.h * MAX_SCALE_PER_FRAME))
            else:
                new_w, new_h = self.w, self.h

        new_cx = float(np.clip(new_cx, 0, im_w))
        new_cy = float(np.clip(new_cy, 0, im_h))
        new_w  = float(np.clip(new_w,  10, im_w))
        new_h  = float(np.clip(new_h,  10, im_h))
        self.cx, self.cy, self.w, self.h = new_cx, new_cy, new_w, new_h

        pred_x1 = new_cx - new_w / 2
        pred_y1 = new_cy - new_h / 2
        return (pred_x1, pred_y1, pred_x1 + new_w, pred_y1 + new_h,
                score,
                sa_x1, sa_y1, sa_x2, sa_y2)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc",         required=True)
    ap.add_argument("--trk",         required=True)
    ap.add_argument("--video",       required=True)
    ap.add_argument("--init_box",    type=int, nargs=4,
                    metavar=("X1", "Y1", "X2", "Y2"), required=True)
    ap.add_argument("--out",         default=None)
    ap.add_argument("--score_thresh",type=float, default=0.20)
    args = ap.parse_args()

    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"ORT providers: {providers}")

    tracker = SiamRPNONNX(args.enc, args.trk)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {args.video}")
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {args.video}  {width}×{height}  {fps:.2f} fps  {total} frames")

    base    = os.path.splitext(os.path.basename(args.video))[0]
    out_vid = args.out or f"{base}_onnx_search_vis.mp4"
    out_csv = out_vid.replace(".mp4", "_results.csv")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_vid, fourcc, fps, (width, height))

    # ── init frame ────────────────────────────────────────────────────────────
    ret, frame = cap.read()
    if not ret:
        sys.exit("Cannot read first frame")

    x1, y1, x2, y2 = args.init_box
    sa = tracker.init(frame, x1, y1, x2, y2)

    vis = frame.copy()
    draw_dashed_rect(vis, sa[0], sa[1], sa[2], sa[3], COLOUR_SEARCH)
    put_label(vis, "search", sa[0], sa[1] - 6, COLOUR_SEARCH)
    draw_solid_rect(vis, x1, y1, x2, y2, COLOUR_INIT)
    put_label(vis, "init", x1, y1 - 6, COLOUR_INIT)
    writer.write(vis)

    # ── tracking loop ─────────────────────────────────────────────────────────
    rows = []
    frame_idx = 1
    import time
    t0_all = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        tx1, ty1, tx2, ty2, score, sx1, sy1, sx2, sy2 = tracker.track(frame)
        ms = (time.perf_counter() - t0) * 1000

        rows.append({
            "frame": frame_idx,
            "pred_x1": round(tx1, 2), "pred_y1": round(ty1, 2),
            "pred_x2": round(tx2, 2), "pred_y2": round(ty2, 2),
            "score":   round(score, 4),
            "search_x1": round(sx1, 2), "search_y1": round(sy1, 2),
            "search_x2": round(sx2, 2), "search_y2": round(sy2, 2),
            "ms": round(ms, 1),
        })

        vis = frame.copy()
        # Search area — dashed cyan
        draw_dashed_rect(vis, sx1, sy1, sx2, sy2, COLOUR_SEARCH)
        put_label(vis, "search area", sx1, sy1 - 6, COLOUR_SEARCH)
        # Tracked bbox — solid yellow
        col = COLOUR_PRED if score >= args.score_thresh else (80, 80, 220)
        draw_solid_rect(vis, tx1, ty1, tx2, ty2, col)
        put_label(vis, f"track {score:.3f}", tx1, ty2 + 16, col)

        writer.write(vis)

        if frame_idx % 100 == 0:
            fps_live = 1000.0 / ms if ms > 0 else 0
            print(f"  frame {frame_idx:5d}/{total}  score={score:.3f}  {fps_live:.1f} fps")

        frame_idx += 1

    cap.release()
    writer.release()

    # ── CSV ───────────────────────────────────────────────────────────────────
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    elapsed = time.perf_counter() - t0_all
    scores  = [r["score"] for r in rows]
    mean_ms = sum(r["ms"] for r in rows) / len(rows)
    print(f"\nDone. {len(rows)} frames in {elapsed:.1f}s  ({len(rows)/elapsed:.1f} fps avg)")
    print(f"Score : mean={sum(scores)/len(scores):.3f}  min={min(scores):.3f}  max={max(scores):.3f}")
    print(f"Latency: {mean_ms:.1f} ms/frame")
    print(f"Video  : {out_vid}")
    print(f"CSV    : {out_csv}")


if __name__ == "__main__":
    main()
