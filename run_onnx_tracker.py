#!/usr/bin/env python3
"""
run_onnx_tracker.py  —  SiamRPN++ ONNX tracker demo on test-split sequences
============================================================================
Loads template_encoder.onnx + tracker.onnx, initialises the tracker with the
GT bounding box from frame 1, then runs fully-open-loop tracking on all
remaining frames.  Draws GT (green) and predicted (yellow) bounding boxes on
each frame, shows per-frame IoU and a running success graph, and writes a
single MP4 demo video.

Usage
-----
  python run_onnx_tracker.py \
      --work_dir  /home/ubuntu/siamrpn_training \
      --onnx_dir  /home/ubuntu/siamrpn_training/exported \
      --out       /home/ubuntu/siamrpn_training/demo/onnx_tracker_demo.mp4 \
      --seqs_per_dataset 3 \
      --max_frames_per_seq 200 \
      --fps 15
"""
import argparse, json, os, random, sys
import numpy as np
import cv2

# ── SiamRPN++ hyperparameters ─────────────────────────────────────────────────
EXEMPLAR_SIZE  = 127        # template crop size
SEARCH_SIZE    = 255        # search crop size
OUTPUT_SIZE    = 25         # RPN output spatial size
STRIDE         = 8          # backbone output stride
CONTEXT_AMOUNT = 0.5        # context padding fraction
ANCHOR_RATIOS  = [0.33, 0.5, 1.0, 2.0, 3.0]
ANCHOR_SCALES  = [8]
PENALTY_K      = 0.04       # change penalty coefficient
WINDOW_INF     = 0.40       # cosine-window influence
LR             = 0.30       # bbox smooth-update learning rate
SCORE_THRESH   = 0.20       # min fg score to accept prediction (else keep prior)

# ImageNet normalisation
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── anchor generation ─────────────────────────────────────────────────────────
def _generate_anchors():
    """
    Generate (A*H*W, 4) anchors in [cx, cy, w, h] format in search-image pixels.
    Anchor ordering matches PySOT's AnchorTarget: A-major → H-major → W-major.
    """
    mid = OUTPUT_SIZE // 2                        # = 12
    offset = SEARCH_SIZE // 2 - mid * STRIDE      # = 127 - 96 = 31
    anchors = []
    for ratio in ANCHOR_RATIOS:
        for scale in ANCHOR_SCALES:
            area = (STRIDE * scale) ** 2          # 64² = 4096
            w = int(np.round(np.sqrt(area / ratio)))
            h = int(np.round(w * ratio))
            for i in range(OUTPUT_SIZE):          # y
                for j in range(OUTPUT_SIZE):      # x
                    cx = offset + j * STRIDE
                    cy = offset + i * STRIDE
                    anchors.append([cx, cy, w, h])
    return np.array(anchors, dtype=np.float32)    # (3125, 4)

ANCHORS = _generate_anchors()

# Cosine window — A-major tiling (same ordering as anchors)
_cos = np.outer(np.hanning(OUTPUT_SIZE), np.hanning(OUTPUT_SIZE))
_cos /= _cos.sum()
WINDOW = np.tile(_cos.flatten(),
                 len(ANCHOR_RATIOS) * len(ANCHOR_SCALES))  # (3125,)

# ── image helpers ─────────────────────────────────────────────────────────────
def to_3ch(img):
    """Ensure image is 3-channel BGR uint8."""
    if img is None:
        return None
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    return img

def get_subwindow(img, cx, cy, model_sz, original_sz, avg_chans):
    """
    Crop a (model_sz × model_sz) patch centred at (cx, cy) in 'img',
    padding with avg_chans where the crop extends outside the image boundary.
    """
    im_h, im_w = img.shape[:2]
    c = (original_sz - 1) / 2.0
    x1, y1 = round(cx - c), round(cy - c)
    x2, y2 = round(cx + c), round(cy + c)

    lp = max(0, -x1);   tp = max(0, -y1)
    rp = max(0, x2 - im_w + 1);  bp = max(0, y2 - im_h + 1)

    x1c = max(0, x1);  y1c = max(0, y1)
    x2c = min(im_w - 1, x2);  y2c = min(im_h - 1, y2)

    patch = img[y1c:y2c + 1, x1c:x2c + 1].copy()

    if any(p > 0 for p in [lp, tp, rp, bp]):
        patch = cv2.copyMakeBorder(patch, tp, bp, lp, rp,
                                   cv2.BORDER_CONSTANT, value=avg_chans.tolist())
    if patch.shape[0] != model_sz or patch.shape[1] != model_sz:
        patch = cv2.resize(patch, (model_sz, model_sz))
    return patch

def preprocess(img_bgr):
    """BGR uint8 (H,W,3) → float32 (1,3,H,W), ImageNet-normalised."""
    img = img_bgr[:, :, ::-1].astype(np.float32) / 255.0   # BGR→RGB
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

# ── post-processing helpers ───────────────────────────────────────────────────
def _softmax2(x):
    """Row-wise softmax for (N, 2) array."""
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def _sz(w, h):
    """PySOT context-aware size metric."""
    pad = (w + h) * 0.5
    return np.sqrt((w + pad) * (h + pad))

def _change(r):
    return np.maximum(r, 1.0 / r)

def decode_outputs(cls_raw, loc_raw):
    """
    Decode ONNX tracker outputs → (fg_scores, pred_bboxes).
      cls_raw: (1, 10, 25, 25)
      loc_raw: (1, 20, 25, 25)
    Returns:
      fg_scores  : (3125,)  fg probability per anchor
      pred_bboxes: (3125, 4) predicted [cx,cy,w,h] in search-image pixels
    """
    # ── fg scores ──────────────────────────────────────────────────────────────
    # Replicate PySOT: score.permute(1,2,3,0).view(2,-1).permute(1,0)
    s = cls_raw.transpose(1, 2, 3, 0)   # (10, 25, 25, 1)
    s = s.reshape(2, -1).T              # (3125, 2)
    fg_scores = _softmax2(s)[:, 1]      # (3125,)

    # ── regression deltas ──────────────────────────────────────────────────────
    d = loc_raw.transpose(1, 2, 3, 0)   # (20, 25, 25, 1)
    d = d.reshape(4, -1).T              # (3125, 4)  [dx, dy, dw, dh]

    # Anchor regression (PySOT convention)
    pred_cx = d[:, 0] * ANCHORS[:, 2] + ANCHORS[:, 0]
    pred_cy = d[:, 1] * ANCHORS[:, 3] + ANCHORS[:, 1]
    pred_w  = np.exp(d[:, 2]) * ANCHORS[:, 2]
    pred_h  = np.exp(d[:, 3]) * ANCHORS[:, 3]
    pred_bboxes = np.stack([pred_cx, pred_cy, pred_w, pred_h], axis=1)  # (3125, 4)

    return fg_scores, pred_bboxes

# ── SiamRPN++ tracker ─────────────────────────────────────────────────────────
class SiamRPNONNX:
    """
    Stateful SiamRPN++ tracker backed by two ONNX sessions:
      - enc_sess:  template_encoder.onnx  (1×3×127×127 → zf_0/1/2)
      - trk_sess:  tracker.onnx           (zf_0/1/2 + 1×3×255×255 → cls, loc)
    """
    def __init__(self, enc_path, trk_path):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.log_severity_level = 3   # suppress ONNX Runtime warnings
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if self._cuda_available() else ["CPUExecutionProvider"])
        self.enc = ort.InferenceSession(enc_path, opts, providers=providers)
        self.trk = ort.InferenceSession(trk_path, opts, providers=providers)
        self.zf  = None   # cached template features
        self.cx = self.cy = self.w = self.h = None
        self.avg_chans = None

    @staticmethod
    def _cuda_available():
        try:
            import onnxruntime as ort
            return "CUDAExecutionProvider" in ort.get_available_providers()
        except Exception:
            return False

    def init(self, img, bbox_x1y1x2y2):
        """Initialise tracker with first frame and GT bbox [x1,y1,x2,y2]."""
        img = to_3ch(img)
        x1, y1, x2, y2 = bbox_x1y1x2y2
        self.cx = (x1 + x2) / 2.0
        self.cy = (y1 + y2) / 2.0
        self.w  = float(x2 - x1)
        self.h  = float(y2 - y1)
        self.avg_chans = img.mean(axis=(0, 1)).astype(np.float32)

        # Template crop size (with context)
        context = CONTEXT_AMOUNT * (self.w + self.h)
        s_z = np.sqrt((self.w + context) * (self.h + context))
        patch = get_subwindow(img, self.cx, self.cy,
                              EXEMPLAR_SIZE, round(s_z), self.avg_chans)
        z_tensor = preprocess(patch)   # (1, 3, 127, 127)
        self.zf = self.enc.run(None, {"template": z_tensor})  # [zf_0, zf_1, zf_2]

    def track(self, img):
        """
        Track one frame.  Returns predicted bbox as [x1, y1, x2, y2] (float).
        """
        assert self.zf is not None, "Call init() before track()"
        img = to_3ch(img)
        im_h, im_w = img.shape[:2]

        # Search region scale
        context   = CONTEXT_AMOUNT * (self.w + self.h)
        s_z       = np.sqrt((self.w + context) * (self.h + context))
        s_x       = s_z * SEARCH_SIZE / EXEMPLAR_SIZE
        scale_x   = SEARCH_SIZE / s_x

        patch = get_subwindow(img, self.cx, self.cy,
                              SEARCH_SIZE, round(s_x), self.avg_chans)
        x_tensor = preprocess(patch)   # (1, 3, 255, 255)

        # Run tracker ONNX
        feed = {"zf_0": self.zf[0], "zf_1": self.zf[1],
                "zf_2": self.zf[2], "search": x_tensor}
        cls_raw, loc_raw = self.trk.run(None, feed)   # (1,10,25,25), (1,20,25,25)

        fg_scores, pred_bboxes = decode_outputs(cls_raw, loc_raw)

        # Scale/ratio change penalty
        tws = self.w * scale_x   # target w in search pixels
        ths = self.h * scale_x   # target h in search pixels
        s_c = _change(_sz(pred_bboxes[:, 2], pred_bboxes[:, 3]) / _sz(tws, ths))
        r_c = _change((pred_bboxes[:, 2] / pred_bboxes[:, 3]) / (tws / ths))
        penalty  = np.exp(-(r_c * s_c - 1) * PENALTY_K)
        pscore   = penalty * fg_scores
        pscore   = (1 - WINDOW_INF) * pscore + WINDOW_INF * WINDOW

        best  = int(np.argmax(pscore))
        score = float(fg_scores[best])

        if score < SCORE_THRESH:
            # Low-confidence: keep previous state
            new_cx, new_cy = self.cx, self.cy
            new_w,  new_h  = self.w, self.h
        else:
            lr = float(penalty[best]) * score * LR

            # Convert best prediction from search-image coords → original-image coords
            pred_cx_img = self.cx + (pred_bboxes[best, 0] - SEARCH_SIZE / 2) / scale_x
            pred_cy_img = self.cy + (pred_bboxes[best, 1] - SEARCH_SIZE / 2) / scale_x
            pred_w_img  = pred_bboxes[best, 2] / scale_x
            pred_h_img  = pred_bboxes[best, 3] / scale_x

            # Smooth update
            new_cx = self.cx * (1 - lr) + pred_cx_img * lr
            new_cy = self.cy * (1 - lr) + pred_cy_img * lr
            new_w  = self.w  * (1 - lr) + pred_w_img  * lr
            new_h  = self.h  * (1 - lr) + pred_h_img  * lr

        # Clamp to image bounds
        new_cx = float(np.clip(new_cx, 0, im_w))
        new_cy = float(np.clip(new_cy, 0, im_h))
        new_w  = float(np.clip(new_w,  10, im_w))
        new_h  = float(np.clip(new_h,  10, im_h))

        self.cx, self.cy, self.w, self.h = new_cx, new_cy, new_w, new_h

        x1 = new_cx - new_w / 2
        y1 = new_cy - new_h / 2
        return [x1, y1, x1 + new_w, y1 + new_h], score

# ── IoU ───────────────────────────────────────────────────────────────────────
def iou(a, b):
    """Axis-aligned IoU of two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0

# ── rendering helpers ─────────────────────────────────────────────────────────
COLOUR_GT   = (50,  220,  50)     # green  — ground truth
COLOUR_PRED = (0,  220, 255)      # yellow — tracker prediction

def draw_tracking_frame(img, gt_box, pred_box, iou_val, score,
                        frame_idx, total_frames, iou_history,
                        dataset, seq, out_w, out_h):
    """Render one tracking frame with GT + predicted boxes and HUD."""
    canvas = cv2.resize(img, (out_w, out_h))
    sh = out_h / img.shape[0];  sw = out_w / img.shape[1]

    # Scale boxes
    def scale(box):
        return [box[0]*sw, box[1]*sh, box[2]*sw, box[3]*sh]

    # GT box (solid green)
    if gt_box is not None:
        b = scale(gt_box)
        x1,y1,x2,y2 = (int(v) for v in b)
        cv2.rectangle(canvas, (x1,y1), (x2,y2), COLOUR_GT, 2)
        cv2.putText(canvas, "GT", (x1+2, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOUR_GT, 1, cv2.LINE_AA)

    # Predicted box (dashed yellow)
    if pred_box is not None:
        b = scale(pred_box)
        x1,y1,x2,y2 = (int(v) for v in b)
        # Draw dashed rectangle
        pts = [(x1,y1),(x2,y1),(x2,y2),(x1,y2),(x1,y1)]
        for k in range(len(pts)-1):
            p1, p2 = pts[k], pts[k+1]
            dx = p2[0]-p1[0]; dy = p2[1]-p1[1]
            dist = max(1, int(np.hypot(dx,dy)))
            dash, gap = 8, 4
            for s in range(0, dist, dash+gap):
                e = min(s+dash, dist)
                pa = (int(p1[0]+dx*s/dist), int(p1[1]+dy*s/dist))
                pb = (int(p1[0]+dx*e/dist), int(p1[1]+dy*e/dist))
                cv2.line(canvas, pa, pb, COLOUR_PRED, 2)
        cv2.putText(canvas, f"Pred ({score:.2f})", (x1+2, y2+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOUR_PRED, 1, cv2.LINE_AA)

    # Top HUD bar
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0,0), (out_w, 38), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    cv2.putText(canvas, f"{dataset}  |  {seq[:28]}",
                (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220,220,220), 1, cv2.LINE_AA)
    iou_col = (50,200,50) if iou_val>=0.5 else (50,200,255) if iou_val>=0.3 else (50,50,220)
    cv2.putText(canvas, f"IoU={iou_val:.3f}   frame {frame_idx}/{total_frames}",
                (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.44, iou_col, 1, cv2.LINE_AA)

    # Bottom IoU sparkline
    if len(iou_history) > 1:
        bar_h, bar_y0 = 30, out_h - 32
        overlay2 = canvas.copy()
        cv2.rectangle(overlay2, (0, bar_y0-2), (out_w, out_h), (0,0,0), -1)
        cv2.addWeighted(overlay2, 0.5, canvas, 0.5, 0, canvas)
        n = len(iou_history)
        for k in range(1, n):
            x_a = int((k-1) / max(n-1, 1) * (out_w-2)) + 1
            x_b = int(k     / max(n-1, 1) * (out_w-2)) + 1
            y_a = bar_y0 + bar_h - int(iou_history[k-1] * bar_h)
            y_b = bar_y0 + bar_h - int(iou_history[k]   * bar_h)
            col = (50,200,50) if iou_history[k]>=0.5 else (50,200,255) if iou_history[k]>=0.3 else (50,50,220)
            cv2.line(canvas, (x_a,y_a), (x_b,y_b), col, 1)
        cv2.putText(canvas, "IoU", (2, out_h-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160,160,160), 1, cv2.LINE_AA)

    return canvas

def make_title_card(text, sub, out_w, out_h, colour, fps, hold_sec=2.5):
    frames = []
    for _ in range(int(fps * hold_sec)):
        card = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        cv2.rectangle(card, (0,0), (14, out_h), colour, -1)
        cv2.putText(card, text, (30, out_h//2-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, colour, 2, cv2.LINE_AA)
        cv2.putText(card, sub, (30, out_h//2+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (170,170,170), 1, cv2.LINE_AA)
        frames.append(card)
    return frames

# ── dataset registry ──────────────────────────────────────────────────────────
def build_registry(work_dir):
    data = os.path.join(work_dir, "data")
    # (name, root, anno, finder_key, min_frames)
    return [
        ("AntiUAV410",
         os.path.join(data,"anti_uav410","val"),          # val split lives in val/
         os.path.join(data,"anti_uav410","val_pysot.json"),
         "std", 30),
        ("AntiUAV300",
         os.path.join(data,"anti_uav300","val"),           # extracted frames in val/
         os.path.join(data,"anti_uav300","val_pysot.json"),
         "std_or_mp4", 30),
        ("DUT-Anti-UAV",
         os.path.join(data,"dut_anti_uav","images"),
         os.path.join(data,"dut_anti_uav","train_pysot.json"),
         "dutantiuav", 10),
    ]

def find_frame_std(root, seq, fid):
    for ext in (".jpg",".png"):
        p = os.path.join(root, seq, f"{fid:06d}{ext}")
        if os.path.isfile(p): return p
    return None

def find_frame_std_or_mp4(root, seq, fid):
    r = find_frame_std(root, seq, fid)
    if r: return r
    mp4 = os.path.join(root, seq, "infrared.mp4")
    if os.path.isfile(mp4):
        cap = cv2.VideoCapture(mp4)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid - 1)
        ok, frm = cap.read(); cap.release()
        if ok: return ("__arr__", frm)
    return None

def find_frame_dutantiuav(root, seq, fid):
    # DUT frames may be 5-digit (00001.jpg) or 6-digit (000001.jpg)
    for ext in (".jpg",".png"):
        for fmt in (f"{fid:06d}", f"{fid:05d}"):
            p = os.path.join(root, seq, f"{fmt}{ext}")
            if os.path.isfile(p): return p
    for wrapper in os.listdir(root):
        w = os.path.join(root, wrapper)
        if os.path.isdir(w):
            for ext in (".jpg",".png"):
                for fmt in (f"{fid:06d}", f"{fid:05d}"):
                    p = os.path.join(w, seq, f"{fmt}{ext}")
                    if os.path.isfile(p): return p
    return None

FINDERS = {
    "std":         find_frame_std,
    "std_or_mp4":  find_frame_std_or_mp4,
    "dutantiuav":  find_frame_dutantiuav,
}

def load(result):
    if result is None: return None
    if isinstance(result, tuple) and result[0] == "__arr__": return result[1]
    img = cv2.imread(result)
    return to_3ch(img)

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
             formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work_dir",  default="/home/ubuntu/siamrpn_training")
    ap.add_argument("--onnx_dir",  default=None)
    ap.add_argument("--out",       default=None)
    ap.add_argument("--seqs_per_dataset",  type=int, default=3)
    ap.add_argument("--max_frames_per_seq",type=int, default=200)
    ap.add_argument("--fps",   type=int, default=15)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height",type=int, default=480)
    ap.add_argument("--seed",  type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    out_w, out_h = args.width, args.height

    if args.onnx_dir is None:
        args.onnx_dir = os.path.join(args.work_dir, "exported")
    demo_dir = os.path.join(args.work_dir, "demo")
    os.makedirs(demo_dir, exist_ok=True)
    if args.out is None:
        args.out = os.path.join(demo_dir, "onnx_tracker_demo.mp4")

    enc_path = os.path.join(args.onnx_dir, "template_encoder.onnx")
    trk_path = os.path.join(args.onnx_dir, "tracker.onnx")
    for p in (enc_path, trk_path):
        if not os.path.isfile(p):
            sys.exit(f"[ERROR] ONNX model not found: {p}")

    print(f"Loading ONNX models from {args.onnx_dir} ...")
    tracker = SiamRPNONNX(enc_path, trk_path)
    print("  Models loaded.\n")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (out_w, out_h))
    if not writer.isOpened():
        sys.exit(f"[ERROR] Cannot open VideoWriter for {args.out}")

    registry = build_registry(args.work_dir)
    global_ious = []

    for (ds_name, root, anno_path, finder_key, min_frames) in registry:
        if not os.path.isfile(anno_path):
            print(f"  [SKIP] {ds_name}: annotation not found")
            continue
        anno = json.load(open(anno_path))
        finder = FINDERS[finder_key]

        # Choose sequences with enough frames
        candidates = []
        for seq, obj_dict in anno.items():
            frames_dict = obj_dict.get("0", {})
            fids = sorted(frames_dict.keys())
            if len(fids) < min_frames:
                continue
            # Verify first frame is loadable
            if load(finder(root, seq, int(fids[0]))) is None:
                continue
            candidates.append((seq, fids, frames_dict))

        if not candidates:
            print(f"  [SKIP] {ds_name}: no usable sequences (min_frames={min_frames})")
            continue

        random.shuffle(candidates)
        chosen = candidates[:args.seqs_per_dataset]
        print(f"[{ds_name}] {len(chosen)} sequences selected")

        # Title card
        for card in make_title_card(
                ds_name,
                "SiamRPN++ ONNX tracker  ·  green=GT  yellow=prediction",
                out_w, out_h,
                (50, 220, 50), args.fps):
            writer.write(card)

        for (seq, fids, frames_dict) in chosen:
            # Subsample to max_frames_per_seq
            if len(fids) > args.max_frames_per_seq:
                step = len(fids) / args.max_frames_per_seq
                fids = [fids[int(i * step)] for i in range(args.max_frames_per_seq)]
            n_total = len(fids)

            # ── Init with frame 1 ──────────────────────────────────────────────
            first_fid = int(fids[0])
            init_img  = load(finder(root, seq, first_fid))
            if init_img is None:
                print(f"    [WARN] {seq}: could not load init frame")
                continue
            gt0 = frames_dict.get(fids[0]) or frames_dict.get(f"{first_fid:06d}")
            if gt0 is None:
                continue

            tracker.init(init_img, gt0)
            print(f"  {seq[:35]:<35} frames={n_total}", end="", flush=True)

            seq_ious  = []
            success_n = 0

            # ── Track each frame ───────────────────────────────────────────────
            for fi, fid_str in enumerate(fids):
                fid = int(fid_str)
                img = load(finder(root, seq, fid))
                if img is None:
                    continue
                gt = (frames_dict.get(fid_str)
                      or frames_dict.get(f"{fid:06d}"))

                if fi == 0:
                    # First frame — show init (prediction == GT)
                    pred_box = gt
                    score_val = 1.0
                else:
                    pred_box, score_val = tracker.track(img)

                iou_val = iou(pred_box, gt) if (gt and pred_box) else 0.0
                seq_ious.append(iou_val)
                if iou_val >= 0.5:
                    success_n += 1

                canvas = draw_tracking_frame(
                    img, gt, pred_box, iou_val, score_val,
                    fi + 1, n_total, seq_ious,
                    ds_name, seq, out_w, out_h)
                writer.write(canvas)

            mean_iou = float(np.mean(seq_ious)) if seq_ious else 0.0
            auc      = float(np.mean([v >= t for v in seq_ious
                                      for t in np.arange(0, 1.01, 0.05)])) if seq_ious else 0.0
            global_ious.extend(seq_ious)
            print(f"   mean_IoU={mean_iou:.3f}  success@0.5={success_n}/{len(seq_ious)}")

        # Per-dataset summary card
        ds_ious   = global_ious[-sum(len(c[1]) for c in chosen):]
        if ds_ious:
            m_iou = float(np.mean(ds_ious))
            succ  = sum(v >= 0.5 for v in ds_ious)
            for card in make_title_card(
                    f"{ds_name}  —  mean IoU {m_iou:.3f}",
                    f"Success@0.5: {succ}/{len(ds_ious)} frames",
                    out_w, out_h, (50, 180, 50), args.fps, hold_sec=2.0):
                writer.write(card)

    writer.release()

    if global_ious:
        mean_iou_all = float(np.mean(global_ious))
        success_all  = sum(v >= 0.5 for v in global_ious)
        print(f"\n{'='*60}")
        print(f"Overall mean IoU  : {mean_iou_all:.4f}")
        print(f"Success@0.5       : {success_all}/{len(global_ious)} "
              f"({100*success_all/len(global_ious):.1f}%)")
        print(f"{'='*60}")

    print(f"\n✓ Tracker demo written: {args.out}")


if __name__ == "__main__":
    main()
