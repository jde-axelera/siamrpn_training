#!/usr/bin/env python3
"""
validate_onnx_vs_pytorch.py
Compare raw PyTorch model outputs vs ONNX model outputs on a single frame.
Also compares anchor arrays: mine vs PySOT's.

Usage:
  python validate_onnx_vs_pytorch.py \
      --cfg   pysot/experiments/siamrpn_r50_alldatasets/config.yaml \
      --ckpt  pysot/snapshot/all_datasets/best_model.pth \
      --enc   exported/template_encoder.onnx \
      --trk   exported/tracker.onnx \
      --video ir_crop.mp4 \
      --init_box 339 148 391 232
"""
import argparse, sys, os
import numpy as np
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "pysot"))

EXEMPLAR_SIZE  = 127
SEARCH_SIZE    = 255
OUTPUT_SIZE    = 25
STRIDE         = 8
CONTEXT_AMOUNT = 0.5
ANCHOR_RATIOS  = [0.33, 0.5, 1.0, 2.0, 3.0]
ANCHOR_SCALES  = [8]
IMAGENET_MEAN  = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD   = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img_bgr):
    img = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)


def get_subwindow(img, cx, cy, model_sz, original_sz, avg_chans):
    im_h, im_w = img.shape[:2]
    c  = (original_sz - 1) / 2.0
    x1, y1 = round(cx - c), round(cy - c)
    x2, y2 = round(cx + c), round(cy + c)
    lp = max(0, -x1);  tp = max(0, -y1)
    rp = max(0, x2 - im_w + 1);  bp = max(0, y2 - im_h + 1)
    patch = img[max(0, y1):min(im_h-1, y2)+1, max(0, x1):min(im_w-1, x2)+1].copy()
    if any(p > 0 for p in [lp, tp, rp, bp]):
        patch = cv2.copyMakeBorder(patch, tp, bp, lp, rp,
                                   cv2.BORDER_CONSTANT, value=avg_chans.tolist())
    if patch.shape[:2] != (model_sz, model_sz):
        patch = cv2.resize(patch, (model_sz, model_sz))
    return patch


# ── Anchor A: my current (buggy) implementation ───────────────────────────────
def anchors_mine():
    mid    = OUTPUT_SIZE // 2
    offset = SEARCH_SIZE // 2 - mid * STRIDE
    anchors = []
    for ratio in ANCHOR_RATIOS:
        for scale in ANCHOR_SCALES:
            area = (STRIDE * scale) ** 2
            w = int(np.round(np.sqrt(area / ratio)))
            h = int(np.round(w * ratio))
            for i in range(OUTPUT_SIZE):
                for j in range(OUTPUT_SIZE):
                    anchors.append([offset + j * STRIDE, offset + i * STRIDE, w, h])
    return np.array(anchors, dtype=np.float32)


# ── Anchor B: PySOT-exact (matches SiamRPNTracker.generate_anchor) ───────────
def anchors_pysot():
    """
    cx/cy = displacement from search-image centre (-96 at j=0, 0 at j=12).
    w/h   = ws_pre*scale using floor(sqrt(stride²/ratio)).
    """
    ori = -(OUTPUT_SIZE // 2) * STRIDE   # -96
    anchors = []
    for ratio in ANCHOR_RATIOS:
        for scale in ANCHOR_SCALES:
            ws_pre = int(np.sqrt(STRIDE ** 2 / ratio))   # floor
            hs_pre = int(ws_pre * ratio)
            w = float(ws_pre * scale)
            h = float(hs_pre * scale)
            for i in range(OUTPUT_SIZE):
                for j in range(OUTPUT_SIZE):
                    anchors.append([ori + j * STRIDE, ori + i * STRIDE, w, h])
    return np.array(anchors, dtype=np.float32)


# ── Anchor C: from PySOT library directly ────────────────────────────────────
def anchors_pysot_lib():
    try:
        from pysot.utils.anchor import Anchors
        from pysot.core.config import cfg
        a = Anchors(cfg.ANCHOR.STRIDE, cfg.ANCHOR.RATIOS, cfg.ANCHOR.SCALES)
        a.generate_all_anchors(im_c=SEARCH_SIZE // 2, size=OUTPUT_SIZE)
        return a.all_anchors  # (N, 4) in [cx, cy, w, h]
    except Exception as e:
        print(f"  [WARN] could not load PySOT Anchors lib: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg",      required=True)
    ap.add_argument("--ckpt",     required=True)
    ap.add_argument("--enc",      required=True)
    ap.add_argument("--trk",      required=True)
    ap.add_argument("--video",    required=True)
    ap.add_argument("--init_box", type=int, nargs=4,
                    metavar=("X1","Y1","X2","Y2"), required=True)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    import onnxruntime as ort
    from pysot.core.config import cfg as pcfg
    from pysot.models.model_builder import ModelBuilder

    sep = "─" * 60

    # ── 1. Load frame and compute patches ─────────────────────────────────────
    print(f"\n{sep}\n  STEP 1: Load frame + compute template/search patches\n{sep}")
    cap = cv2.VideoCapture(args.video)
    ret, frame0 = cap.read()
    ret, frame1 = cap.read()
    cap.release()

    x1, y1, x2, y2 = args.init_box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w  = float(x2 - x1)
    h  = float(y2 - y1)
    avg = frame0.mean(axis=(0, 1)).astype(np.float32)

    ctx = CONTEXT_AMOUNT * (w + h)
    s_z = np.sqrt((w + ctx) * (h + ctx))
    s_x = s_z * SEARCH_SIZE / EXEMPLAR_SIZE
    scale_x = SEARCH_SIZE / s_x

    z_patch = get_subwindow(frame0, cx, cy, EXEMPLAR_SIZE, round(s_z), avg)
    x_patch = get_subwindow(frame1, cx, cy, SEARCH_SIZE,   round(s_x), avg)
    z_np = preprocess(z_patch)
    x_np = preprocess(x_patch)
    print(f"  Template patch: {z_np.shape}  Search patch: {x_np.shape}")
    print(f"  scale_x = {scale_x:.4f}  s_x = {s_x:.2f}")

    # ── 2. Run PyTorch model ───────────────────────────────────────────────────
    print(f"\n{sep}\n  STEP 2: PyTorch model outputs\n{sep}")
    pcfg.merge_from_file(args.cfg)
    pcfg.CUDA = torch.cuda.is_available()
    model = ModelBuilder().eval()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    if pcfg.CUDA:
        model = model.cuda()

    device = "cuda" if pcfg.CUDA else "cpu"
    z_t = torch.from_numpy(z_np).to(device)
    x_t = torch.from_numpy(x_np).to(device)

    with torch.no_grad():
        zf  = model.neck(model.backbone(z_t))
        xf  = model.neck(model.backbone(x_t))
        cls_pt, loc_pt = model.rpn_head(list(zf), list(xf))

    cls_pt = cls_pt.cpu().numpy()
    loc_pt = loc_pt.cpu().numpy()
    print(f"  cls shape: {cls_pt.shape}   loc shape: {loc_pt.shape}")
    print(f"  cls  min/max/mean: {cls_pt.min():.4f} / {cls_pt.max():.4f} / {cls_pt.mean():.4f}")
    print(f"  loc  min/max/mean: {loc_pt.min():.4f} / {loc_pt.max():.4f} / {loc_pt.mean():.4f}")

    # ── 3. Run ONNX model ──────────────────────────────────────────────────────
    print(f"\n{sep}\n  STEP 3: ONNX model outputs\n{sep}")
    opts = ort.SessionOptions(); opts.log_severity_level = 3
    enc = ort.InferenceSession(args.enc, opts, providers=["CPUExecutionProvider"])
    trk = ort.InferenceSession(args.trk, opts, providers=["CPUExecutionProvider"])

    zf_onnx = enc.run(None, {"template": z_np})
    feed = {"zf_0": zf_onnx[0], "zf_1": zf_onnx[1],
            "zf_2": zf_onnx[2], "search": x_np}
    cls_onnx, loc_onnx = trk.run(None, feed)
    print(f"  cls shape: {cls_onnx.shape}   loc shape: {loc_onnx.shape}")
    print(f"  cls  min/max/mean: {cls_onnx.min():.4f} / {cls_onnx.max():.4f} / {cls_onnx.mean():.4f}")
    print(f"  loc  min/max/mean: {loc_onnx.min():.4f} / {loc_onnx.max():.4f} / {loc_onnx.mean():.4f}")

    # ── 4. Compare raw outputs ─────────────────────────────────────────────────
    print(f"\n{sep}\n  STEP 4: Raw output comparison (PyTorch vs ONNX)\n{sep}")
    cls_diff = np.abs(cls_pt - cls_onnx)
    loc_diff = np.abs(loc_pt - loc_onnx)
    print(f"  cls diff  max={cls_diff.max():.6f}  mean={cls_diff.mean():.8f}")
    print(f"  loc diff  max={loc_diff.max():.6f}  mean={loc_diff.mean():.8f}")
    if cls_diff.max() < 1e-4 and loc_diff.max() < 1e-4:
        print("  ✓ ONNX outputs match PyTorch within tolerance — ONNX graph is correct")
    else:
        print("  ✗ ONNX outputs differ — possible ONNX export issue")

    # ── 5. Compare anchor arrays ───────────────────────────────────────────────
    print(f"\n{sep}\n  STEP 5: Anchor comparison (mine vs PySOT)\n{sep}")
    A_mine  = anchors_mine()
    A_pysot = anchors_pysot()
    A_lib   = anchors_pysot_lib()

    print(f"  my anchors     shape: {A_mine.shape}")
    print(f"  pysot-exact    shape: {A_pysot.shape}")
    diff_mine_pysot = np.abs(A_mine - A_pysot)
    print(f"  mine vs pysot-exact: max_diff={diff_mine_pysot.max():.4f}")
    print(f"  First 3 anchors (mine):\n{A_mine[:3]}")
    print(f"  First 3 anchors (pysot-exact):\n{A_pysot[:3]}")

    # The Anchors.all_anchors tuple uses absolute coordinates (generate_all_anchors
    # with im_c=127).  SiamRPNTracker.generate_anchor uses center-relative coords.
    # These are different conventions; we compare against the tracker's convention.
    if A_lib is not None:
        print("\n  Note: Anchors.all_anchors[1] uses absolute coords (cx=31 at j=0).")
        print("  SiamRPNTracker.generate_anchor uses center-relative (cx=-96 at j=0).")
        print("  We match the tracker — skipping direct library comparison.")

    # ── 6a. Run PyTorch SiamRPNTracker directly on same frame ─────────────────
    print(f"\n{sep}\n  STEP 6a: PyTorch SiamRPNTracker direct prediction\n{sep}")
    from pysot.tracker.siamrpn_tracker import SiamRPNTracker as PTTracker
    pt_tracker = PTTracker(model)
    pt_tracker.init(frame0, [x1, y1, x2 - x1, y2 - y1])  # [x,y,w,h]
    pt_out = pt_tracker.track(frame1)
    pt_bbox = pt_out['bbox']  # [x, y, w, h]
    pt_cx = pt_bbox[0] + pt_bbox[2]/2
    pt_cy = pt_bbox[1] + pt_bbox[3]/2
    print(f"  PyTorch tracker: cx={pt_cx:.2f}  cy={pt_cy:.2f}  "
          f"w={pt_bbox[2]:.2f}  h={pt_bbox[3]:.2f}  score={pt_out['best_score']:.4f}")

    # ── 6. Decode both with correct anchors and compare ───────────────────────
    print(f"\n{sep}\n  STEP 6b: ONNX decoded bbox comparison (mine vs pysot-exact anchors)\n{sep}")
    def softmax2(x):
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def decode_with_anchors(cls_raw, loc_raw, anchors):
        s = cls_raw.transpose(1,2,3,0).reshape(2,-1).T
        fg = softmax2(s)[:,1]
        d  = loc_raw.transpose(1,2,3,0).reshape(4,-1).T
        pcx = d[:,0]*anchors[:,2] + anchors[:,0]
        pcy = d[:,1]*anchors[:,3] + anchors[:,1]
        pw  = np.exp(d[:,2])*anchors[:,2]
        ph  = np.exp(d[:,3])*anchors[:,3]
        return fg, np.stack([pcx,pcy,pw,ph],axis=1)

    for label, anc in [("mine (buggy)", A_mine), ("pysot-exact", A_pysot)]:
        fg_pt,   bb_pt   = decode_with_anchors(cls_pt,   loc_pt,   anc)
        fg_onnx, bb_onnx = decode_with_anchors(cls_onnx, loc_onnx, anc)
        best_pt   = int(np.argmax(fg_pt))
        best_onnx = int(np.argmax(fg_onnx))
        print(f"\n  [{label}]")
        print(f"    PyTorch  best anchor={best_pt:4d}  "
              f"score={fg_pt[best_pt]:.4f}  "
              f"pred_cx={bb_pt[best_pt,0]/scale_x:.1f}  cy={bb_pt[best_pt,1]/scale_x:.1f}  "
              f"w={bb_pt[best_pt,2]/scale_x:.1f}  h={bb_pt[best_pt,3]/scale_x:.1f}")
        print(f"    ONNX     best anchor={best_onnx:4d}  "
              f"score={fg_onnx[best_onnx]:.4f}  "
              f"pred_cx={bb_onnx[best_onnx,0]/scale_x:.1f}  cy={bb_onnx[best_onnx,1]/scale_x:.1f}  "
              f"w={bb_onnx[best_onnx,2]/scale_x:.1f}  h={bb_onnx[best_onnx,3]/scale_x:.1f}")
        print(f"    bbox diff:  cx={abs(bb_pt[best_pt,0]-bb_onnx[best_onnx,0])/scale_x:.2f}  "
              f"cy={abs(bb_pt[best_pt,1]-bb_onnx[best_onnx,1])/scale_x:.2f}  "
              f"w={abs(bb_pt[best_pt,2]-bb_onnx[best_onnx,2])/scale_x:.2f}  "
              f"h={abs(bb_pt[best_pt,3]-bb_onnx[best_onnx,3])/scale_x:.2f}")

    print(f"\n{sep}\n  Done.\n{sep}\n")


if __name__ == "__main__":
    main()
