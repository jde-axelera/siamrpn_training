#!/usr/bin/env python3
"""
eval_onnx.py — Fast IoU evaluation of SiamRPN++ ONNX tracker on test sequences
===============================================================================
Loads template_encoder.onnx + tracker.onnx, runs tracking on test sequences
(GT-initialised on frame 1), and reports per-dataset and overall IoU stats.
Writes results to a JSON file for trend tracking.

Usage
-----
  python eval_onnx.py \
      --work_dir  /home/ubuntu/siamrpn_training \
      --onnx_dir  /home/ubuntu/siamrpn_training/exported \
      --out_json  /home/ubuntu/siamrpn_training/eval_results/epoch_150.json \
      --epoch     150 \
      --max_seqs  20        # sequences per dataset (0 = all)
      --max_frames 150      # frames per sequence
"""
import argparse, json, os, random, sys, time
import numpy as np
import cv2

# ── copy the same hyperparameters and helpers from run_onnx_tracker.py ────────
EXEMPLAR_SIZE  = 127
SEARCH_SIZE    = 255
OUTPUT_SIZE    = 25
STRIDE         = 8
CONTEXT_AMOUNT = 0.5
ANCHOR_RATIOS  = [0.33, 0.5, 1.0, 2.0, 3.0]
ANCHOR_SCALES  = [8]
PENALTY_K      = 0.04
WINDOW_INF     = 0.40
LR             = 0.30
SCORE_THRESH   = 0.20

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _generate_anchors():
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

ANCHORS = _generate_anchors()
_cos    = np.outer(np.hanning(OUTPUT_SIZE), np.hanning(OUTPUT_SIZE))
_cos   /= _cos.sum()
WINDOW  = np.tile(_cos.flatten(), len(ANCHOR_RATIOS) * len(ANCHOR_SCALES))

def to_3ch(img):
    if img is None: return None
    if len(img.shape) == 2:          return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 1:            return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    return img

def get_subwindow(img, cx, cy, model_sz, original_sz, avg_chans):
    im_h, im_w = img.shape[:2]
    c  = (original_sz - 1) / 2.0
    x1, y1 = round(cx - c), round(cy - c)
    x2, y2 = round(cx + c), round(cy + c)
    lp = max(0, -x1);   tp = max(0, -y1)
    rp = max(0, x2 - im_w + 1);  bp = max(0, y2 - im_h + 1)
    patch = img[max(0,y1):min(im_h-1,y2)+1, max(0,x1):min(im_w-1,x2)+1].copy()
    if any(p > 0 for p in [lp, tp, rp, bp]):
        patch = cv2.copyMakeBorder(patch, tp, bp, lp, rp,
                                   cv2.BORDER_CONSTANT, value=avg_chans.tolist())
    if patch.shape[0] != model_sz or patch.shape[1] != model_sz:
        patch = cv2.resize(patch, (model_sz, model_sz))
    return patch

def preprocess(img_bgr):
    img = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

def _softmax2(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def _sz(w, h):
    pad = (w + h) * 0.5
    return np.sqrt((w + pad) * (h + pad))

def _change(r):
    return np.maximum(r, 1.0 / r)

def decode_outputs(cls_raw, loc_raw):
    s = cls_raw.transpose(1, 2, 3, 0).reshape(2, -1).T
    fg_scores = _softmax2(s)[:, 1]
    d = loc_raw.transpose(1, 2, 3, 0).reshape(4, -1).T
    pred_cx = d[:, 0] * ANCHORS[:, 2] + ANCHORS[:, 0]
    pred_cy = d[:, 1] * ANCHORS[:, 3] + ANCHORS[:, 1]
    pred_w  = np.exp(np.clip(d[:, 2], -4, 4)) * ANCHORS[:, 2]
    pred_h  = np.exp(np.clip(d[:, 3], -4, 4)) * ANCHORS[:, 3]
    return fg_scores, np.stack([pred_cx, pred_cy, pred_w, pred_h], axis=1)

class SiamRPNONNX:
    def __init__(self, enc_path, trk_path):
        import onnxruntime as ort
        opts = ort.SessionOptions(); opts.log_severity_level = 3
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if "CUDAExecutionProvider" in ort.get_available_providers()
                     else ["CPUExecutionProvider"])
        self.enc = ort.InferenceSession(enc_path, opts, providers=providers)
        self.trk = ort.InferenceSession(trk_path, opts, providers=providers)
        self.zf = None; self.cx = self.cy = self.w = self.h = self.avg_chans = None

    def init(self, img, bbox):
        img = to_3ch(img)
        x1, y1, x2, y2 = bbox
        self.cx, self.cy = (x1+x2)/2., (y1+y2)/2.
        self.w, self.h   = float(x2-x1), float(y2-y1)
        self.avg_chans   = img.mean(axis=(0,1)).astype(np.float32)
        context = CONTEXT_AMOUNT * (self.w + self.h)
        s_z = np.sqrt((self.w + context) * (self.h + context))
        patch = get_subwindow(img, self.cx, self.cy, EXEMPLAR_SIZE, round(s_z), self.avg_chans)
        self.zf = self.enc.run(None, {"template": preprocess(patch)})

    def track(self, img):
        img = to_3ch(img); im_h, im_w = img.shape[:2]
        context = CONTEXT_AMOUNT * (self.w + self.h)
        s_z  = np.sqrt((self.w + context) * (self.h + context))
        s_x  = s_z * SEARCH_SIZE / EXEMPLAR_SIZE
        scale_x = SEARCH_SIZE / s_x
        patch = get_subwindow(img, self.cx, self.cy, SEARCH_SIZE, round(s_x), self.avg_chans)
        cls_raw, loc_raw = self.trk.run(None, {"zf_0": self.zf[0], "zf_1": self.zf[1],
                                               "zf_2": self.zf[2], "search": preprocess(patch)})
        fg_scores, pred_bboxes = decode_outputs(cls_raw, loc_raw)
        tws, ths = self.w * scale_x, self.h * scale_x
        s_c = _change(_sz(pred_bboxes[:,2], pred_bboxes[:,3]) / _sz(tws, ths))
        r_c = _change((pred_bboxes[:,2]/pred_bboxes[:,3]) / (tws/ths))
        penalty = np.exp(-(r_c * s_c - 1) * PENALTY_K)
        pscore  = (1 - WINDOW_INF) * penalty * fg_scores + WINDOW_INF * WINDOW
        best    = int(np.argmax(pscore))
        score   = float(fg_scores[best])
        if score < SCORE_THRESH:
            new_cx, new_cy, new_w, new_h = self.cx, self.cy, self.w, self.h
        else:
            lr = float(penalty[best]) * score * LR
            pcx = self.cx + (pred_bboxes[best,0] - SEARCH_SIZE//2) / scale_x
            pcy = self.cy + (pred_bboxes[best,1] - SEARCH_SIZE//2) / scale_x
            pw  = pred_bboxes[best,2] / scale_x
            ph  = pred_bboxes[best,3] / scale_x
            new_cx = self.cx*(1-lr) + pcx*lr
            new_cy = self.cy*(1-lr) + pcy*lr
            new_w  = self.w *(1-lr) + pw *lr
            new_h  = self.h *(1-lr) + ph *lr
        self.cx = float(np.clip(new_cx, 0, im_w))
        self.cy = float(np.clip(new_cy, 0, im_h))
        self.w  = float(np.clip(new_w,  10, im_w))
        self.h  = float(np.clip(new_h,  10, im_h))
        x1 = self.cx - self.w/2; y1 = self.cy - self.h/2
        return [x1, y1, x1+self.w, y1+self.h], score

def iou(a, b):
    ix1,iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2,iy2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    ua = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua > 0 else 0.0

# ── frame finders (same as run_onnx_tracker.py) ───────────────────────────────
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
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid-1)
        ok, frm = cap.read(); cap.release()
        if ok: return ("__arr__", frm)
    return None

def find_frame_dutantiuav(root, seq, fid):
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

FINDERS = {"std": find_frame_std, "std_or_mp4": find_frame_std_or_mp4,
           "dutantiuav": find_frame_dutantiuav}

def load(result):
    if result is None: return None
    if isinstance(result, tuple) and result[0] == "__arr__": return result[1]
    img = cv2.imread(result)
    return to_3ch(img) if img is not None else None

def build_registry(work_dir):
    data = os.path.join(work_dir, "data")
    return [
        ("AntiUAV410",   os.path.join(data,"anti_uav410","val"),
         os.path.join(data,"anti_uav410","val_pysot.json"),   "std",          20),
        ("AntiUAV300",   os.path.join(data,"anti_uav300","val"),
         os.path.join(data,"anti_uav300","val_pysot.json"),   "std_or_mp4",   20),
        ("MSRS",         os.path.join(data,"msrs","test"),
         os.path.join(data,"msrs","test_pysot.json"),         "std",          10),
        ("DUT-Anti-UAV", os.path.join(data,"dut_anti_uav","images"),
         os.path.join(data,"dut_anti_uav","train_pysot.json"),"dutantiuav",    5),
        ("MassMIND",     os.path.join(data,"massmind","images"),
         os.path.join(data,"massmind","test_pysot.json"),     "std",          10),
    ]

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir",   default="/home/ubuntu/siamrpn_training")
    ap.add_argument("--onnx_dir",   default=None)
    ap.add_argument("--out_json",   default=None)
    ap.add_argument("--epoch",      type=int, default=0)
    ap.add_argument("--max_seqs",   type=int, default=20,
                    help="Max sequences per dataset (0=all)")
    ap.add_argument("--max_frames", type=int, default=150,
                    help="Max frames per sequence")
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    if args.onnx_dir is None:
        args.onnx_dir = os.path.join(args.work_dir, "exported")
    if args.out_json is None:
        d = os.path.join(args.work_dir, "eval_results")
        os.makedirs(d, exist_ok=True)
        args.out_json = os.path.join(d, f"epoch_{args.epoch:04d}.json")

    enc_path = os.path.join(args.onnx_dir, "template_encoder.onnx")
    trk_path = os.path.join(args.onnx_dir, "tracker.onnx")
    for p in (enc_path, trk_path):
        if not os.path.isfile(p):
            sys.exit(f"[ERROR] {p} not found")

    t0 = time.time()
    print(f"\n{'='*65}")
    print(f"  SiamRPN++ ONNX Evaluation   epoch={args.epoch}")
    print(f"{'='*65}")
    tracker = SiamRPNONNX(enc_path, trk_path)

    registry  = build_registry(args.work_dir)
    results   = {"epoch": args.epoch, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "datasets": {}, "overall": {}}
    all_ious  = []

    for (ds_name, root, anno_path, finder_key, min_frames) in registry:
        if not os.path.isfile(anno_path):
            print(f"  [{ds_name}] annotation not found — skipped")
            continue
        anno   = json.load(open(anno_path))
        finder = FINDERS[finder_key]

        # Build candidate list
        candidates = []
        for seq, obj_dict in anno.items():
            frames_dict = obj_dict.get("0", {})
            fids = sorted(frames_dict.keys())
            if len(fids) < min_frames: continue
            if load(finder(root, seq, int(fids[0]))) is None: continue
            candidates.append((seq, fids, frames_dict))

        if not candidates:
            print(f"  [{ds_name}] no usable sequences — skipped")
            continue

        random.shuffle(candidates)
        chosen = candidates[:args.max_seqs] if args.max_seqs > 0 else candidates
        print(f"\n  [{ds_name}]  {len(chosen)} sequences")

        ds_ious = []; seq_results = []
        for (seq, fids, frames_dict) in chosen:
            # Subsample
            if len(fids) > args.max_frames:
                step = len(fids) / args.max_frames
                fids = [fids[int(i*step)] for i in range(args.max_frames)]

            init_img = load(finder(root, seq, int(fids[0])))
            if init_img is None: continue
            gt0 = frames_dict.get(fids[0]) or frames_dict.get(f"{int(fids[0]):06d}")
            if gt0 is None: continue

            tracker.init(init_img, gt0)
            seq_ious = []

            for fi, fid_str in enumerate(fids):
                img = load(finder(root, seq, int(fid_str)))
                if img is None: continue
                gt = frames_dict.get(fid_str) or frames_dict.get(f"{int(fid_str):06d}")
                pred = gt if fi == 0 else tracker.track(img)[0]
                iv = iou(pred, gt) if (gt and pred) else 0.0
                seq_ious.append(iv)

            if not seq_ious: continue
            mean_iou  = float(np.mean(seq_ious))
            succ_05   = sum(v >= 0.5 for v in seq_ious)
            succ_rate = succ_05 / len(seq_ious)
            auc       = float(np.mean([sum(v >= t for v in seq_ious)/len(seq_ious)
                                       for t in np.arange(0, 1.01, 0.05)]))
            ds_ious.extend(seq_ious)
            seq_results.append({"seq": seq, "n_frames": len(seq_ious),
                                 "mean_iou": round(mean_iou,4),
                                 "success_rate@0.5": round(succ_rate,4),
                                 "auc": round(auc,4)})
            flag = "✓" if mean_iou >= 0.5 else ("~" if mean_iou >= 0.3 else "✗")
            print(f"    {flag} {seq[:35]:<35}  IoU={mean_iou:.3f}  "
                  f"succ={succ_05}/{len(seq_ious)}  AUC={auc:.3f}")

        if not ds_ious:
            results["datasets"][ds_name] = {"status": "no_data"}
            continue

        ds_mean   = float(np.mean(ds_ious))
        ds_succ   = sum(v >= 0.5 for v in ds_ious) / len(ds_ious)
        ds_auc    = float(np.mean([sum(v >= t for v in ds_ious)/len(ds_ious)
                                   for t in np.arange(0, 1.01, 0.05)]))
        results["datasets"][ds_name] = {
            "mean_iou": round(ds_mean,4), "success_rate@0.5": round(ds_succ,4),
            "auc": round(ds_auc,4), "n_frames": len(ds_ious),
            "n_seqs": len(seq_results), "sequences": seq_results
        }
        all_ious.extend(ds_ious)

        print(f"    ── {ds_name} aggregate: "
              f"IoU={ds_mean:.3f}  succ@0.5={ds_succ:.1%}  AUC={ds_auc:.3f} "
              f"({len(ds_ious)} frames)")

    # Overall
    if all_ious:
        ov_mean = float(np.mean(all_ious))
        ov_succ = sum(v >= 0.5 for v in all_ious) / len(all_ious)
        ov_auc  = float(np.mean([sum(v >= t for v in all_ious)/len(all_ious)
                                  for t in np.arange(0, 1.01, 0.05)]))
        results["overall"] = {"mean_iou": round(ov_mean,4),
                              "success_rate@0.5": round(ov_succ,4),
                              "auc": round(ov_auc,4),
                              "n_frames": len(all_ious),
                              "eval_time_s": round(time.time()-t0,1)}

        print(f"\n{'='*65}")
        print(f"  OVERALL  epoch={args.epoch:>4d}   "
              f"IoU={ov_mean:.4f}   succ@0.5={ov_succ:.1%}   AUC={ov_auc:.4f}")
        print(f"  frames evaluated: {len(all_ious)}   "
              f"time: {time.time()-t0:.0f}s")
        print(f"{'='*65}\n")
    else:
        results["overall"] = {"mean_iou": 0, "success_rate@0.5": 0, "auc": 0,
                              "n_frames": 0}
        print("  No sequences evaluated.")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(results, open(args.out_json, "w"), indent=2)
    print(f"  Results saved → {args.out_json}")

if __name__ == "__main__":
    main()
