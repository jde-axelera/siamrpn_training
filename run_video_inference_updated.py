#!/usr/bin/env python3
"""
run_video_inference_updated.py  —  SiamRPN++ with recommended fixes applied.

Recommendations implemented vs baseline run_video_inference.py:
  R1  EMA template update (alpha=0.015) gated on score>0.75 AND size<1.5×init
  R2  Hard cap on search-area growth (s_x <= 2.5 × s_z_init)
  R3  Score-gated bbox size update (freeze size when score < 0.55)

These three changes alone break the positive-feedback loop (bbox growth →
larger search area → background contamination → score drop → more growth).

Usage:
  python run_video_inference_updated.py \
      --cfg   pysot/experiments/siamrpn_r50_alldatasets/config.yaml \
      --ckpt  pysot/snapshot/all_datasets/best_model.pth \
      --video ir_crop.mp4 \
      --init_box 339 148 391 232 \
      --rotate -90 \
      --out   ir_crop_updated_rot90ccw.mp4
"""
import argparse, os, sys, csv
import cv2, numpy as np, torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYSOT_DIR  = os.path.join(SCRIPT_DIR, 'pysot')
if os.path.isdir(PYSOT_DIR):
    sys.path.insert(0, PYSOT_DIR)

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.siamrpn_tracker import SiamRPNTracker


# ── Recommendation constants ──────────────────────────────────────────────────
R1_UPDATE_THRESH  = 0.75   # score threshold for template EMA update
R1_MAX_SIZE_RATIO = 1.5    # don't update if bbox has grown > 1.5× initial size
R1_EMA_ALPHA      = 0.015  # blending rate (very slow)
R2_MAX_SX_FACTOR  = 2.5    # cap search area at 2.5× initial s_z
R3_SIZE_THRESH    = 0.55   # freeze bbox size below this score


class UpdatedTracker(SiamRPNTracker):
    """SiamRPNTracker + R1/R2/R3 recommendations."""

    def init(self, img, bbox):
        super().init(img, bbox)
        self._init_size = self.size.copy()     # [w, h] at frame 0
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        self._init_s_z  = np.sqrt(w_z * h_z)  # frozen template crop scale
        self._init_channel_avg = self.channel_average.copy()
        self._update_count = 0

    def track(self, img):
        # ── compute search crop size (same as parent) ────────────────────────
        w_z     = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z     = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z     = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z

        # R2 — cap s_x to break growth feedback loop
        s_x_raw = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        s_x     = min(s_x_raw, R2_MAX_SX_FACTOR * self._init_s_z)
        if s_x < s_x_raw:
            # recalculate scale_z to be consistent with clamped s_x
            s_z_eff = s_x * cfg.TRACK.EXEMPLAR_SIZE / cfg.TRACK.INSTANCE_SIZE
            scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z_eff

        x_crop  = self.get_subwindow(img, self.center_pos,
                                     cfg.TRACK.INSTANCE_SIZE,
                                     round(s_x), self.channel_average)
        outputs   = self.model.track(x_crop)
        score     = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):  return np.maximum(r, 1. / r)
        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        s_c     = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                         sz(self.size[0] * scale_z, self.size[1] * scale_z))
        r_c     = change((self.size[0] / self.size[1]) /
                         (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore  = penalty * score
        pscore  = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                  self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox_cand = pred_bbox[:, best_idx] / scale_z
        lr        = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox_cand[0] + self.center_pos[0]
        cy = bbox_cand[1] + self.center_pos[1]

        # R3 — only update size when confident
        if score[best_idx] >= R3_SIZE_THRESH:
            width  = self.size[0] * (1 - lr) + bbox_cand[2] * lr
            height = self.size[1] * (1 - lr) + bbox_cand[3] * lr
        else:
            width, height = self.size[0], self.size[1]   # freeze size

        cx, cy, width, height = self._bbox_clip(cx, cy, width, height,
                                                img.shape[:2])

        # R1 — EMA template update (gated on score AND size ratio)
        size_ratio = max(width / self._init_size[0], height / self._init_size[1])
        if score[best_idx] > R1_UPDATE_THRESH and size_ratio < R1_MAX_SIZE_RATIO:
            # Crop at FROZEN initial scale — never the inflated current scale
            new_z = self.get_subwindow(img, np.array([cx, cy]),
                                       cfg.TRACK.EXEMPLAR_SIZE,
                                       round(self._init_s_z),
                                       self._init_channel_avg)
            with torch.no_grad():
                new_zf = self.model.neck(self.model.backbone(new_z))
            self.model.zf = [(1 - R1_EMA_ALPHA) * o + R1_EMA_ALPHA * n
                             for o, n in zip(self.model.zf, new_zf)]
            self._update_count += 1

        self.center_pos = np.array([cx, cy])
        self.size       = np.array([width, height])

        return {
            'bbox': [cx - width/2, cy - height/2, width, height],
            'best_score': float(score[best_idx]),
            'r1_updated': self._update_count,
            's_x': s_x,
        }


# ── helpers ───────────────────────────────────────────────────────────────────
def rotate_frame(frame, deg):
    if deg == -90:  return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if deg ==  90:  return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:  return cv2.rotate(frame, cv2.ROTATE_180)
    return frame

def transform_box(x1, y1, x2, y2, rotate, raw_w, raw_h):
    if rotate == -90:
        return y1, raw_w - 1 - x2, y2, raw_w - 1 - x1
    if rotate ==  90:
        return raw_h - 1 - y2, x1, raw_h - 1 - y1, x2
    if rotate == 180:
        return raw_w - 1 - x2, raw_h - 1 - y2, raw_w - 1 - x1, raw_h - 1 - y1
    return x1, y1, x2, y2

def draw_box(frame, bbox, score, color=(0, 230, 255), thickness=2):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    label = f"score:{score:.3f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    lx, ly = max(x1, 0), max(y1 - 4, th + 4)
    cv2.rectangle(frame, (lx, ly - th - 4), (lx + tw + 4, ly + 2), (0,0,0), -1)
    cv2.putText(frame, label, (lx + 2, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


# ── main ──────────────────────────────────────────────────────────────────────
def run(args):
    cfg.merge_from_file(args.cfg)
    cfg.CUDA = torch.cuda.is_available()
    model = ModelBuilder()
    sd = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(sd.get('state_dict', sd))
    model.eval()
    if cfg.CUDA: model = model.cuda()

    tracker = UpdatedTracker(model)

    cap    = cv2.VideoCapture(args.video)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rotate = args.rotate
    out_w  = height if abs(rotate) == 90 else width
    out_h  = width  if abs(rotate) == 90 else height

    print(f"Model  : {args.ckpt}")
    print(f"Video  : {args.video}  ({width}×{height}  {fps:.1f}fps  {total} frames)")
    print(f"Rotate : {rotate:+d}°  output {out_w}×{out_h}")
    print(f"R1 EMA update: thresh={R1_UPDATE_THRESH}  alpha={R1_EMA_ALPHA}  "
          f"max_size_ratio={R1_MAX_SIZE_RATIO}")
    print(f"R2 s_x cap   : {R2_MAX_SX_FACTOR}× s_z_init")
    print(f"R3 size gate : score >= {R3_SIZE_THRESH}")

    base      = os.path.splitext(os.path.basename(args.video))[0]
    out_video = args.out or f"{base}_updated.mp4"
    out_csv   = out_video.replace('.mp4', '_results.csv')
    writer    = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'mp4v'),
                                fps, (out_w, out_h))

    ret, frame0 = cap.read()
    frame0_rot  = rotate_frame(frame0, rotate)
    x1, y1, x2, y2 = args.init_box
    if rotate:
        ix1, iy1, ix2, iy2 = transform_box(x1, y1, x2, y2, rotate, width, height)
    else:
        ix1, iy1, ix2, iy2 = x1, y1, x2, y2
    tracker.init(frame0_rot, [ix1, iy1, ix2 - ix1, iy2 - iy1])
    print(f"Init   : [{ix1},{iy1},{ix2},{iy2}]")

    csv_rows = []
    for fi in range(1, total):
        ret, fr = cap.read()
        if not ret: break
        frr = rotate_frame(fr, rotate)
        out = tracker.track(frr)
        bx, by, bw, bh = out['bbox']
        sc = out['best_score']
        draw_box(frr, [int(bx), int(by), int(bx+bw), int(by+bh)], sc)
        # show s_x search area
        cx_t, cy_t = tracker.center_pos
        sx = int(round(out['s_x']))
        cv2.rectangle(frr,
                      (int(cx_t - sx//2), int(cy_t - sx//2)),
                      (int(cx_t + sx//2), int(cy_t + sx//2)),
                      (0, 255, 255), 1)
        writer.write(frr)
        csv_rows.append([fi, bx, by, bw, bh, sc])
        if fi % 500 == 0:
            print(f"  frame {fi:5d}/{total}  score={sc:.3f}  "
                  f"w={bw:.0f} h={bh:.0f}  "
                  f"s_x={out['s_x']:.0f}  updates={out['r1_updated']}")

    cap.release()
    writer.release()

    with open(out_csv, 'w', newline='') as f:
        csv.writer(f).writerows([['frame','x1','y1','w','h','score']] + csv_rows)

    print(f"\nSaved  : {out_video}")
    print(f"CSV    : {out_csv}")
    print(f"Total template updates (R1): {tracker._update_count}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg',      required=True)
    ap.add_argument('--ckpt',     required=True)
    ap.add_argument('--video',    required=True)
    ap.add_argument('--init_box', type=int, nargs=4,
                    metavar=('X1','Y1','X2','Y2'), required=True)
    ap.add_argument('--rotate',   type=int, default=0, choices=[0,90,-90,180])
    ap.add_argument('--out',      default=None)
    run(ap.parse_args())
