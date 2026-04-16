#!/usr/bin/env python3
"""
debug_video.py  —  Per-frame debug visualization video for SiamRPN++.

Renders a 3-panel annotated video showing:
  Panel L (full frame):  rotated frame + bbox (green) + search area (yellow)
                         + score bar + frame counter + template thumbnail
  Panel M (search crop): 255×255 search patch with correlation heatmap overlaid
                         + best anchor red-cross + decoded best bbox (blue)
  Panel R (regression):  dw heatmap (left) and dh heatmap (right) side-by-side
                         on search crop — shows where model predicts size changes

Usage:
  python debug_video.py \
      --cfg   pysot/experiments/siamrpn_r50_alldatasets/config.yaml \
      --ckpt  pysot/snapshot/all_datasets/best_model.pth \
      --video ir_crop.mp4 \
      --out   debug_annotated.mp4 \
      --init_box 339 148 391 232
"""
import os, sys, argparse
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

SCORE_SIZE = 25
ANCHOR_NUM = 5
SEARCH_SZ  = 255


# ── colour helpers ────────────────────────────────────────────────────────────

def jet_overlay(bgr, heat2d, alpha=0.60):
    """Overlay normalised 2-D heatmap (jet) on BGR image."""
    h, w   = bgr.shape[:2]
    mn, mx = heat2d.min(), heat2d.max()
    norm   = (heat2d - mn) / (mx - mn + 1e-8)
    norm_r = cv2.resize(norm, (w, h), interpolation=cv2.INTER_LINEAR)
    colmap = cv2.applyColorMap((norm_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 1 - alpha, colmap, alpha, 0)

def score_bar(canvas, score, x, y, w=120, h=12):
    """Draw a horizontal score bar on canvas."""
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (60, 60, 60), -1)
    fill = int(w * np.clip(score, 0, 1))
    color = (0, 200, 80) if score > 0.5 else (0, 165, 255) if score > 0.25 else (0, 0, 220)
    cv2.rectangle(canvas, (x, y), (x + fill, y + h), color, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (200, 200, 200), 1)
    cv2.putText(canvas, f'{score:.3f}', (x + w + 5, y + h - 1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

def put_text(canvas, text, x, y, scale=0.42, color=(220,220,220), thickness=1):
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


# ── capture tracker ───────────────────────────────────────────────────────────

class CaptureTracker(SiamRPNTracker):
    """Like SiamRPNTracker but captures all intermediates every frame."""

    def __init__(self, model):
        super().__init__(model)
        self.last = {}
        self._z_np = None
        self._neck_cap = {'xf': None}
        def _hook(mod, inp, out):
            sp = out[0].shape[-1] if isinstance(out, list) else out.shape[-1]
            if sp > 10:
                self._neck_cap['xf'] = [t.detach() for t in out] \
                    if isinstance(out, list) else [out.detach()]
        self._hook = model.neck.register_forward_hook(_hook)

    def remove_hook(self):
        self._hook.remove()

    def init(self, img, bbox):
        super().init(img, bbox)
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))
        z_t = self.get_subwindow(img, self.center_pos,
                                 cfg.TRACK.EXEMPLAR_SIZE, s_z,
                                 self.channel_average)
        arr = z_t.squeeze(0).permute(1,2,0).cpu().numpy()
        self._z_np = np.clip(arr, 0, 255).astype(np.uint8)

    def track(self, img):
        w_z     = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z     = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z     = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x     = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop  = self.get_subwindow(img, self.center_pos,
                                     cfg.TRACK.INSTANCE_SIZE,
                                     round(s_x), self.channel_average)

        outputs   = self.model.track(x_crop)
        score     = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):  return np.maximum(r, 1. / r)
        def sz(w, h):
            pad = (w + h) * 0.5; return np.sqrt((w + pad) * (h + pad))

        s_c     = change(sz(pred_bbox[2,:], pred_bbox[3,:]) /
                         sz(self.size[0]*scale_z, self.size[1]*scale_z))
        r_c     = change((self.size[0]/self.size[1]) /
                         (pred_bbox[2,:]/pred_bbox[3,:]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore  = penalty * score
        pscore  = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                  self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)
        bbox     = pred_bbox[:, best_idx] / scale_z
        lr       = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx     = bbox[0] + self.center_pos[0]
        cy     = bbox[1] + self.center_pos[1]
        width  = self.size[0] * (1-lr) + bbox[2]*lr
        height = self.size[1] * (1-lr) + bbox[3]*lr
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        # extract intermediates
        loc_np = outputs['loc'].detach().cpu().numpy()[0]  # (20,25,25)
        loc_4d = loc_np.reshape(4, ANCHOR_NUM, SCORE_SIZE, SCORE_SIZE)
        dw_map = loc_4d[2].mean(0)
        dh_map = loc_4d[3].mean(0)
        score_map = score.reshape(ANCHOR_NUM, SCORE_SIZE, SCORE_SIZE)
        n = SCORE_SIZE * SCORE_SIZE
        ba = best_idx // n
        br = (best_idx % n) // SCORE_SIZE
        bc = best_idx % SCORE_SIZE

        x_np = x_crop.squeeze(0).permute(1,2,0).cpu().numpy()
        x_np = np.clip(x_np, 0, 255).astype(np.uint8)

        pb   = pred_bbox[:, best_idx]
        self.last = dict(
            x_np      = x_np,
            score_map = score_map.max(0),    # (25,25) max over anchors
            dw_map    = np.clip(dw_map, -2, 2),
            dh_map    = np.clip(dh_map, -2, 2),
            best_px   = int((bc + 0.5) * SEARCH_SZ / SCORE_SIZE),
            best_py   = int((br + 0.5) * SEARCH_SZ / SCORE_SIZE),
            pb        = pb,
            s_x       = s_x,
        )
        return {'bbox': [cx - width/2, cy - height/2, width, height],
                'best_score': float(score[best_idx])}


# ── frame renderer ────────────────────────────────────────────────────────────

PANEL_H = 480   # height of all panels
LEFT_W  = 480   # left panel width  (will be resized from rotated frame)
MID_W   = 280   # centre + right each
RIGHT_W = 280

def render_frame(frame_rot, tracker, score, frame_idx, out_w, out_h):
    """Build the 3-panel debug frame."""
    d = tracker.last
    TOTAL_W = LEFT_W + MID_W + RIGHT_W + 2   # 2px separator

    canvas = np.zeros((PANEL_H, TOTAL_W, 3), dtype=np.uint8)

    # ── LEFT panel: full frame ────────────────────────────────────────────────
    bx, by, bw, bh = [tracker.center_pos[0] - tracker.size[0]/2,
                       tracker.center_pos[1] - tracker.size[1]/2,
                       tracker.size[0], tracker.size[1]]
    frame_d = frame_rot.copy()
    # bbox
    cv2.rectangle(frame_d, (int(bx), int(by)), (int(bx+bw), int(by+bh)), (0,230,80), 2)
    # search area
    sx = int(round(d['s_x']))
    cx_i = int(tracker.center_pos[0]); cy_i = int(tracker.center_pos[1])
    cv2.rectangle(frame_d,
                  (cx_i - sx//2, cy_i - sx//2),
                  (cx_i + sx//2, cy_i + sx//2), (0,230,230), 1)
    # template thumbnail (top-right corner)
    z_s = 48
    z_th = cv2.resize(tracker._z_np, (z_s, z_s))
    fh, fw = frame_d.shape[:2]
    frame_d[4:4+z_s, fw-z_s-4:fw-4] = z_th
    cv2.rectangle(frame_d, (fw-z_s-5, 3), (fw-3, z_s+5), (200,200,0), 1)

    left_panel = cv2.resize(frame_d, (LEFT_W, PANEL_H))

    # annotations
    put_text(left_panel, f'Frame {frame_idx:5d}', 8, 18, 0.50, (220,220,50))
    put_text(left_panel, f'Score', 8, 40, 0.38)
    score_bar(left_panel, score, 55, 28, 90, 13)
    put_text(left_panel, f'W={bw:.0f} H={bh:.0f}', 8, 58, 0.38, (160,220,160))
    put_text(left_panel, f's_x={d["s_x"]:.0f}', 8, 74, 0.38, (160,200,220))
    put_text(left_panel, 'template', LEFT_W - 52, PANEL_H - 6, 0.30, (200,200,0))

    canvas[:, :LEFT_W] = left_panel
    canvas[:, LEFT_W] = 80   # separator

    # ── CENTRE panel: search crop + cls heatmap ───────────────────────────────
    heat_blend = jet_overlay(d['x_np'], d['score_map'], alpha=0.60)
    # best bbox on search
    pb  = d['pb']
    c_x = SEARCH_SZ/2 + pb[0]; c_y = SEARCH_SZ/2 + pb[1]
    bx1 = int(np.clip(c_x - pb[2]/2, 0, SEARCH_SZ-1))
    by1 = int(np.clip(c_y - pb[3]/2, 0, SEARCH_SZ-1))
    bx2 = int(np.clip(c_x + pb[2]/2, 0, SEARCH_SZ-1))
    by2 = int(np.clip(c_y + pb[3]/2, 0, SEARCH_SZ-1))
    cv2.rectangle(heat_blend, (bx1, by1), (bx2, by2), (255, 50, 50), 2)
    # best anchor cross
    cv2.line(heat_blend, (d['best_px']-8, d['best_py']), (d['best_px']+8, d['best_py']), (0,0,255), 2)
    cv2.line(heat_blend, (d['best_px'], d['best_py']-8), (d['best_px'], d['best_py']+8), (0,0,255), 2)

    mid_panel = cv2.resize(heat_blend, (MID_W, PANEL_H - 20))
    mid_canvas = np.zeros((PANEL_H, MID_W, 3), dtype=np.uint8)
    mid_canvas[20:, :] = mid_panel
    put_text(mid_canvas, 'CLS heatmap (fg prob, max/anchor)', 4, 14, 0.35, (180,220,255))

    canvas[:, LEFT_W+1:LEFT_W+1+MID_W] = mid_canvas
    canvas[:, LEFT_W+1+MID_W] = 80  # separator

    # ── RIGHT panel: dw (top) + dh (bottom) ──────────────────────────────────
    dw_blend = jet_overlay(d['x_np'], d['dw_map'], alpha=0.65)
    dh_blend = jet_overlay(d['x_np'], d['dh_map'], alpha=0.65)
    half_h   = (PANEL_H - 30) // 2
    dw_s     = cv2.resize(dw_blend, (RIGHT_W, half_h))
    dh_s     = cv2.resize(dh_blend, (RIGHT_W, half_h))

    right_canvas = np.zeros((PANEL_H, RIGHT_W, 3), dtype=np.uint8)
    right_canvas[20:20+half_h, :] = dw_s
    right_canvas[20+half_h+8:20+half_h*2+8, :] = dh_s
    cv2.line(right_canvas, (0, 20+half_h+4), (RIGHT_W, 20+half_h+4), (80,80,80), 1)

    dw_mean = d['dw_map'].mean()
    dh_mean = d['dh_map'].mean()
    put_text(right_canvas, f'dw map  mean={dw_mean:.3f} exp={np.exp(dw_mean):.2f}x', 4, 14, 0.32, (180,255,180))
    put_text(right_canvas, f'dh map  mean={dh_mean:.3f} exp={np.exp(dh_mean):.2f}x', 4, 20+half_h+2, 0.32, (255,200,180))

    canvas[:, LEFT_W+2+MID_W:] = right_canvas

    return canvas


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg',      required=True)
    ap.add_argument('--ckpt',     required=True)
    ap.add_argument('--video',    required=True)
    ap.add_argument('--init_box', type=int, nargs=4,
                    metavar=('X1','Y1','X2','Y2'), default=[339,148,391,232])
    ap.add_argument('--rotate',   type=int, default=-90)
    ap.add_argument('--out',      default='debug_annotated.mp4')
    args = ap.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.CUDA = torch.cuda.is_available()
    device   = 'cuda' if cfg.CUDA else 'cpu'
    model    = ModelBuilder()
    sd       = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(sd.get('state_dict', sd))
    model.eval().to(device)
    print(f"Model loaded  : {args.ckpt}  device={device}")

    tracker = CaptureTracker(model)

    cap    = cv2.VideoCapture(args.video)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rotate = args.rotate
    print(f"Video         : {args.video}  {width}×{height}  {fps:.1f}fps  {total} frames")

    # output canvas size
    TOTAL_W = LEFT_W + MID_W + RIGHT_W + 2
    writer  = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'),
                              fps, (TOTAL_W, PANEL_H))

    # init
    ret, frame0 = cap.read()
    if rotate == -90: frame0_rot = cv2.rotate(frame0, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:             frame0_rot = frame0
    x1, y1, x2, y2 = args.init_box
    if rotate == -90:
        ix1, iy1 = y1, width - 1 - x2
        ix2, iy2 = y2, width - 1 - x1
    else:
        ix1, iy1, ix2, iy2 = x1, y1, x2, y2

    tracker.init(frame0_rot, [ix1, iy1, ix2-ix1, iy2-iy1])

    # Fake 'last' for frame 0
    tracker.last = {
        'x_np': np.zeros((SEARCH_SZ, SEARCH_SZ, 3), np.uint8),
        'score_map': np.zeros((SCORE_SIZE, SCORE_SIZE)),
        'dw_map': np.zeros((SCORE_SIZE, SCORE_SIZE)),
        'dh_map': np.zeros((SCORE_SIZE, SCORE_SIZE)),
        'best_px': SEARCH_SZ//2, 'best_py': SEARCH_SZ//2,
        'pb': np.zeros(4),
        's_x': 0.0,
    }
    init_frame = render_frame(frame0_rot, tracker, 1.0, 0, TOTAL_W, PANEL_H)
    writer.write(init_frame)

    for fi in range(1, total):
        ret, fr = cap.read()
        if not ret: break
        if rotate == -90: fr_rot = cv2.rotate(fr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:             fr_rot = fr
        out  = tracker.track(fr_rot)
        sc   = out['best_score']
        frm  = render_frame(fr_rot, tracker, sc, fi, TOTAL_W, PANEL_H)
        writer.write(frm)
        if fi % 500 == 0:
            print(f"  frame {fi:5d}/{total}  score={sc:.3f}  "
                  f"dw_mean={tracker.last['dw_map'].mean():.3f}")

    cap.release()
    writer.release()
    tracker.remove_hook()
    print(f"\nDone → {args.out}")


if __name__ == '__main__':
    main()
