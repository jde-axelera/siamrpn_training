#!/usr/bin/env python3
"""
run_video_inference_advanced.py
SiamRPN++ with dual-template blending and template gallery.

Dual template
-------------
Maintains zf_init (frozen at frame 0, never modified) alongside a slowly
EMA-updated zf_dyn. When both templates score the search image independently
and the results are blended, the frozen init anchors the response against
catastrophic drift while zf_dyn provides gradual appearance adaptation.

Template gallery
----------------
A ring buffer of up to GALLERY_SIZE high-confidence template feature sets.
Every frame all gallery slots are evaluated through the RPN head (backbone +
neck run only ONCE — the light RPN head runs once per slot). The slot whose
penalised score peaks highest wins, giving the tracker access to the best
appearance snapshot seen so far. zf_init is always evaluated alongside the
gallery so the original template can never be crowded out.

Combined behaviour (default: both enabled)
-----------------------------------------
1.  Extract search features xf once (backbone + neck).
2.  Evaluate zf_init  →  score_init, pscore_init, bbox_init.
3.  Evaluate each gallery slot; pick the one with highest peak pscore
    →  score_gal, pscore_gal, bbox_gal.
4.  Blend: score_final = W_INIT * score_init + W_DYN * score_gal
           pen_final   = W_INIT * pen_init   + W_DYN * pen_gal
           pscore_final = recomputed from blended score + penalty
5.  best_idx = argmax(pscore_final).
6.  Loc (pred_bbox) from whichever of {init, gallery winner} scored
    higher at best_idx — most confident template drives the box.
7.  R3: freeze box size when best score < R3_SIZE_THRESH.
8.  R2: cap s_x ≤ R2_MAX_SX_FACTOR × s_z_init.
9.  Gallery + EMA update when score > GALLERY_UPD_THRESH
    AND size_ratio < GALLERY_MAX_SIZE_R.

Modes (via flags):
  --no_gallery   disable gallery; fall back to EMA dual-template only
  --no_dual      disable score blending; gallery winner used directly
  (both off)     reduces to init-only + R2/R3, no template update

Usage:
  python run_video_inference_advanced.py \\
      --cfg   pysot/experiments/siamrpn_r50_alldatasets/config.yaml \\
      --ckpt  pysot/snapshot/all_datasets/best_model.pth \\
      --video ir_crop.mp4 \\
      --init_box 339 148 391 232 \\
      --rotate -90 \\
      --out   ir_crop_advanced.mp4
"""
import argparse, os, sys, csv
from collections import deque
import cv2, numpy as np, torch
import torch.nn.functional as F


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYSOT_DIR  = os.path.join(SCRIPT_DIR, 'pysot')
if os.path.isdir(PYSOT_DIR):
    sys.path.insert(0, PYSOT_DIR)

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.siamrpn_tracker import SiamRPNTracker


# ── tunable constants ─────────────────────────────────────────────────────────
# Fix 1 — Decouple s_x from bbox size.
# The positive feedback loop: large bbox → large s_x → object appears tiny in
# the 255×255 crop → only large anchors have high IoU → regression predicts
# only small delta from large anchor → bbox stays large → repeat.
# Locking s_x to the initial scale breaks this loop entirely.  The model
# always processes the search patch at the same scale as during initialisation,
# so predictions for smaller/larger objects are made at the correct scale.
FIX1_SX_LOCK       = True   # lock s_x to initial scale (breaks feedback loop)
SX_LOCK_FACTOR     = 1.0    # s_x = SX_LOCK_FACTOR × s_x_init  (1.0 = full lock)

R2_MAX_SX_FACTOR   = 2.5    # fallback cap when FIX1_SX_LOCK is False: s_x ≤ this × s_z_init
R3_SIZE_THRESH     = 0.55   # freeze bbox size when score drops below this

GALLERY_SIZE       = 5      # ring-buffer capacity (slots)
GALLERY_UPD_THRESH = 0.75   # min score required to admit a new template
GALLERY_MAX_SIZE_R = 1.5    # reject update if bbox grew > this × initial size
EMA_ALPHA          = 0.015  # blend rate for zf_dyn (used in no-gallery mode)

DUAL_W_INIT        = 0.60   # weight for frozen init template in blend
DUAL_W_DYN         = 0.40   # weight for gallery winner / zf_dyn

GALLERY_MIN_GAP    = 50     # minimum frames between gallery admissions.
                            # Prevents burst-filling the ring buffer with
                            # near-identical templates from one short tracking
                            # run, which causes contamination: the stale
                            # cluster keeps winning after the tracker drifts.
                            # At 25fps this gives ≤20 admissions per 1000f.
# ─────────────────────────────────────────────────────────────────────────────


class AdvancedTracker(SiamRPNTracker):
    """
    SiamRPNTracker with dual-template blending and template gallery.

    Both features are enabled by default and can be disabled independently
    via the use_gallery / use_dual constructor arguments.
    """

    def __init__(self, model, use_gallery=True, use_dual=True):
        super().__init__(model)
        self.use_gallery = use_gallery
        self.use_dual    = use_dual

    # ── initialisation ────────────────────────────────────────────────────────
    def init(self, img, bbox):
        super().init(img, bbox)

        # Geometry bookkeeping
        self._init_size        = self.size.copy()
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        self._init_s_z         = np.sqrt(w_z * h_z)
        self._init_channel_avg = self.channel_average.copy()

        # Dual template: frozen + mutable EMA copy
        self._zf_init = [t.detach().clone() for t in self.model.zf]
        self._zf_dyn  = [t.detach().clone() for t in self.model.zf]

        # Gallery: ring buffer seeded with the init template
        self._gallery = deque(maxlen=GALLERY_SIZE)
        self._gallery.append(
            ([t.detach().clone() for t in self.model.zf], 0)
        )

        self._frame_idx       = 0
        self._ema_updates     = 0
        self._gallery_adds    = 0
        self._last_gallery_add = -GALLERY_MIN_GAP   # allow first add immediately

    # ── internal helpers ──────────────────────────────────────────────────────
    def _extract_xf(self, x_crop):
        """Backbone + neck: run once per frame."""
        with torch.no_grad():
            return self.model.neck(self.model.backbone(x_crop))

    def _rpn_forward(self, zf, xf):
        """RPN head only — cheap, called once per gallery slot."""
        with torch.no_grad():
            cls, loc = self.model.rpn_head(zf, xf)
        return cls, loc

    def _decode(self, cls, loc, scale_z):
        """
        Convert raw RPN tensors to:
          score     - fg probability for each of the 3125 anchors  (3125,)
          pred_bbox - decoded (cx, cy, w, h) candidates            (4, 3125)
          penalty   - size/ratio consistency penalty                (3125,)
          pscore    - penalised + windowed score (used for ranking) (3125,)
        """
        score     = self._convert_score(cls)
        pred_bbox = self._convert_bbox(loc, self.anchors)

        def change(r): return np.maximum(r, 1. / r)
        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     sz(self.size[0] * scale_z, self.size[1] * scale_z))
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore  = penalty * score
        pscore  = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                  self.window * cfg.TRACK.WINDOW_INFLUENCE
        return score, pred_bbox, penalty, pscore

    # ── main tracking step ────────────────────────────────────────────────────
    def track(self, img):

        if FIX1_SX_LOCK:
            # Fix 1: Decouple s_x from current bbox estimate.
            # Lock s_x and scale_z to the initial values so the model always
            # processes the search patch at the scale it saw during init.
            # This breaks the positive feedback loop: bbox grows → s_x grows
            # → target appears tiny in crop → only large anchors fire →
            # bbox refuses to shrink.  With s_x locked, the target appears
            # at its true relative size, allowing the model to predict the
            # correct (smaller) bbox when it moves away.
            s_x     = SX_LOCK_FACTOR * self._init_s_z * (
                cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
            scale_z = cfg.TRACK.EXEMPLAR_SIZE / (s_x * cfg.TRACK.EXEMPLAR_SIZE
                                                 / cfg.TRACK.INSTANCE_SIZE)
        else:
            # Original path: R2 hard cap (retained for ablation via --no_fix1)
            w_z     = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
            h_z     = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
            s_z     = np.sqrt(w_z * h_z)
            scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z

            s_x_raw = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
            s_x     = min(s_x_raw, R2_MAX_SX_FACTOR * self._init_s_z)
            if s_x < s_x_raw:
                s_z_eff = s_x * cfg.TRACK.EXEMPLAR_SIZE / cfg.TRACK.INSTANCE_SIZE
                scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z_eff

        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        # Backbone + neck — run ONCE regardless of gallery size
        xf = self._extract_xf(x_crop)

        # ── evaluate frozen init template (always) ────────────────────────────
        cls_i, loc_i = self._rpn_forward(self._zf_init, xf)
        score_i, bbox_i, pen_i, pscore_i = self._decode(cls_i, loc_i, scale_z)

        # ── gallery: race all slots, keep the winner ───────────────────────────
        # Initialise to init template so init can also win the race.
        gal_score  = score_i
        gal_bbox   = bbox_i
        gal_pen    = pen_i
        gal_pscore = pscore_i
        gal_winner = -1     # -1 → init template won

        if self.use_gallery:
            for gidx, (zf_g, _) in enumerate(self._gallery):
                cls_g, loc_g = self._rpn_forward(zf_g, xf)
                s_g, b_g, p_g, ps_g = self._decode(cls_g, loc_g, scale_z)
                if ps_g.max() > gal_pscore.max():
                    gal_score, gal_bbox, gal_pen, gal_pscore = s_g, b_g, p_g, ps_g
                    gal_winner = gidx

        # ── dual-template score blend ─────────────────────────────────────────
        if self.use_dual:
            if self.use_gallery:
                # Blend frozen init with gallery winner
                score_b = DUAL_W_INIT * score_i + DUAL_W_DYN * gal_score
                pen_b   = DUAL_W_INIT * pen_i   + DUAL_W_DYN * gal_pen
            else:
                # Blend frozen init with EMA-updated dynamic template
                cls_d, loc_d = self._rpn_forward(self._zf_dyn, xf)
                score_d, _, pen_d, _ = self._decode(cls_d, loc_d, scale_z)
                score_b = DUAL_W_INIT * score_i + DUAL_W_DYN * score_d
                pen_b   = DUAL_W_INIT * pen_i   + DUAL_W_DYN * pen_d

            # Recompute pscore from blended score + blended penalty
            pscore_final = (pen_b * score_b) * (1 - cfg.TRACK.WINDOW_INFLUENCE) \
                           + self.window * cfg.TRACK.WINDOW_INFLUENCE
            score_final  = score_b
            pen_final    = pen_b
        else:
            # No blending — use gallery winner (or init if gallery is off)
            pscore_final = gal_pscore
            score_final  = gal_score
            pen_final    = gal_pen

        best_idx = np.argmax(pscore_final)

        # Loc from whichever template scored higher at best_idx —
        # the more confident template drives the bounding-box regression.
        if gal_score[best_idx] >= score_i[best_idx]:
            pred_bbox = gal_bbox
        else:
            pred_bbox = bbox_i

        bbox_cand = pred_bbox[:, best_idx] / scale_z
        lr        = pen_final[best_idx] * score_final[best_idx] * cfg.TRACK.LR

        cx = bbox_cand[0] + self.center_pos[0]
        cy = bbox_cand[1] + self.center_pos[1]

        # R3 — size gate
        if score_final[best_idx] >= R3_SIZE_THRESH:
            width  = self.size[0] * (1 - lr) + bbox_cand[2] * lr
            height = self.size[1] * (1 - lr) + bbox_cand[3] * lr
        else:
            width, height = self.size[0], self.size[1]

        cx, cy, width, height = self._bbox_clip(cx, cy, width, height,
                                                img.shape[:2])

        # ── template update: gallery admission + EMA ──────────────────────────
        size_ratio = max(width  / self._init_size[0],
                         height / self._init_size[1])
        frames_since_last = self._frame_idx - self._last_gallery_add
        if (score_final[best_idx] > GALLERY_UPD_THRESH and
                size_ratio < GALLERY_MAX_SIZE_R and
                frames_since_last >= GALLERY_MIN_GAP):
            new_z = self.get_subwindow(img, np.array([cx, cy]),
                                       cfg.TRACK.EXEMPLAR_SIZE,
                                       round(self._init_s_z),
                                       self._init_channel_avg)
            with torch.no_grad():
                new_zf = self.model.neck(self.model.backbone(new_z))
            new_zf = [t.detach() for t in new_zf]

            # Admit to gallery (ring buffer auto-evicts oldest slot)
            self._gallery.append((new_zf, self._frame_idx))
            self._gallery_adds += 1
            self._last_gallery_add = self._frame_idx

            # Keep zf_dyn updated too (used in no-gallery + dual mode)
            self._zf_dyn = [(1 - EMA_ALPHA) * o + EMA_ALPHA * n
                            for o, n in zip(self._zf_dyn, new_zf)]
            self._ema_updates += 1

        self.center_pos = np.array([cx, cy])
        self.size       = np.array([width, height])
        self._frame_idx += 1

        return {
            'bbox':         [cx - width/2, cy - height/2, width, height],
            'best_score':   float(score_final[best_idx]),
            's_x':          s_x,
            'gallery_size': len(self._gallery),
            'gallery_adds': self._gallery_adds,
            'gallery_winner': gal_winner,   # -1=init, 0..N-1=gallery slot
            'ema_updates':  self._ema_updates,
        }


# ── helpers (identical to run_video_inference_updated.py) ─────────────────────
def rotate_frame(frame, deg):
    if deg == -90: return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if deg ==  90: return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180: return cv2.rotate(frame, cv2.ROTATE_180)
    return frame

def transform_box(x1, y1, x2, y2, rotate, raw_w, raw_h):
    if rotate == -90: return y1, raw_w - 1 - x2, y2, raw_w - 1 - x1
    if rotate ==  90: return raw_h - 1 - y2, x1, raw_h - 1 - y1, x2
    if rotate == 180: return raw_w - 1 - x2, raw_h - 1 - y2, raw_w - 1 - x1, raw_h - 1 - y1
    return x1, y1, x2, y2

def draw_box(frame, bbox, score, gallery_winner, color=(0, 230, 255), thickness=2):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    # Cyan = gallery template winning; yellow = init template winning
    box_color = (255, 200, 0) if gallery_winner >= 0 else color
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
    label = f"s:{score:.3f} gal:{gallery_winner}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    lx, ly = max(x1, 0), max(y1 - 4, th + 4)
    cv2.rectangle(frame, (lx, ly - th - 4), (lx + tw + 4, ly + 2), (0, 0, 0), -1)
    cv2.putText(frame, label, (lx + 2, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA)
    return frame


# ── main ─────────────────────────────────────────────────────────────────────
def run(args):
    cfg.merge_from_file(args.cfg)
    cfg.CUDA = torch.cuda.is_available()

    model = ModelBuilder()
    sd    = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(sd.get('state_dict', sd))
    model.eval()
    if cfg.CUDA:
        model = model.cuda()

    # Apply Fix 1 setting from CLI
    global FIX1_SX_LOCK
    FIX1_SX_LOCK = not args.no_fix1

    use_gallery = not args.no_gallery
    use_dual    = not args.no_dual
    tracker     = AdvancedTracker(model, use_gallery=use_gallery, use_dual=use_dual)

    cap    = cv2.VideoCapture(args.video)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rotate = args.rotate
    out_w  = height if abs(rotate) == 90 else width
    out_h  = width  if abs(rotate) == 90 else height

    print(f"Model     : {args.ckpt}")
    print(f"Video     : {args.video}  ({width}×{height}  {fps:.1f}fps  {total} frames)")
    print(f"Rotate    : {rotate:+d}°  output {out_w}×{out_h}")
    print(f"Fix 1 s_x : {'LOCKED to ' + str(SX_LOCK_FACTOR) + '× init scale' if FIX1_SX_LOCK else 'OFF  (R2 cap=' + str(R2_MAX_SX_FACTOR) + '× init)'}")
    print(f"Gallery   : {'ON  size=' + str(GALLERY_SIZE) + '  thresh=' + str(GALLERY_UPD_THRESH) if use_gallery else 'OFF'}")
    print(f"Dual tmpl : {'ON  W_init=' + str(DUAL_W_INIT) + '  W_dyn=' + str(DUAL_W_DYN) + '  EMA_α=' + str(EMA_ALPHA) if use_dual else 'OFF'}")
    print(f"R3 size gate: score >= {R3_SIZE_THRESH}")

    base      = os.path.splitext(os.path.basename(args.video))[0]
    suffix    = '_fix1' if FIX1_SX_LOCK else '_advanced'
    out_video = args.out or f"{base}{suffix}.mp4"
    out_csv   = out_video.replace('.mp4', '_results.csv')
    writer    = cv2.VideoWriter(out_video,
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                fps, (out_w, out_h))

    ret, frame0 = cap.read()
    frame0_rot  = rotate_frame(frame0, rotate)
    x1, y1, x2, y2 = args.init_box
    if rotate:
        ix1, iy1, ix2, iy2 = transform_box(x1, y1, x2, y2, rotate, width, height)
    else:
        ix1, iy1, ix2, iy2 = x1, y1, x2, y2
    tracker.init(frame0_rot, [ix1, iy1, ix2 - ix1, iy2 - iy1])
    print(f"Init bbox : [{ix1},{iy1},{ix2},{iy2}]")

    csv_rows = []
    for fi in range(1, total):
        ret, fr = cap.read()
        if not ret:
            break
        frr = rotate_frame(fr, rotate)
        out = tracker.track(frr)

        bx, by, bw, bh = out['bbox']
        sc = out['best_score']
        gw = out['gallery_winner']

        draw_box(frr, [int(bx), int(by), int(bx + bw), int(by + bh)], sc, gw)

        # Search area rectangle (yellow dashed look via cyan)
        cx_t, cy_t = tracker.center_pos
        sx = int(round(out['s_x']))
        cv2.rectangle(frr,
                      (int(cx_t - sx // 2), int(cy_t - sx // 2)),
                      (int(cx_t + sx // 2), int(cy_t + sx // 2)),
                      (0, 255, 255), 1)

        # HUD
        hud = (f"f:{fi:5d}  sc:{sc:.3f}  "
               f"gal:{out['gallery_size']}/{GALLERY_SIZE}  "
               f"adds:{out['gallery_adds']}  ema:{out['ema_updates']}")
        cv2.putText(frr, hud, (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        writer.write(frr)
        csv_rows.append([fi, bx, by, bw, bh, sc,
                         out['gallery_size'], out['gallery_adds'], gw])

        if fi % 500 == 0:
            print(f"  frame {fi:5d}/{total}  score={sc:.3f}  "
                  f"w={bw:.0f} h={bh:.0f}  s_x={out['s_x']:.0f}  "
                  f"gallery={out['gallery_size']}  adds={out['gallery_adds']}  "
                  f"ema={out['ema_updates']}  winner={'init' if gw < 0 else f'slot{gw}'}")

    cap.release()
    writer.release()

    with open(out_csv, 'w', newline='') as f:
        csv.writer(f).writerows(
            [['frame', 'x', 'y', 'w', 'h', 'score',
              'gallery_size', 'gallery_adds', 'gallery_winner']]
            + csv_rows
        )

    # Summary
    wins = [r[8] for r in csv_rows]
    init_wins    = sum(1 for w in wins if w < 0)
    gallery_wins = sum(1 for w in wins if w >= 0)
    print(f"\nSaved     : {out_video}")
    print(f"CSV       : {out_csv}")
    print(f"Gallery additions : {tracker._gallery_adds}")
    print(f"EMA updates       : {tracker._ema_updates}")
    print(f"Init template wins: {init_wins:5d} ({100*init_wins/len(wins):.1f}%)")
    print(f"Gallery wins      : {gallery_wins:5d} ({100*gallery_wins/len(wins):.1f}%)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg',       required=True)
    ap.add_argument('--ckpt',      required=True)
    ap.add_argument('--video',     required=True)
    ap.add_argument('--init_box',  type=int, nargs=4,
                    metavar=('X1', 'Y1', 'X2', 'Y2'), required=True)
    ap.add_argument('--rotate',    type=int, default=0, choices=[0, 90, -90, 180])
    ap.add_argument('--out',       default=None)
    ap.add_argument('--no_gallery', action='store_true',
                    help='disable template gallery; use EMA dual-template only')
    ap.add_argument('--no_dual',    action='store_true',
                    help='disable dual-template blending; gallery winner used directly')
    ap.add_argument('--no_fix1',    action='store_true',
                    help='disable Fix1 s_x locking; revert to R2 hard cap at 2.5× init')
    run(ap.parse_args())
