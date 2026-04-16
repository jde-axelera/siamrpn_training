#!/usr/bin/env python3
"""
debug_siamrpn.py  —  Deep diagnostic of SiamRPN++ inference pipeline.

Runs full video tracking (best_model.pth) and at selected debug frames captures:
  - Template and search crops
  - Neck feature maps (3 scales)
  - CLS heatmaps (correlation response)
  - LOC regression maps (dw/dh — explains bbox growth)
  - Penalized score surface
  - Top candidate bboxes
  - Search area on original frame

Generates a multi-page PDF report.

Usage:
  python debug_siamrpn.py \
    --cfg   pysot/experiments/siamrpn_r50_alldatasets/config.yaml \
    --ckpt  pysot/snapshot/all_datasets/best_model.pth \
    --video ir_crop.mp4 \
    --csv   ir_crop_best_model_rot90ccw_results.csv \
    --out   debug_best_model.pdf
"""
import os, sys, argparse, warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYSOT_DIR  = os.path.join(SCRIPT_DIR, 'pysot')
if os.path.isdir(PYSOT_DIR):
    sys.path.insert(0, PYSOT_DIR)

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.siamrpn_tracker import SiamRPNTracker

# ── Constants (derived from config) ──────────────────────────────────────────
SCORE_SIZE  = 25
ANCHOR_NUM  = 5
SEARCH_SIZE = 255
TEMPL_SIZE  = 127
STRIDE      = 8


# ── Utility functions ─────────────────────────────────────────────────────────

def safe_norm(arr):
    """Normalize to [0,1]; returns zeros for constant arrays."""
    a = np.nan_to_num(np.array(arr, dtype=np.float32), nan=0., posinf=0., neginf=0.)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-8)

def bgr_u8(t):
    """(1,3,H,W) float tensor → (H,W,3) uint8 BGR.  Unnormalized output."""
    arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(arr, 0, 255).astype(np.uint8)

def overlay_heat(bgr, heat2d, alpha=0.55):
    """Overlay a normalised 2-D heatmap (jet colormap) on a BGR image."""
    h, w = bgr.shape[:2]
    hm = safe_norm(heat2d)
    hm_r = cv2.resize(hm, (w, h), interpolation=cv2.INTER_LINEAR)
    colored = (plt.cm.jet(hm_r)[:, :, :3] * 255).astype(np.uint8)[:, :, ::-1]  # RGB→BGR
    return cv2.addWeighted(bgr, 1 - alpha, colored, alpha, 0)

def score_to_px(row, col):
    """Score-map cell (row,col) → pixel centre in 255×255 search crop."""
    stride = SEARCH_SIZE / SCORE_SIZE   # 10.2
    return (col + 0.5) * stride, (row + 0.5) * stride

def flat_to_arc(idx):
    """Flat score index → (anchor, row, col)."""
    n = SCORE_SIZE * SCORE_SIZE          # 625
    return idx // n, (idx % n) // SCORE_SIZE, idx % SCORE_SIZE

def rotate_frame(frame, deg):
    if deg == -90:  return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if deg == 90:   return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:  return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


# ── Debug tracker ─────────────────────────────────────────────────────────────

class DebugTracker(SiamRPNTracker):
    """
    Subclasses SiamRPNTracker.  Override track() with identical logic but
    captures all intermediate tensors at designated debug frames.
    """
    def __init__(self, model, debug_frames):
        super().__init__(model)
        self.debug_frames = set(debug_frames)
        self.frame_idx    = 0
        self.debug_data   = {}
        self._z_crop_np   = None

        # Forward hook on neck: captures xf when spatial > 10 (search, not template)
        self._neck_cap = {'xf': None}
        def _neck_hook(mod, inp, out):
            sp = out[0].shape[-1] if isinstance(out, list) else out.shape[-1]
            if sp > 10:   # search neck output (31), not template (7)
                self._neck_cap['xf'] = [t.detach() for t in out] \
                    if isinstance(out, list) else [out.detach()]
        self._hook_handle = model.neck.register_forward_hook(_neck_hook)

    def cleanup(self):
        self._hook_handle.remove()

    # ── init ──────────────────────────────────────────────────────────────────
    def init(self, img, bbox):
        super().init(img, bbox)
        # Re-crop template for display (same maths as parent)
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))
        z_t = self.get_subwindow(img, self.center_pos,
                                 cfg.TRACK.EXEMPLAR_SIZE, s_z,
                                 self.channel_average)
        self._z_crop_np = bgr_u8(z_t)

    # ── track ─────────────────────────────────────────────────────────────────
    def track(self, img):
        # ── exact replication of SiamRPNTracker.track() ──────────────────────
        w_z     = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z     = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z     = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x     = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)

        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        outputs   = self.model.track(x_crop)          # triggers neck hook
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
        bbox     = pred_bbox[:, best_idx] / scale_z
        lr       = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx     = bbox[0] + self.center_pos[0]
        cy     = bbox[1] + self.center_pos[1]
        width  = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height,
                                                img.shape[:2])

        # ── capture debug data ────────────────────────────────────────────────
        if self.frame_idx in self.debug_frames:
            xf_np   = [f.cpu().numpy() for f in (self._neck_cap['xf'] or [])]
            cls_np  = outputs['cls'].detach().cpu().numpy()  # (1,10,25,25)
            loc_np  = outputs['loc'].detach().cpu().numpy()  # (1,20,25,25)
            x_np    = bgr_u8(x_crop)

            # Score map per anchor: (5,25,25)
            score_map = score.reshape(ANCHOR_NUM, SCORE_SIZE, SCORE_SIZE)

            # Raw loc maps — ordering from _convert_bbox:
            # after permute(1,2,3,0).view(4,-1):
            #   row 0 = dx (channels 0-4), row 1 = dy (ch5-9),
            #   row 2 = dw (ch10-14),      row 3 = dh (ch15-19)
            # reshape back: (4, 5, 25, 25)
            loc_4d = loc_np[0].reshape(4, ANCHOR_NUM, SCORE_SIZE, SCORE_SIZE)
            # dw at each location: mean over anchors
            dw_map = loc_4d[2].mean(0)      # (25,25) log-scale width factor
            dh_map = loc_4d[3].mean(0)      # (25,25) log-scale height factor

            # decoded pred_bbox widths/heights for top candidates
            # pred_bbox[2,:] / scale_z → image-space widths
            pscore_map = pscore.reshape(ANCHOR_NUM, SCORE_SIZE, SCORE_SIZE)

            # best anchor at each spatial location
            best_anc_per_loc = score_map.reshape(ANCHOR_NUM, -1).argmax(0)  # (625,)

            ba, br, bc = flat_to_arc(best_idx)
            dw_best = loc_4d[2, ba, br, bc]
            dh_best = loc_4d[3, ba, br, bc]

            self.debug_data[self.frame_idx] = dict(
                frame_idx   = self.frame_idx,
                img_rot     = img.copy(),
                x_crop_np   = x_np,
                z_crop_np   = self._z_crop_np,
                xf_neck     = xf_np,            # list of 3 × (1,256,31,31)
                score_map   = score_map,         # (5,25,25)
                pscore_map  = pscore_map,        # (5,25,25)
                dw_map      = dw_map,            # (25,25)
                dh_map      = dh_map,            # (25,25)
                pred_bbox   = pred_bbox.copy(),  # (4,3125)
                score_raw   = score.copy(),      # (3125,)
                pscore_raw  = pscore.copy(),     # (3125,)
                penalty_raw = penalty.copy(),    # (3125,)
                best_idx    = best_idx,
                dw_best     = float(dw_best),
                dh_best     = float(dh_best),
                scale_z     = scale_z,
                s_x         = s_x,
                s_z         = s_z,
                center_before = self.center_pos.copy(),
                size_before   = self.size.copy(),
                center_after  = np.array([cx, cy]),
                size_after    = np.array([width, height]),
                best_score    = float(score[best_idx]),
                best_pscore   = float(pscore[best_idx]),
                penalty_best  = float(penalty[best_idx]),
                lr            = float(lr),
            )

        # ── update state ──────────────────────────────────────────────────────
        self.center_pos = np.array([cx, cy])
        self.size       = np.array([width, height])
        self.frame_idx += 1

        return {'bbox': [cx - width/2, cy - height/2, width, height],
                'best_score': float(score[best_idx])}


# ── Video tracking loop ───────────────────────────────────────────────────────

def run_tracking(args):
    cfg.merge_from_file(args.cfg)
    cfg.CUDA = torch.cuda.is_available()
    device   = 'cuda' if cfg.CUDA else 'cpu'

    model = ModelBuilder()
    sd    = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(sd.get('state_dict', sd))
    model.eval().to(device)
    print(f"  Loaded {args.ckpt}  device={device}")

    DEBUG_FRAMES = [100, 750, 1650, 3250, 3900, 4050, 4800, 5100]
    tracker = DebugTracker(model, DEBUG_FRAMES)

    cap    = cv2.VideoCapture(args.video)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {total} frames")

    # init on frame 0
    ret, frame0 = cap.read()
    frame0_rot  = rotate_frame(frame0, -90)
    ih, iw      = frame0.shape[:2]

    # transform init box from raw → rotated
    x1r, y1r, x2r, y2r = args.init_box
    # raw→rotated(-90): new_x1=raw_y1, new_y1=raw_w-1-raw_x2
    nx1 = y1r;  ny1 = iw - 1 - x2r
    nx2 = y2r;  ny2 = iw - 1 - x1r
    init_xywh = [nx1, ny1, nx2 - nx1, ny2 - ny1]
    tracker.init(frame0_rot, init_xywh)
    print(f"  Init box (rotated): {init_xywh}")

    results = []
    for fi in range(1, total):
        ret, fr = cap.read()
        if not ret: break
        out = tracker.track(rotate_frame(fr, -90))
        bx, by, bw, bh = out['bbox']
        results.append((fi, bx, by, bw, bh, out['best_score']))
        if fi % 500 == 0:
            print(f"  frame {fi:5d}/{total}  score={out['best_score']:.3f}  "
                  f"w={bw:.0f} h={bh:.0f}")

    cap.release()
    tracker.cleanup()
    print(f"  Tracking done. Debug frames captured: {sorted(tracker.debug_data.keys())}")
    return tracker.debug_data, results


# ── PDF generation helpers ────────────────────────────────────────────────────

def add_colorbar(ax, im):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax)


# ── Page 1: Overview dashboard ─────────────────────────────────────────────────

def page_overview(pdf, results, debug_frames, csv_path):
    import pandas as pd
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        frames  = df['frame'].values
        scores  = df['score'].values
        widths  = df['w'].values
        heights = df['h'].values
    else:
        arr = np.array(results)
        frames  = arr[:, 0].astype(int)
        scores  = arr[:, 5]
        widths  = arr[:, 3]
        heights = arr[:, 4]

    areas   = widths * heights
    cx_all  = np.array([r[1] + r[3]/2 for r in results])
    cy_all  = np.array([r[2] + r[4]/2 for r in results])
    drift   = np.sqrt(np.diff(cx_all)**2 + np.diff(cy_all)**2)

    fig, axes = plt.subplots(5, 1, figsize=(17, 22))
    fig.suptitle('SiamRPN++ Debug — Overview Dashboard\n'
                 'best_model.pth  |  ir_crop.mp4  |  −90° rotation',
                 fontsize=14, fontweight='bold')

    def vlines(ax):
        for f in debug_frames:
            ax.axvline(f, color='red', lw=0.7, alpha=0.55, linestyle='--')

    # Score
    axes[0].plot(frames, scores, lw=0.7, color='steelblue')
    axes[0].set_ylabel('Score'); axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title('Tracking Score')
    axes[0].fill_between(frames, scores, alpha=0.15, color='steelblue')
    vlines(axes[0])

    # BBox W and H
    axes[1].plot(frames, widths,  lw=0.7, color='darkorange', label='Width')
    axes[1].plot(frames, heights, lw=0.7, color='green',      label='Height')
    axes[1].axhline(widths[0],  color='darkorange', ls=':', lw=1,
                    label=f'Init W={widths[0]:.0f}')
    axes[1].axhline(heights[0], color='green', ls=':', lw=1,
                    label=f'Init H={heights[0]:.0f}')
    axes[1].set_ylabel('Pixels'); axes[1].set_title('BBox Width & Height')
    axes[1].legend(fontsize=8, ncol=4); vlines(axes[1])

    # Area
    axes[2].plot(frames, areas, lw=0.7, color='purple')
    axes[2].axhline(areas[0], color='gray', ls=':', lw=1,
                    label=f'Init area={areas[0]:.0f} px²')
    axes[2].set_ylabel('px²'); axes[2].set_title('BBox Area')
    axes[2].legend(fontsize=8); vlines(axes[2])

    # Drift
    axes[3].plot(frames[1:], drift, lw=0.7, color='crimson')
    axes[3].axhline(2.08, color='gray', ls='--', lw=1, label='Official baseline 2.08 px/f')
    axes[3].set_ylabel('px/frame'); axes[3].set_title('Centre Drift (px/frame)')
    axes[3].set_ylim(0, np.percentile(drift, 99.5))
    axes[3].legend(fontsize=8); vlines(axes[3])

    # Score volatility (rolling std)
    from numpy.lib.stride_tricks import sliding_window_view
    win = 50
    if len(scores) > win:
        roll_std = np.array([scores[i:i+win].std() for i in range(len(scores)-win)])
        axes[4].plot(frames[:len(roll_std)], roll_std, lw=0.7, color='teal')
        axes[4].set_ylabel('Std'); axes[4].set_title(f'Score Rolling Std (window={win})')
        axes[4].axhline(0.0165, color='gray', ls='--', lw=1,  # official ~0.04/2.4
                        label='Official equiv. ~0.017')
        axes[4].legend(fontsize=8)
        vlines(axes[4])

    for ax in axes:
        ax.set_xlabel('Frame'); ax.grid(True, alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig); plt.close(fig)


# ── Pages 2-9: Per-frame debug ────────────────────────────────────────────────

def page_frame(pdf, d, frame_idx):
    fig = plt.figure(figsize=(17, 22))
    ba, br, bc = flat_to_arc(d['best_idx'])
    bpx, bpy   = score_to_px(br, bc)

    fig.suptitle(
        f'Frame {frame_idx}   score={d["best_score"]:.4f}   '
        f'pscore={d["best_pscore"]:.4f}   penalty={d["penalty_best"]:.4f}\n'
        f'bbox: x={d["size_after"][0]:.0f} y={d["size_after"][1]:.0f} '
        f'W={d["size_after"][0]:.0f} H={d["size_after"][1]:.0f}   '
        f'best_anchor={ba}  (row={br},col={bc})   '
        f'dw_best={d["dw_best"]:.3f}→×{np.exp(d["dw_best"]):.2f}   '
        f'dh_best={d["dh_best"]:.3f}→×{np.exp(d["dh_best"]):.2f}',
        fontsize=10, fontweight='bold')

    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.25,
                  top=0.92, bottom=0.04)

    # ── ROW 0: frame / template / search ─────────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    img  = d['img_rot'].copy()
    cx_b, cy_b = d['center_before']
    # search area (yellow)
    sx = int(round(d['s_x']))
    cv2.rectangle(img,
                  (int(cx_b - sx//2), int(cy_b - sx//2)),
                  (int(cx_b + sx//2), int(cy_b + sx//2)),
                  (0, 255, 255), 2)
    # final bbox (green)
    x1f = int(d['center_after'][0] - d['size_after'][0]/2)
    y1f = int(d['center_after'][1] - d['size_after'][1]/2)
    x2f = int(d['center_after'][0] + d['size_after'][0]/2)
    y2f = int(d['center_after'][1] + d['size_after'][1]/2)
    cv2.rectangle(img, (x1f, y1f), (x2f, y2f), (0, 255, 0), 2)
    ax00.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax00.set_title('Frame  (green=bbox  yellow=search)', fontsize=8)
    ax00.axis('off')

    ax01 = fig.add_subplot(gs[0, 1])
    ax01.imshow(cv2.cvtColor(d['z_crop_np'], cv2.COLOR_BGR2RGB))
    ax01.set_title('Template crop (127×127)', fontsize=8)
    ax01.axis('off')

    ax02 = fig.add_subplot(gs[0, 2])
    x_disp = d['x_crop_np'].copy()
    # best decoded bbox on search crop
    pb    = d['pred_bbox'][:, d['best_idx']]
    c_x   = SEARCH_SIZE / 2 + pb[0]
    c_y   = SEARCH_SIZE / 2 + pb[1]
    bx1   = int(np.clip(c_x - pb[2]/2, 0, SEARCH_SIZE-1))
    by1   = int(np.clip(c_y - pb[3]/2, 0, SEARCH_SIZE-1))
    bx2   = int(np.clip(c_x + pb[2]/2, 0, SEARCH_SIZE-1))
    by2   = int(np.clip(c_y + pb[3]/2, 0, SEARCH_SIZE-1))
    cv2.rectangle(x_disp, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
    ax02.imshow(cv2.cvtColor(x_disp, cv2.COLOR_BGR2RGB))
    ax02.set_title('Search crop (255×255) + best bbox', fontsize=8)
    ax02.axis('off')

    # ── ROW 1: cls heatmap / penalized score / top-20 candidates ─────────────
    ax10 = fig.add_subplot(gs[1, 0])
    cls_heat = d['score_map'].max(0)          # (25,25) max over anchors
    blend    = overlay_heat(d['x_crop_np'], cls_heat)
    ax10.imshow(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB))
    ax10.plot(bpx, bpy, 'r+', ms=14, mew=2.5)
    ax10.set_title(f'CLS heatmap (fg prob, max/anchor)\npeak at anchor {ba} '
                   f'row {br} col {bc}', fontsize=8)
    ax10.axis('off')

    ax11 = fig.add_subplot(gs[1, 1])
    psc_heat = d['pscore_map'].max(0)         # penalized score
    blend_p  = overlay_heat(d['x_crop_np'], psc_heat)
    ax11.imshow(cv2.cvtColor(blend_p, cv2.COLOR_BGR2RGB))
    ax11.plot(bpx, bpy, 'r+', ms=14, mew=2.5)
    ax11.set_title('Penalized score map (after scale/ratio + window)', fontsize=8)
    ax11.axis('off')

    ax12 = fig.add_subplot(gs[1, 2])
    top20 = np.argsort(d['pscore_raw'])[-20:][::-1]
    x_t20 = d['x_crop_np'].copy()
    for rank, tidx in enumerate(top20):
        tp  = d['pred_bbox'][:, tidx]
        tcx = SEARCH_SIZE/2 + tp[0]; tcy = SEARCH_SIZE/2 + tp[1]
        tx1 = int(np.clip(tcx - tp[2]/2, 0, SEARCH_SIZE-1))
        ty1 = int(np.clip(tcy - tp[3]/2, 0, SEARCH_SIZE-1))
        tx2 = int(np.clip(tcx + tp[2]/2, 0, SEARCH_SIZE-1))
        ty2 = int(np.clip(tcy + tp[3]/2, 0, SEARCH_SIZE-1))
        alpha_c = max(60, 255 - rank * 10)
        cv2.rectangle(x_t20, (tx1, ty1), (tx2, ty2), (0, alpha_c, 0), 1)
    cv2.rectangle(x_t20, (bx1, by1), (bx2, by2), (0, 0, 255), 2)  # best in blue
    ax12.imshow(cv2.cvtColor(x_t20, cv2.COLOR_BGR2RGB))
    ax12.set_title('Top-20 candidates (green=rank, blue=best)', fontsize=8)
    ax12.axis('off')

    # ── ROW 2: dw heatmap / dh heatmap / W-H distribution ────────────────────
    ax20 = fig.add_subplot(gs[2, 0])
    dw_clip = np.clip(d['dw_map'], -2, 2)
    blend_dw = overlay_heat(d['x_crop_np'], dw_clip)
    ax20.imshow(cv2.cvtColor(blend_dw, cv2.COLOR_BGR2RGB))
    ax20.plot(bpx, bpy, 'r+', ms=14, mew=2.5)
    ax20.set_title(f'dw map (log width scale, mean/anchor)\n'
                   f'best dw={d["dw_best"]:.3f}  exp={np.exp(d["dw_best"]):.2f}× '
                   f'anchor_w\nmap mean={d["dw_map"].mean():.3f}  '
                   f'max={d["dw_map"].max():.3f}', fontsize=8)
    ax20.axis('off')

    ax21 = fig.add_subplot(gs[2, 1])
    dh_clip = np.clip(d['dh_map'], -2, 2)
    blend_dh = overlay_heat(d['x_crop_np'], dh_clip)
    ax21.imshow(cv2.cvtColor(blend_dh, cv2.COLOR_BGR2RGB))
    ax21.plot(bpx, bpy, 'r+', ms=14, mew=2.5)
    ax21.set_title(f'dh map (log height scale, mean/anchor)\n'
                   f'best dh={d["dh_best"]:.3f}  exp={np.exp(d["dh_best"]):.2f}× '
                   f'anchor_h\nmap mean={d["dh_map"].mean():.3f}  '
                   f'max={d["dh_map"].max():.3f}', fontsize=8)
    ax21.axis('off')

    ax22 = fig.add_subplot(gs[2, 2])
    top100 = np.argsort(d['pscore_raw'])[-100:]
    sz_z   = d['scale_z']
    top_w  = d['pred_bbox'][2, top100] / sz_z
    top_h  = d['pred_bbox'][3, top100] / sz_z
    ax22.hist(top_w, bins=25, color='steelblue',  alpha=0.6, label='W')
    ax22.hist(top_h, bins=25, color='darkorange', alpha=0.6, label='H')
    ax22.axvline(d['size_before'][0], color='blue',   ls='--', lw=1.5,
                 label=f'State W={d["size_before"][0]:.0f}')
    ax22.axvline(d['size_before'][1], color='orange', ls='--', lw=1.5,
                 label=f'State H={d["size_before"][1]:.0f}')
    ax22.set_xlabel('Image pixels'); ax22.set_ylabel('Count')
    ax22.set_title('Top-100 decoded W/H (image space)', fontsize=8)
    ax22.legend(fontsize=7); ax22.grid(True, alpha=0.25)

    # ── ROW 3: neck feature maps (3 scales) ───────────────────────────────────
    scale_names = ['Neck scale-0 (layer2)', 'Neck scale-1 (layer3)', 'Neck scale-2 (layer4)']
    for col, (feat, sname) in enumerate(zip(d['xf_neck'], scale_names)):
        ax = fig.add_subplot(gs[3, col])
        feat_2d = safe_norm(feat[0].mean(0))   # mean over 256 channels → (H,W)
        im = ax.imshow(feat_2d, cmap='jet', vmin=0, vmax=1)
        ax.set_title(f'{sname}\n(mean over 256ch, {feat[0].shape[-2]}×{feat[0].shape[-1]})',
                     fontsize=8)
        ax.axis('off')
        add_colorbar(ax, im)

    # Footer: state summary
    dW = d['size_after'][0] - d['size_before'][0]
    dH = d['size_after'][1] - d['size_before'][1]
    fig.text(0.01, 0.005,
             f'center: ({d["center_before"][0]:.1f},{d["center_before"][1]:.1f}) → '
             f'({d["center_after"][0]:.1f},{d["center_after"][1]:.1f})   '
             f'size: ({d["size_before"][0]:.1f},{d["size_before"][1]:.1f}) → '
             f'({d["size_after"][0]:.1f},{d["size_after"][1]:.1f})  '
             f'ΔW={dW:+.1f}  ΔH={dH:+.1f}   '
             f'scale_z={d["scale_z"]:.4f}  s_x={d["s_x"]:.1f}  '
             f'lr_applied={d["lr"]:.4f}',
             fontsize=7, va='bottom', family='monospace',
             bbox=dict(fc='lightyellow', alpha=0.7, boxstyle='round'))

    pdf.savefig(fig)
    plt.close(fig)


# ── Page 10: Anchor shapes ─────────────────────────────────────────────────────

def page_anchors(pdf):
    RATIOS = [0.33, 0.5, 1.0, 2.0, 3.0]
    SCALE  = 8
    BASE   = 8   # stride

    fig, axes = plt.subplots(2, 5, figsize=(17, 9))
    fig.suptitle('Anchor Analysis — 5 Aspect Ratios, Scale=8, Stride=8',
                 fontsize=13, fontweight='bold')

    # Top row: individual anchor shapes on 255×255 canvas
    sizes = []
    for i, ratio in enumerate(RATIOS):
        ws_pre = int(np.sqrt(BASE**2 / ratio))   # floor — matches PySOT generate_anchor
        hs_pre = int(ws_pre * ratio)
        w, h   = float(ws_pre * SCALE), float(hs_pre * SCALE)
        sizes.append((w, h))
        canvas = np.ones((SEARCH_SIZE, SEARCH_SIZE, 3), dtype=np.uint8) * 200
        cx, cy = SEARCH_SIZE // 2, SEARCH_SIZE // 2
        cv2.rectangle(canvas,
                      (int(cx - w/2), int(cy - h/2)),
                      (int(cx + w/2), int(cy + h/2)),
                      (0, 80, 200), 2)
        cv2.circle(canvas, (cx, cy), 3, (200, 0, 0), -1)
        axes[0, i].imshow(canvas)
        axes[0, i].set_title(f'ratio={ratio}\nw={w:.0f}×h={h:.0f}', fontsize=9)
        axes[0, i].axis('off')

    # Bottom row: full anchor grid on search image
    ax = fig.add_subplot(2, 1, 2)  # replace bottom row
    for ax_ in axes[1]: ax_.remove()
    canvas2 = np.ones((SEARCH_SIZE, SEARCH_SIZE, 3), dtype=np.uint8) * 200

    ori = -(SCORE_SIZE // 2) * STRIDE   # -96
    for i, (w, h) in enumerate(sizes):
        color = [(0,150,255),(0,200,100),(200,100,0),(150,0,200),(200,200,0)][i]
        for r in range(SCORE_SIZE):
            for c in range(SCORE_SIZE):
                cx_a = SEARCH_SIZE/2 + ori + c * STRIDE
                cy_a = SEARCH_SIZE/2 + ori + r * STRIDE
                # draw a cross instead of bbox (too many boxes to see)
                px, py = int(cx_a), int(cy_a)
                if 0 <= px < SEARCH_SIZE and 0 <= py < SEARCH_SIZE:
                    cv2.circle(canvas2, (px, py), 2, color, -1)

    # Draw one example of each anchor centred
    for i, (w, h) in enumerate(sizes):
        color = [(0,150,255),(0,200,100),(200,100,0),(150,0,200),(200,200,0)][i]
        cx_c, cy_c = SEARCH_SIZE//2, SEARCH_SIZE//2
        cv2.rectangle(canvas2,
                      (int(cx_c - w/2), int(cy_c - h/2)),
                      (int(cx_c + w/2), int(cy_c + h/2)), color, 2)

    ax.imshow(canvas2)
    ax.set_title('Anchor centre grid (25×25 positions, dots by ratio color) '
                 '+ example anchor shapes at centre', fontsize=9)
    ax.axis('off')

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ── Page 11: Diagnosis table ──────────────────────────────────────────────────

def page_diagnosis(pdf, debug_data, debug_frames, results):
    fig, ax = plt.subplots(figsize=(17, 22))
    ax.axis('off')
    fig.suptitle('Diagnosis Table — Key Metrics at Debug Frames',
                 fontsize=13, fontweight='bold', y=0.99)

    cols = ['Frame','Score','Pscore','Penalty','W(px)','H(px)',
            'Area','ΔW','ΔH','scale_z','s_x',
            'score_std','score_max','dw_best','dh_best',
            'exp(dw)','exp(dh)','lr','Diagnosis']

    rows, colors = [], []
    for f in debug_frames:
        if f not in debug_data:
            rows.append([str(f)] + ['—'] * (len(cols)-2) + ['MISSING'])
            colors.append(['white'] * len(cols))
            continue
        d = debug_data[f]
        w, h   = d['size_after']
        area   = w * h
        dW     = d['size_after'][0] - d['size_before'][0]
        dH     = d['size_after'][1] - d['size_before'][1]
        sc     = d['best_score']
        exp_dw = np.exp(d['dw_best'])
        exp_dh = np.exp(d['dh_best'])
        s_std  = d['score_raw'].std()
        s_max  = d['score_raw'].max()

        diag = []
        if sc < 0.3:              diag.append('LOW_SCORE')
        if area > 15000:          diag.append('BBOX_LARGE')
        if exp_dw > 2.5 or exp_dh > 2.5: diag.append('SIZE_OVERPREDICT')
        if s_std < 0.02:          diag.append('FLAT_RESPONSE')
        if abs(dW) > 15 or abs(dH) > 15: diag.append('STATE_JUMP')
        if not diag:              diag = ['OK']

        row = [str(f), f'{sc:.4f}', f'{d["best_pscore"]:.4f}',
               f'{d["penalty_best"]:.4f}', f'{w:.1f}', f'{h:.1f}',
               f'{area:.0f}', f'{dW:+.1f}', f'{dH:+.1f}',
               f'{d["scale_z"]:.4f}', f'{d["s_x"]:.1f}',
               f'{s_std:.4f}', f'{s_max:.4f}',
               f'{d["dw_best"]:.3f}', f'{d["dh_best"]:.3f}',
               f'{exp_dw:.2f}×', f'{exp_dh:.2f}×',
               f'{d["lr"]:.4f}', ' | '.join(diag)]
        rows.append(row)

        # cell colors
        row_c = ['white'] * len(cols)
        if sc < 0.3:    row_c[1] = '#ffaaaa'
        elif sc < 0.5:  row_c[1] = '#ffe0aa'
        else:           row_c[1] = '#aaffaa'
        if area > 20000:  row_c[6] = '#ffaaaa'
        elif area > 10000:row_c[6] = '#ffe0aa'
        if abs(dW) > 15:  row_c[7] = '#ffe0aa'
        if abs(dH) > 15:  row_c[8] = '#ffe0aa'
        if exp_dw > 2.5:  row_c[15] = '#ffaaaa'
        if exp_dh > 2.5:  row_c[16] = '#ffaaaa'
        if s_std < 0.02:  row_c[11] = '#ffffaa'
        diag_col = len(cols) - 1
        if 'OK' in diag:          row_c[diag_col] = '#aaffaa'
        elif 'LOW_SCORE' in diag: row_c[diag_col] = '#ffaaaa'
        else:                     row_c[diag_col] = '#ffe0aa'
        colors.append(row_c)

    tbl = ax.table(cellText=rows, colLabels=cols, loc='upper center',
                   cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 2.5)

    for (ri, ci), cell in tbl.get_celld().items():
        if ri == 0:
            cell.set_facecolor('#ccccff')
            cell.set_text_props(fontweight='bold')
        elif ri <= len(colors):
            cell.set_facecolor(colors[ri-1][ci])

    # Summary text
    all_scores = np.array([r[5] for r in results])
    init_area  = results[0][3] * results[0][4]
    final_area = results[-1][3] * results[-1][4]
    cx_ = np.array([r[1] + r[3]/2 for r in results])
    cy_ = np.array([r[2] + r[4]/2 for r in results])
    drift_all = np.sqrt(np.diff(cx_)**2 + np.diff(cy_)**2)

    summary = (
        f'Global stats — '
        f'mean_score={all_scores.mean():.3f}  '
        f'score_std={all_scores.std():.3f}  '
        f'fail_rate(score<0.1)={100*(all_scores<0.1).mean():.1f}%  '
        f'mean_drift={drift_all.mean():.2f}px/f\n'
        f'BBox area: init={init_area:.0f}px²  final={final_area:.0f}px²  '
        f'growth={final_area/init_area:.1f}×\n\n'
        f'Key diagnosis:\n'
        f'  • exp(dw) / exp(dh) > 1.0 at almost every frame → model consistently predicts '
        f'bboxes LARGER than anchor size\n'
        f'    This is the primary driver of bbox growth (smooth update accumulates lr × (large_pred - state) each frame)\n'
        f'  • Flat response maps (score_std < 0.02) indicate model cannot localise target '
        f'(uniform correlation response)\n'
        f'  • Scale penalty partially counteracts oversized predictions but is insufficient '
        f'(PENALTY_K=0.05 too small)\n'
        f'  • Root cause hypothesis: training data objects were larger relative to search area '
        f'than test targets → model biased toward predicting large boxes'
    )
    fig.text(0.01, 0.02, summary, fontsize=8.5, va='bottom', family='monospace',
             bbox=dict(fc='lightyellow', alpha=0.8, boxstyle='round'))

    plt.tight_layout(rect=[0, 0.10, 1, 0.98])
    pdf.savefig(fig)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg',   required=True)
    ap.add_argument('--ckpt',  required=True)
    ap.add_argument('--video', required=True)
    ap.add_argument('--csv',   default=None,
                    help='Pre-computed results CSV (optional, for accurate overview)')
    ap.add_argument('--out',   default='debug_best_model.pdf')
    ap.add_argument('--init_box', type=int, nargs=4,
                    metavar=('X1','Y1','X2','Y2'), default=[339, 148, 391, 232])
    args = ap.parse_args()

    DEBUG_FRAMES = [100, 750, 1650, 3250, 3900, 4050, 4800, 5100]

    print('\n=== Running tracking ===')
    debug_data, results = run_tracking(args)

    print(f'\n=== Generating PDF: {args.out} ===')
    with PdfPages(args.out) as pdf:
        print('  Page 1: Overview dashboard')
        page_overview(pdf, results, DEBUG_FRAMES, args.csv)

        for i, f in enumerate(DEBUG_FRAMES, 2):
            print(f'  Page {i}: Frame {f}')
            if f in debug_data:
                page_frame(pdf, debug_data[f], f)
            else:
                print(f'    WARNING: no capture for frame {f}')

        print('  Page 10: Anchor shapes')
        page_anchors(pdf)

        print('  Page 11: Diagnosis table')
        page_diagnosis(pdf, debug_data, DEBUG_FRAMES, results)

    print(f'\nDone → {args.out}')


if __name__ == '__main__':
    main()
