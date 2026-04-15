#!/usr/bin/env python3
"""
debug_dataflow.py — Full dataflow visualisation for SiamRPN++ training pipeline.

For each active dataset, takes N samples and plots every step:
  Step 0 : Raw full frame  +  ground-truth annotation
  Step 1 : After _get_center_crop  (template 254×254  /  search 510×510)
  Step 2 : After Augmentation      (template 127×127  /  search 255×255)
  Step 3 : AnchorTarget labels     (cls heatmap, positive anchors on search)
  Step 4 : Final model tensors     (tensor stats + display)

Output : /data/siamrpn_training/debug_dataflow.pdf
"""
import sys, os, json, random, textwrap
sys.path.insert(0, '/data/siamrpn_training/pysot')
sys.path.insert(0, '/data/siamrpn_training')

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

from pysot.core.config import cfg
cfg.merge_from_file('/data/siamrpn_training/pysot/experiments/siamrpn_r50_alldatasets/config.yaml')

import train_siamrpn_aws as T
from pysot.datasets.augmentation import Augmentation
from pysot.datasets.anchor_target import AnchorTarget
from pysot.utils.bbox import corner2center, center2corner, Center

random.seed(0); np.random.seed(0)

# ─── colour palette ───────────────────────────────────────────────────────────
C_RAWANNO  = '#00FF41'   # bright green  — GT annotation on raw frame
C_CENTANNO = '#FF6B35'   # orange        — annotation on center-crop
C_GETBBOX  = '#FFD700'   # gold          — _get_bbox output on crop
C_AUGBBOX  = '#00BFFF'   # deep-sky blue — bbox after augmentation
C_POS      = '#FF2D55'   # red           — positive anchor centres
C_NEG      = '#636366'   # grey          — negative
C_IGN      = '#FFD60A'   # yellow        — ignored

SAMPLES_PER_DS = 3       # samples to visualise per dataset
PDF_PATH   = '/data/siamrpn_training/debug_dataflow.pdf'
PNG_DIR    = '/data/siamrpn_training/docs/debug_images'
os.makedirs(PNG_DIR, exist_ok=True)

# ─── helpers ──────────────────────────────────────────────────────────────────
def bgr2rgb(img):
    arr = np.array(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr[:, :, ::-1].copy()

def draw_box(ax, x1, y1, x2, y2, color, lw=2, label=None):
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                               linewidth=lw, edgecolor=color,
                               facecolor='none', linestyle='-')
    ax.add_patch(rect)
    if label:
        ax.text(x1+2, y1-4, label, color=color, fontsize=6.5,
                fontweight='bold',
                path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])

def draw_cross(ax, cx, cy, color, size=6):
    ax.plot([cx-size, cx+size], [cy, cy], color=color, lw=1.2)
    ax.plot([cx, cx], [cy-size, cy+size], color=color, lw=1.2)

def show_img(ax, img_bgr, title, fontsize=8):
    """Display BGR image, clean axes."""
    ax.imshow(bgr2rgb(img_bgr))
    ax.set_title(title, fontsize=fontsize, pad=3)
    ax.axis('off')

def anno_to_xywh(anno):
    if len(anno) == 4:
        x1,y1,x2,y2 = [float(v) for v in anno]
        return x1, y1, x2-x1, y2-y1, (x1+x2)/2, (y1+y2)/2
    else:
        w,h = float(anno[0]), float(anno[1])
        return 0, 0, w, h, 0, 0

def make_aug():
    ta = Augmentation(cfg.DATASET.TEMPLATE.SHIFT, cfg.DATASET.TEMPLATE.SCALE,
                      cfg.DATASET.TEMPLATE.BLUR,  cfg.DATASET.TEMPLATE.FLIP,
                      cfg.DATASET.TEMPLATE.COLOR)
    sa = Augmentation(cfg.DATASET.SEARCH.SHIFT,   cfg.DATASET.SEARCH.SCALE,
                      cfg.DATASET.SEARCH.BLUR,    cfg.DATASET.SEARCH.FLIP,
                      cfg.DATASET.SEARCH.COLOR)
    return ta, sa

def cls_map_image(cls, output_size=25, anchor_num=5):
    """Aggregate cls labels (any anchor positive = positive) → color image."""
    # cls shape: (anchor_num, output_size, output_size)  values: 1/0/-1
    if cls.ndim == 1:
        cls = cls.reshape(anchor_num, output_size, output_size)
    # max over anchors
    max_cls = cls.max(axis=0)   # (25,25)
    rgb = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    rgb[max_cls ==  1] = [220, 50, 50]    # positive  → red
    rgb[max_cls ==  0] = [60, 60, 60]     # negative  → dark grey
    rgb[max_cls == -1] = [220, 200, 30]   # ignore    → yellow
    return rgb

def positive_anchor_overlay(search_img, cls, delta, output_size=25, anchor_num=5,
                              stride=8, base_size=8):
    """Draw positive anchor centres on the 255×255 search image."""
    import copy
    vis = copy.deepcopy(search_img)
    if cls.ndim == 1:
        cls = cls.reshape(anchor_num, output_size, output_size)
    
    # Centre of each grid cell in the 255×255 image
    # PySOT anchors: centre at (col*stride + base_size, row*stride + base_size) roughly
    # More precisely: centre_x = col*stride + (255 - output_size*stride)//2
    offset = (cfg.TRAIN.SEARCH_SIZE - output_size * stride) // 2  # typically 8
    
    for a in range(anchor_num):
        for r in range(output_size):
            for c in range(output_size):
                if cls[a, r, c] == 1:
                    px = c * stride + offset + stride // 2
                    py = r * stride + offset + stride // 2
                    cv2.circle(vis, (int(px), int(py)), 4, (255, 45, 50), -1)
                    cv2.circle(vis, (int(px), int(py)), 4, (255, 255, 255), 1)
    return vis

def tensor_stats_text(arr):
    a = arr.astype(np.float32)
    return (f"shape : {arr.shape}\n"
            f"dtype : {arr.dtype}\n"
            f"min   : {a.min():.3f}\n"
            f"max   : {a.max():.3f}\n"
            f"mean  : {a.mean():.3f}\n"
            f"std   : {a.std():.3f}\n"
            f"range : {a.max()-a.min():.3f}")

# ─── dataset registry ─────────────────────────────────────────────────────────
DATASETS = [
    dict(name="Anti-UAV 410",
         cls=T.AntiUAV410Dataset,
         kwargs=dict(root='/data/siamrpn_training/data/anti_uav410/train',
                     anno_path='/data/siamrpn_training/data/anti_uav410/train_pysot.json')),
    dict(name="MSRS",
         cls=T.MSRSDataset,
         kwargs=dict(root='/data/siamrpn_training/data/msrs/train',
                     anno_path='/data/siamrpn_training/data/msrs/train_pysot.json')),
    dict(name="MassMIND",
         cls=T.MassMINDDataset,
         kwargs=dict(root='/data/siamrpn_training/data/massmind/images',
                     anno_path='/data/siamrpn_training/data/massmind/train_pysot.json')),
    dict(name="DUT-VTUAV",
         cls=T.DUTVTUAVDataset,
         kwargs=dict(root='/data/siamrpn_training/data/dut_vtuav/train',
                     anno_path='/data/siamrpn_training/data/dut_vtuav/train_pysot.json')),
]

anchor_target = AnchorTarget()

# ─── per-sample page builder ──────────────────────────────────────────────────
def visualise_sample(pdf, ds_name, sample_idx, ds_obj):
    """
    Build one full PDF page showing every step of the dataflow for one sample.
    We need to re-run each step manually to capture intermediates.
    """
    # ── pick a valid sequence ─────────────────────────────────────────────────
    for attempt in range(30):
        seq_idx = random.randint(0, len(ds_obj.sequences)-1)
        seq, frames = ds_obj.sequences[seq_idx]
        if len(frames) < 2:
            continue
        track = ds_obj.labels[seq]["0"]

        t_idx = random.randint(0, len(frames)-1)
        lo = max(t_idx - ds_obj.frame_range, 0)
        hi = min(t_idx + ds_obj.frame_range, len(frames)-1) + 1
        s_idx = random.randint(lo, hi-1)
        tf, sf = frames[t_idx], frames[s_idx]

        # load images
        if hasattr(ds_obj, '_find_image'):
            tp = ds_obj._find_image(seq, tf)
            sp = ds_obj._find_image(seq, sf)
            t_img = ds_obj._load_image(tp) if tp else None
            s_img = ds_obj._load_image(sp) if sp else None
        else:
            tp = os.path.join(ds_obj.root, seq, f"{tf:06d}.jpg")
            sp = os.path.join(ds_obj.root, seq, f"{sf:06d}.jpg")
            t_img = cv2.imread(tp)
            s_img = cv2.imread(sp)

        if t_img is None or s_img is None:
            continue

        t_anno_key = f"{tf:06d}"
        s_anno_key = f"{sf:06d}"
        if t_anno_key not in track or s_anno_key not in track:
            continue

        t_anno = track[t_anno_key]
        s_anno = track[s_anno_key]
        break
    else:
        print(f"  [WARN] {ds_name}: could not find valid sample after 30 attempts")
        return

    # ── compute every intermediate ────────────────────────────────────────────
    ez   = cfg.TRAIN.EXEMPLAR_SIZE   # 127
    ssz  = cfg.TRAIN.SEARCH_SIZE     # 255

    # step 1: center crops
    t_crop, t_anno_c = T._get_center_crop(t_img,  t_anno, ez*2,  ez)
    s_crop, s_anno_c = T._get_center_crop(s_img,  s_anno, ssz*2, ez)

    # _get_bbox on crops (same logic as training)
    t_bbox_obj = ds_obj._get_bbox(t_crop, t_anno_c)
    s_bbox_obj = ds_obj._get_bbox(s_crop, s_anno_c)

    # step 2: augmentation (deterministic seed for reproducibility)
    ta, sa = make_aug()
    random.seed(sample_idx); np.random.seed(sample_idx)
    t_aug, t_aug_bbox = ta(t_crop.copy(), t_bbox_obj, ez, gray=False)
    s_aug, s_aug_bbox = sa(s_crop.copy(), s_bbox_obj, ssz, gray=False)

    # step 3: anchor targets
    cls, delta, delta_weight, _ = anchor_target(s_aug_bbox, cfg.TRAIN.OUTPUT_SIZE, neg=False)

    # ── layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 28))
    fig.patch.set_facecolor('#0D0D0D')

    # title bar
    title_ax = fig.add_axes([0.0, 0.955, 1.0, 0.045])
    title_ax.set_facecolor('#1A1A2E')
    title_ax.axis('off')
    ann_str = str(t_anno)
    title_ax.text(0.5, 0.5,
                  f"Dataset: {ds_name}   |   Sample {sample_idx+1}/{SAMPLES_PER_DS}   |   "
                  f"Seq: {seq}   |   Template frame: {tf}   Search frame: {sf}",
                  ha='center', va='center', color='white',
                  fontsize=12, fontweight='bold', transform=title_ax.transAxes)

    # ── define rows ───────────────────────────────────────────────────────────
    gs = gridspec.GridSpec(5, 6, figure=fig,
                           left=0.02, right=0.98,
                           top=0.95, bottom=0.02,
                           hspace=0.38, wspace=0.28)

    # helper: next subplot with dark bg
    def ax(r, c, colspan=1, rowspan=1):
        a = fig.add_subplot(gs[r:r+rowspan, c:c+colspan])
        a.set_facecolor('#1C1C1E')
        for spine in a.spines.values():
            spine.set_edgecolor('#3A3A3C')
        return a

    def row_label(fig, y, text):
        fig.text(0.01, y, text, color='#FFD60A', fontsize=9, fontweight='bold',
                 va='center', rotation=90)

    # ── ROW 0 : section headers ───────────────────────────────────────────────
    row_label(fig, 0.875, "STEP 0 ─ RAW INPUT")
    row_label(fig, 0.71,  "STEP 1 ─ CENTER CROP")
    row_label(fig, 0.545, "STEP 2 ─ AUGMENTATION")
    row_label(fig, 0.38,  "STEP 3 ─ ANCHOR TARGET")
    row_label(fig, 0.15,  "STEP 4 ─ MODEL TENSORS")

    # ── ROW 0 : raw frames ────────────────────────────────────────────────────
    ax00 = ax(0, 0, colspan=2)
    ax00.imshow(bgr2rgb(t_img))
    ax00.set_title(f"Template frame {tf}  |  {t_img.shape[1]}×{t_img.shape[0]}",
                   color='white', fontsize=8, pad=3)
    ax00.axis('off')
    if len(t_anno) == 4:
        x1,y1,x2,y2 = [float(v) for v in t_anno]
        draw_box(ax00, x1, y1, x2, y2, C_RAWANNO, lw=2,
                 label=f"GT [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
        draw_cross(ax00, (x1+x2)/2, (y1+y2)/2, C_RAWANNO)

    ax01 = ax(0, 2, colspan=2)
    ax01.imshow(bgr2rgb(s_img))
    ax01.set_title(f"Search frame {sf}  |  {s_img.shape[1]}×{s_img.shape[0]}",
                   color='white', fontsize=8, pad=3)
    ax01.axis('off')
    if len(s_anno) == 4:
        x1,y1,x2,y2 = [float(v) for v in s_anno]
        draw_box(ax01, x1, y1, x2, y2, C_RAWANNO, lw=2,
                 label=f"GT [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
        draw_cross(ax01, (x1+x2)/2, (y1+y2)/2, C_RAWANNO)

    # raw annotation info box
    ax02 = ax(0, 4, colspan=2)
    ax02.axis('off')
    if len(t_anno) == 4:
        tx1,ty1,tx2,ty2 = [float(v) for v in t_anno]
        tw, th = tx2-tx1, ty2-ty1
        sx1,sy1,sx2,sy2 = [float(v) for v in s_anno]
        sw, sh = sx2-sx1, sy2-sy1
        info = (
            f"━━ TEMPLATE ANNOTATION ━━\n"
            f"  Format : [x1,y1,x2,y2]\n"
            f"  Values : [{tx1:.1f}, {ty1:.1f}, {tx2:.1f}, {ty2:.1f}]\n"
            f"  Size   : {tw:.1f} × {th:.1f} px\n"
            f"  Center : ({(tx1+tx2)/2:.1f}, {(ty1+ty2)/2:.1f})\n"
            f"  AR     : {tw/max(th,1):.2f}\n"
            f"  Image  : {t_img.shape[1]}×{t_img.shape[0]}\n\n"
            f"━━ SEARCH ANNOTATION ━━\n"
            f"  Values : [{sx1:.1f}, {sy1:.1f}, {sx2:.1f}, {sy2:.1f}]\n"
            f"  Size   : {sw:.1f} × {sh:.1f} px\n"
            f"  Center : ({(sx1+sx2)/2:.1f}, {(sy1+sy2)/2:.1f})\n"
            f"  Image  : {s_img.shape[1]}×{s_img.shape[0]}"
        )
    else:
        info = f"Anno (w,h): {t_anno}"
    ax02.text(0.05, 0.95, info, transform=ax02.transAxes,
              color='#E5E5EA', fontsize=7.5, va='top', fontfamily='monospace',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='#2C2C2E', edgecolor='#48484A'))

    # ── ROW 1 : center crops ──────────────────────────────────────────────────
    ax10 = ax(1, 0, colspan=2)
    ax10.imshow(bgr2rgb(t_crop))
    ax10.set_title(f"Template center crop  {t_crop.shape[1]}×{t_crop.shape[0]}",
                   color='white', fontsize=8, pad=3)
    ax10.axis('off')
    cx0, cy0 = t_crop.shape[1]//2, t_crop.shape[0]//2
    if len(t_anno_c) == 4:
        cx1,cy1,cx2,cy2 = [float(v) for v in t_anno_c]
        draw_box(ax10, cx1, cy1, cx2, cy2, C_CENTANNO, lw=2,
                 label=f"anno_c [{cx1:.0f},{cy1:.0f},{cx2:.0f},{cy2:.0f}]")
        draw_cross(ax10, (cx1+cx2)/2, (cy1+cy2)/2, C_CENTANNO)
    # show image center crosshair
    draw_cross(ax10, cx0, cy0, '#FFFFFF', size=8)

    ax11 = ax(1, 2, colspan=2)
    ax11.imshow(bgr2rgb(s_crop))
    ax11.set_title(f"Search center crop  {s_crop.shape[1]}×{s_crop.shape[0]}",
                   color='white', fontsize=8, pad=3)
    ax11.axis('off')
    sx0, sy0 = s_crop.shape[1]//2, s_crop.shape[0]//2
    if len(s_anno_c) == 4:
        sx1,sy1,sx2,sy2 = [float(v) for v in s_anno_c]
        draw_box(ax11, sx1, sy1, sx2, sy2, C_CENTANNO, lw=2,
                 label=f"anno_c [{sx1:.0f},{sy1:.0f},{sx2:.0f},{sy2:.0f}]")
        draw_cross(ax11, (sx1+sx2)/2, (sy1+sy2)/2, C_CENTANNO)
    draw_cross(ax11, sx0, sy0, '#FFFFFF', size=8)

    ax12 = ax(1, 4, colspan=2)
    ax12.axis('off')
    # _get_bbox output
    def fmt_corner(b):
        return f"[{b.x1:.1f}, {b.y1:.1f}, {b.x2:.1f}, {b.y2:.1f}]"
    def corner_center(b):
        cx = (b.x1+b.x2)/2; cy = (b.y1+b.y2)/2
        w = b.x2-b.x1; h = b.y2-b.y1
        return cx, cy, w, h
    tcx,tcy,tbw,tbh = corner_center(t_bbox_obj)
    scx,scy,sbw,sbh = corner_center(s_bbox_obj)
    info2 = (
        f"━━ _get_center_crop ━━\n"
        f"  Output size (tmpl): {t_crop.shape[1]}×{t_crop.shape[0]}\n"
        f"  Output size (srch): {s_crop.shape[1]}×{s_crop.shape[0]}\n"
        f"  White cross = image centre\n"
        f"  Orange box  = new annotation\n\n"
        f"━━ _get_bbox output ━━\n"
        f"  TEMPLATE (corner):\n"
        f"    {fmt_corner(t_bbox_obj)}\n"
        f"    cx={tcx:.1f} cy={tcy:.1f}\n"
        f"    w={tbw:.1f} h={tbh:.1f}\n\n"
        f"  SEARCH (corner):\n"
        f"    {fmt_corner(s_bbox_obj)}\n"
        f"    cx={scx:.1f} cy={scy:.1f}\n"
        f"    w={sbw:.1f} h={sbh:.1f}"
    )
    ax12.text(0.05, 0.95, info2, transform=ax12.transAxes,
              color='#E5E5EA', fontsize=7.5, va='top', fontfamily='monospace',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='#2C2C2E', edgecolor='#48484A'))

    # ── ROW 2 : after augmentation ────────────────────────────────────────────
    ax20 = ax(2, 0, colspan=2)
    ax20.imshow(bgr2rgb(t_aug))
    ax20.set_title(f"Template after aug  {ez}×{ez}  (template_aug)",
                   color='white', fontsize=8, pad=3)
    ax20.axis('off')
    # draw _get_bbox box (gold) and aug bbox (blue)
    draw_box(ax20, t_bbox_obj.x1*(ez/t_crop.shape[1]),
                    t_bbox_obj.y1*(ez/t_crop.shape[0]),
                    t_bbox_obj.x2*(ez/t_crop.shape[1]),
                    t_bbox_obj.y2*(ez/t_crop.shape[0]),
                    C_GETBBOX, lw=1.5, label='_get_bbox (scaled)')
    # aug bbox is relative to the 127×127 output
    draw_box(ax20, float(t_aug_bbox.x1), float(t_aug_bbox.y1),
                    float(t_aug_bbox.x2), float(t_aug_bbox.y2),
                    C_CENTANNO, lw=2, label='aug_bbox')
    draw_cross(ax20, ez//2, ez//2, '#FFFFFF', size=5)

    ax21 = ax(2, 2, colspan=2)
    ax21.imshow(bgr2rgb(s_aug))
    ax21.set_title(f"Search after aug  {ssz}×{ssz}  (search_aug)",
                   color='white', fontsize=8, pad=3)
    ax21.axis('off')
    draw_box(ax21, float(s_aug_bbox.x1), float(s_aug_bbox.y1),
                    float(s_aug_bbox.x2), float(s_aug_bbox.y2),
                    C_AUGBBOX, lw=2, label='aug_bbox → anchor_target input')
    draw_cross(ax21, ssz//2, ssz//2, '#FFFFFF', size=5)

    ax22 = ax(2, 4, colspan=2)
    ax22.axis('off')
    aug_info = (
        f"━━ Augmentation config ━━\n"
        f"  TEMPLATE:\n"
        f"    SHIFT={cfg.DATASET.TEMPLATE.SHIFT}px\n"
        f"    SCALE=±{cfg.DATASET.TEMPLATE.SCALE*100:.0f}%\n"
        f"    BLUR={cfg.DATASET.TEMPLATE.BLUR}\n"
        f"    FLIP={cfg.DATASET.TEMPLATE.FLIP}\n\n"
        f"  SEARCH:\n"
        f"    SHIFT={cfg.DATASET.SEARCH.SHIFT}px\n"
        f"    SCALE=±{cfg.DATASET.SEARCH.SCALE*100:.0f}%\n"
        f"    BLUR={cfg.DATASET.SEARCH.BLUR}\n"
        f"    FLIP={cfg.DATASET.SEARCH.FLIP}\n\n"
        f"━━ aug_bbox (to anchor_target) ━━\n"
        f"  x1={s_aug_bbox.x1:.2f}  y1={s_aug_bbox.y1:.2f}\n"
        f"  x2={s_aug_bbox.x2:.2f}  y2={s_aug_bbox.y2:.2f}\n"
        f"  w={s_aug_bbox.x2-s_aug_bbox.x1:.2f}\n"
        f"  h={s_aug_bbox.y2-s_aug_bbox.y1:.2f}\n"
        f"  cx={(s_aug_bbox.x1+s_aug_bbox.x2)/2:.2f}\n"
        f"  cy={(s_aug_bbox.y1+s_aug_bbox.y2)/2:.2f}"
    )
    ax22.text(0.05, 0.95, aug_info, transform=ax22.transAxes,
              color='#E5E5EA', fontsize=7.5, va='top', fontfamily='monospace',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='#2C2C2E', edgecolor='#48484A'))

    # ── ROW 3 : anchor target ─────────────────────────────────────────────────
    _na = cfg.ANCHOR.ANCHOR_NUM   # 5
    _os = cfg.TRAIN.OUTPUT_SIZE   # 25

    # cls map
    ax30 = ax(3, 0, colspan=2)
    cls_np = cls.numpy() if hasattr(cls, 'numpy') else cls
    cls_rgb = cls_map_image(cls_np, _os, _na)
    ax30.imshow(cls_rgb, interpolation='nearest')
    ax30.set_title(f"AnchorTarget cls map  {_os}×{_os}  (max over {_na} anchors)",
                   color='white', fontsize=8, pad=3)
    ax30.axis('off')
    # count
    if cls_np.ndim == 1:
        cls_np2 = cls_np.reshape(_na, _os, _os)
    else:
        cls_np2 = cls_np
    n_pos = int((cls_np2 == 1).sum())
    n_neg = int((cls_np2 == 0).sum())
    n_ign = int((cls_np2 == -1).sum())
    ax30.text(0.02, 0.02, f"pos={n_pos}  neg={n_neg}  ign={n_ign}",
              transform=ax30.transAxes, color='white', fontsize=7,
              bbox=dict(facecolor='#000000AA', edgecolor='none', boxstyle='round'))
    # coloured legend
    for i,(txt,col) in enumerate([("■ Positive", C_POS), ("■ Negative", C_NEG), ("■ Ignore", C_IGN)]):
        ax30.text(0.02, 0.14 + i*0.07, txt, transform=ax30.transAxes,
                  color=col, fontsize=6.5, fontweight='bold')

    # positive anchors on search
    ax31 = ax(3, 2, colspan=2)
    pos_vis = positive_anchor_overlay(s_aug, cls_np, None, _os, _na,
                                       stride=cfg.ANCHOR.STRIDE, base_size=cfg.TRAIN.BASE_SIZE)
    ax31.imshow(bgr2rgb(pos_vis))
    ax31.set_title("Positive anchors on search image  (red dots)",
                   color='white', fontsize=8, pad=3)
    draw_box(ax31, float(s_aug_bbox.x1), float(s_aug_bbox.y1),
                    float(s_aug_bbox.x2), float(s_aug_bbox.y2),
                    C_AUGBBOX, lw=1.5, label='aug_bbox')
    ax31.axis('off')

    # delta / regression targets
    ax32 = ax(3, 4, colspan=2)
    delta_np = delta.numpy() if hasattr(delta, 'numpy') else delta
    # show dx component averaged over anchors
    if delta_np.ndim == 3:
        dx_map = delta_np[0].mean(axis=0)   # (output_size, output_size)
    else:
        dx_map = delta_np.reshape(4, _na, _os, _os)[0].mean(axis=0)
    dw_np = delta_weight.numpy() if hasattr(delta_weight, 'numpy') else delta_weight

    ax32.set_facecolor('#1C1C1E')
    im32 = ax32.imshow(dx_map, cmap='RdBu_r', interpolation='nearest',
                       vmin=-3, vmax=3)
    ax32.set_title("Regression delta[dx] map  (avg over anchors)",
                   color='white', fontsize=8, pad=3)
    ax32.axis('off')
    plt.colorbar(im32, ax=ax32, fraction=0.046, pad=0.04)
    ax32.text(0.02, 0.02, f"delta_weight non-zero: {int((dw_np!=0).sum())}",
              transform=ax32.transAxes, color='white', fontsize=7,
              bbox=dict(facecolor='#000000AA', edgecolor='none', boxstyle='round'))

    # ── ROW 4 : final model tensors ───────────────────────────────────────────
    # template tensor
    t_tensor = t_aug.transpose(2,0,1).astype(np.float32)  # (3,127,127)
    s_tensor = s_aug.transpose(2,0,1).astype(np.float32)  # (3,255,255)

    ax40 = ax(4, 0)
    # show channel 0 (B channel)
    ax40.imshow(t_tensor[0], cmap='magma', vmin=0, vmax=255)
    ax40.set_title("Template tensor\nchannel 0 (B)", color='white', fontsize=7.5, pad=3)
    ax40.axis('off')

    ax41 = ax(4, 1)
    ax41.imshow(t_tensor[1], cmap='magma', vmin=0, vmax=255)
    ax41.set_title("Template tensor\nchannel 1 (G)", color='white', fontsize=7.5, pad=3)
    ax41.axis('off')

    ax42 = ax(4, 2)
    ax42.imshow(s_tensor[0], cmap='magma', vmin=0, vmax=255)
    ax42.set_title("Search tensor\nchannel 0 (B)", color='white', fontsize=7.5, pad=3)
    ax42.axis('off')

    ax43 = ax(4, 3)
    ax43.imshow(s_tensor[1], cmap='magma', vmin=0, vmax=255)
    ax43.set_title("Search tensor\nchannel 1 (G)", color='white', fontsize=7.5, pad=3)
    ax43.axis('off')

    ax44 = ax(4, 4, colspan=2)
    ax44.axis('off')
    tensor_info = (
        f"━━ Template tensor (→ model) ━━\n"
        f"{tensor_stats_text(t_tensor)}\n\n"
        f"━━ Search tensor (→ model) ━━\n"
        f"{tensor_stats_text(s_tensor)}\n\n"
        f"━━ Note ━━\n"
        f"  Raw 0–255 BGR float32\n"
        f"  NO /255, NO mean-sub\n"
        f"  (matches training-time\n"
        f"   PySOT convention)"
    )
    ax44.text(0.05, 0.95, tensor_info, transform=ax44.transAxes,
              color='#E5E5EA', fontsize=7.5, va='top', fontfamily='monospace',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='#2C2C2E', edgecolor='#48484A'))

    # ── legend strip at bottom ────────────────────────────────────────────────
    leg_ax = fig.add_axes([0.02, 0.002, 0.96, 0.016])
    leg_ax.set_facecolor('#1A1A2E')
    leg_ax.axis('off')
    legend_items = [
        (C_RAWANNO,  "GT annotation (raw frame)"),
        (C_CENTANNO, "Annotation after center-crop / aug_bbox"),
        (C_GETBBOX,  "_get_bbox output (scaled to aug input)"),
        (C_AUGBBOX,  "aug_bbox → anchor_target input"),
        ('#FFFFFF',  "Image centre crosshair"),
        (C_POS,      "Positive anchor"),
    ]
    for i,(col,label) in enumerate(legend_items):
        x = 0.01 + i * 0.165
        leg_ax.text(x, 0.5, f"● {label}", color=col, fontsize=7,
                    va='center', transform=leg_ax.transAxes)

    pdf.savefig(fig, bbox_inches='tight', facecolor=fig.get_facecolor())
    _slug = ds_name.lower().replace(' ', '_').replace('-', '')
    fig.savefig(os.path.join(PNG_DIR, 'dataflow_' + _slug + '_s' + str(sample_idx+1) + '.png'),
               dpi=90, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [{ds_name}] sample {sample_idx+1} → page written  (seq={seq} t={tf} s={sf})")


# ─── title page ───────────────────────────────────────────────────────────────
def title_page(pdf):
    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor('#0D0D0D')
    ax = fig.add_axes([0.05, 0.1, 0.9, 0.8])
    ax.set_facecolor('#0D0D0D')
    ax.axis('off')

    ax.text(0.5, 0.92, "SiamRPN++ Training Pipeline — Dataflow Debug",
            ha='center', va='top', color='white', fontsize=22, fontweight='bold',
            transform=ax.transAxes)
    ax.text(0.5, 0.84, "Full visualisation of every preprocessing step for each active training dataset",
            ha='center', va='top', color='#8E8E93', fontsize=13,
            transform=ax.transAxes)

    # active datasets summary
    ds_info = [
        ("Anti-UAV 410", "200 sequences, 1500 frames each, 640×512 LWIR",
         "Primary IR SOT dataset — drones against sky"),
        ("MSRS",          "520 sequences, paired IR/visible, 480×640",
         "Multi-spectral road-scene pairs"),
        ("MassMIND",      "1801 sequences, 640×512 LWIR",
         "Aerial IR tracking — large diverse set"),
        ("DUT-VTUAV",     "225 sequences, 1920×1080 thermal UAV",
         "IR UAV tracking, 10fps video / 1fps annotations"),
    ]
    ax.text(0.08, 0.73, "Active Datasets in Current Training Run:",
            color='#FFD60A', fontsize=13, fontweight='bold', transform=ax.transAxes)
    for i,(name,desc,note) in enumerate(ds_info):
        y = 0.65 - i * 0.12
        ax.text(0.08, y,     f"  {i+1}. {name}", color='#30D158', fontsize=11,
                fontweight='bold', transform=ax.transAxes)
        ax.text(0.08, y-0.04, f"      {desc}", color='#E5E5EA', fontsize=9,
                transform=ax.transAxes)
        ax.text(0.08, y-0.075, f"      ↳ {note}", color='#8E8E93', fontsize=8.5,
                transform=ax.transAxes)

    # dataflow steps
    ax.text(0.55, 0.73, "Dataflow Steps per Sample:",
            color='#FFD60A', fontsize=13, fontweight='bold', transform=ax.transAxes)
    steps = [
        ("Step 0", "Raw full frame + ground-truth annotation box"),
        ("Step 1", "_get_center_crop()  →  254×254 template  /  510×510 search"),
        ("Step 2", "Augmentation()     →  127×127 template  /  255×255 search"),
        ("Step 3", "AnchorTarget()     →  cls map (pos/neg/ign) + regression delta"),
        ("Step 4", "Final model tensors (raw 0-255 float32, no normalisation)"),
    ]
    for i,(step, desc) in enumerate(steps):
        y = 0.65 - i * 0.085
        ax.text(0.55, y, f"  {step}: {desc}", color='#E5E5EA', fontsize=9.5,
                transform=ax.transAxes)

    # key config
    ax.text(0.08, 0.17, "Key Config:", color='#FFD60A', fontsize=11, fontweight='bold',
            transform=ax.transAxes)
    cfg_text = (f"  EXEMPLAR_SIZE={cfg.TRAIN.EXEMPLAR_SIZE}    SEARCH_SIZE={cfg.TRAIN.SEARCH_SIZE}"
                f"    OUTPUT_SIZE={cfg.TRAIN.OUTPUT_SIZE}    ANCHOR_NUM={cfg.ANCHOR.ANCHOR_NUM}"
                f"    BATCH={cfg.TRAIN.BATCH_SIZE}    EPOCHS={cfg.TRAIN.EPOCH}")
    ax.text(0.08, 0.12, cfg_text, color='#E5E5EA', fontsize=9,
            fontfamily='monospace', transform=ax.transAxes)

    ax.text(0.5, 0.04,
            f"Samples per dataset: {SAMPLES_PER_DS}   |   Total pages: {len(DATASETS)*SAMPLES_PER_DS + 1}",
            ha='center', color='#636366', fontsize=9, transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


# ─── main ─────────────────────────────────────────────────────────────────────
print(f"Generating {PDF_PATH} …")
print(f"Datasets: {[d['name'] for d in DATASETS]}")
print(f"Samples per dataset: {SAMPLES_PER_DS}")
print(f"Expected pages: {1 + len(DATASETS)*SAMPLES_PER_DS}")
print()

with PdfPages(PDF_PATH) as pdf:
    title_page(pdf)

    for ds_cfg_entry in DATASETS:
        name = ds_cfg_entry['name']
        print(f"\n── {name} ──")
        try:
            ds = ds_cfg_entry['cls'](**ds_cfg_entry['kwargs'])
            if len(ds.sequences) == 0:
                print(f"  [SKIP] no sequences loaded")
                continue
            for i in range(SAMPLES_PER_DS):
                visualise_sample(pdf, name, i, ds)
        except Exception as e:
            import traceback
            print(f"  [ERROR] {e}")
            traceback.print_exc()

print(f"\nDone! PDF saved to {PDF_PATH}")
