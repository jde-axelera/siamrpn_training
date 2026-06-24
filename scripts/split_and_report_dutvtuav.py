#!/usr/bin/env python3
"""
DUT-VTUAV annotation check — thick-line PDF report (20 images, 2-col layout).
"""
import json, os, random, io
import numpy as np
import cv2
from pathlib import Path
from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm

DSET   = Path("/data/siamrpn_training/data/dut_vtuav")
TRAIN  = DSET / "train"
TEST   = DSET / "test"
REPORT = Path("/data/siamrpn_training/report/dutvtuav_annotation_check.pdf")
random.seed(42)

# ── load existing JSONs ───────────────────────────────────────────────────────
train_annos = json.load(open(DSET / "train_pysot.json"))
test_annos  = json.load(open(DSET / "test_pysot.json"))
print(f"Loaded: {len(train_annos)} train / {len(test_annos)} test sequences")

# ── collect sample frames ─────────────────────────────────────────────────────
def sample_frames(split_dir, annos, n):
    split_dir = Path(split_dir)
    samples   = []
    seq_list  = list(annos.keys())
    random.shuffle(seq_list)
    for seq in seq_list:
        if len(samples) >= n:
            break
        fdict = annos[seq]["0"]
        fids  = sorted(fdict.keys())
        mid   = fids[len(fids) // 2]
        fpath = split_dir / seq / "infrared" / f"{mid}.jpg"
        if not fpath.exists():
            imgs = sorted((split_dir / seq / "infrared").glob("*.jpg"))
            if not imgs:
                continue
            fpath = imgs[len(imgs) // 2]
            mid   = f"{int(fpath.stem):06d}"
        img = cv2.imread(str(fpath))
        if img is None:
            continue
        bbox = fdict.get(mid)
        if bbox is None:
            continue
        samples.append((seq, img, bbox))
    return samples

train_samples = sample_frames(TRAIN, train_annos, 14)
test_samples  = sample_frames(TEST,  test_annos,   6)
all_samples   = train_samples + test_samples
print(f"Collected {len(all_samples)} frames")

# ── annotate with thick lines ─────────────────────────────────────────────────
THUMB_W, THUMB_H = 840, 480   # larger canvas so text/lines stay crisp

def annotate(img, bbox, seq, split):
    img = cv2.resize(img.copy(), (THUMB_W, THUMB_H))
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    oh, ow = img.shape[:2]  # same as THUMB after resize, but track original for scale
    # original image size before resize
    orig_h, orig_w = img.shape[:2]   # already resized — use for bbox scaling
    # bbox was in original resolution — need to scale to THUMB
    # We must re-read original to get original dims; approximate from first read
    sx = THUMB_W / (bbox[2] - bbox[0] + THUMB_W)  # fallback — recalc below
    # Re-open to get original dims
    pass

    colour  = (0, 230, 70)  if split == "train" else (0, 200, 255)
    outline = (0,   0,  0)

    # Scale bbox from original image coords to thumb coords
    # We stored the original image before resize — use the raw img passed in
    return img, bbox, colour, seq, split

# Redo: keep original dims before resize
def annotate2(orig_img, bbox, seq, split):
    oh, ow = orig_img.shape[:2]
    img = cv2.resize(orig_img.copy(), (THUMB_W, THUMB_H))
    sx, sy = THUMB_W / ow, THUMB_H / oh
    x1 = int(bbox[0] * sx);  y1 = int(bbox[1] * sy)
    x2 = int(bbox[2] * sx);  y2 = int(bbox[3] * sy)

    colour = (0, 230, 70) if split == "train" else (0, 200, 255)

    # thick box with dark outline for contrast
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,0),   8)
    cv2.rectangle(img, (x1, y1), (x2, y2), colour,     4)

    # corner ticks
    tick = 18
    for px, py, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(img, (px, py), (px+dx*tick, py),       (0,0,0), 6)
        cv2.line(img, (px, py), (px,         py+dy*tick),(0,0,0), 6)
        cv2.line(img, (px, py), (px+dx*tick, py),       colour,  3)
        cv2.line(img, (px, py), (px,         py+dy*tick),colour, 3)

    # semi-transparent header bar
    overlay = img.copy()
    cv2.rectangle(overlay, (0,0), (THUMB_W, 42), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    tag   = "TRAIN" if split == "train" else "TEST"
    label = f"[{tag}]  {seq}"
    cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0,0,0),   4, cv2.LINE_AA)
    cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_DUPLEX, 0.75, colour,    1, cv2.LINE_AA)

    # bbox info bottom bar
    overlay2 = img.copy()
    cv2.rectangle(overlay2, (0, THUMB_H-36), (THUMB_W, THUMB_H), (0,0,0), -1)
    cv2.addWeighted(overlay2, 0.55, img, 0.45, 0, img)
    info = f"bbox  x1={int(bbox[0])}  y1={int(bbox[1])}  x2={int(bbox[2])}  y2={int(bbox[3])}    w={int(bbox[2]-bbox[0])}  h={int(bbox[3]-bbox[1])}"
    cv2.putText(img, info, (10, THUMB_H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1, cv2.LINE_AA)
    return img

annotated = []
for i, (seq, img, bbox) in enumerate(all_samples):
    split = "train" if i < len(train_samples) else "test"
    annotated.append(annotate2(img, bbox, seq, split))

# ── build PDF ─────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()
doc    = SimpleDocTemplate(str(REPORT), pagesize=A4,
                           leftMargin=1*cm, rightMargin=1*cm,
                           topMargin=1.2*cm, bottomMargin=1*cm)
story  = []
story.append(Paragraph("DUT-VTUAV — Annotation Check", styles["Title"]))
story.append(Paragraph(
    f"<b>{len(train_annos)} train</b> / <b>{len(test_annos)} test</b> sequences (90/10 split, 250 total).  "
    "Green box = train · Cyan box = test.", styles["Normal"]))
story.append(Spacer(1, 0.4*cm))

PAGE_W   = A4[0] - 2*cm
CELL_W   = PAGE_W / 2 - 0.3*cm
CELL_H   = CELL_W * THUMB_H / THUMB_W

rows = []
for k in range(0, len(annotated), 2):
    row = []
    for img_bgr in annotated[k:k+2]:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        buf = io.BytesIO()
        PILImage.fromarray(img_rgb).save(buf, format="JPEG", quality=92)
        buf.seek(0)
        row.append(RLImage(buf, width=CELL_W, height=CELL_H))
    if len(row) == 1:
        row.append("")
    rows.append(row)

t = Table(rows, colWidths=[PAGE_W/2, PAGE_W/2])
t.setStyle(TableStyle([
    ("VALIGN",  (0,0), (-1,-1), "MIDDLE"),
    ("ALIGN",   (0,0), (-1,-1), "CENTER"),
    ("ROWPADDING", (0,0), (-1,-1), 5),
]))
story.append(t)
doc.build(story)
print(f"\n✓ PDF: {REPORT}")
