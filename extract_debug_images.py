#!/usr/bin/env python3
"""
extract_debug_images.py
Extract key pages from the debug PDF as JPEG images for the README.
"""
import fitz, os, sys

PDF_PATH = '/tmp/debug_best_model.pdf'
OUT_DIR  = '/data/siamrpn_training/docs'

# page_index (0-based) -> output filename
PAGES = {
    0:  'debug_overview.jpg',       # Overview dashboard
    1:  'debug_frame_100.jpg',      # Frame 100 (good tracking)
    2:  'debug_frame_750.jpg',      # Frame 750 (first drop)
    4:  'debug_frame_3250.jpg',     # Frame 3250 (very low)
    5:  'debug_frame_3900.jpg',     # Frame 3900 (recovery)
    8:  'debug_frame_5100.jpg',     # Frame 5100 (total loss)
    9:  'debug_anchors.jpg',        # Anchor shapes
    10: 'debug_diagnosis.jpg',      # Diagnosis table
}

doc = fitz.open(PDF_PATH)
for page_idx, fname in PAGES.items():
    page = doc[page_idx]
    mat  = fitz.Matrix(2.0, 2.0)   # 2× resolution
    pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    out  = os.path.join(OUT_DIR, fname)
    pix.save(out)
    print(f'  saved {fname}  ({pix.width}×{pix.height})')

doc.close()
print(f'Done — {len(PAGES)} images written to {OUT_DIR}')
