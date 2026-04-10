#!/usr/bin/env python3
"""
make_test_demo.py  —  Combined GT-annotated demo video from all test splits
===========================================================================
Loads test-split annotations for every available dataset, samples sequences,
draws ground-truth bounding boxes with dataset labels, and assembles a single
MP4 demo video with title cards between dataset sections.

Usage
-----
  python make_test_demo.py \
      --work_dir /home/ubuntu/siamrpn_training \
      --out      /home/ubuntu/siamrpn_training/demo/test_gt_demo.mp4 \
      --seqs_per_dataset 4 \
      --max_frames_per_seq 120 \
      --fps 15 \
      --width 640 --height 480
"""
import argparse, json, os, random, sys, textwrap
import numpy as np
import cv2

# ── colour palette ────────────────────────────────────────────────────────────
COLOURS = {
    "AntiUAV410":  (0,  255,  80),   # green
    "AntiUAV300":  (0,  200, 255),   # cyan
    "MSRS":        (255, 180,  0),   # amber
    "MassMIND":    (200,   0, 240),  # purple
    "DUT-Anti-UAV":(255,  60,  60),  # red
}
DEFAULT_COLOUR = (200, 200, 200)

# ── dataset registry ──────────────────────────────────────────────────────────
# Each entry: (display_name, root_dir, anno_json, frame_finder_key)
def build_registry(work_dir):
    data = os.path.join(work_dir, "data")
    return [
        ("AntiUAV410",
         os.path.join(data, "anti_uav410", "val"),          # val split lives here
         os.path.join(data, "anti_uav410", "val_pysot.json"),
         "antiuav410"),
        ("AntiUAV300",
         os.path.join(data, "anti_uav300", "val"),           # extracted frames in val/
         os.path.join(data, "anti_uav300", "val_pysot.json"),
         "antiuav300"),
        ("MSRS",
         os.path.join(data, "msrs", "test"),
         os.path.join(data, "msrs", "test_pysot.json"),
         "msrs"),
        ("MassMIND",
         os.path.join(data, "massmind", "images"),
         os.path.join(data, "massmind", "test_pysot.json"),
         "massmind"),
        ("DUT-Anti-UAV",
         os.path.join(data, "dut_anti_uav", "images"),
         os.path.join(data, "dut_anti_uav", "train_pysot.json"),
         "dutantiuav"),
    ]

# ── frame finders ─────────────────────────────────────────────────────────────
def _find_frame_antiuav410(root, seq, fid):
    for ext in (".jpg", ".png"):
        p = os.path.join(root, seq, f"{fid:06d}{ext}")
        if os.path.isfile(p):
            return p
    return None

def _find_frame_antiuav300(root, seq, fid):
    # Try extracted frames first
    for ext in (".jpg", ".png"):
        p = os.path.join(root, seq, f"{fid:06d}{ext}")
        if os.path.isfile(p):
            return p
    # Fall back to reading from mp4
    mp4 = os.path.join(root, seq, "infrared.mp4")
    if os.path.isfile(mp4):
        cap = cv2.VideoCapture(mp4)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid - 1)
        ok, frm = cap.read()
        cap.release()
        if ok:
            return ("__mp4__", frm)
    return None

def _find_frame_msrs(root, seq, fid):
    ir_dir = os.path.join(root, "ir")
    if not os.path.isdir(ir_dir):
        ir_dir = root
    imgs = sorted(f for f in os.listdir(ir_dir)
                  if f.lower().endswith((".png", ".jpg", ".bmp")))
    idx = fid - 1
    if 0 <= idx < len(imgs):
        return os.path.join(ir_dir, imgs[idx])
    return None

def _find_frame_massmind(root, seq, fid):
    stem = seq.replace("massmind_", "", 1)
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[0] == stem:
                return os.path.join(dirpath, fn)
    return None

def _find_frame_dutantiuav(root, seq, fid):
    # Handle wrapped layout (images/Anti-UAV-Tracking-V0/seqname/)
    # DUT frames may be 5-digit (00001.jpg) or 6-digit (000001.jpg)
    for base in (root,):
        for subdir in ("", ):
            for ext in (".jpg", ".png"):
                for fmt in (f"{fid:06d}", f"{fid:05d}"):
                    p = os.path.join(base, seq, f"{fmt}{ext}")
                    if os.path.isfile(p):
                        return p
    # Try one wrapper level
    for wrapper in os.listdir(root):
        wpath = os.path.join(root, wrapper)
        if os.path.isdir(wpath):
            for ext in (".jpg", ".png"):
                for fmt in (f"{fid:06d}", f"{fid:05d}"):
                    p = os.path.join(wpath, seq, f"{fmt}{ext}")
                    if os.path.isfile(p):
                        return p
    return None

FRAME_FINDERS = {
    "antiuav410":  _find_frame_antiuav410,
    "antiuav300":  _find_frame_antiuav300,
    "msrs":        _find_frame_msrs,
    "massmind":    _find_frame_massmind,
    "dutantiuav":  _find_frame_dutantiuav,
}

# ── image loading ─────────────────────────────────────────────────────────────
def load_frame(result):
    """Load a frame from a path or a pre-decoded (tag, array) tuple."""
    if result is None:
        return None
    if isinstance(result, tuple) and result[0] == "__mp4__":
        return result[1]
    img = cv2.imread(result)
    if img is None:
        return None
    # Grayscale → 3-channel
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img.squeeze(), cv2.COLOR_GRAY2BGR)
    return img

# ── drawing helpers ───────────────────────────────────────────────────────────
def draw_bbox(img, bbox, colour, label="", thickness=2):
    """Draw bbox (x1,y1,x2,y2) with an optional label."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness)
    if label:
        fs, ft = 0.45, 1
        tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)[0]
        ty = max(y1 - 4, th + 4)
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), colour, -1)
        cv2.putText(img, label, (x1 + 2, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), ft, cv2.LINE_AA)

def draw_hud(img, dataset, seq, fid, total):
    """Overlay dataset / sequence info at the top-left."""
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    txt = f"{dataset}  |  {seq[:30]}  |  frame {fid}/{total}"
    cv2.putText(img, txt, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)

def make_title_card(dataset, n_seqs, out_w, out_h, colour, fps, hold_sec=2):
    """Return a list of identical title-card frames (hold_sec × fps frames)."""
    n_frames = int(fps * hold_sec)
    frames = []
    for _ in range(n_frames):
        card = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        # Coloured left bar
        cv2.rectangle(card, (0, 0), (12, out_h), colour, -1)
        # Dataset name
        cv2.putText(card, dataset, (30, out_h // 2 - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, colour, 2, cv2.LINE_AA)
        info = f"{n_seqs} sequences  ·  ground-truth annotations"
        cv2.putText(card, info, (30, out_h // 2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        frames.append(card)
    return frames

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
             formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work_dir", default="/home/ubuntu/siamrpn_training")
    ap.add_argument("--out",      default=None, help="Output MP4 path")
    ap.add_argument("--seqs_per_dataset", type=int, default=4)
    ap.add_argument("--max_frames_per_seq", type=int, default=120)
    ap.add_argument("--fps",    type=int, default=15)
    ap.add_argument("--width",  type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--seed",   type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_w, out_h = args.width, args.height
    demo_dir = os.path.join(args.work_dir, "demo")
    os.makedirs(demo_dir, exist_ok=True)
    if args.out is None:
        args.out = os.path.join(demo_dir, "test_gt_demo.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (out_w, out_h))
    if not writer.isOpened():
        sys.exit(f"[ERROR] Cannot open VideoWriter for {args.out}")

    registry = build_registry(args.work_dir)
    total_frames_written = 0

    for (ds_name, root, anno_path, finder_key) in registry:
        if not os.path.isfile(anno_path):
            print(f"  [SKIP] {ds_name}: annotation not found ({anno_path})")
            continue
        anno = json.load(open(anno_path))
        if not anno:
            print(f"  [SKIP] {ds_name}: empty annotation file")
            continue

        # Sample sequences that have at least one resolvable frame
        finder = FRAME_FINDERS[finder_key]
        colour = COLOURS.get(ds_name, DEFAULT_COLOUR)

        all_seqs = list(anno.keys())
        random.shuffle(all_seqs)

        chosen = []
        for seq in all_seqs:
            if len(chosen) >= args.seqs_per_dataset:
                break
            frames_dict = anno[seq].get("0", {})
            fids = sorted(frames_dict.keys())
            if not fids:
                continue
            # Quick probe: check first frame is loadable
            first_fid = int(fids[0])
            result = finder(root, seq, first_fid)
            if load_frame(result) is None:
                continue
            chosen.append(seq)

        if not chosen:
            print(f"  [SKIP] {ds_name}: no loadable sequences found")
            continue

        print(f"\n  [{ds_name}] {len(chosen)} sequences → {args.out}")
        # Title card
        for card in make_title_card(ds_name, len(chosen), out_w, out_h,
                                    colour, args.fps, hold_sec=2.5):
            writer.write(card)
            total_frames_written += 1

        for seq in chosen:
            frames_dict = anno[seq].get("0", {})
            fids = sorted(frames_dict.keys())

            # Subsample to max_frames_per_seq
            if len(fids) > args.max_frames_per_seq:
                step = len(fids) / args.max_frames_per_seq
                fids = [fids[int(i * step)] for i in range(args.max_frames_per_seq)]

            n_total = len(fids)
            for i, fid_str in enumerate(fids):
                fid = int(fid_str)
                result = finder(root, seq, fid)
                img = load_frame(result)
                if img is None:
                    continue

                bbox = frames_dict.get(fid_str)
                if bbox is None:
                    # Try zero-padded key
                    bbox = frames_dict.get(f"{fid:06d}")
                if bbox is None:
                    continue

                # Resize to output resolution
                canvas = cv2.resize(img, (out_w, out_h))
                sx = out_w / img.shape[1]
                sy = out_h / img.shape[0]
                scaled_bbox = [bbox[0]*sx, bbox[1]*sy, bbox[2]*sx, bbox[3]*sy]

                draw_bbox(canvas, scaled_bbox, colour,
                          label=f"{ds_name} | GT")
                draw_hud(canvas, ds_name, seq, i + 1, n_total)

                writer.write(canvas)
                total_frames_written += 1

            print(f"    seq={seq[:30]}  frames={len(fids)}")

    writer.release()
    dur = total_frames_written / args.fps
    print(f"\n✓ Demo video written: {args.out}")
    print(f"  Total frames : {total_frames_written}")
    print(f"  Duration     : {dur:.1f}s  at {args.fps} fps")


if __name__ == "__main__":
    main()
