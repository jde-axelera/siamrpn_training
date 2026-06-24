"""
Convert Anti-UAV410 annotations to PySOT SubDataset JSON format.

Anti-UAV410 format (per sequence):
    IR_label.json: {"exist": [1,0,...], "gt_rect": [[x,y,w,h],...], ...}

PySOT SubDataset format:
    {video_name: {track_id: {frame_str: [x1,y1,x2,y2]}}}
"""

import json
import os
import argparse


def convert(data_root, split='train', output_path=None):
    split_dir = os.path.join(data_root, split)
    sequences = sorted(os.listdir(split_dir))

    pysot_annos = {}
    skipped = 0
    for seq in sequences:
        label_path = os.path.join(split_dir, seq, 'IR_label.json')
        if not os.path.isfile(label_path):
            skipped += 1
            continue

        with open(label_path) as f:
            label = json.load(f)

        exist  = label['exist']
        gt_rect = label['gt_rect']

        frames_dict = {}
        for i, (ex, box) in enumerate(zip(exist, gt_rect)):
            if ex == 0 or box is None:
                continue
            x, y, w, h = box
            if w <= 0 or h <= 0:
                continue
            frame_str = '{:06d}'.format(i + 1)   # frames are 1-indexed filenames
            frames_dict[frame_str] = [x, y, x + w, y + h]   # [x1, y1, x2, y2]

        if frames_dict:
            pysot_annos[seq] = {'0': frames_dict}   # single track per sequence

    if output_path is None:
        output_path = os.path.join(data_root, f'{split}_pysot.json')

    with open(output_path, 'w') as f:
        json.dump(pysot_annos, f)

    print(f"[{split}] {len(pysot_annos)} sequences converted → {output_path}")
    if skipped:
        print(f"  Skipped {skipped} sequences (missing IR_label.json)")
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='/Users/jaydeepde/work/Research/Arquimea/data/anti_uav410')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'])
    args = parser.parse_args()

    for split in args.splits:
        convert(args.data_root, split)
