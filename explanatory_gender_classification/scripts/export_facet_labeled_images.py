"""
Copy FACET test images to two folders (female / male) by ground-truth label,
with the annotated person's bounding box drawn on each image.

Useful for manual inspection: you can immediately see which person carries
the label and whether multi-person images are labelled correctly.

Output layout:
  <out_dir>/female/<image_name>
  <out_dir>/male/<image_name>

Usage:
  python scripts/export_facet_labeled_images.py
  python scripts/export_facet_labeled_images.py --max_height 600 --out_dir /some/path
"""

import os
import ast
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

FACET_DIR  = r'c:\Users\Queby\Research\xil-gender-classification\facet_dataset\facet_processed'
OUTPUT_DIR = r'c:\Users\Queby\Research\xil-gender-classification\facet_dataset\labeled_images'


def _parse_bbox(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    s = str(value).strip()
    if not s or s.lower() == 'nan':
        return None
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return None


def draw_bbox_on_image(pil_img, bbox_xywh, label_text, max_height=800):
    """Return a copy of pil_img (optionally downscaled) with a labelled bbox."""
    orig_w, orig_h = pil_img.size

    # Downscale to max_height while preserving aspect ratio
    if orig_h > max_height:
        scale = max_height / orig_h
        new_w = int(orig_w * scale)
        pil_img = pil_img.resize((new_w, max_height), Image.LANCZOS)
    else:
        scale = 1.0

    draw = ImageDraw.Draw(pil_img)

    if bbox_xywh is not None:
        x, y, bw, bh = [v * scale for v in bbox_xywh[:4]]
        x1, y1, x2, y2 = x, y, x + bw, y + bh
        # Lime green bounding box
        draw.rectangle([x1, y1, x2, y2], outline='lime', width=max(2, int(3 * scale)))

        # Label tag above the box
        tag_h = max(14, int(18 * scale))
        tag_box = [x1, max(0, y1 - tag_h), x1 + len(label_text) * 8 * scale, max(0, y1)]
        draw.rectangle(tag_box, fill='lime')
        try:
            font = ImageFont.truetype("arial.ttf", size=max(10, int(14 * scale)))
        except Exception:
            font = ImageFont.load_default()
        draw.text((x1 + 2, max(0, y1 - tag_h)), label_text,
                  fill='black', font=font)

    return pil_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--facet_dir',  default=FACET_DIR)
    parser.add_argument('--out_dir',    default=OUTPUT_DIR)
    parser.add_argument('--split',      default='test')
    parser.add_argument('--max_height', type=int, default=800,
                        help='Max image height in pixels (preserves aspect ratio). '
                             '0 = keep original size.')
    args = parser.parse_args()

    split_csv = os.path.join(args.facet_dir, 'splits', f'{args.split}.csv')
    img_dir   = os.path.join(args.facet_dir, 'images')

    df = pd.read_csv(split_csv)
    print(f"Loaded {len(df)} rows from {split_csv}")

    female_dir = os.path.join(args.out_dir, 'female')
    male_dir   = os.path.join(args.out_dir, 'male')
    os.makedirs(female_dir, exist_ok=True)
    os.makedirs(male_dir,   exist_ok=True)

    max_h = args.max_height if args.max_height > 0 else 99999
    n_ok, n_miss = 0, 0

    for i, row in df.iterrows():
        img_name = str(row['image_name'])
        label    = str(row.get('label', 'unknown'))      # 'male' / 'female'
        encoded  = int(row.get('encoded_label', -1))    # 0=female, 1=male
        bbox     = _parse_bbox(row.get('bbox'))

        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            n_miss += 1
            continue

        try:
            pil = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"  [WARN] cannot open {img_name}: {e}")
            n_miss += 1
            continue

        tag = label  # shown on the bbox tag
        pil = draw_bbox_on_image(pil, bbox, tag, max_height=max_h)

        out_folder = female_dir if encoded == 0 else male_dir
        out_path   = os.path.join(out_folder, img_name)
        pil.save(out_path, quality=90)
        n_ok += 1

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(df)} done...", flush=True)

    print(f"\nDone. {n_ok} saved ({n_miss} missing/skipped)")
    print(f"  Female: {female_dir}")
    print(f"  Male:   {male_dir}")


if __name__ == '__main__':
    main()
