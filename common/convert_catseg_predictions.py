"""
convert_catseg_predictions.py
Convert CAT-Seg's val prediction json (COCO style: one background RLE per image + one RLE per foreground class) into
5 categories of semantic mask (0=background, 1..4=foreground category), output <stem>.png to the specified directory.

Usage:
    python common/convert_catseg_predictions.py         --json 12_Cat-Seg/output/plant/inference/sem_seg_predictions.json         --out_dir predictions/catseg
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    data = json.load(open(args.json))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_image = {}
    for e in data:
        name = Path(e["file_name"]).name
        per_image.setdefault(name, []).append(e)

    for name, entries in per_image.items():
        h, w = entries[0]["segmentation"]["size"]
        mask = np.zeros((h, w), dtype=np.uint8)
        for e in entries:
            rle = e["segmentation"]
            m = coco_mask.decode(rle).astype(bool)
            mask[m] = e["category_id"]
        out_path = out_dir / name
        Image.fromarray(mask, mode="L").save(out_path)

    print(f"output {len(per_image)} Zhang mask -> {out_dir}")


if __name__ == "__main__":
    main()
