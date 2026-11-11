"""
convert_gs2_predictions.py
Convert Grounded-SAM2's val prediction json (COCO style: bbox + RLE mask + category_id) to
5 categories of semantic mask (0=background, 1..4=foreground category), output <stem>.png to the specified directory.

Usage:
    python common/convert_gs2_predictions.py         --json 11_Grounded-SAM2/Grounded-SAM-2/outputs/grounded_sam2_val_predictions_fmsam_fixed.json         --out_dir predictions/grounded_sam2
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

    # image_id -> file_name
    id2name = {img["id"]: img["file_name"] for img in data["images"]}
    # annotation is sorted by (image_id, score desc), the overlapping area is won by the high confidence box
    anns = sorted(data["annotations"], key=lambda a: (a["image_id"], -a["score"]))

    per_image = {}
    for a in anns:
        per_image.setdefault(a["image_id"], []).append(a)

    for img_id, items in per_image.items():
        name = id2name[img_id]
        h, w = items[0]["segmentation"]["size"]
        mask = np.zeros((h, w), dtype=np.uint8)
        for a in items:
            rle = a["segmentation"]
            m = coco_mask.decode(rle).astype(bool)
            mask[m] = a["category_id"]
        out_path = out_dir / name
        Image.fromarray(mask, mode="L").save(out_path)

    print(f"output {len(per_image)} Zhang mask -> {out_dir}")


if __name__ == "__main__":
    main()
