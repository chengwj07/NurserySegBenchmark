"""
convert_sam3_predictions.py
Convert SAM3 val dump (coco_predictions_segm.json, example COCO predictions) into 5-category semantic masks
(0=background, 1..4=foreground category), output <file_name>.png to the specified directory.

image_id<->filename mapping comes from the validation set COCO json (--coco-json, default sam3_data/val/annotations.json).
The merging strategy is consistent with build_pred_map in 10_SAM3/eval_plant_metrics.py: write pixels from high to low confidence,
High confidence in the overlapping area is given priority, and uncovered pixels are background 0.

Usage:
    python common/convert_sam3_predictions.py         --pred-json work_dirs/SAM3_semantic/dumps/roboflow/trees/coco_predictions_segm.json         --coco-json sam3_data/val/annotations.json         --out_dir predictions/sam3
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask


def decode_rle(rle: dict) -> np.ndarray:
    if isinstance(rle.get("counts"), list):
        rle = coco_mask.frPyObjects([rle], rle["size"][0], rle["size"][1])[0]
    return coco_mask.decode(rle).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-json", type=str, required=True)
    ap.add_argument("--coco-json", type=str,
                    default="sam3_data/val/annotations.json")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    preds_all = json.load(open(args.pred_json))
    coco = json.load(open(args.coco_json))
    hw = {img["id"]: (img["height"], img["width"]) for img in coco["images"]}
    id2fname = {img["id"]: img["file_name"] for img in coco["images"]}

    pred_by_img = {}
    for p in preds_all:
        pred_by_img.setdefault(int(p["image_id"]), []).append(p)

    # image_id offset automatic alignment (reindex scenario)
    gt_ids = set(hw.keys())
    pred_ids = set(pred_by_img.keys())
    shift = next((s for s in (0, 1, -1)
                  if pred_ids and all(i - s in gt_ids for i in pred_ids)), 0)
    if shift != 0:
        print(f"[WARN] image_id offset +{shift}, automatically aligned")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for image_id, preds in sorted(pred_by_img.items()):
        key = image_id - shift
        if key not in hw:
            continue
        h, w = hw[key]
        mask = np.zeros((h, w), dtype=np.uint8)
        for p in sorted(preds, key=lambda x: -x["score"]):
            cat = int(p["category_id"])
            if not 1 <= cat <= 4:
                continue
            m = decode_rle(p["segmentation"])
            if m.shape[:2] != (h, w):
                import cv2
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = np.where((mask == 0) & (m > 0), cat, mask)
        Image.fromarray(mask, mode="L").save(out_dir / id2fname[key])
        n += 1

    print(f"output {n} Zhang mask -> {out_dir}")


if __name__ == "__main__":
    main()
