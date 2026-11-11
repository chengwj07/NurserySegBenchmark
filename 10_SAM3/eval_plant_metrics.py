#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluation after SAM3 fine-tuning is completed: Convert mode:val prediction dump to mmseg style accuracy table.

Outputs the same console table as mmsegmentation IoUMetric:
    per class results: (Class + IoU/Acc/Dice/Fscore/Precision/Recall, percentage)
    Evaluating metrics:  aAcc / mIoU / mAcc / mDice / mFscore / mPrecision / mRecall

Usage (two steps):
    # 1) Run SAM3 with its own val (automatically load fine-tuning checkpoint, dump prediction)
    conda activate sam3
    cd 10_SAM3
    python sam3/train/train.py -c configs/plant/plant_text_val.yaml         --use-cluster 0 --num-gpus 1
    # -> dump to <experiment_log_dir>/dumps/plant_segm/coco_predictions_segm.json

    # 2) Use this script to convert dump + truth value into accuracy table (GT is reconstructed from the polygon annotated by COCO)
    python eval_plant_metrics.py         --pred-json work_dirs/plant_ft/dumps/plant_segm/coco_predictions_segm.json

The indicator calculation is the same as that of 01~08 (mmseg) / 09_FM-SAM: the entire set of confusion matrices are accumulated and calculated uniformly.
Class-by-class IoU/Acc/Dice/Fscore/Precision/Recall + population mean (nanmean, zero-shot class exclusion).
"""
import argparse
import json
import pathlib
import sys

import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
# Reuse FM-SAM indicator code (pure numpy, no ultralytics dependency) to ensure consistent caliber
FM_SAM_SRC = SCRIPT_DIR.parent / "09_FM-SAM" / "src"
sys.path.insert(0, str(FM_SAM_SRC))
from metrics import SegConfusionMatrix, format_metrics  # noqa: E402


def decode_rle(rle: dict) -> np.ndarray:
    'Decode COCO RLE into HxW uint8 mask (compatible with counts as list/str).'
    from pycocotools import mask as mask_util
    if isinstance(rle.get("counts"), list):
        rle = mask_util.frPyObjects([rle], rle["size"][0], rle["size"][1])[0]
    return mask_util.decode(rle).astype(np.uint8)


def ann_to_mask(ann: dict, h: int, w: int) -> np.ndarray:
    'Convert annotation segmentation (polygon or RLE) into HxW binary mask.'
    from pycocotools import mask as mask_util
    segm = ann["segmentation"]
    if isinstance(segm, list):
        rle = mask_util.merge(mask_util.frPyObjects(segm, h, w))
        return mask_util.decode(rle).astype(np.uint8)
    return decode_rle(segm)


def build_gt_maps(coco: dict) -> dict:
    """Reconstruct the semantic category GT map of each image from the COCO annotations (0=background, 1..4=species).

    Returns:
        {image_id: HxW uint8 GT image}, and {image_id: (h, w)}.
    """
    hw = {img["id"]: (img["height"], img["width"]) for img in coco["images"]}
    gt_maps = {img_id: np.zeros((h, w), dtype=np.uint8)
               for img_id, (h, w) in hw.items()}
    for ann in coco["annotations"]:
        img_id = int(ann["image_id"])
        if img_id not in gt_maps:
            continue
        h, w = hw[img_id]
        m = ann_to_mask(ann, h, w)
        cat = int(ann["category_id"])
        if not 1 <= cat <= 4:
            continue
        # When categories overlap, the one that appears first takes precedence (species are spatially separated and rarely conflict in practice)
        gt_maps[img_id] = np.where(
            (gt_maps[img_id] == 0) & (m > 0), cat, gt_maps[img_id]
        )
    return gt_maps, hw


def build_pred_map(image_id: int, preds: list, key: int,
                   hw: dict) -> np.ndarray:
    """Merge the instance predictions of a certain graph into a semantic category index graph.

    Write pixels from high to low confidence (high confidence occupies overlapping areas first), category = predicted species id
    (1..4), the uncovered pixel is background 0. Consistent with the merging strategy of 09_FM-SAM.

    Args:
        image_id: image_id in prediction.
        preds: all prediction instances for this graph.
        key: the real image_id of the image in COCO json (already offset-aligned).
        hw: {image_id: (h, w)}。
    """
    h, w = hw[key]
    pred_map = np.zeros((h, w), dtype=np.uint8)
    for p in sorted(preds, key=lambda x: -x["score"]):
        cat = int(p["category_id"])
        if not 1 <= cat <= 4:
            continue
        mask = decode_rle(p["segmentation"])
        # Size Defense: Crop/Scale Mask to Image Dimensions
        if mask.shape[:2] != (h, w):
            import cv2
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        # Only fill in pixels not occupied by higher confidence instances
        pred_map = np.where((pred_map == 0) & (mask > 0), cat, pred_map)
    return pred_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-json", type=str, required=True,
                        help="SAM3 val 模式 dump 的 coco_predictions_segm.json")
    parser.add_argument("--coco-json", type=str, default=None,
                        help="验证集 COCO 标注 json（默认: 脚本目录下 "
                             "data/trees/valid/_annotations.coco.json），"
                             "用于 image_id<->文件名 映射和重建 GT")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="精度表输出目录（默认: pred-json 所在目录）")
    args = parser.parse_args()

    coco_json = pathlib.Path(
        args.coco_json or SCRIPT_DIR / "data" / "trees" / "valid" / "_annotations.coco.json"
    ).resolve()
    out_dir = pathlib.Path(args.out_dir or pathlib.Path(args.pred_json).parent)

    # ---- Load prediction + COCO (GT + id<-> file name mapping) ----
    with open(args.pred_json, encoding="utf-8") as f:
        preds_all = json.load(f)
    with open(coco_json, encoding="utf-8") as f:
        coco = json.load(f)

    id2fname = {img["id"]: img["file_name"] for img in coco["images"]}  # noqa: F841
    cat_id2name = {c["id"]: c["name"] for c in coco["categories"]}
    gt_maps, hw = build_gt_maps(coco)
    num_classes = len(cat_id2name) + 1  # With background
    class_names = ["background"] + [cat_id2name[i] for i in sorted(cat_id2name)]
    print(f"[评估] 预测实例数: {len(preds_all)}, GT 图像数: {len(gt_maps)}")

    # ---- Merge prediction by graph + cumulative confusion matrix ----
    pred_by_img = {}
    for p in preds_all:
        pred_by_img.setdefault(int(p["image_id"]), []).append(p)

    # Infer image_id global offset: reindex_coco_to_temp will add 0 to the overall index source JSON +1,
    # This causes the predicted image_id to be 1 greater than the id in COCO json (if the source JSON is indexed by 1, there will be no offset).
    # Find the offset at once to avoid guessing by ID (guessing will mismatch the image when the IDs overlap).
    gt_ids = set(gt_maps.keys())
    pred_ids = set(pred_by_img.keys())
    shift = next((s for s in (0, 1, -1)
                  if pred_ids and all(i - s in gt_ids for i in pred_ids)), 0)
    if shift != 0:
        print(f"[评估] 检测到 image_id 偏移 +{shift}（reindex），已自动对齐")

    cm = SegConfusionMatrix(num_classes, class_names=class_names)
    n_imgs = 0
    unmatched = []
    for image_id, preds in sorted(pred_by_img.items()):
        key = image_id - shift
        if key not in gt_maps:
            unmatched.append(image_id)
            continue
        gt = gt_maps[key]
        pred_map = build_pred_map(image_id, preds, key, hw)
        cm.update(pred_map, gt)
        n_imgs += 1

    if unmatched:
        print(f"  警告: {len(unmatched)} 个 image_id 未匹配到 GT（如 {unmatched[:5]}）")
    if n_imgs == 0:
        print("[评估] 未得到任何有效结果，请检查 --pred-json / --coco-json")
        sys.exit(1)

    # ---- Output mmseg style precision table ----
    metrics = cm.compute()
    text = format_metrics(metrics, title=f"SAM3 验证集评估结果 ({n_imgs} 张图像)")
    print(text)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "plant_pixel_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    (out_dir / "plant_pixel_metrics.txt").write_text(text, encoding="utf-8")
    print(f"[评估] 精度表已保存: {out_dir / 'plant_pixel_metrics.txt'}")


if __name__ == "__main__":
    main()
