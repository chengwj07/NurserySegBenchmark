"""
summarize_metrics.py
Calculate pixel indicators (IoU/Dice/F1...) and boundary indicators (Boundary IoU/F1) uniformly for the prediction masks of a batch of models,
Summarize output JSON and markdown tables (common for Phase 2 verification + Phase 4 delivery).

Usage:
    python common/summarize_metrics.py --gt_dir data/ann_dir/val         --out multiclass_segmentation_comparison/boundary_metrics_comparison.json         [--models \"unet:predictions/unet:_mask.png\" \"catseg:predictions/catseg:\" ...]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from eval_boundary import per_image_boundary_metrics

CLASS_NAMES = ["background", "Podocarpus macrophyllus", "Camellia sasanqua", "Loropetalum chinense", "Ligustrum quihoui"]
STRUCT = np.ones((3, 3), dtype=bool)


def pixel_confusion(gt_files, pred_dir, num_classes, pred_suffix):
    'Global confusion matrix (same conventions as mmseg IoUMetric).'
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    n_pred = 0
    for gt_f in gt_files:
        pred_name = f"{gt_f.stem}{pred_suffix}" if pred_suffix else gt_f.name
        pred_f = pred_dir / pred_name
        if not pred_f.exists():
            continue
        gt = np.array(Image.open(gt_f))
        pred = np.array(Image.open(pred_f))
        if gt.shape != pred.shape:
            continue
        conf += np.bincount(gt.ravel() * num_classes + pred.ravel(),
                            minlength=num_classes * num_classes).reshape(
                                num_classes, num_classes)
        n_pred += 1
    return conf, n_pred


def pixel_metrics(conf, num_classes):
    tp = np.diag(conf)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    iou = tp / (tp + fp + fn + 1e-9)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-9)
    acc = tp / (tp + fn + 1e-9)
    prec = tp / (tp + fp + 1e-9)
    aacc = tp.sum() / (conf.sum() + 1e-9)
    return {
        "aAcc": aacc * 100,
        "mIoU": iou.mean() * 100,
        "mAcc": acc.mean() * 100,
        "mDice": dice.mean() * 100,
        "mPrecision": prec.mean() * 100,
        "per_class_IoU": [round(x * 100, 2) for x in iou],
    }


def boundary_metrics(gt_files, pred_dir, num_classes, pred_suffix,
                     dilation, tau):
    acc = {c: ([], []) for c in range(num_classes)}
    n_pred = 0
    for gt_f in gt_files:
        pred_name = f"{gt_f.stem}{pred_suffix}" if pred_suffix else gt_f.name
        pred_f = pred_dir / pred_name
        if not pred_f.exists():
            continue
        gt = np.array(Image.open(gt_f))
        pred = np.array(Image.open(pred_f))
        if gt.shape != pred.shape:
            continue
        res = per_image_boundary_metrics(gt, pred, num_classes, dilation, tau)
        for c, (bi, bf) in res.items():
            acc[c][0].append(bi)
            acc[c][1].append(bf)
        n_pred += 1

    b_iou, b_f1, n_img = {}, {}, {}
    for c in range(num_classes):
        b_iou[c] = float(np.nanmean(acc[c][0])) * 100 if acc[c][0] else float("nan")
        b_f1[c] = float(np.nanmean(acc[c][1])) * 100 if acc[c][1] else float("nan")
        n_img[c] = len(acc[c][0])

    def m(d, fg_only=False):
        vals = [v for k, v in d.items() if not np.isnan(v) and (not fg_only or k >= 1)]
        return float(np.mean(vals))

    return {
        "mBoundaryIoU_5": m(b_iou),
        "mBoundaryIoU_fg4": m(b_iou, True),
        "mBoundaryF1_5": m(b_f1),
        "mBoundaryF1_fg4": m(b_f1, True),
        "per_class_BoundaryIoU": {str(c): round(b_iou[c], 2) for c in b_iou},
        "per_class_BoundaryF1": {str(c): round(b_f1[c], 2) for c in b_f1},
        "n_images": {str(c): n_img[c] for c in n_img},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", type=str, default="data/ann_dir/val")
    ap.add_argument("--num_classes", type=int, default=5)
    ap.add_argument("--dilation", type=int, default=14)
    ap.add_argument("--tau", type=int, default=3)
    ap.add_argument("--out", type=str, default='multiclass_segmentation_comparison/boundary_metrics_comparison.json')
    ap.add_argument("--models", nargs="+", default=[
        "unet:predictions/unet:_mask.png",
        "deeplabv3plus:predictions/deeplabv3plus:_mask.png",
        "hrnet:predictions/hrnet:_mask.png",
        "segformer:predictions/segformer:_mask.png",
        "swin:predictions/swin:_mask.png",
        "oneformer:predictions/oneformer:_mask.png",
        "internimage:predictions/internimage:_mask.png",
        "vmamba:predictions/vmamba:_mask.png",
        "fmsam:predictions/fmsam:",
        "sam3:predictions/sam3:",
        "grounded_sam2:predictions/gs2_default:",
        "catseg:predictions/catseg:",
    ], help='label:pred_dir:pred_suffix list')
    args = ap.parse_args()

    gt_dir = Path(args.gt_dir)
    gt_files = sorted(gt_dir.glob("*.png"))
    results = {}
    for item in args.models:
        label, pred_dir, suffix = item.split(":")
        pred_dir = Path(pred_dir)
        conf, n_pred = pixel_confusion(gt_files, pred_dir, args.num_classes, suffix)
        pm = pixel_metrics(conf, args.num_classes)
        bm = boundary_metrics(gt_files, pred_dir, args.num_classes, suffix,
                              args.dilation, args.tau)
        results[label] = {"pixel": pm, "boundary": bm, "n_pred": n_pred}
        print(f"[{label}] n_pred={n_pred}  mIoU={pm['mIoU']:.2f}  "
              f"mB-IoU={bm['mBoundaryIoU_5']:.2f}  mB-F1={bm['mBoundaryF1_5']:.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"config": {"dilation": args.dilation, "tau": args.tau},
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
