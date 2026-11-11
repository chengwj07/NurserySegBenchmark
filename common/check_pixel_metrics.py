"""
check_pixel_metrics.py
Recompute pixel-level metrics (global confusion matrix conventions, consistent with mmseg IoUMetric) from the saved prediction mask directory,
Used to check whether the inference pipeline is consistent with the values in the comparison document.

Usage:
    python common/check_pixel_metrics.py --pred_dir predictions/unet         --gt_dir data/ann_dir/val --num_classes 5
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", type=str, required=True)
    ap.add_argument("--gt_dir", type=str, required=True)
    ap.add_argument("--num_classes", type=int, default=5)
    ap.add_argument("--pred_suffix", type=str, default="_mask.png",
                    help='The suffix appended to the prediction file after stem, None means the same name as gt')
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    gt_files = sorted(gt_dir.glob("*.png"))
    n = len(gt_files)
    print(f"GT quantity: {n}")

    conf = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    missing = []
    for gt_f in gt_files:
        pred_name = f"{gt_f.stem}{args.pred_suffix}" if args.pred_suffix else gt_f.name
        pred_f = pred_dir / pred_name
        if not pred_f.exists():
            missing.append(pred_name)
            continue
        gt = load_gray(gt_f)
        pred = load_gray(pred_f)
        if gt.shape != pred.shape:
            print(f"[WARN] shape mismatch {gt_f.name}: gt{gt.shape} pred{pred.shape}")
            continue
        valid = (gt >= 0) & (gt < args.num_classes)
        conf += np.bincount(
            gt[valid].ravel() * args.num_classes + pred[valid].ravel(),
            minlength=args.num_classes * args.num_classes,
        ).reshape(args.num_classes, args.num_classes)

    if missing:
        print(f"[WARN] Missing predictions {len(missing)} each: {missing[:5]} ...")
        if len(missing) > n - n // 2:
            print('Too few prediction files, aborted.')
            return

    tp = np.diag(conf)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    iou = tp / (tp + fp + fn + 1e-9)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-9)
    acc = tp / (tp + fn + 1e-9)
    prec = tp / (tp + fp + 1e-9)
    total = conf.sum()
    aacc = tp.sum() / (total + 1e-9)

    names = ["background", "Podocarpus macrophyllus", "Camellia sasanqua", "Loropetalum chinense", "Ligustrum quihoui"]
    print(f"\n{'Class':<10}{'IoU':>8}{'Dice':>8}{'Fscore':>9}{'Acc':>8}{'Precision':>10}")
    for c in range(args.num_classes):
        nm = names[c] if c < len(names) else f"class{c}"
        print(f"{nm:<10}{iou[c]*100:>7.2f}{dice[c]*100:>7.2f}{dice[c]*100:>8.2f}"
              f"{acc[c]*100:>7.2f}{prec[c]*100:>9.2f}")
    m = lambda a: a.mean() * 100  # noqa: E731
    print(f"\naAcc:   {aacc*100:.2f}")
    print(f"mIoU:   {m(iou):.2f}")
    print(f"mAcc:   {m(acc):.2f}")
    print(f"mDice:  {m(dice):.2f}")
    print(f"mFscore:{m(dice):.2f}")
    print(f"mPrecision: {m(prec):.2f}")
    print(f"mRecall:{m(acc):.2f}")


if __name__ == "__main__":
    main()
