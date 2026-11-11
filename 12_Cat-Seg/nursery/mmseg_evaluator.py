# -*- coding: utf-8 -*-
# mmseg-style semantic segmentation accuracy evaluator (based on detectron2 SemSegEvaluator extension)
#
# Differences from the original SemSegEvaluator:
# - The original version only outputs mIoU / fwIoU / mACC / pACC (see detectron2 sem_seg_evaluation.py)
# - This evaluator additionally calculates each class and mean from the confusion matrix: IoU / Dice / F1 / Precision / Recall / Acc,
# And output a summary table consistent with mmsegmentation IoUMetric (mIoU / mDice / mFscore / mPrecision /
# mRecall / mAcc / aAcc), the category name uses stuff_classes in the registration data set.
#
# Usage (via eval_mmseg.py entry, command line parameters are consistent with eval.sh):
#   python miaomu/eval_mmseg.py --config-file configs/plant_vitb_384.yaml \
#       --eval-only --num-gpus 1 OUTPUT_DIR output/plant_eval MODEL.WEIGHTS output/plant/model_final.pth
import os
from collections import OrderedDict

import numpy as np

from detectron2.evaluation.sem_seg_evaluation import SemSegEvaluator
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager


def _nanmean(x):
    "Ignore the mean of NaNs; return 0 for all NaNs (consistent with mmseg's nanmean)."
    return float(np.nanmean(x)) if np.isfinite(x).any() else 0.0


def _safe_div(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def compute_metrics(conf_matrix, class_names):
    """Calculate all metrics from the confusion matrix in mmseg IoUMetric style (pure functions for easy testing/reuse).

    Args:
        conf_matrix: (K+1, K+1) matrix, row=GT, column=pred, the last row/column is ignore class.
        class_names: list of class names, length K.
    Returns:
        (per_class dict, mean dict)
    """
    cm = conf_matrix[:-1, :-1].astype(np.float64)
    num_classes = len(class_names)

    tp = np.diag(cm)
    gt_pos = cm.sum(axis=1)
    pred_pos = cm.sum(axis=0)
    fp = pred_pos - tp
    fn = gt_pos - tp

    iou = _safe_div(tp, tp + fp + fn)
    dice = _safe_div(2 * tp, 2 * tp + fp + fn)
    precision = _safe_div(tp, pred_pos)
    recall = _safe_div(tp, gt_pos)
    acc = _safe_div(tp, gt_pos)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    valid = gt_pos > 0
    mean = OrderedDict()
    mean["aAcc"] = float(np.sum(tp) / np.sum(gt_pos)) * 100 if np.sum(gt_pos) > 0 else 0.0
    mean["mIoU"] = _nanmean(iou[valid]) * 100
    mean["mDice"] = _nanmean(dice[valid]) * 100
    mean["mFscore"] = _nanmean(f1[valid]) * 100
    mean["mPrecision"] = _nanmean(precision[valid]) * 100
    mean["mRecall"] = _nanmean(recall[valid]) * 100
    mean["mAcc"] = _nanmean(acc[valid]) * 100

    per_class = OrderedDict()
    for i, name in enumerate(class_names):
        per_class[f"IoU-{name}"] = iou[i] * 100
        per_class[f"Dice-{name}"] = dice[i] * 100
        per_class[f"F1-{name}"] = f1[i] * 100
        per_class[f"Precision-{name}"] = precision[i] * 100
        per_class[f"Recall-{name}"] = recall[i] * 100
        per_class[f"Acc-{name}"] = acc[i] * 100
        per_class[f"Ratio-{name}"] = (gt_pos[i] / np.sum(gt_pos)) * 100 if np.sum(gt_pos) > 0 else 0.0

    return per_class, mean


class MMSegStyleEvaluator(SemSegEvaluator):
    'Outputs complete semantic segmentation metrics consistent with mmsegmentation IoUMetric.'

    def evaluate(self):
        # ---------- Distributed summary (logic consistent with the original version) ----------
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            if not is_main_process():
                return
            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

        per_class, res = compute_metrics(self._conf_matrix, self._class_names)

        # ---------- Console table (similar to mmseg training log style) ----------
        self._print_table(self._class_names, per_class, res)

        # Summary results: overall indicator + each type of indicator (key name consistent with mmseg IoUMetric style)
        full = OrderedDict()
        full.update(res)
        full.update(per_class)

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.json")
            with PathManager.open(file_path, "w") as f:
                import json

                json.dump({k: float(v) for k, v in full.items()}, f, indent=2)
            self._logger.info(f"Results saved to {file_path}")

        results = OrderedDict({"sem_seg": full})
        self._logger.info(results)
        return results

    def _print_table(self, class_names, per_class, res):
        # Use detectron2's logger to print (self._logger has been configured with handler/level by detectron2)
        logger = self._logger
        header = f"{'Class':<28}{'IoU':>8}{'Dice':>8}{'F1':>8}{'Precision':>11}{'Recall':>9}{'Acc':>8}{'Ratio%':>9}"
        logger.info("\n" + "=" * len(header))
        logger.info("Semantic Segmentation Results (mmseg style)")
        logger.info("=" * len(header))
        logger.info(header)
        logger.info("-" * len(header))
        for name in class_names:
            logger.info(
                f"{name:<28}"
                f"{per_class[f'IoU-{name}']:>8.2f}{per_class[f'Dice-{name}']:>8.2f}"
                f"{per_class[f'F1-{name}']:>8.2f}{per_class[f'Precision-{name}']:>11.2f}"
                f"{per_class[f'Recall-{name}']:>9.2f}{per_class[f'Acc-{name}']:>8.2f}"
                f"{per_class[f'Ratio-{name}']:>9.2f}"
            )
        logger.info("-" * len(header))
        logger.info(
            f"{'mIoU':<28}{res['mIoU']:>8.2f}   (mean IoU over classes with GT pixels)"
        )
        logger.info(f"{'mDice':<28}{res['mDice']:>8.2f}")
        logger.info(f"{'mFscore':<28}{res['mFscore']:>8.2f}")
        logger.info(f"{'mPrecision':<28}{res['mPrecision']:>8.2f}")
        logger.info(f"{'mRecall':<28}{res['mRecall']:>8.2f}")
        logger.info(f"{'mAcc':<28}{res['mAcc']:>8.2f}")
        logger.info(f"{'aAcc':<28}{res['aAcc']:>8.2f}")
        logger.info("=" * len(header))
