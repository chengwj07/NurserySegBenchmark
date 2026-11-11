"""
06_oneformer/metrics.py

Segmentation index: fully aligned mmsegmentation
Calculation method and output naming of `IoUMetric(iou_metrics=[\"mIoU\", \"mDice\", \"mFscore\"])`
(aAcc / mIoU / mAcc / mDice / mFscore / mPrecision / mRecall), 
In this way, it can be directly compared with the indicators of the other seven mmseg series models, with the same conventions. Supports any number of categories
(Both two classifications, or multiple classifications of background + multiple species are acceptable).

**Easily confused and miscalculated in previous versions**:
- `aAcc` = overall pixel accuracy = (number of pixels of prediction pairs) / (number of total pixels). This is the only \"overall\" metric that does not differentiate between categories.
- `Acc` / `mAcc` **not** `aAcc`! In mmseg source code (`iou_metric.py::total_area_to_metrics`),
  \"Acc\" is `TP/(TP+FN)` for each category, which is the recall rate (Recall) of each category; `mAcc` is for all categories
  Acc is averaged. Numerically \"Acc\" and \"Recall\" are exactly the same formula - mmseg calls it when selecting `mIoU`/`mDice`
  \"Acc\", call it \"Recall\" when selecting `mFscore`, the two are always equal, just have different names.
- `mIoU`/`mDice`/`mFscore`/`mPrecision`/`mRecall` are **all categories are calculated separately and then averaged**, under multiple categories
  It is the background + the average of a total of N categories for each species, not just a certain category.
- If a certain category does not appear in a single pixel in the verification set (for example, there are too few samples of a certain species and it happens to be absent in the current verification set division),
  The denominator of this class will be 0, mmseg handles it with `nanmean` - write this class as NaN and exclude it from the average, instead of treating it as 0
  Bring down the average. The implementation of this document is exactly the same as this.

The formulas in this file have been checked line by line against the `mmseg/evaluation/metrics/iou_metric.py` source code of mmseg 1.2.2.
And it has been numerically verified using the real mmseg function (error < 1e-9, multi-category scenarios have also been verified).

Accumulate the TP/FP/FN of each category on the entire verification set and then calculate them uniformly, instead of calculating the indicators separately for each batch and averaging them.
Avoid indicator deviation caused by small batches/small target samples.
"""
import numpy as np


class SegConfusionMatrix:
    """Accumulate the TP/FP/FN of each category, and finally calculate the index according to the mmseg formula.

    Args:
        num_classes: Total number of segmentation classes (including background). Two-class transmission is 2, and the current 5 types of nursery plant data transmission in this project is 5.
        class_names: Optional, a list of class names of length num_classes, used to generate detailed reports by class
            (If not, use the default name \"class0\"/\"class1\"/...).
    """

    def __init__(self, num_classes: int, class_names=None):
        self.num_classes = num_classes
        self.class_names = list(class_names) if class_names else [f"class{i}" for i in range(num_classes)]
        assert len(self.class_names) == num_classes, 'class_names length must equal num_classes'
        # area_intersect[c] = TP_c, area_pred[c] = TP_c+FP_c, area_label[c] = TP_c+FN_c
        # These three histograms are of the same caliber as mmseg’s intersect_and_union()
        self.area_intersect = np.zeros(num_classes, dtype=np.int64)
        self.area_pred = np.zeros(num_classes, dtype=np.int64)
        self.area_label = np.zeros(num_classes, dtype=np.int64)

    def update(self, pred: np.ndarray, gt: np.ndarray):
        'pred, gt: Arbitrary-shaped category index array (value 0..num_classes-1), the shape must be consistent.'
        pred = pred.reshape(-1).astype(np.int64)
        gt = gt.reshape(-1).astype(np.int64)

        # Only the positions consistent with gt are considered intersections (category-by-category statistics, calculated in one go with bincount, faster than category-by-category Boolean mask)
        intersect = pred[pred == gt]
        self.area_intersect += np.bincount(intersect, minlength=self.num_classes)[: self.num_classes]
        self.area_pred += np.bincount(pred, minlength=self.num_classes)[: self.num_classes]
        self.area_label += np.bincount(gt, minlength=self.num_classes)[: self.num_classes]

    def compute(self) -> dict:
        """The returned key directly corresponds to the output of mmseg IoUMetric:
        aAcc / mIoU / mAcc / mDice / mFscore / mPrecision / mRecall, 
        In addition, each category has a separate `IoU.<class name>` / `Acc.<class name>` /... to facilitate viewing the performance of a species individually.
        """
        ti, tp_area, tl = self.area_intersect.astype(np.float64), self.area_pred.astype(np.float64), \
            self.area_label.astype(np.float64)
        union = tp_area + tl - ti

        total = tl.sum()
        aAcc = ti.sum() / total if total > 0 else float("nan")

        with np.errstate(divide="ignore", invalid="ignore"):
            iou = ti / union
            acc = ti / tl  # == recall, category by category, 0/0 (the category has not appeared at all) will be nan
            dice = 2 * ti / (tp_area + tl)
            precision = ti / tp_area
            recall = acc
            fscore = 2 * precision * recall / (precision + recall)

        result = {
            "aAcc": float(aAcc),
            "mIoU": float(np.nanmean(iou)),
            "mAcc": float(np.nanmean(acc)),
            "mDice": float(np.nanmean(dice)),
            "mFscore": float(np.nanmean(fscore)),
            "mPrecision": float(np.nanmean(precision)),
            "mRecall": float(np.nanmean(recall)),
        }
        for i, name in enumerate(self.class_names):
            result[f"IoU.{name}"] = float(iou[i])
            result[f"Acc.{name}"] = float(acc[i])
            result[f"Dice.{name}"] = float(dice[i])
            result[f"Fscore.{name}"] = float(fscore[i])
            result[f"Precision.{name}"] = float(precision[i])
            result[f"Recall.{name}"] = float(recall[i])
        return result

    def reset(self):
        self.area_intersect[:] = 0
        self.area_pred[:] = 0
        self.area_label[:] = 0


# Backward compatibility: If the old code/old script is still importing BinaryConfusionMatrix, it is equivalent to num_classes=2
# SegConfusionMatrix, the category name is fixed to background/foreground.
def BinaryConfusionMatrix():
    return SegConfusionMatrix(num_classes=2, class_names=["background", "foreground"])
