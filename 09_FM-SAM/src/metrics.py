"""Semantic segmentation evaluation index module (consistent with mmseg IoUMetric conventions, aligned with 01~08 models).

Two categories: 0=background, 1=foreground. Accumulate the confusion matrix on the entire validation set and calculate it uniformly (without averaging on a picture-by-picture basis):

  IoU_c       = TP_c / (TP_c + FP_c + FN_c)
  Acc_c = TP_c / (TP_c + FN_c) # Class-by-class accuracy, equivalent to recall
  Dice_c = 2*TP_c / (2*TP_c + FP_c + FN_c) # Equivalent to F1
  Precision_c = TP_c / (TP_c + FP_c)
  Recall_c    = TP_c / (TP_c + FN_c)
  Fscore_c    = 2*Precision_c*Recall_c / (Precision_c + Recall_c)
  aAcc = sum of TP of all categories / total pixels = (TP+TN)/total
  mIoU/mAcc/mDice/mFscore/mPrecision/mRecall = unweighted mean of various types (nanmean)

When a class has zero samples (or zero predictions) in the validation set, the class index is labeled nan and excluded by nanmean (consistent with mmseg;
Empty/missed images will not be regarded as a full score of 1.0 like the old version of _safe_div).
"""
import numpy as np


class SegConfusionMatrix:
    'Accumulate TP / TP+FP / TP+FN for each category, and finally calculate the indicators according to the mmseg formula.'

    def __init__(self, num_classes: int = 2, class_names=None):
        self.num_classes = num_classes
        self.class_names = (
            list(class_names) if class_names else [f"class{i}" for i in range(num_classes)]
        )
        assert len(self.class_names) == num_classes, 'class_names length must equal num_classes'
        # area_intersect[c] = TP_c, area_pred[c] = TP_c+FP_c, area_label[c] = TP_c+FN_c
        self.area_intersect = np.zeros(num_classes, dtype=np.int64)
        self.area_pred = np.zeros(num_classes, dtype=np.int64)
        self.area_label = np.zeros(num_classes, dtype=np.int64)

    def update(self, pred: np.ndarray, gt: np.ndarray):
        'pred, gt: Arbitrary-shaped category index array (value 0..num_classes-1), the shape must be consistent.'
        pred = pred.reshape(-1).astype(np.int64)
        gt = gt.reshape(-1).astype(np.int64)
        intersect = pred[pred == gt]
        self.area_intersect += np.bincount(intersect, minlength=self.num_classes)[: self.num_classes]
        self.area_pred += np.bincount(pred, minlength=self.num_classes)[: self.num_classes]
        self.area_label += np.bincount(gt, minlength=self.num_classes)[: self.num_classes]

    def compute(self) -> dict:
        ti = self.area_intersect.astype(np.float64)
        tp = self.area_pred.astype(np.float64)
        tl = self.area_label.astype(np.float64)
        union = tp + tl - ti

        total = tl.sum()
        aAcc = ti.sum() / total if total > 0 else float("nan")

        with np.errstate(divide="ignore", invalid="ignore"):
            iou = ti / union
            acc = ti / tl            # == recall
            dice = 2 * ti / (tp + tl)
            precision = ti / tp
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


def _display_width(s: str) -> int:
    """The terminal display width of the string (full-width characters such as Chinese are counted as 2), consistent with prettytable.

    When wcwidth is unavailable it falls back to the number of characters len() (in which case prettytable behaves the same).
    """
    try:
        from wcwidth import wcswidth
        w = wcswidth(s)
        return w if w and w > 0 else len(s)
    except Exception:
        return len(s)


def _center(s: str, w: int) -> str:
    """Center according to the terminal display width, consistent with wcwidth.center (same as prettytable) character by character.

    When the margin is an odd number, if the content display width is an even number, the extra space will be placed on the left, otherwise it will be placed on the right.
    """
    dlen = _display_width(s)
    marg = w - dlen
    left = marg // 2 + (1 if (marg % 2 == 1 and dlen % 2 == 0) else 0)
    return " " * left + s + " " * (marg - left)


def _simple_table(columns: dict) -> str:
    'Simple ASCII table without prettytable dependency (consistent with PrettyTable default output).'
    headers = list(columns.keys())
    rows = list(zip(*columns.values()))
    widths = {}
    for h, vals in columns.items():
        widths[h] = max(_display_width(h),
                        max(_display_width(str(v)) for v in vals))

    def cell(s, w):
        return " " + _center(str(s), w) + " "

    sep = "+" + "+".join("-" * (widths[h] + 2) for h in headers) + "+"
    lines = [sep]
    lines.append("|" + "|".join(cell(h, widths[h]) for h in headers) + "|")
    lines.append(sep)
    for row in rows:
        lines.append("|" + "|".join(
            cell(v, widths[h]) for h, v in zip(headers, row)) + "|")
    lines.append(sep)
    return "\n".join(lines)


def format_metrics(metrics: dict, title: str = None) -> str:
    """Output metrics in console format for mmsegmentation IoUMetric (consistent with mmseg 1.2.2).

    First print the per class result table (percentage, 2 decimal places), and then print the overall indicators
    aAcc / mIoU / mAcc / mDice / mFscore / mPrecision / mRecall.

    Args:
        metrics: Dictionary of metrics returned by SegConfusionMatrix.compute()
            (7 overall + IoU./Acc./Dice./Fscore./Precision./Recall. per category).
        title: Compatible with old calls, not used in output (mmseg style without title).

    Returns:
        A string in the same format as mmseg IoUMetric.
    """
    # Class-by-class indicators: IoU/Acc/Dice/Fscore/Precision/Recall in mmseg order, converted to percentage
    class_names = [k[4:] for k in metrics if k.startswith("IoU.")]
    columns = {"Class": class_names}
    for m in ("IoU", "Acc", "Dice", "Fscore", "Precision", "Recall"):
        columns[m] = [round(metrics[f"{m}.{name}"] * 100, 2)
                      for name in class_names]

    lines = ["per class results:", ""]
    try:
        from prettytable import PrettyTable
        table = PrettyTable()
        for key, vals in columns.items():
            table.add_column(key, vals)
        lines.append(table.get_string())
    except ImportError:
        lines.append(_simple_table(columns))
    lines.append("")
    lines.append("Evaluating metrics:")
    for key in ("aAcc", "mIoU", "mAcc", "mDice", "mFscore",
                "mPrecision", "mRecall"):
        v = round(metrics.get(key, float("nan")) * 100, 2)
        lines.append(f"{key}: {v:.4f}")
    return "\n".join(lines)
