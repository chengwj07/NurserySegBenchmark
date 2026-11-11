"""Evaluation module: YOLO detection + SAM2 segmentation + indicator calculation.

Complete evaluation process:
  1. Load the trained YOLO26 detection model
  2. Load the SAM2 segmentation model (official weights)
  3. For each image in the val set:
     a. YOLO detects foreground targets -> detection frames
     b. Calculate the center point of each detection frame
     c. Send the center point as prompt to SAM2 -> segmentation mask
     d. Union of multiple masks -> final prediction mask
  4. Calculate aAcc / mIoU / mAcc / mDice / uniformly after accumulating the confusion matrix on the entire validation set.
     mFscore / mPrecision / mRecall and category-by-category indicators (consistent with mmseg IoUMetric conventions,
     Comparable to 01~08 models).
"""
import pathlib
import cv2
import numpy as np
from tqdm import tqdm

from .sam_segment import SAM2Segmenter
from .metrics import SegConfusionMatrix, format_metrics


def evaluate(
    yolo_weights: str,
    sam_model_path: str,
    data_root: str,
    ann_dir: str = "ann_dir",
    img_dir: str = "img_dir",
    split: str = "val",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_det: int = 300,
    device: str = "cpu",
    num_classes: int = 2,
    class_names: list[str] | None = None,
    save_dir: str | None = None,
    save_masks: str | None = None,
) -> dict:
    """Evaluate the YOLO+SAM2 semantic segmentation effect on the validation set.

    Args:
        yolo_weights: trained YOLO weight path (best.pt).
        sam_model_path: SAM2 weight path.
        data_root: data root directory.
        ann_dir: Mark the directory name.
        img_dir: image directory name.
        split: The data partition to evaluate (usually \"val\").
        conf_threshold: YOLO detection confidence threshold.
        iou_threshold: YOLO NMS IoU threshold.
        max_det: The maximum number of detections per picture.
        device: inference device.
        num_classes: Total number of categories (including background). 2=two classification, when >2, the true value mask is based on the original
            The category index is read (0=background), and the predicted category is determined by the YOLO detection box category number.
        class_names: list of class names (including background, length = num_classes).
        save_dir: If given, write the indicator results to this directory (metrics.json + metrics.txt),
            Contains overall aAcc/mIoU/mAcc/mDice/mFscore/mPrecision/mRecall and class-by-class metrics.
        save_masks: If given, write the prediction masks (0..num_classes-1 class index) for each image to this directory.

    Returns:
        Average metric dictionary (7 items overall + IoU/Acc/Dice/Fscore/Precision/Recall per category).
    """
    from ultralytics import YOLO

    if class_names is None:
        class_names = ["background"] + [f"class{i}" for i in range(1, num_classes)]
    if len(class_names) != num_classes:
        raise ValueError(
            f"class_names length {len(class_names)} Must equal num_classes {num_classes}"
        )

    root = pathlib.Path(data_root).resolve()
    img_val_dir = root / img_dir / split
    ann_val_dir = root / ann_dir / split

    if not img_val_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {img_val_dir}")
    if not ann_val_dir.exists():
        raise FileNotFoundError(f"The annotation directory does not exist: {ann_val_dir}")

    # Load model
    print(f"[Evaluation] Load YOLO model: {yolo_weights}")
    yolo = YOLO(yolo_weights)
    print(f"[Evaluation] Load SAM2 model: {sam_model_path}")
    segmenter = SAM2Segmenter(sam_model_path, device=device)

    # Collect images and masks
    img_files = sorted(
        f for f in img_val_dir.iterdir()
        if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    )
    print(f"[Evaluation] Total {len(img_files)} Verification image")

    cm = SegConfusionMatrix(num_classes=num_classes, class_names=class_names)
    n_images = 0
    if save_masks:
        pathlib.Path(save_masks).mkdir(parents=True, exist_ok=True)
    for img_file in tqdm(img_files, desc='Assessment'):
        # read image
        image = cv2.imread(str(img_file))
        if image is None:
            print(f"  Warning: Unable to read image {img_file}")
            continue

        # Read ground truth mask (multiclass retains original class index 0..num_classes-1)
        mask_file = ann_val_dir / img_file.name
        gt_mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            print(f"  Warning: Unable to read mask {mask_file}")
            continue
        if num_classes == 2:
            gt_mask = (gt_mask > 0).astype(np.uint8)
        else:
            gt_mask = np.clip(gt_mask, 0, num_classes - 1).astype(np.uint8)

        # YOLO detection (with category and confidence)
        results = yolo.predict(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            max_det=max_det,
            verbose=False,
        )
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            box_classes = results[0].boxes.cls.cpu().numpy().astype(np.int64)
            box_confs = results[0].boxes.conf.cpu().numpy()
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            box_classes = np.zeros((0,), dtype=np.int64)
            box_confs = np.zeros((0,), dtype=np.float32)

        # SAM2 segmentation (using the center point of the detection frame as the prompt, the category comes from YOLO)
        pred_mask = segmenter.segment_with_boxes_labeled(
            image, boxes, box_classes, box_confs, num_classes=num_classes
        )

        if save_masks:
            cv2.imwrite(str(pathlib.Path(save_masks) / img_file.name), pred_mask)

        # Cumulative confusion matrix (pred/gt are both 0..num_classes-1 category index)
        cm.update(pred_mask, gt_mask)
        n_images += 1

    # Summary: Calculate once after the entire verification set is accumulated (no picture-by-picture averaging)
    if n_images == 0:
        print('[Assessment] Warning: No valid assessment results obtained')
        return {}

    metrics = cm.compute()
    print(format_metrics(metrics, title=f"Validation set evaluation results ({n_images} images)"))

    # Save complete indicators (overall + category-by-category) for easy review later
    if save_dir:
        import json
        save_path = pathlib.Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        (save_path / "metrics.txt").write_text(
            format_metrics(metrics, title=f"Validation set evaluation results ({n_images} images)"),
            encoding="utf-8",
        )
        print(f"[Evaluation] Metrics saved: {save_path / 'metrics.json'} / metrics.txt")

    return metrics


if __name__ == "__main__":
    import argparse
    import yaml
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    import src  # noqa: F401 (triggers environment variable setting)

    parser = argparse.ArgumentParser(description='Evaluating YOLO+SAM2 semantic segmentation')
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--save_masks", default=None,
                        help='If given, write each image prediction mask (0..num_classes-1) to this directory')
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Infer YOLO optimal weight path
    yolo_project = cfg["yolo"]["project"]
    yolo_name = cfg["yolo"]["name"]
    best_pt = str(pathlib.Path(yolo_project) / yolo_name / "weights" / "best.pt")

    # Category configuration: num_classes=number of foreground categories (config), +1 is the total number of categories including background
    num_fg = int(cfg["data"].get("num_classes", 1))
    fg_names = cfg["data"].get("class_names") or []
    if fg_names and len(fg_names) != num_fg:
        print(f"WARNING: class_names length {len(fg_names)} != num_classes {num_fg}")
        fg_names = []
    class_names = ["background"] + list(fg_names)

    evaluate(
        yolo_weights=best_pt,
        sam_model_path=cfg["sam"]["model"],
        data_root=cfg["data"]["data_root"],
        ann_dir=cfg["data"]["ann_dir"],
        img_dir=cfg["data"]["img_dir"],
        conf_threshold=cfg["eval"]["conf_threshold"],
        iou_threshold=cfg["eval"]["iou_threshold"],
        max_det=cfg["eval"]["max_det"],
        device=cfg["yolo"]["device"],
        num_classes=num_fg + 1,
        class_names=class_names,
        save_dir=str(pathlib.Path(yolo_project) / yolo_name / "eval"),
        save_masks=args.save_masks,
    )
