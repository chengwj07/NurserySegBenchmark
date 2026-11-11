"""End-to-end pipeline orchestration: data preparation -> YOLO training -> evaluation.

Usage:
    python -m src.pipeline --config configs/default.yaml
    python run.py --config configs/default.yaml
"""
import pathlib
import sys

import yaml

# Make sure the project root is in sys.path
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import src  # noqa: F401 (triggers environment variable setting)

from src.data_prep import prepare_dataset
from src.train_yolo import train_yolo
from src.evaluate import evaluate


def load_config(config_path: str) -> dict:
    'Load YAML configuration file.'
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(config_path: str = "configs/default.yaml") -> dict:
    """Run the complete YOLO+SAM2 training and evaluation pipeline.

    Steps:
      1. Data preparation: mask -> YOLO detection label + data.yaml
      2. YOLO26 fine-tuning training
      3. Evaluation: YOLO detection + SAM2 segmentation + indicator calculation

    Args:
        config_path: Configuration file path.

    Returns:
        Dictionary of evaluation metrics.
    """
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    yolo_cfg = cfg["yolo"]
    sam_cfg = cfg["sam"]
    eval_cfg = cfg["eval"]

    # ---- Step 1: Data preparation ----
    print("\n" + "=" * 60)
    print('  Step 1: Data preparation (mask -> YOLO detection label)')
    print("=" * 60)
    num_fg = int(data_cfg.get("num_classes", 1))
    fg_names = data_cfg.get("class_names") or []
    if fg_names and len(fg_names) != num_fg:
        print(f"WARNING: class_names length {len(fg_names)} != num_classes {num_fg}")
        fg_names = []
    data_yaml = prepare_dataset(
        data_root=data_cfg["data_root"],
        ann_dir=data_cfg["ann_dir"],
        img_dir=data_cfg["img_dir"],
        num_classes=num_fg,
        class_names=list(fg_names) or None,
    )

    # ---- Step 2: YOLO training ----
    print("\n" + "=" * 60)
    print('  Step 2: YOLO26 detection fine-tuning')
    print("=" * 60)
    best_pt = train_yolo(
        data_yaml=data_yaml,
        model=yolo_cfg["model"],
        epochs=yolo_cfg["epochs"],
        imgsz=yolo_cfg["imgsz"],
        batch=yolo_cfg["batch"],
        device=yolo_cfg["device"],
        project=yolo_cfg["project"],
        name=yolo_cfg["name"],
        patience=yolo_cfg.get("patience", 20),
    )

    # ---- Step 3: Evaluation ----
    print("\n" + "=" * 60)
    print('  Step 3: Evaluation (YOLO detection + SAM2 segmentation + metrics)')
    print("=" * 60)
    metrics = evaluate(
        yolo_weights=best_pt,
        sam_model_path=sam_cfg["model"],
        data_root=data_cfg["data_root"],
        ann_dir=data_cfg["ann_dir"],
        img_dir=data_cfg["img_dir"],
        conf_threshold=eval_cfg["conf_threshold"],
        iou_threshold=eval_cfg["iou_threshold"],
        max_det=eval_cfg["max_det"],
        device=yolo_cfg["device"],
        num_classes=num_fg + 1,
        class_names=["background"] + list(fg_names),
        save_dir=str(pathlib.Path(yolo_cfg["project"]) / yolo_cfg["name"] / "eval"),
    )

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run YOLO+SAM2 complete pipeline')
    parser.add_argument("--config", default="configs/default.yaml", help="Configuration file path")
    args = parser.parse_args()
    run_pipeline(args.config)
