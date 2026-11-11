"""YOLO26 + SAM2 semantic segmentation - command line entry.

Usage:
    python run.py pipeline --config configs/default.yaml # Complete pipeline
    python run.py prep --data-root data                     # Prepare data only
    python run.py train --config configs/default.yaml       # Train YOLO only
    python run.py eval --config configs/default.yaml        # Evaluate only
    python run.py smoke                                     # Smoke test
"""
import argparse
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import src  # noqa: F401


def cmd_pipeline(args):
    from src.pipeline import run_pipeline
    metrics = run_pipeline(args.config)
    return metrics


def cmd_prep(args):
    from src.data_prep import prepare_dataset
    names = args.class_names.split(",") if args.class_names else None
    return prepare_dataset(
        data_root=args.data_root,
        ann_dir=args.ann_dir,
        img_dir=args.img_dir,
        num_classes=args.num_classes,
        class_names=names,
    )


def cmd_train(args):
    import yaml
    from src.train_yolo import train_yolo
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_yaml = str(pathlib.Path(cfg["data"]["data_root"]).resolve() / "data.yaml")
    return train_yolo(
        data_yaml=data_yaml,
        model=args.model or cfg["yolo"]["model"],
        epochs=args.epochs or cfg["yolo"]["epochs"],
        imgsz=cfg["yolo"]["imgsz"],
        batch=cfg["yolo"]["batch"],
        device=cfg["yolo"]["device"],
        project=cfg["yolo"]["project"],
        name=cfg["yolo"]["name"],
        resume=args.resume,
    )


def cmd_eval(args):
    import yaml
    from src.evaluate import evaluate
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    yolo_project = cfg["yolo"]["project"]
    yolo_name = cfg["yolo"]["name"]
    best_pt = str(pathlib.Path(yolo_project) / yolo_name / "weights" / "best.pt")
    num_fg = int(cfg["data"].get("num_classes", 1))
    fg_names = cfg["data"].get("class_names") or []
    if fg_names and len(fg_names) != num_fg:
        print(f"WARNING: class_names length {len(fg_names)} != num_classes {num_fg}")
        fg_names = []
    return evaluate(
        yolo_weights=args.yolo_weights or best_pt,
        sam_model_path=cfg["sam"]["model"],
        data_root=cfg["data"]["data_root"],
        ann_dir=cfg["data"]["ann_dir"],
        img_dir=cfg["data"]["img_dir"],
        conf_threshold=cfg["eval"]["conf_threshold"],
        iou_threshold=cfg["eval"]["iou_threshold"],
        max_det=cfg["eval"]["max_det"],
        device=cfg["yolo"]["device"],
        num_classes=num_fg + 1,
        class_names=["background"] + list(fg_names),
        save_dir=str(pathlib.Path(yolo_project) / yolo_name / "eval"),
    )


def cmd_smoke(args):
    from smoke_test import run_smoke_test
    return run_smoke_test()


def main():
    parser = argparse.ArgumentParser(
        description='YOLO26 + SAM2 semantic segmentation pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="Subcommands")

    # pipeline
    p_pipe = sub.add_parser("pipeline", help='Complete pipeline: data preparation + training + evaluation')
    p_pipe.add_argument("--config", default="configs/default.yaml")
    p_pipe.set_defaults(func=cmd_pipeline)

    # prep
    p_prep = sub.add_parser("prep", help="Prepare data only")
    p_prep.add_argument("--data-root", required=True)
    p_prep.add_argument("--ann-dir", default="ann_dir")
    p_prep.add_argument("--img-dir", default="img_dir")
    p_prep.add_argument("--num-classes", type=int, default=1,
                        help='Number of foreground categories (1=two categories, >1=multiple categories)')
    p_prep.add_argument("--class-names", default=None,
                        help='Comma separated foreground category names')
    p_prep.set_defaults(func=cmd_prep)

    # train
    p_train = sub.add_parser("train", help="Train YOLO only")
    p_train.add_argument("--config", default="configs/default.yaml")
    p_train.add_argument("--model", default=None, help="Override the model name or weights")
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--resume", action="store_true",
                         help='Continue training from the breakpoint of last.pt in yolo.project/yolo.name')
    p_train.set_defaults(func=cmd_train)

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate only")
    p_eval.add_argument("--config", default="configs/default.yaml")
    p_eval.add_argument("--yolo-weights", default=None, help="Override the YOLO weights path")
    p_eval.set_defaults(func=cmd_eval)

    # smoke
    p_smoke = sub.add_parser("smoke", help="Smoke test")
    p_smoke.set_defaults(func=cmd_smoke)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
