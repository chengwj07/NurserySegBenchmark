# -*- coding: utf-8 -*-
# Four types of seedlings segmentation single picture/folder reasoning script (CAT-Seg)
# Usage (run in the 12_Cat-Seg root directory to ensure that the relative path class json is available):
#   python miaomu/infer.py --config-file configs/plant_vitb_384.yaml \
#       --weights output/plant/model_final.pth \
#       --input /path/to/img.png \
#       --output-dir output/plant/infer
# # Folder: --input /path/to/img_dir/val (whole directory traversal)
# Output: <name>_pred.png (palette mask) and <name>_vis.png (overlay visualization) are generated for each image.
import argparse
import glob
import os
import sys

import numpy as np
import torch
import cv2

# fmt: off
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))
# fmt: on

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from cat_seg import add_cat_seg_config


def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="CAT-Seg 苗木分割推理")
    parser.add_argument("--config-file", default="configs/plant_vitb_384.yaml", metavar="FILE")
    parser.add_argument("--weights", required=True, help="模型权重 .pth")
    parser.add_argument("--input", required=True, help="图片文件或图片目录")
    parser.add_argument("--output-dir", default="output/plant/infer")
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER,
                        help="额外的 detectron2 配置覆盖，例如 MODEL.WEIGHTS ...")
    return parser


def main():
    args = get_parser().parse_args()
    setup_logger(name="detectron2")
    logger = setup_logger(name="catseg_miaomu")

    cfg = setup_cfg(args)
    # Use the test class name (json) during inference, and Predictor provides stuff_classes through MetadataCatalog
    meta = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    logger.info(f"类别: {meta.stuff_classes}")

    predictor = DefaultPredictor(cfg)

    if os.path.isdir(args.input):
        files = sorted(glob.glob(os.path.join(args.input, "*.png")) +
                       glob.glob(os.path.join(args.input, "*.jpg")) +
                       glob.glob(os.path.join(args.input, "*.jpeg")))
        if not files:
            logger.error(f"{args.input} 下没有图片")
            return
    else:
        files = [args.input]
    logger.info(f"共 {len(files)} 张图片")

    os.makedirs(args.output_dir, exist_ok=True)
    palette = np.array(meta.stuff_colors, dtype=np.uint8)

    for path in files:
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            logger.warning(f"读取失败: {path}")
            continue
        out = predictor(img)
        sem_seg = out["sem_seg"].argmax(dim=0).cpu().numpy().astype(np.uint8)
        # Ignore the category (255) and set it to 0 (background) before coloring
        sem_seg_vis = np.where(sem_seg == 255, 0, sem_seg)
        color_mask = palette[sem_seg_vis]
        name = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(args.output_dir, f"{name}_pred.png"), color_mask)
        # Overlay visualization (original image + semi-transparent mask)
        overlay = img.copy()
        cv2.addWeighted(color_mask, 0.5, overlay, 0.5, 0, overlay)
        cv2.imwrite(os.path.join(args.output_dir, f"{name}_vis.png"), overlay)
        logger.info(f"done: {name}")


if __name__ == "__main__":
    main()
