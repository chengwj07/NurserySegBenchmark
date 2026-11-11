# Copyright (c) Facebook, Inc. and its affiliates.
# Semantic segmentation dataset registration of seedlings (5 categories: 0=background, 1=Podocarpus macrophyllus, 2=Camellia sasanqua, 3=Loropetalum chinense, 4=Ligustrum quihoui)
# Data directory structure (consistent with load_sem_seg requirements of detectron2, image/mask has the same name):
# $PLANT_DATA_ROOT/img_dir/{train,val}/*.png original image (RGB, 1024x1024)
# $PLANT_DATA_ROOT/ann_dir/{train,val}/*.png mask (pixel value 0~4, 255 is ignored)
# PLANT_DATA_ROOT defaults to DETECTRON2_DATASETS, if not, "datasets/plant".
#
# Note: The class names in PLANT_CLASSES must be consistent with datasets/plant.json (index one-to-one correspondence),
# And ", " cannot appear in the class name (CAT-Seg code will split the class name according to ", ").

import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

PLANT_CLASSES = ("background", "podocarpus macrophyllus", "sasanqua camellia",
                 "loropetalum chinense", "ligustrum quihoui")
PLANT_COLORS = [
    [0, 0, 0],        # background - black
    [220, 20, 60],    # Podocarpus macrophyllus - red
    [60, 179, 113],   # Camellia sasanqua - green
    [30, 144, 255],   # Loropetalum chinense - blue
    [255, 215, 0],    # Ligustrum quihoui - Yellow
]


def _get_plant_meta():
    return {"stuff_classes": list(PLANT_CLASSES), "stuff_colors": PLANT_COLORS}


def register_all_plant(root):
    meta = _get_plant_meta()
    for split in ("train", "val"):
        image_dir = os.path.join(root, "img_dir", split)
        gt_dir = os.path.join(root, "ann_dir", split)
        name = f"plant_{split}_sem_seg"
        DatasetCatalog.register(
            name,
            lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="png"),
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            seg_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )


_root = os.getenv("PLANT_DATA_ROOT", os.getenv("DETECTRON2_DATASETS", "datasets/plant"))
register_all_plant(_root)
