"""
Register the five nursery plant classes with MMSegmentation.

Class indices are 0=background, 1=Podocarpus macrophyllus,
2=Camellia sasanqua, 3=Loropetalum chinense and 4=Ligustrum quihoui.

Import this module from each model configuration with

    custom_imports = dict(imports=['common.plant_dataset'], allow_failed_imports=False)

Set dataset_type to "PlantSegDataset" and set num_classes to 5 in both
decode_head and auxiliary_head. Apply these settings to all model configurations.

Generate the dataset with common/prepare_data.py --num_classes 5.
Images are stored in img_dir/{train,val} and masks in ann_dir/{train,val}.
Mask pixels contain class indices 0 through 4 in METAINFO class order.

To use different classes, update METAINFO.classes and METAINFO.palette.
The class order must match mask pixel values. Provide one RGB color per class.
When changing the class count, also update num_classes in each model configuration.
"""
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class PlantSegDataset(BaseSegDataset):
    '5 categories of nursery plant segmentation dataset: 0=background, 1=Podocarpus macrophyllus, 2=Camellia sasanqua, 3=Loropetalum chinense, 4=Ligustrum quihoui.'

    METAINFO = dict(
        classes=("background", "Podocarpus macrophyllus", "Camellia sasanqua", "Loropetalum chinense", "Ligustrum quihoui"),
        # The colors used for visualization (RGB) correspond to classes one-to-one, and can be color matched at will, regardless of accuracy.
        palette=[
            [0, 0, 0],        # background - black
            [220, 20, 60],    # Podocarpus macrophyllus - red
            [60, 179, 113],   # Camellia sasanqua - green
            [30, 144, 255],   # Loropetalum chinense - blue
            [255, 215, 0],    # Ligustrum quihoui - Yellow
        ],
    )

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix=".png",
            seg_map_suffix=".png",
            reduce_zero_label=False,  # background(0) is a real category, not "ignored", it must be kept False
            **kwargs,
        )
