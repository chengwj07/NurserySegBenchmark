# Nursery Stock Segmentaton Benchmark

[English](README.md) · [简体中文](README_zh-CN.md)

This repository contains the UAV nursery stock segmentation experiments for *From UAV Imagery to Asset Value with Vision Foundation Models and an LLM Agent*. The experiments compare semantic segmentation models for identifying four nursery stock categories and background in UAV imagery. The segmentation results support the downstream valuation workflow described in the paper.

## Repository structure

| Directory | Model | Framework |
| --- | --- | --- |
| [01_unet](01_unet/README.md) | U-Net | MMSegmentation |
| [02_deeplabv3plus](02_deeplabv3plus/README.md) | DeepLabV3+ | MMSegmentation |
| [03_hrnet](03_hrnet/README.md) | HRNet | MMSegmentation |
| [04_segformer](04_segformer/README.md) | SegFormer | MMSegmentation |
| [05_swin_transformer](05_swin_transformer/README.md) | Swin Transformer | MMSegmentation |
| [06_oneformer](06_oneformer/README.md) | OneFormer | Hugging Face Transformers |
| [07_internimage](07_internimage/README.md) | InternImage | MMSegmentation |
| [08_vmamba](08_vmamba/README.md) | VMamba | MMSegmentation |
| [09_FM-SAM](09_FM-SAM/README.md) | FM-SAM | Ultralytics YOLO26 + SAM 2 |
| [10_SAM3](10_SAM3/README.md) | SAM 3 | Meta SAM 3 |
| [11_Grounded-SAM2](11_Grounded-SAM2/README.md) | Grounded-SAM2 | GroundingDINO + SAM 2 |
| [12_Cat-Seg](12_Cat-Seg/README.md) | CAT-Seg | CAT-Seg / Detectron2 |

Each directory links to its experiment documentation. The MMSegmentation models use the [official installation guide](https://mmsegmentation.readthedocs.io/en/latest/get_started.html). The SAM 3, Grounded-SAM2 and CAT-Seg experiments adapt their respective upstream repositories.

## Shared evaluation

Use the same validation split and class indices for all models. Class IDs are 0 for background, 1 for *Podocarpus macrophyllus*, 2 for *Camellia sasanqua*, 3 for *Loropetalum chinense* and 4 for *Ligustrum quihoui*.

- `common/prepare_data.py` prepares paired images and class-index masks.
- `common/convert_*_predictions.py` converts SAM 3, Grounded-SAM2 and CAT-Seg predictions into class-index masks.
- `common/check_pixel_metrics.py` computes pixel metrics including overall accuracy, mean IoU, mean Dice and mean F-score.
- `common/eval_boundary.py` computes Boundary IoU and Boundary F1. `common/summarize_metrics.py` combines the results and `common/make_boundary_report.py` generates the Markdown report.

## Acknowledgments

The experiments build on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [Transformers](https://github.com/huggingface/transformers), [InternImage](https://github.com/OpenGVLab/InternImage), [VMamba](https://github.com/MzeroMiko/VMamba), [Ultralytics](https://github.com/ultralytics/ultralytics), [SAM 2](https://github.com/facebookresearch/sam2), [SAM 3](https://github.com/facebookresearch/sam3), [Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2) and [CAT-Seg](https://github.com/cvlab-kaist/CAT-Seg).
