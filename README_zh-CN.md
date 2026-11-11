# NurserySegBenchmark

[English](README.md) · [简体中文](README_zh-CN.md)

本仓库收录论文 *From UAV Imagery to Nursery Stock Value with Vision Foundation Models and an LLM Agent* 中基于无人机影像的苗木分割实验。实验比较不同语义分割模型识别影像中四类苗木及背景的表现。分割结果为论文所述的后续价值评估流程提供支持。

## 目录结构

| 目录 | 模型 | 框架 |
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

各目录的链接指向对应的实验说明。MMSegmentation 模型按[官方安装指南](https://mmsegmentation.readthedocs.io/en/latest/get_started.html)配置环境。SAM 3、Grounded-SAM2 和 CAT-Seg 实验均基于各自的上游仓库进行适配。

## 统一评估

所有模型使用同一验证集划分和类别索引。类别编号为 0（背景）、1（罗汉松，*Podocarpus macrophyllus*）、2（茶梅，*Camellia sasanqua*）、3（红花檵木，*Loropetalum chinense*）和 4（亮晶女贞，*Ligustrum quihoui*）。

- `common/prepare_data.py` 生成配对影像和类别索引掩膜。
- `common/convert_*_predictions.py` 将 SAM 3、Grounded-SAM2 和 CAT-Seg 的预测转换为类别索引掩膜。
- `common/check_pixel_metrics.py` 计算整体像素准确率、平均 IoU、平均 Dice 和平均 F-score 等像素级指标。
- `common/eval_boundary.py` 计算 Boundary IoU 和 Boundary F1。`common/summarize_metrics.py` 汇总结果，`common/make_boundary_report.py` 生成 Markdown 报告。

## 致谢

实验基于 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)、[Transformers](https://github.com/huggingface/transformers)、[InternImage](https://github.com/OpenGVLab/InternImage)、[VMamba](https://github.com/MzeroMiko/VMamba)、[Ultralytics](https://github.com/ultralytics/ultralytics)、[SAM 2](https://github.com/facebookresearch/sam2)、[SAM 3](https://github.com/facebookresearch/sam3)、[Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2) 和 [CAT-Seg](https://github.com/cvlab-kaist/CAT-Seg)。
