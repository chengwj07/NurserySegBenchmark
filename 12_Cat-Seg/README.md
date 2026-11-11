# CAT-Seg: Cost Aggregation for Open-Vocabulary Semantic Segmentation [CVPR 2024 Highlight]

## Contents of this project: Fine-tuning of segmentation of five types of seedlings

> This catalog is based on [KU-CVLAB/CAT-Seg](https://github.com/cvlab-kaist/CAT-Seg) (CVPR 2024 Highlight,
> Pure CLIP open vocabulary semantic segmentation) official warehouse, on which **five types of seedling semantic segmentation** (0=background, 1=Podocarpus macrophyllus,
> Fine-tuning, evaluation and reasoning of 2=Camellia sasanqua, 3=Loropetalum chinense, 4=Ligustrum quihoui). For complete operating instructions see
> **[`nursery/README.md`](nursery/README.md)**, this section is a summary. Data division is consistent with 01~10
> (train 1948 pictures / val 486 pictures, 1024x1024).

### 1. New content of the project

| Documentation | Description |
|---|---|
| `configs/plant_vitb_384.yaml` | **New**, seedling fine-tuning configuration: ViT-B/16, `NUM_CLASSES=5`, 384 resolution, 30000 iter, fine-tuned from `weights/model_base.pth` (COCO-Stuff pre-training) |
| `datasets/plant.json` | **New**, 5 categories of English category names (`background / podocarpus / camellia sasanqua / loropetalum / ligustrum lucidum`) |
| `cat_seg/data/datasets/register_plant.py` | **New**, register `plant_train_sem_seg` / `plant_val_sem_seg` data set (`$PLANT_DATA_ROOT/img_dir|ann_dir/{train,val}` structure) |
| `nursery/train.sh` / `eval.sh` / `infer.py` | **New**, training/evaluation/inference entrance |
| `nursery/mmseg_evaluator.py` | **New**, mmseg style complete metric evaluator (IoU/Dice/F1/Precision/Recall/Acc + mean per class, nanmean semantics consistent with mmseg `IoUMetric`) |
| `nursery/eval_mmseg.py` | **New**, mmseg style evaluation entrance (reuse official Trainer, replace evaluator) |
| `weights/model_base.pth` | **New**, CAT-Seg-B pre-training weights (downloaded, 819MB) |

The official `plain_train_net.py` refers to the non-existent `add_mask_former_config`. This project uses `train_net.py`.
No need to modify the official code. Fine-tuning points: `CLIP_FINETUNE=attention` (only fine-tuning the Q/V projection and position encoding of CLIP attention,
Text encoder freeze), `CLIP_MULTIPLIER=0.01`, pixel-wise multi-label BCE loss.

### 2. Quick start

```bash
conda activate catseg && cd 12_Cat-Seg
export PLANT_DATA_ROOT=/data/ssd/haoweimu/zhenong_data/nursery/seg_exp/data

# Training (single card, default 30000 iter; dual card plus SOLVER.IMS_PER_BATCH 8)
sh nursery/train.sh configs/plant_vitb_384.yaml 1 output/plant

# Evaluation (mmseg style full accuracy table, recommended)
python nursery/eval_mmseg.py --config-file configs/plant_vitb_384.yaml --eval-only --num-gpus 1 \
    OUTPUT_DIR output/plant_eval MODEL.WEIGHTS output/plant/model_final.pth

# Single image/folder reasoning
python nursery/infer.py --config-file configs/plant_vitb_384.yaml \
    --weights output/plant/model_final.pth --input /path/to/img_or_dir --output-dir output/plant/infer
```

### 3. Validation set results

mmseg style complete accuracy table (`output/plant_eval/inference/sem_seg_evaluation.json`):
**aAcc 96.53 / mIoU 80.19 / mDice 88.64 / mPrecision 88.23 / mRecall 89.07**
Class-by-class IoU: background 96.00 | Podocarpus macrophyllus 80.41 | Camellia sasanqua 81.85 | Loropetalum chinense 79.10 | Ligustrum quihoui 63.61
(Official SemSegEvaluator same evaluation: mIoU 80.34 / fwIoU 93.59 / mACC 88.15 / pACC 96.59)

The category imbalance is obvious: background accounts for about 84.8%, Ligustrum quihoui only about 0.8%, and its IoU is also the lowest.

### 4. Environmental attention

- **RTX 50 series (sm_120) must have PyTorch ≥ 2.6 + cu128**, and detectron2 must be recompiled after upgrading torch;
- Do not directly `pip install -r requirements.txt` (old and will pull detectron2 from GitHub),
  For manual installation steps, see Section 1 of `nursery/README.md`.

---

## Official README original text (upstream content, unmodified)

This is our official implementation of CAT-Seg! 

[[arXiv](https://arxiv.org/abs/2303.11797)] [[Project](https://ku-cvlab.github.io/CAT-Seg/)] [[HuggingFace Demo](https://huggingface.co/spaces/hamacojr/CAT-Seg)] [[Segment Anything with CAT-Seg](https://huggingface.co/spaces/hamacojr/SAM-CAT-Seg)]<br>

by [Seokju Cho](https://seokju-cho.github.io/)\*, [Heeseong Shin](https://github.com/hsshin98)\*, [Sunghwan Hong](https://sunghwanhong.github.io), [Anurag Arnab](https://anuragarnab.github.io), [Paul Hongsuck Seo](https://phseo.github.io), [Seungryong Kim](https://cvlab.korea.ac.kr)

## Introduction
![](assets/fig1.png)
We introduce cost aggregation to open-vocabulary semantic segmentation, which jointly aggregates both image and text modalities within the matching cost.

For further details and visualization results, please check out our [paper](https://arxiv.org/abs/2303.11797) and our [project page](https://ku-cvlab.github.io/CAT-Seg/).

**❗️Update:** We released the code and pre-trained weights for CVPR version of CAT-Seg! 
Some major updates are:
- We now solely utilize CLIP as the pre-trained encoders, without additional backbones (ResNet, Swin)!
- We also fine-tune the text encoder of CLIP, yielding significantly improved performance!

For further details, please check out our updated [paper](https://arxiv.org/abs/2303.11797).
Note that the demos are still running on our previous version, and will be updated soon!

## :fire:TODO
- [x] Train/Evaluation Code (Mar 21, 2023)
- [x] Pre-trained weights (Mar 30, 2023)
- [x] Code of interactive demo (Jul 13, 2023)
- [x] Release code for CVPR version (Apr 4, 2024)
- [x] Release checkpoints for CVPR version (Apr 11, 2024)
- [ ] Demo update

## Installation
Please follow [installation](INSTALL.md). 

## Data Preparation
Please follow [dataset preperation](datasets/README.md).

## Demo
If you want to try your own images locally, please try [interactive demo](https://github.com/KU-CVLAB/CAT-Seg/tree/demo).

## Training
We provide shell scripts for training and evaluation. ```run.sh``` trains the model in default configuration and evaluates the model after training. 

To train or evaluate the model in different environments, modify the given shell script and config files accordingly.

### Training script
```bash
sh run.sh [CONFIG] [NUM_GPUS] [OUTPUT_DIR] [OPTS]

# For ViT-B variant
sh run.sh configs/vitb_384.yaml 4 output/
# For ViT-L variant
sh run.sh configs/vitl_336.yaml 4 output/
```

## Evaluation
```eval.sh``` automatically evaluates the model following our evaluation protocol, with weights in the output directory if not specified.
To individually run the model in different datasets, please refer to the commands in ```eval.sh```.

### Evaluation script
```bash
sh run.sh [CONFIG] [NUM_GPUS] [OUTPUT_DIR] [OPTS]

sh eval.sh configs/vitl_336.yaml 4 output/ MODEL.WEIGHTS path/to/weights.pth
```

## Pretrained Models
We provide pretrained weights for our models reported in the paper. All of the models were evaluated with 4 NVIDIA RTX 3090 GPUs, and can be reproduced with the evaluation script above.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">CLIP</th>
<th valign="bottom">A-847</th>
<th valign="bottom">PC-459</th>
<th valign="bottom">A-150</th>
<th valign="bottom">PC-59</th>
<th valign="bottom">PAS-20</th>
<th valign="bottom">PAS-20b</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: CAT-Seg (B) -->
<tr>
<td align="left">CAT-Seg (B)</a></td>
<td align="center">ViT-B/16</td>
<td align="center">12.0</td>
<td align="center">19.0</td>
<td align="center">31.8</td>
<td align="center">57.5</td>
<td align="center">94.6</td>
<td align="center">77.3</td>
<td align="center"><a href="https://huggingface.co/spaces/hamacojr/CAT-Seg-weights/resolve/main/model_base.pth">ckpt</a>&nbsp;
</tr>
<!-- ROW: CAT-Seg (L) -->
<tr>
<td align="left">CAT-Seg (L)</a></td>
<td align="center">ViT-L/14</td>
<td align="center">16.0</td>
<td align="center">23.8</td>
<td align="center">37.9</td>
<td align="center">63.3</td>
<td align="center">97.0</td>
<td align="center">82.5</td>
<td align="center"><a href="https://huggingface.co/spaces/hamacojr/CAT-Seg-weights/resolve/main/model_large.pth">ckpt</a>&nbsp;
</tr>

</tbody></table>


## Acknowledgement
We would like to acknowledge the contributions of public projects, such as [Zegformer](https://github.com/dingjiansw101/ZegFormer), whose code has been utilized in this repository.
We also thank [Benedikt](mailto:benedikt.blumenstiel@student.kit.edu) for finding an error in our inference code and evaluating CAT-Seg over various datasets!
## Citing CAT-Seg :cat::pray:

```BibTeX
@misc{cho2024catseg,
      title={CAT-Seg: Cost Aggregation for Open-Vocabulary Semantic Segmentation}, 
      author={Seokju Cho and Heeseong Shin and Sunghwan Hong and Anurag Arnab and Paul Hongsuck Seo and Seungryong Kim},
      year={2024},
      eprint={2303.11797},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
