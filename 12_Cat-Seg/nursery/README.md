# CAT-Seg Four types of seedling segmentation fine-tuning guide

This directory is [CAT-Seg](https://github.com/cvlab-kaist/CAT-Seg) (CVPR 2024 Highlight, open vocabulary semantic segmentation of pure CLIP) in
**Fine-tuned implementation of four types of seedling segmentation** (0=background, 1=Podocarpus macrophyllus, 2=Camellia sasanqua, 3=Loropetalum chinense, 4=Ligustrum quihoui) tasks.

## 1. Environment dependencies (you need to install them yourself)

> Note: The `requirements.txt` that comes with the warehouse is relatively old and will pull detectron2 from GitHub. **It is not recommended to directly `pip install -r requirements.txt`**.
> It is recommended to follow the following manual installation steps in order.

### 1.1 Basic environment

> ⚠️ **RTX 50 series graphics cards (5090/5080, etc. Blackwell, sm_120) must use PyTorch ≥ 2.6 + cu128**,
> Old versions of torch (1.13, etc.) only support sm_86 and will get stuck/report `not compatible` during training.
> For this type of graphics card, please directly follow the command after 2.1 below to install torch 2.6 or 2.8 (cu128).

```bash
# Create a new independent conda environment (Python 3.9~3.11, 3.8 is too old to install the new torch)
conda create -n catseg python=3.10 -y
conda activate catseg

# RTX 50 series (sm_120): install PyTorch 2.6+ / cu128
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu128
# Or conda method (recommended):
# conda install pytorch==2.6.0 torchvision==0.21.0 pytorch-cuda=12.8 -c pytorch -c nvidia

# Old graphics card (30/40 series): install the version that matches your CUDA, for example
# pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

Verify that the GPU is available and the architecture is recognized:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
# RTX 5090 should output something like: 2.6.0+cu128 True (12, 0)
```

### 1.2 detectron2

detectron2 must match your torch/CUDA version. It is recommended to compile from source code (the official wheel only covers some version combinations).
NOTE: **After upgrading torch, detectron2 must be recompiled** (it is bound to the compiled version of torch).

```bash
# If GitHub can be connected directly:
git clone https://github.com/facebookresearch/detectron2.git # Or use the existing detectron2/ subdirectory in the warehouse
cd detectron2
pip install -e . # Compilation requires gcc/g++/ninja and the CUDA toolkit (nvcc) matching torch
```

If your torch/CUDA combination has an official precompiled wheel, you can also use it (for example: torch 1.13+cu117):
```bash
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu117/torch1.13/index.html
```

Verification:
```bash
python -c "import detectron2, torch; print(detectron2.__version__, torch.__version__, torch.cuda.is_available())"
```

### 1.3 Other Python dependencies

```bash
pip install einops opencv-python pycocotools scipy ftfy regex timm imageio yacs fvcore
pip install open_clip_torch # The top level of the warehouse code import open_clip, which must be installed (when inferring that only OpenAI ViT-B is used, its weights are not actually downloaded)
```

### 1.4 Model weight

Two types of weights are used in training/inference:

1. **CLIP ViT-B/16 weight** (`clip.load("ViT-B/16")` is automatically downloaded from OpenAI, and `~/.cache/clip/` is saved).
   If the network cannot access OpenAI, you can manually put `ViT-B-16.pt` to `~/.cache/clip/`.
2. **CAT-Seg pre-training weights** (ViT-B model trained on COCO-Stuff, as a starting point for fine-tuning, has been downloaded to
   `weights/model_base.pth`, 819MB). If you need to re-download:
   ```bash
   wget -P weights https://huggingface.co/spaces/hamacojr/CAT-Seg-weights/resolve/main/model_base.pth
   ```

## 2. Data preparation

The data directory structure is consistent with the requirements of detectron2 `load_sem_seg` (the image/mask has the same name and the same suffix as `.png`):

```
<PLANT_DATA_ROOT>/
├── img_dir/
│ ├── train/ # Original image .png (RGB)
│ └── val/ # Original image .png (RGB)
└── ann_dir/
    ├── train/ # mask .png (pixel value 0~4, corresponding to the category order below)
    └── val/ # mask .png
```

This project already has ready-made data: `/data/ssd/haoweimu/zhenong_data/miaomu/seg_exp/data`
(train 1948 pictures, val 486 pictures, mask pixels 0~4 = background/Podocarpus macrophyllus/Camellia sasanqua/Loropetalum chinense/Ligustrum quihoui), you can use it directly.

Class map (`datasets/plant.json`, index = mask pixel value, text CLIP encoded, in English):

| Index | Chinese | English name in plant.json |
|----|------|----------------------|
| 0 | background | `background` |
| 1 | Podocarpus macrophyllus | `podocarpus macrophyllus` |
| 2 | Camellia sasanqua | `sasanqua camellia` |
| 3 | Loropetalum chinense | `loropetalum chinense` |
| 4 | Ligustrum quihoui | `ligustrum quihoui` |

> If you want to change the class name/number of classes: edit `datasets/plant.json` and
> `PLANT_CLASSES`/`PLANT_COLORS` in `cat_seg/data/datasets/register_plant.py` (the two class names must be consistent),
> And simultaneously change `configs/plant_vitb_384.yaml` to `NUM_CLASSES`. Note that `", "` cannot appear in the category name (the code will split the category name by commas).

## 3. Fine-tuning training

Run in the root directory of `12_Cat-Seg` (the class json and weight paths of relative paths are subject to the root directory):

```bash
cd /data/ssd/haoweimu/zhenong_data/miaomu/seg_exp/12_Cat-Seg
conda activate catseg

export PLANT_DATA_ROOT=/data/ssd/haoweimu/zhenong_data/miaomu/seg_exp/data

# Single card training (default 30000 iter, batch 4, 384 resolution, fine-tuned from model_base.pth)
sh miaomu/train.sh configs/plant_vitb_384.yaml 1 output/plant

# Dual SIM
sh miaomu/train.sh configs/plant_vitb_384.yaml 2 output/plant SOLVER.IMS_PER_BATCH 8

# Cover more hyperparameters, such as increasing resolution/number of iterations
sh miaomu/train.sh configs/plant_vitb_384.yaml 1 output/plant \
    INPUT.MIN_SIZE_TRAIN "(640,)" INPUT.CROP.SIZE "(640,640)" INPUT.SIZE_DIVISIBILITY 640 \
    SOLVER.MAX_ITER 50000
```

Key fine-tuning configuration (see `configs/plant_vitb_384.yaml`):

- `MODEL.WEIGHTS=weights/model_base.pth`: Fine-tuning of pre-trained weights from COCO-Stuff. head output layer when loading
  (Category 171→5) Mismatches will be automatically skipped and a warning will be printed, which is normal.
- `CLIP_FINETUNE=attention`: Only fine-tune the Q/V projection and position encoding of CLIP attention (official default),
  The text encoder is frozen throughout; `CLIP_MULTIPLIER=0.01` gives the CLIP branch a learning rate of 1/100.
- The loss is per-pixel multi-label BCE, `IGNORE_VALUE=255` is used for padding.

## 4. Evaluation

```bash
# Official SemSegEvaluator (mIoU / mACC / pACC, simple version)
sh miaomu/eval.sh configs/plant_vitb_384.yaml 1 output/plant_eval output/plant/model_final.pth
# Indicator output is in output/plant_eval/log.txt, including mIoU/mAcc, etc.
```

### 4.1 mmseg style complete accuracy table (recommended)

The official evaluator only outputs mIoU/mACC/pACC. Extended evaluator for `miaomu/eval_mmseg.py`
(`miaomu/mmseg_evaluator.py`) Calculate the complete metric consistent with **mmsegmentation IoUMetric** from the confusion matrix,
And print a table like mmseg training log + save JSON:

```bash
python miaomu/eval_mmseg.py --config-file configs/plant_vitb_384.yaml \
    --eval-only --num-gpus 1 \
    OUTPUT_DIR output/plant_eval MODEL.WEIGHTS output/plant/model_final.pth
```

Example console output:

```
==========================================================
Semantic Segmentation Results (mmseg style)
==========================================================
Class                       IoU     Dice      F1  Precision   Recall      Acc   Ratio%
----------------------------------------------------------
background                 97.32    98.64    98.64      98.90     98.39    98.39    84.05
podocarpus macrophyllus    85.10    91.95    91.95      93.02     90.91    90.91     5.20
sasanqua camellia          80.25    89.04    89.04      91.30     86.84    86.84     6.10
loropetalum chinense       88.44    93.87    93.87      95.00     92.77    92.77     3.02
ligustrum quihoui          76.91    86.95    86.95      89.47     84.53    84.53     1.63
----------------------------------------------------------
mIoU       85.60   (mean IoU over classes with GT pixels)
mDice      92.09
mFscore    92.09
mPrecision 93.54
mRecall    90.71
mAcc       90.71
aAcc       97.10
==========================================================
```

- Each type of indicator is defined by mmseg: `IoU=TP/(TP+FP+FN)`, `Dice=F1=2·TP/(2·TP+FP+FN)`,
  `Precision=TP/(TP+FP)`, `Recall=Acc=TP/(TP+FN)`; `Ratio%` is the proportion of GT pixels of this type
  (Used to observe category imbalance, this data set background is about 84%, Ligustrum quihoui is about 1.6%).
- The mean only counts categories with GT pixels in the validation set (mmseg's nanmean semantics). Zero sample categories display NaN and do not drag down the mean.
- Complete values (7 items per category + 7 averages, 42 keys in total) are saved to `OUTPUT_DIR/inference/sem_seg_evaluation.json` at the same time,
  To facilitate subsequent drawing/comparison; average keys such as `mIoU` and keys of each type such as `IoU-<Class name>` coexist.

## 5. Single picture/folder reasoning

```bash
python miaomu/infer.py --config-file configs/plant_vitb_384.yaml \
    --weights output/plant/model_final.pth \
    --input /path/to/img.png --output-dir output/plant/infer

# Folder batch inference
python miaomu/infer.py --config-file configs/plant_vitb_384.yaml \
    --weights output/plant/model_final.pth \
    --input /data/ssd/haoweimu/zhenong_data/miaomu/seg_exp/data/img_dir/val \
    --output-dir output/plant/infer_val
```

Output `*_pred.png` (palette mask) and `*_vis.png` (semi-transparent overlay).

## 6. Code change instructions (relative to official warehouse)

| Documentation | Changes |
|------|------|
| `cat_seg/data/datasets/register_plant.py` | **New**, register `plant_train_sem_seg`/`plant_val_sem_seg` data set |
| `cat_seg/data/datasets/__init__.py` | Mount the above registration module |
| `datasets/plant.json` | **New**, 5 English category names |
| `configs/plant_vitb_384.yaml` | **New**, seedling fine-tuning configuration |
| `miaomu/train.sh` / `eval.sh` / `infer.py` | **New**, training/evaluation/inference entrance |
| `miaomu/mmseg_evaluator.py` | **New**, mmseg style complete metric evaluator (per type IoU/Dice/F1/Precision/Recall/Acc) |
| `miaomu/eval_mmseg.py` | **New**, mmseg style evaluation entrance (reuse official Trainer, replace evaluator) |
| `weights/model_base.pth` | **New**, CAT-Seg-B pre-trained weights (downloaded) |

The official `plain_train_net.py` references the non-existent `add_mask_former_config`. This project uses `train_net.py` without modifying the official code.
