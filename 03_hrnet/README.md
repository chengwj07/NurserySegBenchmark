# HRNet (HRNetV2-W48) segmentation (default is 5 categories of seedlings, the number of categories can be switched, see the root directory README)

## 1. Model structure

The backbone has been upgraded to **HRNetV2-W48** (the one with the largest number of channels and the highest official benchmark accuracy in the mmseg HRNet series
Standard configuration, formerly W18):

```
Input (3, H, W)
   │
   ▼
Stem (2x stride-2 conv, /4 resolution)
   │
   ▼
Stage1: Single branch Bottleneck x4 (1/4 resolution, 64ch)
   │ Split out a second low-resolution branch
   ▼
Stage2: 2 branches in parallel (48ch@1/4, 96ch@1/8), multi-scale feature fusion is repeated between branches
   │ Split into a third branch
   ▼
Stage3: 3 branches parallel (48/96/192ch @ 1/4,1/8,1/16), 4 fusion modules
   │ Split out the fourth branch
   ▼
Stage4: 4 branches parallel (48/96/192/384ch @ 1/4,1/8,1/16,1/32), 3 fusion modules
   │
   ▼
FCNHead: Concat (720 channels) after upsampling all 4 branch features to 1/4 resolution
   → 1x1 conv → N class logits (N=num_classes, default 5)
```

Different from the encoding and decoding structure of "downsampling and then upsampling" such as U-Net/DeepLab, HRNet maintains multi-resolution branches in parallel throughout the entire process and fuses them repeatedly.
High-resolution details are better preserved and are suitable for segmentation tasks with fine edges (such as farmland/plot boundaries).

- Backbone network **loads ImageNet pre-trained weights by default** (`open-mmlab://msra/hrnetv2_w48`)
- Training/testing resolution 1024x1024; testing phase sliding window inference (crop 1024x1024, stride 768x768)
- Evaluation indicators: Accuracy (aAcc), mIoU, mDice, mFscore (including Precision/Recall) one-time output

## 2. Environmental dependencies

```bash
pip install -U openmim
mim install mmengine mmcv==2.1.0 mmsegmentation==1.2.2
pip install tifffile pillow numpy
```

## 3. Data preparation

The original data set is `class name/tif/*.tif` + `class name/train/*.tif` organized by category folders (see the root directory README for details).

```bash
python ../common/prepare_data.py --data_root /path/to/original dataset --out_dir /path/to/work_data_root --num_classes 5
```

## 4. Training code

```bash
cd 03_hrnet
python train.py \
    --data_root /path/to/work_data_root \
    --work_dir ./work_dirs/hrnet \
    --max_epochs 50 --batch_size 2 --lr 0.01
```

> The default is **epoch training** (50 epochs, once for each round of verification), LR is scheduled according to epoch;
> **mIoU early stop** (patience=15) is enabled by default and can be turned off by adding `--no_early_stop`.
> Background accounts for about 84% of the pixels. CrossEntropyLoss uses inverse-frequency class weights to prevent background-only predictions.

## 5. Inference code

```bash
cd 03_hrnet
python infer.py \
    --checkpoint ./work_dirs/hrnet/best_mIoU_epoch_xx.pth \
    --input /path/to/image_or_dir \
    --out_dir ./predictions
```
