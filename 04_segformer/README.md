# SegFormer (MiT-B5) segmentation (default 5 types of seedlings, the number of categories can be switched, see the root directory README)

## 1. Model structure

The backbone has been upgraded to **MiT-B5** (the one with the largest number of parameters and the highest official benchmark accuracy among the mmseg SegFormer series.
Standard configuration, formerly MiT-B0):

```
Input (3, H, W)
   │
   ▼
MixVisionTransformer encoder (4 stages, hierarchical downsampling 1/4, 1/8, 1/16, 1/32)
   ├─ Overlapped Patch Embedding (overlapped patch segmentation, replacing ViT’s hard segmentation)
   ├─ Efficient Self-Attention (sr_ratio downsampling of K/V to reduce the amount of calculation)
   ├─ Mix-FFN (3x3 depthwise conv introduces position information without explicit position encoding)
   ├─Number of layers of each stage: 3 / 6 / 40 / 3 (B0 is 2/2/2/2, B5 is greatly deepened)
   └─ 4 stage output channels: 64 / 128 / 320 / 512 (B0 is 32/64/160/256)
   │
   ▼
SegformerHead (full MLP decoder)
   ├─ The characteristics of each stage are unified into 768 channels through Linear.
   ├─ Concat after upsampling to 1/4 resolution
   ├─ 1-layer MLP fusion + 1x1 conv → N class logits (N=num_classes, default 5)
```

Lightweight, global receptive field (self-attention), minimalist decoder (no convolution), usually better than pure CNN baseline in remote sensing boundary blur scenes;
B5 has a significantly improved depth/width compared to B0 and is the SOTA configuration in the SegFormer series.

- Backbone network **loads ImageNet pre-trained weights by default** (mmseg official `mit_b5_20220624-658746d9.pth`,
  It will be automatically downloaded from `download.openmmlab.com`; if there is no network, please download it in advance and put the `config.py` in
  `pretrained` changed to local path)
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
cd 04_segformer
python train.py \
    --data_root /path/to/work_data_root \
    --work_dir ./work_dirs/segformer \
    --max_epochs 50 --batch_size 1 --lr 6e-5
```

> The default is **epoch training** (50 epochs, once for each round of verification), LR is scheduled according to epoch;
> **mIoU early stop** (patience=15) is enabled by default and can be turned off by adding `--no_early_stop`.
> Background accounts for about 84% of the pixels. CrossEntropyLoss uses inverse-frequency class weights to prevent background-only predictions.
> The memory usage of MiT-B5 is significantly higher than that of B0. If the memory is insufficient, crop_size can be further reduced or gradient accumulation can be used.

## 5. Inference code

```bash
cd 04_segformer
python infer.py \
    --checkpoint ./work_dirs/segformer/best_mIoU_epoch_xx.pth \
    --input /path/to/image_or_dir \
    --out_dir ./predictions
```
