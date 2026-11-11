# Swin Transformer (Swin-Large) + UpperNet segmentation (default 5 categories of seedlings, the number of categories can be switched, see the root directory README)

## 1. Model structure

The backbone has been upgraded to **Swin-Large (window12, input pre-training resolution 384)** (the number of parameters in the mmseg Swin series
The largest, official benchmark standard configuration with the highest accuracy, formerly Swin-Tiny):

```
Input (3, H, W)
   │
   ▼
Patch Embedding (4x4 patch)
   │
   ▼
Stage1 (192ch, 2 blocks, window self-attn + shifted-window self-attn alternation, window=12)
   │ Patch Merging (2x downsampling, doubling the channels)
   ▼
Stage2 (384ch,  2  blocks)
   │  Patch Merging
   ▼
Stage3 (768ch, 18 blocks) ← Main capacity, deeper than Tiny version (only 6 blocks)
   │  Patch Merging
   ▼
Stage4 (1536ch, 2  blocks)
   │
   ├─────────────────────────────┐ (4 stage feature output: 192/384/768/1536ch)
   ▼                             │
UPerHead (main head) │
   ├─ PPM (Pooling Pyramid, scales 1/2/3/6) processes the deepest features
   ├─ FPN top-down fusion of 4 stage features
   └─ 1x1 conv → N class logits (N=num_classes, default 5) │
                                            │
FCNHead (auxiliary head, stage3 feature, weight=0.4) ◄────┘
```

The number of channels/depth of Swin-Large is more than 2 times that of the Tiny version, making it the most accurate standard configuration in the Swin series.
Suitable as a strong baseline in comparative experiments.

- Backbone network** loads ImageNet-22K pre-trained weights by default** (mmseg official
  `swin_large_patch4_window12_384_22k_20220412-6580f57d.pth` will be automatically downloaded from
  `download.openmmlab.com` download; if there is no network, please download in advance and put `pretrained` in `config.py`
  Change to local path)
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
cd 05_swin_transformer
python train.py \
    --data_root /path/to/work_data_root \
    --work_dir ./work_dirs/swin \
    --max_epochs 50 --batch_size 1 --lr 6e-5
```

> The default is **epoch training** (50 epochs, once for each round of verification), LR is scheduled according to epoch;
> **mIoU early stop** (patience=15) is enabled by default and can be turned off by adding `--no_early_stop`.
> Background accounts for about 84% of the pixels. CrossEntropyLoss uses inverse-frequency class weights to prevent background-only predictions.
> Swin-Large has one of the highest memory usage among the 8 models. If the memory is insufficient, crop_size can be further reduced or gradient accumulation can be used.

## 5. Inference code

```bash
cd 05_swin_transformer
python infer.py \
    --checkpoint ./work_dirs/swin/best_mIoU_epoch_xx.pth \
    --input /path/to/image_or_dir \
    --out_dir ./predictions
```
