# DeepLabV3+ segmentation (default 5 categories of seedlings, the number of categories can be switched, see the root directory README)

## 1. Model structure

Using mmsegmentation native implementation, the backbone has been upgraded to **ResNetV1c-101** (mmseg DeepLabV3+ series
The standard configuration with the largest number of parameters and the highest official benchmark accuracy, originally ResNet-50):

```
Input (3, H, W)
   │
   ▼
ResNetV1c-101 backbone (dilated convolution, stem replaced by 3 3x3conv, 101 layers, twice as deep as R50)
   ├─ stage1 (stride 1, out 256ch) ──────────┐ Low-level features (c1)
   ├─ stage2 (stride 2)                       │
   ├─ stage3 (dilation=2, out 1024ch) ─────┐ │ (auxiliary header input)
   └─ stage4 (dilation=4, out 2048ch)      │   │
              │                            │   │
              ▼                            │   │
   DepthwiseSeparableASPPHead (main header) │ │
     ├─ ASPP: Hole rate (1,12,24,36) parallel convolution captures multi-scale context │
     ├─ Concat with low-level feature c1 (1x1 conv reduced to 48 channels) │
     └─ Depth separable convolution decoding → 1x1 conv → N class logits (N=num_classes, default 5) │
                                             │   │
   FCNHead (auxiliary head, deep supervision during training, weight=0.4) ◄──┘ │
                                                 ◄──┘
```

- Backbone network **loads ImageNet pre-trained weights by default** (`open-mmlab://resnet101_v1c`)
- Atrous convolution maintains the stage3/4 output resolution as 1/16, taking into account both the receptive field and details.
- Training/testing resolution 1024x1024; testing phase sliding window inference (crop 1024x1024, stride 768x768), supports large images of any size
- Evaluation indicators: Accuracy (aAcc), mIoU, mDice, mFscore (including Precision/Recall) one-time output

## 2. Environmental dependencies

```bash
pip install -U openmim
mim install mmengine mmcv==2.1.0 mmsegmentation==1.2.2
pip install tifffile pillow numpy
```

## 3. Data preparation

The original data set is `class name/tif/*.tif` + `class name/train/*.tif` organized by category folders (see the root directory README for details).
See root directory `common/prepare_data.py` (common to all models, only needs to be run once):

```bash
python ../common/prepare_data.py --data_root /path/to/original dataset --out_dir /path/to/work_data_root --num_classes 5
```

## 4. Training code

```bash
cd 02_deeplabv3plus
python train.py \
    --data_root /path/to/work_data_root \
    --work_dir ./work_dirs/deeplabv3plus \
    --max_epochs 50 --batch_size 2 --lr 0.01
```

> The default is **epoch training** (50 epochs, once for each round of verification), LR is scheduled according to epoch;
> **mIoU early stop** (patience=15) is enabled by default and can be turned off by adding `--no_early_stop`.
> Background accounts for about 84% of the pixels. CrossEntropyLoss uses inverse-frequency class weights to prevent background-only predictions.

## 5. Inference code

```bash
cd 02_deeplabv3plus
python infer.py \
    --checkpoint ./work_dirs/deeplabv3plus/best_mIoU_epoch_xx.pth \
    --input /path/to/image_or_dir \
    --out_dir ./predictions
```

Output `<name>_mask.png` (0/255 binary mask) and `<name>_overlay.png` (overlay visualization).
