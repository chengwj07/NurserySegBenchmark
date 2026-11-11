# U-Net segmentation (default is 5 categories of seedlings, the number of categories can be switched, see the root directory README)

## 1. Model structure

This implementation directly uses **mmsegmentation**’s built-in `UNet` backbone + `FCNHead`:

```
Input (3, H, W)
   │
   ▼
UNet Encoder-Decoder (5 levels of downsampling, base_channels=64)
   ├─ Encoder: 5 stages, 2 3x3 conv + BN + ReLU per stage, 2x2 maxpool downsampling between stages
   ├─ Decoder: 4 stages, each stage upsampling (transposed convolution/bilinear) + skip connection concat + 2 3x3 conv
   │
   ├──> Decode Head: FCNHead (in_channels=64, take the deepest decoding feature) → 1x1 conv → N class logits (N=num_classes, default 5)
   └──> Auxiliary Head: FCNHead (in_channels=128, take the penultimate layer features, loss weight 0.4, deep supervision accelerates convergence, only enabled during training)
```

- Loss function: `CrossEntropyLoss` (decode_head weight 1.0, auxiliary_head weight 0.4)
- Output: N-channel logits (N = num_classes, with 5 classes by default: background, Podocarpus macrophyllus, Camellia sasanqua, Loropetalum chinense and Ligustrum quihoui).
  argmax gets the final pixel-by-pixel category index mask
- Training/testing resolution 1024x1024 (consistent with the original slice size); sliding window inference is used in the testing phase (`slide` mode,
  crop 1024x1024, stride 768x768) to support large images of any size
- Evaluation indicators: Accuracy (aAcc), mIoU, mDice, mFscore (including Precision/Recall) one-time output

The number of parameters is small and training is fast, making it suitable as a baseline. **Known limitations**: mmseg's `UNet` backbone is a segmentation-specific encoding and decoding structure.
Non-standard classification backbone, the official does not provide ImageNet pre-training weights, so this model is still randomly initialized and trained (cannot meet the "default loading"
pre-trained weights").

## 2. Environmental dependencies

```bash
pip install -U openmim
mim install mmengine mmcv==2.1.0 mmsegmentation==1.2.2
pip install tifffile pillow numpy
```

## 3. Data preparation (common to all models, only needs to be run once)

Original data set structure (organized by category folder, tif and mask are both .tif, 1024x1024, see root directory README for details):
```
data_root/class name/tif/*.tif (original image slice)
data_root/class name/train/*.tif (mask, 0/1 for two categories, 0..N-1 category index for multiple categories)
```

```bash
cd seg_experiment
python common/prepare_data.py \
    --data_root /path/to/original dataset \
    --out_dir /path/to/work_data_root \
    --num_classes 5 \
    --val_ratio 0.2
```

## 4. Training code

```bash
cd 01_unet
python train.py \
    --data_root /path/to/work_data_root \
    --work_dir ./work_dirs/unet \
    --max_epochs 50 \
    --batch_size 2 \
    --lr 1e-4
```

- The default is **epoch training** (`EpochBasedTrainLoop`, 50 epochs, once for each round of verification), and the learning rate is scheduled according to epoch (2 epoch linear warmup + PolyLR)
- CrossEntropyLoss uses inverse-frequency class weights from training pixels to address class imbalance (about 84% background and 1% Ligustrum quihoui) and prevent background-only predictions.
- **mIoU early stopping** is enabled by default (`EarlyStoppingHook`, patience=15: training will stop if mIoU does not improve for 15 consecutive epochs), if you need to turn it off, add `--no_early_stop`
- Training logs, tensorboard/visualization, and checkpoints are all saved under `work_dir`
- By default, save the best checkpoint by `mIoU` (`save_best="mIoU"`, named `best_mIoU_epoch_x.pth`)
- Resume training at breakpoint: add `--resume`
- Load pre-trained/trained weights to continue training: `--load_from xxx.pth`

## 5. Inference code

```bash
cd 01_unet
python infer.py \
    --checkpoint ./work_dirs/unet/best_mIoU_epoch_xx.pth \
    --input /path/to/image_or_dir \
    --out_dir ./predictions
```

Output:
- `<name>_mask.png`: 0/255 binary mask
- `<name>_overlay.png`: a visualization of the prediction results superimposed on the original image
