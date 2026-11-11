# InternImage-XL segmentation (default is 5 categories of seedlings, the number of categories can be switched, see the root directory README)

> **Note**: InternImage is not in the mmsegmentation official built-in model library, it relies on the author's customized
> **DCNv3 deformable convolution CUDA operator**. **Important**: The official `segmentation/mmseg_custom` registration code is based on
> mmcv 1.x + mmsegmentation 0.27.0 (old API) developed ([Official README](https://github.com/OpenGVLab/InternImage/tree/master/segmentation) Original words: "Our segmentation code is developed on top of MMSegmentation v0.27.0"),
> The mmcv 2.1.0 / mmsegmentation 1.2.2 combination used directly with this project will be in
> `mmseg_custom/core/__init__.py` reports `ModuleNotFoundError: No module named 'mmseg.core'`
> ——mmseg 1.x has completely dismantled and reorganized `mmseg.core`. This is caused by the fact that the official warehouse has not yet adapted to the new version of mmseg.
> Incompatible, it’s not because the configuration is wrong. Therefore, this project does not directly use the official `mmseg_custom`, but comes with an adapted backbone.
> Registration code (`common/internimage_backbone.py`, see Section 2 for details).

## 1. Model structure

The backbone uses **InternImage-XL** (335M parameters) - it can be used directly with UPerHead in the mmseg ecosystem, and there is an official
Maximum standard configuration of release weight. The larger InternImage-H (1.07B) officially only provides the Mask2Former version, which requires additional
Exclusive parameters such as `center_feature_scale`/`res_post_norm`/`level2_post_norm` are not implemented in this project (please refer to
[Official config](https://github.com/OpenGVLab/InternImage/tree/master/segmentation/configs) self-expanded):

```
Input (3, H, W)
   │
   ▼
Stem (2x stride-2 conv)
   │
   ▼
Stage1 (192ch, 5 blocks, DCNv3 deformable convolution + LayerScale + post-norm)
   │ Downsampling (2x2 conv stride2)
   ▼
Stage2 (384ch, 5  blocks)
   │ Downsampling
   ▼
Stage3 (768ch, 24 blocks) ← Main capacity
   │ Downsampling
   ▼
Stage4 (1536ch, 5  blocks)
   │
   ├───────────────────────────────┐ (4 stage feature output)
   ▼                              │
UPerHead (main head, PPM+FPN integration, GroupNorm)│
   → 1x1 conv → N class logits (N=num_classes, default 5) │
                                   │
FCNHead (auxiliary header, stage3 feature, GroupNorm)◄┘
```

DCNv3 deformable convolution allows the sampling position of each convolution kernel to be adaptively offset according to the input content, which combines the efficient inductive bias of CNN and
The ability to model long distance/irregular shapes similar to the attention mechanism has natural advantages in modeling irregular plot boundaries.

- Backbone network **loads official ImageNet-22K pre-trained weights by default** (`internimage_xl_22k_192to384.pth`, from
  [OpenGVLab official ADE20K UpperNet config](https://github.com/OpenGVLab/InternImage/blob/master/segmentation/configs/ade20k/upernet_internimage_xl_640_160k_ade20k.py),
  The first run will automatically download from `huggingface.co/OpenGVLab/InternImage`)
- Turn on `with_cp=True` (gradient checkpoint), and trade a small amount of speed for video memory with 335M parameters.
- The norm of decode_head/auxiliary_head uses **GroupNorm** instead of SyncBN: the PSPModule of UPerHead contains
  When "global pooling to 1x1" branch, `batch_size=1` (default for this configuration, limited by video memory), SyncBN training mode will report an error due to insufficient number of samples.
- Training/testing resolution 1024x1024; testing phase sliding window inference (crop 1024x1024, stride 768x768)
- Evaluation indicators: Accuracy (aAcc), mIoU, mDice, mFscore (including Precision/Recall) one-time output

## 2. Environment dependencies (important, additional custom operators need to be compiled)

```bash
# 1. Basic environment
pip install -U openmim
mim install mmengine mmcv==2.1.0 mmsegmentation==1.2.2
pip install tifffile pillow numpy timm

# 2. Obtain the DCNv3 CUDA operator (only the operator itself is needed, the Python registration code of backbone is included in this project)
git clone https://github.com/OpenGVLab/InternImage.git
cd InternImage/segmentation/ops_dcnv3
sh make.sh # Compile and install DCNv3 CUDA operator (needs to match the native CUDA/PyTorch version)
                     # After success, `python -c "import ops_dcnv3"` should be able to import normally
```

**Not required** `pip install -e .` installs the official `mmseg_custom` package, and there is no need to
`InternImage/segmentation` added to `PYTHONPATH`——`config.py`
`custom_imports = dict(imports=["common.plant_dataset", "common.internimage_backbone"])`
It has pointed to the backbone registration code that comes with this project and is adapted to mmcv 2.x/mmseg 1.x
(`common/internimage_backbone.py` is a verbatim version of the official 725-line source code, with only 4 old and new changes replaced.
API import does not change any model structure, and there is a complete change description at the top of the file). This step only requires the official warehouse to provide
DCNv3 operator itself.

> If DCNv3 cannot be compiled (for example, the CUDA version does not match, there is no GPU compilation environment), you can refer to the InternImage repository.
> "DCNv3_pytorch" pure PyTorch fallback implementation (slower but no need to compile), replaces in `config.py`
> `core_op="DCNv3"` is `core_op="DCNv3_pytorch"`.

## 3. Data preparation

The original data set is `class name/tif/*.tif` + `class name/train/*.tif` organized by category folders (see the root directory README for details).

```bash
python ../common/prepare_data.py --data_root /path/to/original dataset --out_dir /path/to/work_data_root --num_classes 5
```

## 4. Training code

```bash
cd 07_internimage
python train.py \
    --data_root /path/to/work_data_root \
    --work_dir ./work_dirs/internimage \
    --max_epochs 50 --batch_size 1 --lr 2e-5 \
    --amp --gpu_mem_fraction 0.4
```

> The default is **epoch training** (50 epochs, once for each round of verification), LR is scheduled according to epoch;
> **mIoU early stop** (patience=15) is enabled by default and can be turned off by adding `--no_early_stop`.
> Background accounts for about 84% of the pixels. CrossEntropyLoss uses inverse-frequency class weights to prevent background-only predictions.
> 335M parameter volume takes up a lot of video memory. If the video memory is insufficient, you can turn on `--amp` (mixed precision) or further reduce crop_size; share the graphics card
> It is recommended to use `--gpu_mem_fraction` to limit the occupancy ratio.

## 5. Inference code

```bash
cd 07_internimage
python infer.py \
    --checkpoint ./work_dirs/internimage/best_mIoU_epoch_xx.pth \
    --input /path/to/image_or_dir \
    --out_dir ./predictions
```
