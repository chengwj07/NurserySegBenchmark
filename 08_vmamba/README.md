# VMamba-Base segmentation (default 5 categories of seedlings, the number of categories can be switched, see the root directory README)

> **Note**: VMamba is also not in the official built-in model library of mmsegmentation and relies on its customized
> **selective_scan CUDA operator** (core of the Mamba state space model). This project follows the official recommended method,
> Overlay [MzeroMiko/VMamba](https://github.com/MzeroMiko/VMamba) provided by the official warehouse `segmentation/` directory
> backbone registration code, trained on mmsegmentation.

## 1. Model structure

Backbone uses **VMamba-Base (s2l15)** (122M parameters) - the largest of the three officially released Tiny/Small/Base levels:

```
Input (3, H, W)
   │
   ▼
Patch Embedding (4x4 patch, 128ch)
   │
   ▼
VSS Block x2 (2D-Selective-Scan: Expand the feature map into a sequence in 4 directions,
   │ Do Mamba selective state space scanning and fusion separately, linear complexity captures global dependencies)
   │ Patch Merging (2x downsampling)
   ▼
VSS Block x2
   │  Patch Merging
   ▼
VSS Block x15 ← Where the main capacity is
   │  Patch Merging
   ▼
VSS Block x2
   │
   ├─────────────────────────────┐ (4 stage feature output: 128/256/512/1024ch)
   ▼                              │
UPerHead (main head, PPM+FPN integration, GroupNorm)│
   → 1x1 conv → N class logits (N=num_classes, default 5) │
                                   │
FCNHead (auxiliary header, stage3 feature, GroupNorm)◄┘
```

VMamba uses a state space model (SSM) to replace self-attention, and the sequence length linear complexity (instead of the quadratic complexity of Transformer),
It has theoretical efficiency advantages over Swin/SegFormer on large high-resolution remote sensing images, and at the same time, through 2D-Selective-Scan
The ability to model spatial continuity of visual data is preserved.

- Backbone key parameters are taken from the official segmentation config (`segmentation/configs/vssm1/upernet_vssm_..._base.py`):
  `ssm_d_state=1`, `ssm_ratio=2.0`, `forward_type="v05_noz"`, `norm_layer="ln2d"`, etc., only change
  Change `dims`/`depths` to Base file (`dims=128, depths=(2,2,15,2)`)
- The norm of decode_head/auxiliary_head uses **GroupNorm** instead of SyncBN: the reason is the same as InternImage/Swin
  (When UPerHead’s PSPModule is set to `batch_size=1`, SyncBN training mode will report an error due to insufficient sample number)
- Training/testing resolution 1024x1024; testing phase sliding window inference (crop 1024x1024, stride 768x768)
- Evaluation indicators: Accuracy (aAcc), mIoU, mDice, mFscore (including Precision/Recall) one-time output

## 2. Environment dependencies (important, additional custom operators need to be compiled)

```bash
# 1. Basic environment
pip install -U openmim
mim install mmengine mmcv==2.1.0 mmsegmentation==1.2.2
pip install tifffile pillow numpy timm einops

# 2. Get the VMamba official repository and selective_scan operator
git clone https://github.com/MzeroMiko/VMamba.git
cd VMamba/kernels/selective_scan
pip install -e . # Compile selective_scan CUDA operator (needs CUDA/PyTorch version matching)

# 3. Install/expose split backbone registration code
cd ../../segmentation
pip install -e . # Or add this directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/VMamba/segmentation
```

> If the target environment cannot compile CUDA operators, you can refer to the pure PyTorch reference implementation of "without CUDA" in the VMamba repository README.
> (slower speed). In addition, the official mentioned that `torch.backends.cudnn.enabled=True` will significantly slow down VMamba on some machines.
> If you find it is abnormally slow, you can close it according to official recommendations in `vmamba.py`.

## 3. Pre-training weights

The `pretrained` field of backbone in `config.py` is left blank (`""`) - this is VMamba’s official way of splitting config itself.
You need to manually download and fill in the local path:

1. Go to [VMamba official releases](https://github.com/MzeroMiko/VMamba/releases) or the warehouse README
   Classification checkpoints table, find the weight file corresponding to **VMamba-Base (s2l15)**
2. After downloading, change `pretrained=""` in `config.py` to the local path, for example:
   ```python
   pretrained="/path/to/vssmbase_xxx_ckpt_epoch_xxx.pth",
   ```
3. **Be sure to confirm that the downloaded weight is consistent with `forward_type`/`downsample_version`/`patchembed_version` in `config.py`
   Waiting for structural parameters to match** - VMamba has iterated through multiple versions (v0/v1/v2), and the backbone structural parameters do not match, resulting in weight loading errors or
   Silence degenerates into random initialization. If you encounter a shape mismatch, check the segmentation config of the corresponding version in the VMamba repository.

## 4. Data preparation

The original data set is `class name/tif/*.tif` + `class name/train/*.tif` organized by category folders (see the root directory README for details).

```bash
python ../common/prepare_data.py --data_root /path/to/original dataset --out_dir /path/to/work_data_root --num_classes 5
```

## 5. Training code

```bash
cd 08_vmamba
python train.py \
    --data_root /path/to/work_data_root \
    --work_dir ./work_dirs/vmamba \
    --max_epochs 50 --batch_size 1 --lr 2e-5 \
    --amp --gpu_mem_fraction 0.4
```

> The default is **epoch training** (50 epochs, once for each round of verification), LR is scheduled according to epoch;
> **mIoU early stop** (patience=15) is enabled by default and can be turned off by adding `--no_early_stop`.
> Background accounts for about 84% of the pixels. CrossEntropyLoss uses inverse-frequency class weights to prevent background-only predictions.

## 6. Inference code

```bash
cd 08_vmamba
python infer.py \
    --checkpoint ./work_dirs/vmamba/best_mIoU_epoch_xx.pth \
    --input /path/to/image_or_dir \
    --out_dir ./predictions
```
