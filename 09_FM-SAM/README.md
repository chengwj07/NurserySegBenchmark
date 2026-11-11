# YOLO26 + SAM2 semantic segmentation

A two-stage semantic segmentation scheme based on YOLO26 detection + SAM2 segmentation.

## Architecture

```
Original image ──> YOLO26 detection ──> Detection frame ──> Calculate center point ──> SAM2 point prompt segmentation ──> Semantic segmentation mask
         (fine-tuning) (official weight)
```

1. **YOLO26 fine-tuning**: Fine-tune the YOLO26 detector on your dataset to locate foreground objects.
2. **SAM2 point prompt segmentation**: Send the center point of the detection frame as a prompt to SAM2 (official weight) to generate an accurate segmentation mask
3. **Evaluation**: Compute IoU, Dice, accuracy, precision and recall on the validation set, including means over foreground and background.

## Directory structure

```
FM-SAM/
├── configs/
│ └── default.yaml # Configuration file
├── src/
│ ├── __init__.py # Environment variable settings
│ ├── data_prep.py # Mask -> YOLO detection label
│ ├── train_yolo.py # YOLO26 fine-tuning training
│ ├── sam_segment.py # SAM2 point prompt segmentation
│ ├── metrics.py # Evaluation indicators
│ ├── evaluate.py # Evaluation process
│ └── pipeline.py # End-to-end orchestration
├── weights/
│ └── sam2_t.pt # SAM2 weight (needs to be placed by yourself)
├── smoke_test.py              # Smoke test
├── run.py # CLI entry
├── requirements.txt
└── README.md
```

## Data format

```
data_root/
├── ann_dir/
│   ├── train/  (000.png, ...)   # Binary masks (0 = background, 255 = foreground)
│   └── val/
├── img_dir/
│ ├── train/ (000.png, ...) # Original image (same name as mask)
│   └── val/
```

There is a one-to-one correspondence between the mask and the original image file name, and both are in PNG format.

## How to use

### Smoke test (no real data required)

```bash
python smoke_test.py
# or
python run.py smoke
```

Automatically generate synthetic data, run the complete pipeline, and verify that all indicator outputs are normal.

### Complete process

1. Edit `configs/default.yaml` and set `data.data_root` to point to your data directory
2. Confirm that the `weights/sam2_t.pt` weights file exists
3. Run:

```bash
python run.py pipeline --config configs/default.yaml
```

### Step by step execution

```bash
# Prepare data only
python run.py prep --data-root /path/to/data

# Train YOLO only
python run.py train --config configs/default.yaml

# Evaluate only
python run.py eval --config configs/default.yaml
```

## Indicator description

For binary segmentation (foreground = 1, background = 0), foreground is the positive class in the confusion matrix.

| Indicators | Formulas | Description |
|------|------|------|
| IoU | TP/(TP+FP+FN) | Foreground intersection over union |
| Dice | 2TP/(2TP+FP+FN) | Foreground Dice coefficient |
| Acc | (TP+TN)/total | Pixel accuracy |
| Precision | TP/(TP+FP) | Foreground precision |
| Recall | TP/(TP+FN) | Foreground recall |
| mIoU | (IoU_fg+IoU_bg)/2 | Mean IoU over foreground and background |
| mDice | (Dice_fg+Dice_bg)/2 | Mean Dice over foreground and background |
| mAcc | (Acc_fg+Acc_bg)/2 | Pixel precision mean (=Acc) |
| mPrecision | (P_fg+P_bg)/2 | Mean precision over foreground and background |
| mRecall | (R_fg+R_bg)/2 | Mean recall over foreground and background |

> Acc is global pixel accuracy and is the same for foreground and background in this table, so mAcc = Acc.

## Configuration instructions

```yaml
data:
  data_root: "data" #Data root directory
  ann_dir: "ann_dir" # Mark directory
  img_dir: "img_dir" # Image directory

yolo:
  model: "yolo26n.pt" # Pretrained weights (or yolo26n.yaml built from scratch)
  epochs: 100
  imgsz: 640
  batch: 16
  device: "cpu" # or "0" use GPU

sam:
  model: "weights/sam2_t.pt" # SAM2 weight path

eval:
  conf_threshold: 0.25 # YOLO detection confidence threshold
  iou_threshold: 0.45 # YOLO NMS threshold
```

## Weight preparation

### SAM2 weight

Place the SAM2 weights file into `weights/sam2_t.pt`. Supported weight names:
- `sam2_t.pt` (SAM2 Tiny)
- `sam2_s.pt` (SAM2 Small)
- `sam2_b.pt` (SAM2 Base)
- `sam2_l.pt` (SAM2 Large)

### YOLO26 weight

- Online environment: use `yolo26n.pt` (automatically downloaded on first run)
- Offline environment: Use `yolo26n.yaml` to build from scratch instead (for testing only)

## Environmental requirements

- Python >= 3.10
- PyTorch >= 2.0
- Ultralytics >= 8.4.0 (requires YOLO26 and SAM2 support)
- OpenCV, NumPy, PyYAML, tqdm

```bash
pip install -r requirements.txt
```

## Technical details

### SAM2 feature cache

SAM2's image encoder is computationally expensive (~2-3 seconds/image), but mask decoding is fast (~0.05 seconds/point).
This project uses `SAM2Predictor.set_image()` to cache image features. Each point prompt only performs mask decoding.
Multi-point inference speed is increased by 20-50 times.

### Data preparation

Connected-component analysis converts each distinct foreground region in the mask into a YOLO bounding box.
The label file is written in the same directory as the image (.txt format), and Ultralytics automatically searches for it.
