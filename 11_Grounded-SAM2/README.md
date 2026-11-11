# Grounded-SAM2: Text prompt detection + segmentation (five types of seedlings)

Grounded-SAM2 = a two-stage combination of **GroundingDINO** (open vocabulary detector) + **SAM 2** (segmentation model):
Given a text description, GroundingDINO detects all boxes matching the instance, and SAM2 uses the center point of the detection box as a prompt to segment the accurate mask.
The instance masks are then merged into semantic category maps.

This project was carried out on five types of seedlings (0=background, 1=Podocarpus macrophyllus, 2=Camellia sasanqua, 3=Loropetalum chinense, 4=Ligustrum quihoui)
**GroundingDINO's LoRA fine-tuning** (allowing the detector to learn to distinguish tree species with similar crown shapes) and full-process evaluation, evaluation caliber
(Full set cumulative confusion matrix + nanmean of aAcc/mIoU/mDice/mFscore/mPrecision/mRecall) vs.
01~10 are consistent and can be directly compared horizontally.

## Directory structure

```
11_Grounded-SAM2/
├── Grounded-SAM-2/ # IDEA-Research/Grounded-SAM-2 official warehouse (clone)
│ ├── eval_plant_metrics.py # This project: two-stage evaluation (mmseg style accuracy table)
│ ├── grounded_sam2_local_demo.py # Single graph inference demo (GDINO detection + SAM2 segmentation)
│ ├── grounded_sam2_tracking_camera_with_continuous_id.py # Camera video continuous ID tracking
│ ├── grounded_sam2_tracking_demo_with_continuous_id_plus.py# Video continuous ID tracking (including reverse tracking)
│ └── outputs/ # Evaluate prediction dump + accuracy table
├── GroundingDINO/ # IDEA-Research/GroundingDINO official warehouse (clone, LoRA micro-call)
├── ckpt/ # GroundingDINO pre-training weights (swint_ogc / swinb_cogcoor) + demo diagram
├── outputs/ # 20 epochs of LoRA adapter + merged complete weights
├── prepare_dataset.py # mask → instance-level bbox JSON
├── grounding_dino_criterion.py # Self-contained training loss (Hungarian matching + focal CE + L1/GIoU)
├── lora_finetune_grounding_dino.py # LoRA fine-tuning entrance (freeze the entire amount, inject adapter, end merge export)
├── smoke_test.py # positive_map / collate / criterion / gradient flow smoke test
└── tools/
    ├── run_eval.py # Namespace mapping package, run Grounded-SAM-2/eval_plant_metrics.py
    ├── box_eval.py # Box-level detection evaluation (P/R@0.5 threshold/NMS scan)
    └── stubs/ # Missing dependency stub (supervision.py, etc.)
```

## 1. Data preparation: `prepare_dataset.py`

Convert the semantic mask of `seg_exp/data/` (`ann_dir`, pixels 0~4) into the one required for GroundingDINO training
Instance-level bbox annotation JSON (connected domain analysis is done for each category value, `--min_area` filters noise connected domain):

```bash
python prepare_dataset.py --data_root ../data --min_area 200
```

- Output `data/train_annotations.json` / `data/val_annotations.json`
  (train 1944 pictures / 108701 boxes, val 485 pictures / 24816 boxes)
- Chinese class name → English Latin scientific name is mapped to `CLASS_NAME_MAP` at the top of the script. For new categories, just add a line here;
  The text backbone of GroundingDINO is bert-base-uncased. **Chinese is not supported and English prompts must be given**

## 2. LoRA fine-tuning GroundingDINO: `lora_finetune_grounding_dino.py`

The goal of fine-tuning is to let GroundingDINO learn to distinguish four tree species with similar crown shapes (inter-category confusion is serious when detected directly with official pre-trained weights).

- **Freezing Strategy**: Freeze all parameters and only inject LoRA into the Linear layer related to "Text-Visual Alignment"
  （`cross_attn` / `text_layers` / `fusion_layers` / `bert`，r=8, alpha=16, dropout=0.05）；
  `bbox_embed` output head remains fully trainable
- **Text structure**: For each picture, scramble all the variety names and spell them into captions, use the official `build_captions_and_token_span` +
  `create_positive_map_from_span` only marks spans that actually appear in the current picture as positive samples.
  Let the model learn to "distinguish similar varieties" instead of remembering the caption order
- **loss**: self-contained `grounding_dino_criterion.py` (`HungarianMatcher` + focal CE + L1 + GIoU),
  Consistent with official GroundingDINOCriterion design
- **Save**: Each epoch saves a small adapter (`outputs/lora_species_epochN.pt`);
  After training, the last adapter is automatically merged into Grounded-SAM-2 `load_model` with complete weights that can be loaded directly.
  `outputs/groundingdino_swint_ogc_finetuned.pth` (There is a "normal model zero missing load" self-check before exporting)

```bash
python lora_finetune_grounding_dino.py \
    --data_root ../data \
    --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --ckpt groundingdino_swint_ogc.pth \
    --epochs 20 --batch_size 4 --out_dir outputs
```

Run `python smoke_test.py` first to verify data alignment and gradient flow (`SMOKE TEST PASSED` should be output).

## 3. Evaluation

### 3.1 Full-process pixel evaluation (recommended)

`Grounded-SAM-2/eval_plant_metrics.py`: GroundingDINO image-by-image text detection for val
(caption = 4 class names) → SAM2 uses the center point of each detection frame as a point prompt (labels=1) to segment frame by frame →
Instance masks are merged into semantic category maps according to detection confidence from high to low → with `ann_dir` ground truth cumulative confusion matrix, output mmseg accuracy table.
GroundingDINO defaults to loading the fine-tuned weight `outputs/groundingdino_swint_ogc_finetuned.pth` first.

```bash
cd Grounded-SAM-2
python eval_plant_metrics.py # Inference + accuracy table
python eval_plant_metrics.py --skip-infer --pred-json outputs/grounded_sam2_val_predictions.json # Only recalculate the accuracy table
```

**Validation set results (fine-tuned GroundingDINO + SAM2, stored in `Grounded-SAM-2/outputs/grounded_sam2_eval/`)**:

```
aAcc 93.81 / mIoU 67.86 / mAcc 85.15 / mDice 79.65 / mFscore 79.65 / mPrecision 77.01 / mRecall 85.15
Class-by-class IoU: Podocarpus macrophyllus 74.52 | Camellia sasanqua 67.59 | Loropetalum chinense 62.52 | Ligustrum quihoui 41.75
```

Ligustrum quihoui (GT pixels account for about 1.6%, with the fewest samples) has the lowest IoU and is the current main bottleneck.

### 3.2 Frame-level detection and evaluation: `tools/box_eval.py`

Only evaluate the detector (no SAM2): the predicted box matches the GT box by IoU≥0.5 + category (COCO style),
Scan the confidence threshold / NMS and output P@0.5 / R@0.5 and the optimal configuration:

```bash
python tools/box_eval.py
python tools/box_eval.py --pred-json <Existing prediction dump> # Skip inference and only scan
```

### 3.3 Namespace packaging

`tools/run_eval.py` maps the `grounding_dino.*` namespace in the vendored eval script to
**`groundingdino.*` package that has been patched to run** under transformers 5.x / numpy 2.x / timm 0.4,
Avoid environment dependency issues when running vendored scripts directly:

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=../tools/stubs \
  /home/server/Python_env/gd/bin/python \
  ../tools/run_eval.py \
  --gdino-ckpt ../outputs/groundingdino_swint_ogc_finetuned.pth \
  --sam2-ckpt ../weights/sam2_l.pt \
  --sam2-config configs/sam2/sam2_hiera_l.yaml
```

## 4. Inference and tracking demo

- `Grounded-SAM-2/grounded_sam2_local_demo.py`: Single image text prompt detection + segmentation visualization
  (Change to `TEXT_PROMPT` / `IMG_PATH`)
- `grounded_sam2_tracking_camera_with_continuous_id.py`: camera real-time video streaming,
  Perform GDINO detection + SAM2 segmentation on each frame, maintain **continuous object ID** across frames and visualize tracking results
- `grounded_sam2_tracking_demo_with_continuous_id_plus.py`: Continuous ID tracking for offline videos
  (Including reverse tracking, ID stable across frames), extension from the official tracking demo

## 5. Environment compatibility fix (relative to upstream warehouse)

| Documentation | Changes |
|---|---|
| `GroundingDINO/groundingdino/models/GroundingDINO/bertwarper.py` | transformers 5.x removes the `get_extended_attention_mask` / `invert_attention_mask` / `get_head_mask` private methods on `PreTrainedModel`, locally replicates the 4.x behavior and skips the useless head_mask branch |
| `GroundingDINO/groundingdino/models/GroundingDINO/ms_deform_attn.py` | CUDA < 12.8 Compatibility Fix |
| `Grounded-SAM-2/sam2/*.yaml` | Configuration update aligned with sam2 package version |
| `tools/stubs/` | Packages that the vendored demo depends on but whose environment is missing (such as supervision) are replaced with stubs |

> The data, category conventions and evaluation standards are consistent with `seg_exp/data/` and `09_FM-SAM`;
> Reuse `09_FM-SAM/weights/sam2_l.pt` when SAM2 weight is required (see 3.3 command),
> Or press `Grounded-SAM-2/checkpoints/download_ckpts.sh` to download.
