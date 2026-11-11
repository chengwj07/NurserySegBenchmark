# OneFormer segmentation (default is 5 categories of seedlings, the number of categories can be switched, see the root directory README)

> **Note**: OneFormer is not officially supported by mmsegmentation (it is a unified panorama/instance/semantic segmentation framework,
> The official implementation is based on Detectron2). This model uses **HuggingFace `transformers`** instead.
> `OneFormerForUniversalSegmentation` implementation, loading semantic segmentation mode (`task_inputs=["semantic"]`).

## 0. File description

- `dataset.py`: Read the `img_dir/ann_dir` structure produced by `common/prepare_data.py`
- `metrics.py`: Metric calculation that is exactly the same as mmsegmentation `IoUMetric` (see Section 7)
- `engine.py`: `train.py`/`eval.py`/`infer.py` shared model building, saving/loading, verification loop, log formatting
- `train.py`: training entrance (mmseg style iter-by-iter/epoch-by-epoch log, only saves best+last, supports breakpoint continuation of training)
- `eval.py`: Use a saved checkpoint to recalculate the indicator on the validation set
- `infer.py`: perform inference on single/entire directory images

## 1. Model structure

Uses **Swin-Large** backbone by default (`shi-labs/oneformer_ade20k_swin_large`) - published on HuggingFace
The configuration with the largest weight among OneFormer weights and can be used directly without additional native operators:

```
Input Image + Text Query "the task is semantic"
        │                        │
        ▼                        ▼
   Swin-Large Backbone Text Encoder (CLIP text tower, frozen/tweakable)
   (Window self-attention + shift window)
        │                        │
        ▼                        ▼
   Pixel Decoder (multi-scale deformable attention, similar to Mask2Former)
        │
        ▼
   Task-Conditioned Transformer Decoder
     ├─ Query and text task token joint modeling (task-dynamic mask/class queries)
     ├─ Multi-layer cross-attention iteratively refines the query on multi-scale pixel features
        │
        ▼
   Predict a set of (class, binary mask) pairs
        │
        ▼
   Semantic segmentation post-processing (post_process_semantic_segmentation):
   Merge the query masks using their class confidence scores to produce a per-pixel class map (0 = background, 1..N-1 = foreground classes).
```

The key features of OneFormer: use the same set of model weights + different text tasks prompt("panoptic"/"instance"/"semantic")
You can switch the segmentation paradigm; this project only uses its semantic segmentation mode. The category number/category name is determined by the `CLASSES` constant at the top of `engine.py`
Determine, the default is 5 types of seedlings (`background, Podocarpus macrophyllus, Camellia sasanqua, Loropetalum chinense, Ligustrum quihoui`), see the root directory README for modifications
Section "Switching Category Number/Name".

**Alternative**: There is also `shi-labs/oneformer_ade20k_dinat_large` (DiNAT-Large backbone) on HuggingFace, OneFormer
The paper reports that its accuracy is slightly better than Swin-Large at the same level, but it requires additional installation of the `natten` library (including custom CUDA operators, which is
CUDA/PyTorch version is more sensitive). If you don't mind installing `natten`, you can replace it with:

```bash
pip install natten
python train.py ... --pretrained shi-labs/oneformer_ade20k_dinat_large
```

## 2. Environmental dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate pillow numpy tifffile
```

The default Swin-Large backbone does not require any additional native operators and can be run after installing the above.

## 3. Data preparation

The original data set is `class name/tif/*.tif` + `class name/train/*.tif` organized by category folders (see the root directory README for details):

```bash
python ../common/prepare_data.py --data_root /path/to/original dataset --out_dir /path/to/work_data_root --num_classes 5
```

`dataset.py` directly reads the converted `img_dir/{split}` + `ann_dir/{split}` structure and shares the same copy with the other 7 models
Preprocessed product, no additional conversion required.

## 4. Training code

```bash
cd 06_oneformer
python train.py \
    --data_root /path/to/work_data_root \
    --output_dir ./work_dirs/oneformer \
    --epochs 50 --batch_size 2 --lr 1e-5 \
    --amp --gpu_mem_fraction 0.4
```

The first run will automatically download the pre-trained weights specified by `--pretrained` from HuggingFace Hub (default
`shi-labs/oneformer_ade20k_swin_large`), and use `ignore_mismatched_sizes=True` to change the classification header from
ADE20K 150 categories are replaced with the number of categories configured in `engine.py::CLASSES` (default 5 categories, you will see a bunch of them in the training log
"newly initialized" prompt, this is a normal phenomenon, not an error).

### Training log (mmseg style)

Each `--log_interval` (default 20) iter prints a line of training log, and prints a line of verification log at the end of each epoch. The format is as follows
The `LoggerHook` of mmsegmentation is close, making it easy to compare with the logs of the other 7 models:

```
2024/01/01 12:00:00 - oneformer - INFO - Epoch(train) [1][20/120]  lr: 1.0000e-05  eta: 0:45:12  time: 0.5123  data_time: 0.0321  memory: 8452  loss: 2.3456
2024/01/01 12:01:30 - oneformer - INFO - Epoch(train) [1] Completed, average loss: 1.8234
2024/01/01 12:02:10 - oneformer - INFO - Epoch(val) [1]  loss: 1.5678  aAcc: 78.32  mIoU: 55.14  mAcc: 62.23  mDice: 68.45  mFscore: 68.02  mPrecision: 70.80  mRecall: 62.23
    per-class IoU -> background: IoU 71.20  |  Podocarpus macrophyllus: IoU 52.10  |  Camellia sasanqua: IoU 48.33  |  Loropetalum chinense: IoU 41.05  |  Ligustrum quihoui: IoU 62.87
2024/01/01 12:02:10 - oneformer - INFO - Epoch(val) [1] The new optimal model has been saved to best/ (mIoU=55.14)
```

(Limited by the public API of HuggingFace `transformers`, the various sub-losses of mask/class/dice/contrastive are not available
Only the total loss can be printed as a separate value; the mmseg model can print each item because it creates the loss by itself. )

### Only save the two weights of best + last

Instead of saving an `epoch_N.pth` for each epoch like earlier versions, only two directories are now maintained:

```
work_dirs/oneformer/
    best/ The highest mIoU in the verification set (model.safetensors + config.json + processor configuration)
    last/ last epoch (used for breakpoint continuation of training)
    trainer_state.pt optimizer/scheduler/AMP scaler state + current epoch + historical optimal mIoU
```

`best/` and `last/` are all self-contained directories saved using HuggingFace’s official `save_pretrained()` (not naked
`.pth` state_dict) - `eval.py`/`infer.py` can be loaded directly by pointing to this directory, no additional specification is required.
`--pretrained`.

### Continue training after breakpoint

```bash
python train.py \
    --data_root /path/to/work_data_root \
    --output_dir ./work_dirs/oneformer \
    --epochs 50 --batch_size 2 --lr 1e-5 \
    --resume
```

`--resume` will automatically restore from `output_dir/last` + `output_dir/trainer_state.pt`: model weights, optimizer state,
Learning rate scheduler status, AMP scaler status, current epoch, historical optimal mIoU, and then training without losing the optimizer momentum to warm up again.
If there is no recoverable checkpoint in `output_dir`, it will directly report an error and exit (it will not be mistaken for re-training).

> **Important**: When continuing training, these parameters `--epochs`/`--lr`/`--batch_size`/`--amp` must be consistent with those in the first training
> ——The learning rate scheduler (cosine schedule) is calculated based on the total number of steps. Changing `--epochs`/`--batch_size` will cause the total number of steps to change.
> The curve after the scheduler is restored does not match; if `--amp` is changed, the restored state of `scaler` does not match whether it is currently enabled or not.

### Video memory/accuracy related

- **Shared graphics card**: `cudnn.benchmark` is turned off by default (`--cudnn_benchmark` can be re-enabled), available
  `--gpu_mem_fraction 0.4` hard limits the maximum proportion of the current GPU's video memory that this process can use. The training script will be based on PyTorch
  The version automatically sets the appropriate `PYTORCH_CUDA_ALLOC_CONF` (≥2.0 uses `expandable_segments`, 1.10~1.13 degenerates to
  `max_split_size_mb`, <1.10 not set)
- **Accuracy**: Default is FP32, `--amp` turns on mixed precision training (fp16 + `GradScaler`), it can be used on large models such as Swin-Large
  Significantly reduces video memory and speeds up

### Known transformers bugs and fixes (already built into `engine.py`, no additional processing is required on your part)

- **`num_text` is `None`**: after `OneFormerProcessor.from_pretrained()` is loaded
  `image_processor.num_text` may be `None`, once `processor(..., segmentation_maps=...)` is called during training
  Will report `TypeError: can't multiply sequence by non-int of type 'NoneType'`
   ([huggingface/transformers#27356](https://github.com/huggingface/transformers/issues/27356), 
  It has not been officially fixed yet). `engine.py::load_model_and_processor()` is set manually after loading the model according to the officially confirmed method:
  `processor.image_processor.num_text = model.config.num_queries - model.config.text_encoder_n_ctx`.
- **`text_queries` is `None`**: The officially released inference uses `config.is_training=False` in the checkpoint. At this time, the model
  The `text_mapper` submodule will not be created at all during construction; but as long as `mask_labels`/`class_labels` is passed (it will definitely be created during training
  Pass), the comparison loss will be calculated unconditionally, so it is reported at `text_queries.flatten(1)`
  `AttributeError: 'NoneType' object has no attribute 'flatten'`. `engine.py` unified in
  Explicitly pass `is_training=True` when `from_pretrained()`.
- **Bare `state_dict()` saving/loading is unstable**: Use `torch.save(model.state_dict(), xxx.pth)` in earlier versions +
  `load_state_dict()`, under the combined action of `ignore_mismatched_sizes=True` and other mechanisms, occurred during reloading
  `Missing key(s) in state_dict` error is reported, and the same code behaves inconsistently under different checkpoint/environment combinations. now
  Uniformly use `save_pretrained()`/`from_pretrained()` officially recommended by HuggingFace (see "Only save best+last" above)
  section), more stable.

## 5. Inference code

```bash
cd 06_oneformer
python infer.py \
    --checkpoint ./work_dirs/oneformer/best \
    --input /path/to/image_or_dir \
    --out_dir ./predictions
```

`--checkpoint` points to the directory where `train.py` is saved (`best` or `last`). There is no need to specify `--pretrained` separately.
——The model structure and processor configuration have been completely saved in this directory. Each image outputs three files:
- `<name>_mask.png`: single channel, the pixel value is the predicted category index (0..N-1), for programmatic post-processing/statistics
- `<name>_color.png`: The segmentation map colored by `engine.py::PALETTE`, the human eye can directly see each category area
- `<name>_overlay.png`: a translucent overlay on the input image, with background left transparent.

## 6. Use the optimal checkpoint to recalculate the indicators on the validation set

The indicator printed at each epoch during the training process is the verification set performance of that checkpoint at that time; if you just want to use a certain
The saved checkpoint (`best` or `last`) can be used to recalculate the complete indicator. There is no need to re-run the training. Use `eval.py`:

```bash
cd 06_oneformer
python eval.py \
    --checkpoint ./work_dirs/oneformer/best \
    --data_root /path/to/work_data_root \
    --out_file ./work_dirs/oneformer/eval_best.json
```

Complete metrics (with respective IoU/Acc/Dice/Fscore/Precision/Recall for each category) are printed and optionally saved.
`train.py`/`eval.py`/`infer.py` The three scripts share the same model construction logic in `engine.py`
(`load_model_and_processor`) to ensure that there will be no problem of "one set of construction methods during training and another set during evaluation" causing the indicators to be inconsistent.

## 7. Evaluation indicator caliber (important, the previous version was missed, be sure to confirm the alignment)

The formula of `metrics.py` has been compared with `mmseg/evaluation/metrics/iou_metric.py` of mmsegmentation 1.2.2
The source code is checked line by line and numerically verified using the real mmseg function (error < 1e-9, including "a certain category has zero samples in the verification set"
Such boundary cases), and `IoUMetric(iou_metrics=["mIoU","mDice","mFscore"])` with the remaining 7 mmseg models
The output calibers are exactly the same and can be directly compared together. Supports any number of categories. The following uses the currently configured 5 categories as an example:

| Indicator | Meaning | Formula |
|---|---|---|
| `aAcc` | **Overall** pixel accuracy, the only metric that does not distinguish between categories | `(Number of pixels for all categories of prediction pairs) / Total number of pixels` |
| `mIoU` | Average of 5 categories of IoU | Each category `TP/(TP+FP+FN)` and then averaged |
| `mAcc` | Average of **recall** for 5 categories (**not** `aAcc`!) | Average of `TP/(TP+FN)` for each category |
| `mDice` | Average of 5 categories of Dice | Each category `2TP/(2TP+FP+FN)` and then average |
| `mFscore` | Average of F1 for 5 categories | `2PR/(P+R)` for each category and then average |
| `mPrecision` | The average of the precision rates of 5 categories | Each category `TP/(TP+FP)` and then averaged |
| `mRecall` | Same as `mAcc`, numerically identical, except that mmseg has a different name under different `iou_metrics` options | Same as `mAcc` |

**The two things that are most easily confused and the ones that were wrong in previous versions**:
1. `mAcc`/`Acc` is not "overall accuracy", it is **the average recall rate of each category**; the real overall pixel accuracy is `aAcc`.
2. `mIoU`/`mDice`/`mFscore`, etc. all calculate the average of all categories separately, not just a certain category. If you only want to see a single
   For species performance, `metrics.py` also provides `IoU.Podocarpus macrophyllus`/`Acc.Camellia sasanqua` such exclusive values ​​by category name.
   (You can see it in the "complete indicator" printed by `eval.py`, or take it from the json saved by `--out_file`. The category name is followed by
   The configuration in `engine.py::CLASSES` is consistent).

**About nan**: If there is not even one pixel of a certain category in the verification set (for example, there are no samples in the current division of a certain species),
Indicators such as `Acc`/`Recall`/`Fscore` for this class will be `nan`, and average values such as `mAcc` will correctly exclude this category.
(Instead of treating it as 0 to lower the average), this is completely consistent with the official method of mmseg. When you see a category is `nan`,
Go to the "Mask Pixel Value Distribution" printed during conversion in `common/prepare_data.py` to confirm whether the sample size of this category is too small.

## 8. Video memory prompt

OneFormer's video memory usage is significantly higher than that of the pure CNN/lightweight Transformer model. If the video memory is insufficient:
- Reduce `--batch_size` to 1 to simulate larger batches with gradient accumulation
- Turn on `--amp` to reduce video memory usage
- Use `--gpu_mem_fraction` to observe how much video memory is actually needed and adjust it gradually
