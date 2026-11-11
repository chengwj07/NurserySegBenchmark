"""
06_oneformer/engine.py

Common parts of train.py / eval.py / infer.py: model + processor construction and saving, collate_fn,
Verification loop, mmseg style training log formatting. It is extracted to ensure \"how to construct/save the model during training\" and
\"How to load the model during evaluation/inference\" is exactly the same.

**About checkpoint format**: This project uses HuggingFace’s official `save_pretrained()`/
`from_pretrained()` to save/load the model and processor (one checkpoint corresponds to a directory, which is
`model.safetensors` + `config.json` + processor configuration file), rather than bare
`torch.save(model.state_dict(), xxx.pth)`. Reason: OneFormer model is in
Under the combined action of `ignore_mismatched_sizes=True` + `low_cpu_mem_usage` and other mechanisms, the bare state_dict
There was an error report of missing key when saving/reloading (the same code behaved under different checkpoint/environment combinations)
inconsistent); `save_pretrained()` is the approach recommended by HuggingFace official documentation, which will save the model structure (config.json)
Packed together with the weights, it does not depend on the original hub weights from_pretrained when reloading, and then reuses them.
\"state_dict to align\" this step is more stable.
"""
import datetime
import os
import time

import numpy as np
import torch
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

from metrics import SegConfusionMatrix

# Class definition: consistent with METAINFO.classes used by mmseg model in common/plant_dataset.py
# (The subscript is the mask pixel value, 0 must be background). If you want to change it to another category number/category name, change both sides together.
CLASSES = ("background", "Podocarpus macrophyllus", "Camellia sasanqua", "Loropetalum chinense", "Ligustrum quihoui")
NUM_CLASSES = len(CLASSES)
ID2LABEL = {i: name for i, name in enumerate(CLASSES)}

# The fixed naming of the checkpoint directory/training status file, train.py / eval.py / infer.py all find files according to this convention
BEST_DIR_NAME = "best"
LAST_DIR_NAME = "last"
TRAINER_STATE_NAME = "trainer_state.pt"

# The palette used for visualization is the same as the METAINFO.palette used by the mmseg model in common/plant_dataset.py
# To keep it consistent (it is convenient to compare the prediction results of different models), the subscripts correspond to CLASSES one-to-one. When changing the category, change both sides together.
PALETTE = [
    [0, 0, 0],        # background - black
    [220, 20, 60],    # Podocarpus macrophyllus - red
    [60, 179, 113],   # Camellia sasanqua - green
    [30, 144, 255],   # Loropetalum chinense - blue
    [255, 215, 0],    # Ligustrum quihoui - Yellow
]


def log(msg: str):
    'A single-line log similar in style to mmengine LoggerHook, making it easy to compare with the training logs of the other seven mmseg series models.'
    now = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print(f"{now} - oneformer - INFO - {msg}", flush=True)


def collate_fn(batch):
    # Each sample output by OneFormerProcessor is already a tensor. Here, the items with the same key are simply stacked/listed.
    # raw_mask/orig_size are our own additional fields (see dataset.py) and are not passed to model.forward().
    out = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if k == "orig_size":
            out[k] = vals  # list of (H, W) tuples
        elif torch.is_tensor(vals[0]):
            try:
                out[k] = torch.stack(vals)
            except Exception:
                out[k] = vals  # Variable length fields (such as mask_labels/class_labels list) keep list
        else:
            out[k] = vals
    return out


def move_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if k in ("raw_mask", "orig_size"):
            continue  # These two fields are only used locally during the verification phase and will not be entered into the model/moved to the GPU to participate in forward
        if torch.is_tensor(v):
            out[k] = v.to(device)
        elif isinstance(v, list):
            out[k] = [x.to(device) if torch.is_tensor(x) else x for x in v]
        else:
            out[k] = v
    return out


def load_model_and_processor(pretrained: str, device: str, checkpoint: str = None):
    """Build processor + model. train.py / eval.py / infer.py all adjust this function to ensure that the construction method is completely consistent.

    Args:
        pretrained: OneFormer pretrained weight id on HuggingFace (e.g.
            'shi-labs/oneformer_ade20k_swin_large'). Load freshly from here when checkpoint is None
            (Transfer learning: The number of categories is changed from the original 150+1 categories to the N+1 categories configured in CLASSES of this project, and the shape of the layer is inconsistent.
            It will be re-initialized randomly, which is normal. You will see the \"newly initialized\" prompt in the training log).
        device: 'cuda' or 'cpu'.
        checkpoint: Optional, the directory path saved by this project using save_checkpoint() (not the .pth file).
            Once uploaded, load the model + processor as a whole directly from this directory. There is no need and should not be uploaded again.
            ignore_mismatched_sizes (shapes have been aligned during training).
    """
    if checkpoint:
        processor = OneFormerProcessor.from_pretrained(checkpoint)
        model = OneFormerForUniversalSegmentation.from_pretrained(checkpoint, is_training=True)
    else:
        processor = OneFormerProcessor.from_pretrained(pretrained)
        model = OneFormerForUniversalSegmentation.from_pretrained(
            pretrained,
            id2label=ID2LABEL,
            ignore_mismatched_sizes=True,  # Number of categories changed from ADE20K-150 category to category 2
            # Known pitfalls of transformers: config.is_training=False in the officially released inference checkpoint.
            # At this time, the text_mapper submodule will not be created at all when constructing the model. It is necessary to compare the loss during training/verification.
            # If text_queries is always None, an error will be reported directly. Unified and explicitly opened here.
            is_training=True,
        )
    # Known HuggingFace transformers bug (huggingface/transformers#27356 / #27572):
    # image_processor.num_text is sometimes not set/retained correctly whenever calling
    # processor(..., segmentation_maps=...) will report
    # "can't multiply sequence by non-int of type 'NoneType'".
    # No matter which path is loaded from, it will be safely recalculated here, the officially confirmed repair method.
    processor.image_processor.num_text = model.config.num_queries - model.config.text_encoder_n_ctx

    model.to(device)
    return model, processor


def save_checkpoint(model, processor, out_dir: str):
    """Save the model + processor into a self-contained directory, and use eval.py/infer.py directly
    `--checkpoint out_dir` is enough, no need to specify additional --pretrained."""
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)


def save_trainer_state(path: str, epoch: int, global_step: int, optimizer, scheduler, scaler,
                        best_miou: float):
    'Save the complete training status required to resume training after the breakpoint (excluding the model weight itself, the model weight is stored separately by save_checkpoint).'
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_miou": best_miou,
        },
        path,
    )


def load_trainer_state(path: str, optimizer, scheduler, scaler):
    'Restore the optimizer/learning rate scheduler/AMP scaler state and return (the next epoch to be run, global_step, best_miou).'
    state = torch.load(path, map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    return state["epoch"] + 1, state["global_step"], state["best_miou"]


def find_resume_checkpoint(output_dir: str):
    """When --resume, automatically find last/ + trainer_state.pt left in the last training in output_dir.
    Only when both are present can it be considered recoverable. If one is missing, it will be deemed that there is no recoverable checkpoint (to avoid recovering a checkpoint).
    A half-baked state in which the model weights and optimizer state do not match).
    """
    last_dir = os.path.join(output_dir, LAST_DIR_NAME)
    state_path = os.path.join(output_dir, TRAINER_STATE_NAME)
    if os.path.isdir(last_dir) and os.path.isfile(state_path):
        return last_dir, state_path
    return None, None


class AvgMeter:
    'Sliding window average is used to calculate the display value of iter_time/data_time, window size = printing interval.'

    def __init__(self):
        self._vals = []

    def update(self, v: float, window: int):
        self._vals.append(v)
        if len(self._vals) > window:
            self._vals.pop(0)

    @property
    def avg(self) -> float:
        return sum(self._vals) / max(1, len(self._vals))


def format_eta(seconds: float) -> str:
    return str(datetime.timedelta(seconds=max(0, int(seconds))))


def colorize_mask(pred_mask: np.ndarray) -> np.ndarray:
    """Mask the category index (H, W), the value range is 0..NUM_CLASSES-1, and color it according to PALETTE into (H, W, 3) RGB image,
    infer.py is used to store visualization results."""
    palette = np.array(PALETTE, dtype=np.uint8)
    return palette[pred_mask]


def gpu_memory_mb() -> str:
    if torch.cuda.is_available():
        return f"{torch.cuda.max_memory_allocated() / 1024 ** 2:.0f}"
    return "N/A"


@torch.no_grad()
def evaluate(model, processor, val_loader, device):
    """Calculate indicators consistent with the mmsegmentation IoUMetric conventions on the validation set, and incidentally return the average loss.
    The returned dict structure can be found in metrics.py::SegConfusionMatrix.compute(). The number of categories/category names can be found in this file.
    CLASSES constant at the top.
    """
    model.eval()
    cm = SegConfusionMatrix(NUM_CLASSES, class_names=CLASSES)
    val_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        raw_masks = batch["raw_mask"]          # list[Tensor(H, W)]
        orig_sizes = batch["orig_size"]         # list[(H, W)]
        model_inputs = move_to_device(batch, device)

        outputs = model(**model_inputs)
        val_loss += outputs.loss.item()
        n_batches += 1

        # Semantic segmentation post-processing: fuse the mask/class predictions output by the model into a pixel-by-pixel category map,
        # And resize back to the original resolution of each image to directly compare with raw_mask.
        pred_maps = processor.post_process_semantic_segmentation(
            outputs, target_sizes=list(orig_sizes)
        )
        for pred, gt in zip(pred_maps, raw_masks):
            pred_np = pred.cpu().numpy().astype(np.uint8)
            gt_np = gt.cpu().numpy().astype(np.uint8)
            cm.update(pred_np, gt_np)

    metrics = cm.compute()
    metrics["loss"] = val_loss / max(1, n_batches)
    return metrics


def format_metrics(metrics: dict) -> str:
    """Aligned with mmseg verification log style: one row for average indicator, table by category (IoU/Acc/Dice/Fscore/Precision/Recall).
    Values ​​are displayed as percentages (consistent with mmseg printing convention). If a certain category does not appear in the validation set, the corresponding value is nan.
    Just press nan to print (it prompts you to check whether this class has too few samples in the verification set division).
    """
    head = (
        f"aAcc: {metrics['aAcc'] * 100:.2f}  mIoU: {metrics['mIoU'] * 100:.2f}  "
        f"mAcc: {metrics['mAcc'] * 100:.2f}  mDice: {metrics['mDice'] * 100:.2f}  "
        f"mFscore: {metrics['mFscore'] * 100:.2f}  mPrecision: {metrics['mPrecision'] * 100:.2f}  "
        f"mRecall: {metrics['mRecall'] * 100:.2f}"
    )
    cols = ["IoU", "Acc", "Dice", "Fscore", "Precision", "Recall"]
    header = "    " + " | ".join(f"{c:>10}" for c in ["Class"] + cols)
    sep = "    " + "-" * (len(header) - 4)
    lines = [header, sep]
    for name in CLASSES:
        vals = [f"{metrics[f'{col}.{name}'] * 100:.2f}" for col in cols]
        lines.append("    " + " | ".join(f"{v:>10}" for v in [name] + vals))
    return f"{head}\n" + "\n".join(lines)
