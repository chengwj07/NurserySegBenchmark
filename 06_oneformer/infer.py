"""
06_oneformer/infer.py

OneFormer inference script: Make semantic segmentation predictions for single or entire catalog images.

Usage:
    python infer.py --checkpoint ./work_dirs/oneformer/best         --input /path/to/image_or_dir --out_dir ./predictions
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

sys.path.insert(0, os.path.dirname(__file__))
from engine import CLASSES, colorize_mask, load_model_and_processor

IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def load_as_rgb(path: Path) -> Image.Image:
    if path.suffix.lower() in (".tif", ".tiff") and HAS_TIFFFILE:
        arr = tifffile.imread(str(path))
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.shape[-1] > 3:
            arr = arr[..., :3]
        return Image.fromarray(arr.astype(np.uint8))
    return Image.open(path).convert("RGB")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                         help='The checkpoint directory saved by train.py, such as ./work_dirs/oneformer/best')
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./predictions")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model, processor = load_model_and_processor(pretrained=None, device=args.device,
                                                  checkpoint=args.checkpoint)
    # Text_inputs is not passed in the inference stage (without segmentation_maps, the processor will not generate text for comparative learning),
    # In load_model_and_processor, in order to make the state_dict of checkpoint match, the model is constructed using
    # is_training=True; turn it off after the weights are loaded, otherwise text_mapper(None) will report an error in forward propagation.
    # is_training is just a normal Boolean attribute at runtime. Turning it off after loading the weights will not affect the loaded parameters.
    model.model.is_training = False
    model.eval()

    in_path = Path(args.input)
    files = [in_path] if in_path.is_file() else sorted(
        [p for p in in_path.iterdir() if p.suffix.lower() in IMG_EXTS]
    )
    print(f"Images to process: {len(files)}, Category: {list(CLASSES)}")

    for f in files:
        image = load_as_rgb(f)
        inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(args.device)
        with torch.no_grad():
            outputs = model(**inputs)
        pred = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
        pred_mask = pred.cpu().numpy().astype(np.uint8)  # Category index 0..len(CLASSES)-1

        # 1) Original category index mask (0..N-1), used for programmatic post-processing/statistics, not directly visible to the human eye
        mask_out = Path(args.out_dir) / f"{f.stem}_mask.png"
        Image.fromarray(pred_mask, mode="L").save(mask_out)

        # 2) The segmentation map colored by category, the human eye can directly see the areas of each category
        color_mask = colorize_mask(pred_mask)
        color_out = Path(args.out_dir) / f"{f.stem}_color.png"
        Image.fromarray(color_mask).save(color_out)

        # 3) Visualization superimposed on the original image (the background class is not superimposed, only other categories are highlighted, semi-transparent)
        img_arr = np.array(image)
        overlay = img_arr.copy()
        fg = pred_mask != 0
        overlay[fg] = (0.5 * overlay[fg] + 0.5 * color_mask[fg]).astype(np.uint8)
        overlay_out = Path(args.out_dir) / f"{f.stem}_overlay.png"
        Image.fromarray(overlay).save(overlay_out)

        print(f"[OK] {f.name} -> {mask_out.name}, {color_out.name}, {overlay_out.name}")


if __name__ == "__main__":
    main()
