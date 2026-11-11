"""
05_swin_transformer/infer.py
swin inference script: perform segmentation prediction on a single image or the entire directory image, and output 0/1 mask (png) and overlay visualization map.

Usage:
    # leaflet
    python infer.py --config config.py --checkpoint work_dirs/swin/best_mIoU_iter_xxx.pth         --input /path/to/image.tif --out_dir ./predictions

    # Entire directory
    python infer.py --config config.py --checkpoint work_dirs/swin/best_mIoU_iter_xxx.pth         --input /path/to/image_dir --out_dir ./predictions
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mmseg.apis import init_model, inference_model

from mmseg.utils import register_all_modules
register_all_modules(init_default_scope=True)

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def load_as_rgb(path: Path) -> np.ndarray:
    if path.suffix.lower() in (".tif", ".tiff") and HAS_TIFFFILE:
        arr = tifffile.imread(str(path))
    else:
        arr = np.array(Image.open(path))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]
    return arr.astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), "config.py"))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Image file or directory of images")
    parser.add_argument("--out_dir", type=str, default="./predictions")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model = init_model(args.config, args.checkpoint, device=args.device)

    in_path = Path(args.input)
    files = [in_path] if in_path.is_file() else sorted(
        [p for p in in_path.iterdir() if p.suffix.lower() in IMG_EXTS]
    )
    print(f"Images to process: {len(files)}")

    for f in files:
        img = load_as_rgb(f)
        # During mmseg training, images are read into BGR through cv2, data_preprocessor(bgr_to_rgb=True)
        # Then convert it to RGB and feed it to the model; here, the ndarray path does not perform channel conversion and needs to be manually transferred to BGR.
        result = inference_model(model, img[..., ::-1].copy())
        pred_mask = result.pred_sem_seg.data.squeeze(0).cpu().numpy().astype(np.uint8)  # 0..num_classes-1

        mask_out = Path(args.out_dir) / f"{f.stem}_mask.png"
        Image.fromarray(pred_mask, mode="L").save(mask_out)

        overlay = img.copy()
        overlay[pred_mask == 1] = (0.5 * overlay[pred_mask == 1] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
        overlay_out = Path(args.out_dir) / f"{f.stem}_overlay.png"
        Image.fromarray(overlay).save(overlay_out)

        print(f"[OK] {f.name} -> {mask_out.name}, {overlay_out.name}")


if __name__ == "__main__":
    main()
