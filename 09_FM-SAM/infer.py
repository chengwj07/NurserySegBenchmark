"""YOLO26 + SAM2 segmentation inference script.

Make segmentation predictions for single images or entire catalog images, and output 0/1 masks (png) and overlay visualizations.

Usage:
    # leaflet
    python infer.py --yolo-weights runs/yolo26_train/weights/best.pt         --sam-weights weights/sam2_t.pt         --input /path/to/image.tif --out_dir ./predictions

    # Entire directory
    python infer.py --yolo-weights runs/yolo26_train/weights/best.pt         --sam-weights weights/sam2_t.pt         --input /path/to/image_dir --out_dir ./predictions

    #Adjust detection threshold
    python infer.py --yolo-weights best.pt --input ./imgs --conf 0.5 --iou 0.5
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Add sys.path to the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import src  # noqa: F401 (triggers environment variable setting)
from src.sam_segment import SAM2Segmenter

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")

# 5 types of nursery stock palette (consistent with common/plant_dataset.py, RGB)
FIVE_CLASS_PALETTE = [
    (0, 0, 0),         # background black
    (220, 20, 60),     # Podocarpus macrophyllus red
    (60, 179, 113),    # Camellia sasanqua green
    (30, 144, 255),    # Loropetalum chinense blue
    (255, 215, 0),     # Ligustrum quihoui yellow
]
FIVE_CLASS_NAMES = ['background', "Podocarpus macrophyllus", "Camellia sasanqua", "Loropetalum chinense", "Ligustrum quihoui"]


def build_palette(num_classes: int) -> list[tuple[int, int, int]]:
    """Returns an RGB palette of length num_classes (0=background black).

    Fixed color matching is used for category 5 (consistent with 01~08), and the remaining categories are generated uniformly according to HSV.
    """
    if num_classes == len(FIVE_CLASS_PALETTE):
        return FIVE_CLASS_PALETTE
    import colorsys
    palette = [(0, 0, 0)]
    for i in range(1, num_classes):
        r, g, b = colorsys.hsv_to_rgb(
            (i - 1) / max(1, num_classes - 1), 0.75, 0.95
        )
        palette.append((int(255 * r), int(255 * g), int(255 * b)))
    return palette


def load_as_rgb(path: Path) -> np.ndarray:
    """Load the image as an RGB uint8 array.

    Supports TIFF (via tifffile), PNG/JPG/BMP (via PIL).
    The grayscale image is automatically copied to three channels, and the first 3 channels of multi-channel are intercepted.
    """
    if path.suffix.lower() in (".tif", ".tiff") and HAS_TIFFFILE:
        arr = tifffile.imread(str(path))
    else:
        arr = np.array(Image.open(path))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]
    return arr.astype(np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO26 + SAM2 segmentation reasoning')
    parser.add_argument("--yolo-weights", type=str, required=True,
                        help='YOLO26 training weights (best.pt)')
    parser.add_argument("--sam-weights", type=str, default="weights/sam2_t.pt",
                        help='SAM2 weight path')
    parser.add_argument("--input", type=str, required=True,
                        help="Image file or directory of images")
    parser.add_argument("--out_dir", type=str, default="./predictions",
                        help='Output directory')
    parser.add_argument("--device", type=str, default="cuda:0",
                        help='Inference device (cuda:0/cpu)')
    parser.add_argument("--conf", type=float, default=0.25,
                        help='YOLO detection confidence threshold')
    parser.add_argument("--iou", type=float, default=0.45,
                        help='YOLO NMS IoU threshold')
    parser.add_argument("--max_det", type=int, default=300,
                        help='Maximum number of detection frames per image')
    parser.add_argument("--num-classes", type=int, default=1,
                        help='Total number of categories (including background). 1=Binary classification output 0/255,'
                             '>1 Output raw class index masks and visualize with palette')
    parser.add_argument("--class-names", type=str, default=None,
                        help='Comma separated category names (with background), for legend/log only')
    return parser.parse_args()


def mask_to_overlay(img_rgb: np.ndarray, pred: np.ndarray,
                    palette: list) -> np.ndarray:
    'Superimpose a category index mask onto the original image (each category is semi-transparently colored by palette).'
    overlay = img_rgb.astype(np.float32)
    h, w = pred.shape
    for c in range(1, len(palette)):
        color = np.array(palette[c], dtype=np.float32)
        m = pred == c
        if m.sum() == 0:
            continue
        overlay[m] = 0.5 * overlay[m] + 0.5 * color
    return overlay.astype(np.uint8)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load YOLO model
    from ultralytics import YOLO
    print(f"[Inference] Load YOLO model: {args.yolo_weights}")
    yolo = YOLO(args.yolo_weights)

    # Load the SAM2 splitter
    print(f"[Inference] Load SAM2 model: {args.sam_weights}")
    segmenter = SAM2Segmenter(args.sam_weights, device=args.device)

    # Collect input images
    in_path = Path(args.input)
    files = [in_path] if in_path.is_file() else sorted(
        [p for p in in_path.iterdir() if p.suffix.lower() in IMG_EXTS]
    )
    print(f"[reasoning] Images to process: {len(files)}")

    for f in files:
        # Load image (RGB)
        img_rgb = load_as_rgb(f)
        # Convert to BGR for use by YOLO/SAM2
        img_bgr = img_rgb[:, :, ::-1].copy()

        # ----YOLO detection (with category and confidence)----
        results = yolo.predict(
            img_bgr,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            verbose=False,
            device=args.device,
        )
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            box_classes = results[0].boxes.cls.cpu().numpy().astype(np.int64)
            box_confs = results[0].boxes.conf.cpu().numpy()
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            box_classes = np.zeros((0,), dtype=np.int64)
            box_confs = np.zeros((0,), dtype=np.float32)

        # ---- SAM2 segmentation (using the center point of the detection frame as the prompt, the category comes from YOLO) ----
        pred_mask = segmenter.segment_with_boxes_labeled(
            img_bgr, boxes, box_classes, box_confs,
            num_classes=args.num_classes,
        )

        # ----Save mask (two categories 0/255; multi-category original category index 0..N-1)----
        mask_out = Path(args.out_dir) / f"{f.stem}_mask.png"
        if args.num_classes > 2:
            Image.fromarray(pred_mask, mode="L").save(mask_out)
        else:
            Image.fromarray(pred_mask * 255, mode="L").save(mask_out)

        # ---- Save Overlay Visualization (Palette by Category) ----
        palette = build_palette(args.num_classes)
        overlay = mask_to_overlay(img_rgb, pred_mask, palette)
        overlay_out = Path(args.out_dir) / f"{f.stem}_overlay.png"
        Image.fromarray(overlay).save(overlay_out)

        n_det = len(boxes)
        fg_ratio = (pred_mask > 0).sum() / pred_mask.size * 100
        print(f"[OK] {f.name} -> {mask_out.name}, {overlay_out.name} "
              f"(detection {n_det} goals, proportion of prospects {fg_ratio:.1f}%)")

    print(f"\\n[Inference] Completed, processed together {len(files)} image, output directory: {args.out_dir}")


if __name__ == "__main__":
    main()
