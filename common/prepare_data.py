"""
common/prepare_data.py

Dataset preprocessing script. Supports two types of masks: binary segmentation (0/1) and multiclass segmentation (0,1,2,...,N-1).

raw_dataset structure (one folder for each category, tif/ under the folder is the original image slice, train/ is the corresponding
Mask, both are in .tif format, the size is 1024x1024, and the file names correspond one to one):

    data_root/
        Class name A/
            tif/ *.tif (original image slice, 1024x1024)
            train/ *.tif (mask, 1024x1024; binary segmentation is 0/1, multiclass segmentation is 0..N-1 category index)
        Class name B/
            tif/
            train/
        ...

(Compatible with the old structure: If data_root is directly under tif/ and train/, it will also be treated as a single category. Here
The \"class name folder\" is only used to aggregate data from multiple sources and does not represent the segmentation category - the segmentation category is always determined by the mask pixel value. )

This script will traverse all category folders, pair tif+mask one by one, and uniformly convert it to mmsegmentation
The agreed standard directory structure is shared by the training/inference scripts of all 8 models:

    work_data_root/
        img_dir/
            train/ *.png (RGB, converted from original tif)
            val/
        ann_dir/
            train/ *.png (single channel, pixel value is category index 0..num_classes-1)
            val/

To prevent files with the same name in different category folders from overwriting each other, the output file name will be added with a category prefix:
    <Class name>__<Original file name>.png

Usage (two categories, default):
    python prepare_data.py         --data_root /path/to/your/dataset         --out_dir /path/to/work_data_root         --val_ratio 0.2 --seed 42

Usage (multiple categories, such as background=0/Podocarpus macrophyllus=1/Camellia sasanqua=2/Loropetalum chinense=3/Ligustrum quihoui=4, 5 categories in total):
    python prepare_data.py         --data_root /path/to/your/dataset         --out_dir /path/to/work_data_root         --num_classes 5         --val_ratio 0.2 --seed 42
"""
import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError as e:
    raise ImportError('Please install Pillow first: pip install pillow') from e

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

IMG_EXTS = (".tif", ".tiff")
MASK_EXTS = (".tif", ".tiff")


def find_class_dirs(data_root: Path):
    """Returns [(class_name, class_dir), ...].
    If data_root itself directly contains tif/ and train/, it is regarded as a single (unnamed) category.
    Otherwise, traverse the first-level subdirectories, and all subdirectories containing both tif/ and train/ are counted as one category.
    """
    if (data_root / "tif").is_dir() and (data_root / "train").is_dir():
        return [("root", data_root)]

    class_dirs = []
    for sub in sorted(data_root.iterdir()):
        if sub.is_dir() and (sub / "tif").is_dir() and (sub / "train").is_dir():
            class_dirs.append((sub.name, sub))
    if not class_dirs:
        raise FileNotFoundError(
            f"in {data_root} No category folders containing tif/ and train/ subdirectories were found, please check the dataset structure."
        )
    return class_dirs


def find_pairs(data_root: Path):
    """Traverse all category folders and match image-mask pairs in tif/ and train/ by file name (without suffix).
    Return [(class_name, img_path, mask_path), ...]
    """
    class_dirs = find_class_dirs(data_root)
    print(f"discover {len(class_dirs)} category folders: {[c[0] for c in class_dirs]}")

    all_pairs = []
    for class_name, class_dir in class_dirs:
        img_dir = class_dir / "tif"
        mask_dir = class_dir / "train"

        img_files = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
        mask_files = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in MASK_EXTS}

        common_keys = sorted(set(img_files.keys()) & set(mask_files.keys()))
        missing_mask = sorted(set(img_files.keys()) - set(mask_files.keys()))
        missing_img = sorted(set(mask_files.keys()) - set(img_files.keys()))
        if missing_mask:
            print(f"[warning][{class_name}] {len(missing_mask)} Image does not have a corresponding mask and will be skipped")
        if missing_img:
            print(f"[warning][{class_name}] {len(missing_img)} masks have no corresponding images and will be skipped")

        for k in common_keys:
            all_pairs.append((class_name, img_files[k], mask_files[k]))
        print(f"[{class_name}] matches {len(common_keys)} pair image/mask")

    print(f"Total matches {len(all_pairs)} For Image/Mask (summary of all categories)")
    return all_pairs


def read_image_any(path: Path) -> np.ndarray:
    if HAS_TIFFFILE:
        return tifffile.imread(str(path))
    return np.array(Image.open(path))


def save_image_as_png(arr: np.ndarray, out_path: Path):
    """Save the original image tif (possibly multiband/singleband) as a 3-channel png that can be read directly by mmseg.
    - 3 channels and above -> take the first 3 bands as RGB
    - Single channel -> Copy to 3 channel grayscale image
    If you want to retain all bands for subsequent scientific research analysis, please expand it to .npy / multi-channel storage solution.
    """
    if arr.ndim == 2:
        img = Image.fromarray(_to_uint8(arr)).convert("L").convert("RGB")
    elif arr.ndim == 3:
        if arr.shape[-1] >= 3:
            img = Image.fromarray(_to_uint8(arr[..., :3]))
        else:
            img = Image.fromarray(_to_uint8(arr[..., 0])).convert("RGB")
    else:
        raise ValueError(f"Unsupported image dimensions: {arr.shape}")
    img.save(out_path)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """linearly stretches any dtype (commonly found in remote sensing tif uint16/float) to uint8,
    to save as png for use with the standard visual backbone. If it is already uint8, it will be returned directly.
    """
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    if hi <= lo:
        lo, hi = arr.min(), arr.max() if arr.max() > arr.min() else arr.min() + 1
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255.0
    return arr.astype(np.uint8)


def save_mask_as_png(arr: np.ndarray, out_path: Path, num_classes: int, class_pixel_counter: Counter):
    """Save the mask as a single channel png, with the pixel values ​​being the class index.
    - num_classes == 2 (default, two categories): strict binarization, (pixel > 0) is regarded as foreground = 1, compatible with old behavior.
    - num_classes > 2 (multiple categories): retain the original class index without binarization; the index will be rounded to the nearest integer and merged
      Crop to the range [0, num_classes-1] (commonly caused by annotation tool interpolation/scaling causing individual pixels to appear non-integer or
      out-of-bounds value), the number of out-of-bounds pixels will be counted and a prompt will be summarized at the end, making it easier for you to go back and check whether there are any problems with the annotation.
    """
    if arr.ndim == 3:
        arr = arr[..., 0]

    if num_classes == 2:
        mask = (arr > 0).astype(np.uint8)
    else:
        raw = np.rint(arr).astype(np.int64)
        out_of_range = int(np.sum((raw < 0) | (raw > num_classes - 1)))
        if out_of_range > 0:
            class_pixel_counter["__out_of_range__"] += out_of_range
        mask = np.clip(raw, 0, num_classes - 1).astype(np.uint8)

    class_pixel_counter.update(Counter(mask.flatten().tolist()))
    Image.fromarray(mask, mode="L").save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                         help='raw_dataset root directory (including several class name/tif, class name/train subdirectories)')
    parser.add_argument("--out_dir", type=str, required=True, help='Output directory of the converted mmseg format dataset')
    parser.add_argument("--num_classes", type=int, default=2,
                         help='Total number of segmentation categories (including background). Default 2 = binary segmentation (0=background, 1=foreground).'
                              'Multiple categories such as Background=0/Podocarpus macrophyllus=1/Camellia sasanqua=2/Loropetalum chinense=3/Ligustrum quihoui=4 Biography 5.')
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)

    pairs = find_pairs(data_root)
    random.seed(args.seed)
    random.shuffle(pairs)

    n_val = max(1, int(len(pairs) * args.val_ratio))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    print(f"train: {len(train_pairs)}  val: {len(val_pairs)}")

    class_pixel_counter = Counter()
    for split, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        img_out = out_dir / "img_dir" / split
        ann_out = out_dir / "ann_dir" / split
        img_out.mkdir(parents=True, exist_ok=True)
        ann_out.mkdir(parents=True, exist_ok=True)

        for class_name, img_path, mask_path in split_pairs:
            out_stem = f"{class_name}__{img_path.stem}"
            img_arr = read_image_any(img_path)
            mask_arr = read_image_any(mask_path)
            save_image_as_png(img_arr, img_out / f"{out_stem}.png")
            save_mask_as_png(mask_arr, ann_out / f"{out_stem}.png", args.num_classes, class_pixel_counter)

    print(f"Finish. Data has been written: {out_dir}")
    print('Directory structure:')
    print(f"  {out_dir}/img_dir/train  ({len(train_pairs)} Zhang)")
    print(f"  {out_dir}/img_dir/val    ({len(val_pairs)} Zhang)")
    print(f"  {out_dir}/ann_dir/train")
    print(f"  {out_dir}/ann_dir/val")

    out_of_range = class_pixel_counter.pop("__out_of_range__", 0)
    total_pixels = sum(class_pixel_counter.values())
    print(f"\\nMask pixel value distribution (num_classes={args.num_classes}, the total number of pixels {total_pixels}):")
    for cls_idx in range(args.num_classes):
        cnt = class_pixel_counter.get(cls_idx, 0)
        pct = 100.0 * cnt / max(total_pixels, 1)
        print(f"  class {cls_idx}: {cnt} ({pct:.2f}%)")
    if out_of_range > 0:
        print(f"[Warning] Yes {out_of_range} pixels whose original value exceeds [0, {args.num_classes - 1}] range,"
              f"It has been cropped to the boundary value. It is recommended to check whether there is any problem with the original mask annotation.")


if __name__ == "__main__":
    main()
