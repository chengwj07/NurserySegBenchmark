"""Data preparation module: Convert binary mask annotations into YOLO detection format labels.

Data directory structure (input):
    data_root/
    +-- ann_dir/
    | +-- train/ (000.png, 001.png, ...) # Binary mask (0-1 or 0-255)
    |   +-- val/
    +-- img_dir/
        +-- train/ (000.png, 001.png, ...) # Original image
        +-- val/

Output:
    - Generate a .txt tag file with the same name under img_dir/{split}/
    - Generate data.yaml under data_root
"""
import pathlib
import cv2
import numpy as np
import yaml


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    """Binarize the mask, compatible with two storage formats: 0-1 and 0-255.

    Args:
        mask: HxW uint8 grayscale mask.

    Returns:
        HxW uint8 binary mask (0=background, 1=foreground).
    """
    return (mask > 0).astype(np.uint8)


def mask_to_yolo_labels(mask: np.ndarray, num_classes: int = 1,
                        min_area: int = 16) -> list[str]:
    """Convert a single mask into a list of rows of YOLO detection labels.

    Connected domain analysis is performed for each foreground category, and a detection frame is generated for each independent region.
    Output format: \"class x_center y_center width height\" (normalized to 0-1),
    class starts from 0, corresponding to the order of names in data.yaml.

    Args:
        mask: HxW uint8 array.
              When num_classes=1, it is regarded as a binary mask (>0 is the foreground, keeping the old behavior);
              When num_classes>1, the pixel value is the class index (0=background, 1..num_classes=foreground).
        num_classes: Number of foreground classes (excluding background).
        min_area: The filter area is too small for the noise connected area.

    Returns:
        YOLO label line list, such as [\"0 0.50 0.50 0.30 0.40\", ...]
    """
    h, w = mask.shape[:2]
    lines = []
    for c in range(1, num_classes + 1):
        # Single category: all non-zero pixels are in the same foreground; multiple categories: taken one by one according to category index
        if num_classes == 1:
            binary = (mask > 0).astype(np.uint8)
        else:
            binary = (mask == c).astype(np.uint8)
        if binary.sum() == 0:
            continue

        cls_idx = c - 1
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        for i in range(1, num_labels):  # Skip 0 (background)
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            # Normalize to 0-1 and crop to legal range
            cx = max(0.0, min(1.0, (x + bw / 2.0) / w))
            cy = max(0.0, min(1.0, (y + bh / 2.0) / h))
            nw = max(0.0, min(1.0, bw / w))
            nh = max(0.0, min(1.0, bh / h))
            lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def prepare_dataset(
    data_root: str,
    ann_dir: str = "ann_dir",
    img_dir: str = "img_dir",
    num_classes: int = 1,
    class_names: list[str] | None = None,
    min_area: int = 16,
) -> str:
    """Convert mask annotations to YOLO detection labels in batches and generate data.yaml.

    The label file is written to img_dir/{split}/ and has the same name as the image but a .txt extension.
    Ultralytics will automatically look for .txt tags in the same directory.

    Args:
        data_root: Data root directory path.
        ann_dir: Mark the directory name (relative to data_root).
        img_dir: image directory name (relative to data_root).
        num_classes: Number of foreground classes (excluding background). 1=two categories, >1=multiple categories,
            Mask pixel values 1..num_classes correspond to class_names in turn.
        class_names: list of foreground class names, the length must be equal to num_classes;
            Names written to data.yaml. By default, class0/class1/... is automatically named.
        min_area: Minimum connected domain area filtering threshold.

    Returns:
        Absolute path to the generated data.yaml file.
    """
    root = pathlib.Path(data_root).resolve()
    ann_base = root / ann_dir
    img_base = root / img_dir

    if not ann_base.exists():
        raise FileNotFoundError(f"ann dir not found: {ann_base}")

    if class_names is None:
        # The second category uses the old default name "foreground"
        class_names = (["foreground"] if num_classes == 1
                       else [f"class{i}" for i in range(num_classes)])
    if len(class_names) != num_classes:
        raise ValueError(
            f"class_names length {len(class_names)} Must equal num_classes {num_classes}"
        )

    splits = ["train", "val"]
    total = 0
    empty_count = 0
    for split in splits:
        ann_split = ann_base / split
        img_split = img_base / split
        if not ann_split.exists():
            print(f"  skip {split} (dir not found)")
            continue
        img_split.mkdir(parents=True, exist_ok=True)
        count = 0
        for mask_file in sorted(ann_split.glob("*.png")):
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"  WARNING: cannot read mask {mask_file}")
                continue
            lines = mask_to_yolo_labels(mask, num_classes=num_classes,
                                        min_area=min_area)
            if not lines:
                empty_count += 1
            label_path = img_split / (mask_file.stem + ".txt")
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""),
                                  encoding="utf-8")
            count += 1
            # First mask prints diagnostic information
            if count == 1 and split == "train":
                unique = np.unique(mask)
                print(f"  [diag] {mask_file.name}: shape={mask.shape}, "
                      f"dtype={mask.dtype}, min={mask.min()}, max={mask.max()}, "
                      f"unique={unique[:10]}")
                print(f"  [diag] -> {len(lines)} labels generated")
        print(f"  {split}: {count} label files ({empty_count if split=='train' else '?'} empty in train)")
        total += count

    if empty_count > 0:
        print(f"  WARNING: {empty_count} masks produced empty labels! "
              f"Check mask pixel values (should be non-zero for foreground).")

    # Generate data.yaml
    data_yaml = root / "data.yaml"
    data_cfg = {
        "path": str(root).replace("\\", "/"),
        "train": f"{img_dir}/train",
        "val": f"{img_dir}/val",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    data_yaml.write_text(yaml.dump(data_cfg, allow_unicode=True, sort_keys=False),
                         encoding="utf-8")
    print(f"  data.yaml -> {data_yaml}")
    print(f"  total {total} label files")
    return str(data_yaml)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="convert masks to YOLO labels")
    parser.add_argument("--data-root", required=True, help="data root dir")
    parser.add_argument("--ann-dir", default="ann_dir", help="annotation dir name")
    parser.add_argument("--img-dir", default="img_dir", help="image dir name")
    parser.add_argument("--num-classes", type=int, default=1,
                        help="foreground class count (1=binary, >1=multiclass)")
    parser.add_argument("--class-names", default=None,
                        help="comma-separated class names (foreground only)")
    args = parser.parse_args()
    names = args.class_names.split(",") if args.class_names else None
    prepare_dataset(args.data_root, args.ann_dir, args.img_dir,
                    num_classes=args.num_classes, class_names=names)
