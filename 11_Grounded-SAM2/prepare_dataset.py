"""
Data preprocessing: Convert mmseg / YOLO style segmentation mask annotations into those required for Grounding DINO training
Instance-level bbox annotations (json).

Enter the directory structure (your actual data, in tiled format):
  data/
    data.yaml # Category name list: names: {0: Podocarpus macrophyllus, 1: Camellia sasanqua, 2: Loropetalum chinense, 3: Ligustrum quihoui}
    img_dir/
      train/*.png # Original image (1024x1024 RGB, the same name as ann)
      val/*.png
    ann_dir/
      train/*.png # Single channel mask: background=0, pixel value 1..N = category id (1 corresponds to category 0 in names)
      val/*.png

Output:
  data/train_annotations.json
  data/val_annotations.json
  Structure:
    {
      "class_names": ["buddhist pine", "sasanqua camellia", ...], # The same order as data.yaml
      "images": [
        {
          "image_path": "...",
          "boxes_xyxy": [[x0,y0,x1,y1], ...], # Pixel coordinates
          "class_ids": [0, 2, 2, ...] # The category index of each box (pointing to class_names)
        }, ...
      ]
    }
  A picture can contain multiple categories at the same time (pixel values 1..N are each connected domain).

Usage:
  python prepare_dataset.py --data_root ./data --min_area 200
"""

import os
import json
import argparse
import re
import cv2
import numpy as np

# ------------------------------------------------------------------
# Chinese class name (name in data.yaml) -> English prompt mapping for training
# The text backbone of GroundingDINO is bert-base-uncased, which does not support Chinese and must be given in English.
# The Latin name you specified is used here; the model usually responds better to natural language phrases, and you want to change to the common name later.
# (such as "buddhist pine") Just change the value here and rerun the script.
# When adding a new category, just add a line here.
# ------------------------------------------------------------------
CLASS_NAME_MAP = {
    "罗汉松": "podocarpus macrophyllus",  # Podocarpus macrophyllus
    "茶梅": "sasanqua camellia",          # Camellia sasanqua
    "红花继木": "loropetalum chinense",   # Loropetalum chinense
    "亮晶女贞": "ligustrum quihoui",      # Ligustrum quihoui
}


def parse_names_from_yaml(yaml_path):
    'Parse names block from mmdet style data.yaml: {idx: name}'
    names = {}
    with open(yaml_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    in_names = False
    for line in lines:
        stripped = line.strip()
        if stripped == "names:":
            in_names = True
            continue
        if in_names:
            m = re.match(r"^(\d+)\s*:\s*(.+?)\s*$", stripped)
            if m:
                names[int(m.group(1))] = m.group(2).strip()
            else:
                in_names = False  # end of names block
    if not names:
        raise ValueError(f"在 {yaml_path} 里没有解析到 names 块, 请检查格式")
    return names


def mask_to_boxes_per_class(mask, num_classes, min_area):
    'Single channel mask -> {class_id: [box_xyxy, ...]} (connected domain analysis for each category value)'
    out = {}
    for v in range(1, num_classes + 1):
        binary = (mask == v).astype(np.uint8)
        if binary.sum() == 0:
            continue
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        boxes = []
        for i in range(1, num_labels):  # 0 is background
            x, y, w, h, area = stats[i]
            if area < min_area:
                continue
            boxes.append([int(x), int(y), int(x + w), int(y + h)])
        if boxes:
            out[v] = boxes
    return out


def build_split(data_root, yaml_path, split, min_area, class_names):
    img_dir = os.path.join(data_root, "img_dir", split)
    ann_dir = os.path.join(data_root, "ann_dir", split)
    if not os.path.isdir(img_dir):
        print(f"[跳过] 目录不存在: {img_dir}")
        return []

    num_classes = len(class_names)
    img_files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".png"))
    images = []
    skipped_empty = 0
    for fname in img_files:
        img_path = os.path.join(img_dir, fname)
        ann_path = os.path.join(ann_dir, fname)
        if not os.path.exists(ann_path):
            print(f"[警告] 找不到对应 mask: {ann_path}, 跳过")
            continue

        mask = cv2.imread(ann_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"[警告] 读取 mask 失败: {ann_path}, 跳过")
            continue

        boxes_by_class = mask_to_boxes_per_class(mask, num_classes, min_area)
        if not boxes_by_class:
            skipped_empty += 1
            continue

        boxes_xyxy, class_ids = [], []
        for v, boxes in sorted(boxes_by_class.items()):
            # The index in pixel value v -> names is v-1
            cid = v - 1
            if cid < 0 or cid >= num_classes:
                print(f"[警告] {ann_path} 出现越界类别像素值 {v}, 忽略该类别")
                continue
            boxes_xyxy.extend(boxes)
            class_ids.extend([cid] * len(boxes))

        if not boxes_xyxy:
            skipped_empty += 1
            continue

        images.append({
            "image_path": img_path,
            "boxes_xyxy": boxes_xyxy,
            "class_ids": class_ids,
        })

    print(f"[{split}] 共 {len(images)} 张有效标注图, 跳过空 mask {skipped_empty} 张")
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--min_area", type=int, default=200,
                        help="小于该像素面积的连通域视为噪声, 过滤掉")
    args = parser.parse_args()

    yaml_path = os.path.join(args.data_root, "data.yaml")
    names = parse_names_from_yaml(yaml_path)
    print(f"从 {yaml_path} 读取到类别: {names}")

    class_names = []
    for idx in sorted(names.keys()):
        cn = names[idx]
        if cn not in CLASS_NAME_MAP:
            raise ValueError(
                f"类别 '{cn}' 还没在 prepare_dataset.py 顶部 CLASS_NAME_MAP 里配英文 prompt, 请补一行"
            )
        class_names.append(CLASS_NAME_MAP[cn])
    print(f"英文 prompt 顺序: {class_names}")

    for split, out_name in [("train", "train_annotations.json"),
                            ("val", "val_annotations.json")]:
        images = build_split(args.data_root, yaml_path, split, args.min_area, class_names)
        out_path = os.path.join(args.data_root, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"class_names": class_names, "images": images},
                      f, ensure_ascii=False, indent=2)
        n_box = sum(len(im["boxes_xyxy"]) for im in images)
        print(f"写出: {out_path}  (图像 {len(images)} 张, 框 {n_box} 个)")


if __name__ == "__main__":
    main()
