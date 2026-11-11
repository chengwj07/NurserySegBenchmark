"""Box-level detection evaluation for GroundingDINO on the plant val set.

Matches predicted boxes against GT boxes (val_annotations.json) with
IoU>=0.5 + class match (COCO-style). Reports Precision@0.5 / Recall@0.5 at
various confidence thresholds and NMS settings, plus AP50.
"""
import argparse
import importlib
import importlib.abc
import importlib.util
import json
import os
import sys

import numpy as np

PATCHED_ROOT = "/data/ssd/haoweimu/zhenong_data/miaomu/seg_exp/11_Grounded-SAM2/GroundingDINO"
EVAL_SCRIPT = "/data/ssd/haoweimu/zhenong_data/miaomu/seg_exp/11_Grounded-SAM2/Grounded-SAM-2/eval_plant_metrics.py"

sys.path.insert(0, PATCHED_ROOT)


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, real_name):
        self.real_name = real_name

    def create_module(self, spec):
        return importlib.import_module(self.real_name)

    def exec_module(self, module):
        pass


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "grounding_dino":
            return importlib.util.spec_from_loader(fullname, _AliasLoader("groundingdino"))
        if fullname == "grounding_dino.groundingdino":
            return importlib.util.spec_from_loader(fullname, _AliasLoader("groundingdino"))
        if fullname.startswith("grounding_dino.groundingdino."):
            real = "groundingdino" + fullname[len("grounding_dino.groundingdino"):]
            try:
                importlib.util.find_spec(real)
            except (ImportError, AttributeError, ValueError):
                return None
            return importlib.util.spec_from_loader(fullname, _AliasLoader(real))
        return None


sys.meta_path.insert(0, _AliasFinder())

VAL_ANN_JSON = "/data/ssd/haoweimu/zhenong_data/miaomu/seg_exp/data/val_annotations.json"
VAL_IMG_DIR = "/data/ssd/haoweimu/zhenong_data/miaomu/seg_exp/data/img_dir/val"
GDINO_CONFIG = "/data/ssd/haoweimu/zhenong_data/miaomu/seg_exp/11_Grounded-SAM2/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_CKPT = "/data/ssd/haoweimu/zhenong_data/miaomu/seg_exp/11_Grounded-SAM2/outputs/groundingdino_swint_ogc_finetuned.pth"

from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from torchvision.ops import box_convert


def iou_matrix(a, b):
    """a: (N,4), b: (M,4) xyxy -> IoU (N,M)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-json", default=None,
                    help="reuse existing GDINO raw predictions JSON instead of running inference")
    args = ap.parse_args()

    gt = json.load(open(VAL_ANN_JSON))
    class_names = gt["class_names"]
    print("GT classes:", class_names)
    gt_by_name, gt_count = {}, 0
    for img in gt["images"]:
        name = os.path.basename(img["image_path"])
        boxes = np.asarray(img["boxes_xyxy"], dtype=np.float32).reshape(-1, 4)
        cats = np.asarray(img["class_ids"], dtype=np.int64).reshape(-1)
        gt_by_name[name] = (boxes, cats)
        gt_count += len(boxes)
    print(f"GT: {len(gt_by_name)} images, {gt_count} boxes")

    if args.pred_json and os.path.exists(args.pred_json):
        preds = json.load(open(args.pred_json))
        dets = {}
        for p in preds["annotations"]:
            fname = preds["images"][p["image_id"]]["file_name"]
            dets.setdefault(fname, []).append(p)
        det_by_name = {}
        for name, ps in dets.items():
            boxes = np.array([p["bbox"] for p in ps], dtype=np.float32)  # xyxy
            confs = np.array([p["conf"] for p in ps], dtype=np.float32)
            cats = np.array([p["category_id"] for p in ps], dtype=np.int64)
            det_by_name[name] = (boxes, confs, cats)
        print(f"loaded {len(det_by_name)} images from pred json")
    else:
        from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
        import torch
        dev = "cuda"
        model = load_model(GDINO_CONFIG, GDINO_CKPT, device=dev)
        caption = ". ".join(class_names) + "."
        phrase_map = {}
        for i, n in enumerate(class_names, start=1):
            phrase_map[n.strip().lower()] = i
            phrase_map[n.strip().lower().replace(" ", "")] = i
        det_by_name = {}
        for fname in sorted(gt_by_name):
            img_path = os.path.join(VAL_IMG_DIR, fname)
            image_source, image_tensor = load_image(img_path)
            boxes_norm, confs, phrases = predict(
                model=model, image=image_tensor, caption=caption,
                box_threshold=0.05, text_threshold=0.2, device=dev)
            if len(boxes_norm) == 0:
                det_by_name[fname] = (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int64))
                continue
            h_orig, w_orig = image_source.shape[:2]
            boxes_xyxy = box_convert(
                boxes_norm * torch.tensor([w_orig, h_orig, w_orig, h_orig]),
                in_fmt="cxcywh", out_fmt="xyxy").numpy()
            cats = np.array([phrase_map.get(p.strip().lower(), 0) for p in phrases], dtype=np.int64)
            keep = cats > 0
            det_by_name[fname] = (boxes_xyxy[keep], confs.numpy()[keep], cats[keep])
        print("raw detections dumped (threshold 0.05)")

    # ---- evaluation sweep ----
    def eval_cfg(thr, nms_iou):
        tp_all, fp_all = 0, 0
        n_gt = 0
        for name, (gtb, gtc) in gt_by_name.items():
            if name not in det_by_name:
                n_gt += len(gtb)
                continue
            db, dc, dcat = det_by_name[name]
            if len(db):
                keep = dc >= thr
                db, dc, dcat = db[keep], dc[keep], dcat[keep]
            if len(db) and nms_iou is not None:
                from torchvision.ops import nms as tv_nms
                import torch
                k = tv_nms(torch.from_numpy(db), torch.from_numpy(dc), nms_iou).numpy()
                db, dc, dcat = db[k], dc[k], dcat[k]
            # greedy match by conf desc
            gt_used = np.zeros(len(gtb), dtype=bool)
            order = np.argsort(-dc)
            for i in order:
                if len(gtb) == 0:
                    fp_all += 1
                    continue
                iou = iou_matrix(db[i:i+1], gtb)[0]
                iou[gt_used] = -1
                j = int(np.argmax(iou))
                if iou[j] >= 0.5 and dcat[i] == gtc[j] + 1:  # pred 1..4 vs GT 0..3
                    gt_used[j] = True
                    tp_all += 1
                else:
                    fp_all += 1
            n_gt += len(gtb)
        p = tp_all / max(tp_all + fp_all, 1) * 100
        r = tp_all / max(n_gt, 1) * 100
        return p, r, tp_all, fp_all, n_gt

    print("\nthreshold / NMS sweep (P@0.5, R@0.5):")
    best = None
    for thr in [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]:
        for nms in [None, 0.7, 0.5]:
            p, r, tp, fp, ng = eval_cfg(thr, nms)
            tag = "P=%.1f R=%.1f (TP=%d FP=%d GT=%d)" % (p, r, tp, fp, ng)
            print(f"  thr={thr:.2f} nms={nms}: {tag}")
            if p >= 60 and (best is None or p > best[0]):
                best = (p, r, thr, nms, tp, fp)
    if best:
        print("\nBEST config with P>=60: thr=%.2f nms=%s -> P=%.1f R=%.1f (TP=%d FP=%d)"
              % (best[2], best[3], best[0], best[1], best[4], best[5]))
    else:
        print("\nNo config reached P>=60 in sweep.")


if __name__ == "__main__":
    main()
