"""
eval_boundary.py
Compute boundary metrics for semantic segmentation on the validation set: Boundary IoU and Boundary F1 (per-class).

Boundary IoU (Cheng et al., CVPR 2021, \"Boundary IoU: Improving Object-Centric
Image Segmentation Evaluation\"): 
    B_d(G) = the band obtained by dilating the GT mask boundary B(G) by d pixels
    B-IoU_c = |(P_c ∩ G_c) ∩ B_d(G_c)| / |(P_c ∪ G_c) ∩ B_d(G_c)|

Boundary F1 (boundary matching F1, common conventions for remote sensing/medical segmentation):
    Extract the prediction boundary B(P_c) and the GT boundary B(G_c) (1 pixel wide), match them according to the Euclidean distance, and score hits if the distance ≤ τ pixels.
    Precision = Number of hit prediction boundary pixels / |B(P_c)|
    Recall = Hit GT boundary pixel number / |B(G_c)|
    B-F1_c    = 2·P·R / (P + R)

Aggregation conventions: Calculate picture by picture and class by class → nanmean across pictures within a class (skip if the GT of this class in the picture is empty) →
    mBoundaryIoU / mBoundaryF1 is the mean value of 5 categories (consistent with the existing mIoU conventions), and outputs the mean value of 4 foreground classes at the same time.

Usage:
    python common/eval_boundary.py --pred_dir predictions/unet --gt_dir data/ann_dir/val         --num_classes 5 --dilation 14 --tau 3 --pred_suffix _mask.png --name unet
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

STRUCT = np.ones((3, 3), dtype=bool)  # 8 connected

CLASS_NAMES = ["background", "Podocarpus macrophyllus", "Camellia sasanqua", "Loropetalum chinense", "Ligustrum quihoui"]


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    '1 pixel wide border: mask minus erosion result (8-connected).'
    eroded = ndimage.binary_erosion(mask, structure=STRUCT)
    return mask & ~eroded


def per_image_boundary_metrics(gt: np.ndarray, pred: np.ndarray,
                               num_classes: int, dilation: int,
                               tau: int) -> dict:
    'The class-by-class boundary indicator of a single image returns {class: (biou, bf1)}. If the class is empty in GT, it will be skipped.'
    out = {}
    for c in range(num_classes):
        G = gt == c
        if G.sum() == 0:
            continue
        P = pred == c
        if P.sum() == 0:
            # The prediction is completely empty: the numerator and denominator of B-IoU are only within the band domain, and P is empty in the band domain → 0
            bg = boundary_mask(G)
            band = ndimage.binary_dilation(bg, structure=STRUCT, iterations=dilation)
            denom = band.sum()
            out[c] = (0.0, 0.0) if denom > 0 else (np.nan, np.nan)
            continue

        # ---- Boundary IoU ----
        bg = boundary_mask(G)
        band = ndimage.binary_dilation(bg, structure=STRUCT, iterations=dilation)
        inter = (P & G & band).sum()
        union = ((P | G) & band).sum()
        biou = inter / union if union > 0 else np.nan

        # ---- Boundary F1 ----
        bp = boundary_mask(P)
        bpr = bp.sum()
        # The distance from each GT boundary pixel to the nearest predicted boundary pixel
        d_gt = ndimage.distance_transform_edt(~bp)
        hit_gt = int((d_gt[bg] <= tau).sum())  # Number of pixels hitting GT boundary
        # The distance from each predicted boundary pixel to the nearest GT boundary pixel
        d_pred = ndimage.distance_transform_edt(~bg)
        hit_pred = int((d_pred[bp] <= tau).sum())  # Number of hit prediction boundary pixels
        prec = hit_pred / bpr if bpr > 0 else 0.0
        rec = hit_gt / bg.sum() if bg.sum() > 0 else 0.0
        bf1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        out[c] = (biou, bf1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", type=str, required=True)
    ap.add_argument("--gt_dir", type=str, required=True)
    ap.add_argument("--num_classes", type=int, default=5)
    ap.add_argument("--dilation", type=int, default=14,
                    help='Boundary IoU boundary expansion radius (px), default 1%% Figure diagonal ≈ 14')
    ap.add_argument("--tau", type=int, default=3,
                    help='Boundary F1 boundary matching tolerance (px)')
    ap.add_argument("--pred_suffix", type=str, default="_mask.png",
                    help='The suffix appended to the stem of the prediction file, None/empty means the same name as gt')
    ap.add_argument("--name", type=str, default="", help='Result label (model name)')
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    gt_files = sorted(gt_dir.glob("*.png"))

    # class -> list of (biou, bf1) over images where class present in GT
    acc = {c: ([], []) for c in range(args.num_classes)}
    missing = []
    for gt_f in gt_files:
        pred_name = f"{gt_f.stem}{args.pred_suffix}" if args.pred_suffix else gt_f.name
        pred_f = pred_dir / pred_name
        if not pred_f.exists():
            missing.append(pred_name)
            continue
        gt = np.array(Image.open(gt_f))
        pred = np.array(Image.open(pred_f))
        if gt.shape != pred.shape:
            print(f"[WARN] shape mismatch {gt_f.name}: gt{gt.shape} pred{pred.shape}")
            continue
        res = per_image_boundary_metrics(gt, pred, args.num_classes,
                                         args.dilation, args.tau)
        for c, (biou, bf1) in res.items():
            acc[c][0].append(biou)
            acc[c][1].append(bf1)

    if missing:
        print(f"[WARN] Missing predictions {len(missing)} each: {missing[:5]} ...")
        if len(missing) > len(gt_files) // 2:
            print('Too few prediction files, aborted.')
            return

    name = args.name or Path(args.pred_dir).name
    print(f"\n=== Boundary metrics: {name} (dilation={args.dilation}, tau={args.tau}) ===")
    print(f"{'Class':<10}{'B-IoU':>9}{'B-F1':>9}{'n_img':>7}")
    row = {"b_iou": {}, "b_f1": {}}
    for c in range(args.num_classes):
        ios, f1s = acc[c]
        bi = float(np.nanmean(ios)) * 100 if ios else float("nan")
        bf = float(np.nanmean(f1s)) * 100 if f1s else float("nan")
        nm = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class{c}"
        print(f"{nm:<10}{bi:>8.2f}{bf:>8.2f}{len(ios):>7d}")
        row["b_iou"][c] = bi
        row["b_f1"][c] = bf

    def m(d):
        vals = [v for v in d.values() if not np.isnan(v)]
        return float(np.mean(vals))

    def m_fg(d):
        vals = [v for k, v in d.items() if k >= 1 and not np.isnan(v)]
        return float(np.mean(vals))

    print(f"\\nmBoundaryIoU(5 categories): {m(row['b_iou']):.2f}   mBoundaryIoU(4foreground): {m_fg(row['b_iou']):.2f}")
    print(f"mBoundaryF1 (Category 5): {m(row['b_f1']):.2f}   mBoundaryF1 (4 foreground): {m_fg(row['b_f1']):.2f}")


if __name__ == "__main__":
    main()
