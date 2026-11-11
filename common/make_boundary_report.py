"""
make_boundary_report.py
Read the JSON output by summarize_metrics.py and generate the \"boundary_metrics_comparison.md\" report.

Usage:
    python common/make_boundary_report.py         --json multiclass_segmentation_comparison/boundary_metrics_comparison.json         --out multiclass_segmentation_comparison/boundary_metrics_comparison.md
"""
import argparse
import json
from pathlib import Path

CLASS_NAMES = ['background', "Podocarpus macrophyllus", "Camellia sasanqua", "Loropetalum chinense", "Ligustrum quihoui"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default='multiclass_segmentation_comparison/boundary_metrics_comparison.json')
    ap.add_argument("--out", type=str, default='multiclass_segmentation_comparison/boundary_metrics_comparison.md')
    args = ap.parse_args()

    data = json.load(open(args.json))
    cfg = data["config"]
    res = data["results"]

    order = ["unet", "deeplabv3plus", "hrnet", "segformer", "swin",
             "oneformer", "internimage", "vmamba", "fmsam", "sam3",
             "grounded_sam2", "catseg"]
    names = {"unet": "UNet", "deeplabv3plus": "DeepLabV3+", "hrnet": "HRNet",
             "segformer": "SegFormer", "swin": "Swin", "oneformer": "OneFormer",
             "internimage": "InternImage", "vmamba": "VMamba", "fmsam": "FM-SAM",
             "sam3": "SAM3", "grounded_sam2": "Grounded-SAM2", "catseg": "CAT-Seg"}

    lines = []
    lines.append(f"# boundary_metrics_comparison (Boundary IoU / Boundary F1)")
    lines.append("")
    lines.append(f"The verification set val has 486 images (the same division as the pixel-level mIoU comparison experiment)."
                 f"For the indicator conventions, see \"Method Description\" at the end of the article.")
    lines.append("")
    lines.append(f"**Parameter**: Boundary IoU expansion radius d={cfg['dilation']}px"
                 f"(Approximately 1% of the image diagonal); Boundary F1 matching tolerance τ={cfg['tau']}px.")
    lines.append("")

    # ---- Summary table ----
    lines.append('## Summary')
    lines.append("")
    lines.append('| Model | mIoU | mDice | mBoundaryIoU | mBoundaryF1 | mBoundaryIoU(4 foreground) | mBoundaryF1(4 foreground) |')
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for k in order:
        r = res[k]
        p = r["pixel"]
        b = r["boundary"]
        lines.append(f"| {names[k]} | {p['mIoU']:.2f} | {p['mDice']:.2f} | "
                     f"{b['mBoundaryIoU_5']:.2f} | {b['mBoundaryF1_5']:.2f} | "
                     f"{b['mBoundaryIoU_fg4']:.2f} | {b['mBoundaryF1_fg4']:.2f} |")
    lines.append("")

    # ---- Category-by-category Boundary IoU ----
    lines.append('## Class-by-class Boundary IoU')
    lines.append("")
    lines.append('| Model | ' + " | ".join(CLASS_NAMES) + " |")
    lines.append("| --- | " + " | ".join(["---"] * 5) + " |")
    for k in order:
        b = res[k]["boundary"]["per_class_BoundaryIoU"]
        vals = [f"{b[str(i)]:.2f}" for i in range(5)]
        lines.append(f"| {names[k]} | " + " | ".join(vals) + " |")
    lines.append("")

    # ---- Category by category Boundary F1 ----
    lines.append('## Category by category Boundary F1')
    lines.append("")
    lines.append('| Model | ' + " | ".join(CLASS_NAMES) + " |")
    lines.append("| --- | " + " | ".join(["---"] * 5) + " |")
    for k in order:
        b = res[k]["boundary"]["per_class_BoundaryF1"]
        vals = [f"{b[str(i)]:.2f}" for i in range(5)]
        lines.append(f"| {names[k]} | " + " | ".join(vals) + " |")
    lines.append("")

    # ----Method description----
    lines.append('## Method description')
    lines.append("")
    lines.append(f"1. **Boundary IoU** (Cheng et al., CVPR 2021): For each category, take the GT mask boundary"
                 f" B(G_c) (1 pixel wide, 8 connected), dilation d={cfg['dilation']}px gets the band;"
                 f"`B-IoU = |(P∩G)∩B_d(G)| / |(P∪G)∩B_d(G)|`, the numerator and denominator are both limited to the band area."
                 f"Calculate picture by picture and class by class, average across pictures within a class (skip if the GT of this class in the picture is empty),"
                 f"`mBoundaryIoU` is the 5-category mean.")
    lines.append("")
    lines.append(f"2. **Boundary F1** (boundary matching F1): Extract prediction boundary B(P_c) and GT boundary B(G_c),"
                 f"Match according to Euclidean distance, distance ≤ τ={cfg['tau']}px marks the hit;"
                 f"`Precision = number of hit prediction boundary pixels/|B(P_c)|`,"
                 f"`Recall = Hit GT boundary pixel number/|B(G_c)|`, `B-F1 = 2PR/(P+R)`."
                 f"Calculated picture by picture and class by class, averaged across pictures within a class, `mBoundaryF1` is the mean of 5 classes.")
    lines.append("")
    lines.append('3. **Boundary extraction**: `mask & ~erosion(mask)` (3×3 all 1 structural elements).')
    lines.append("")
    lines.append('4. **Metric conventions**:'
                 '(a) The 5-category mean value of mBoundaryIoU/mBoundaryF1 is consistent with the existing mIoU conventions (including background);'
                 'Also gives the 4 foreground class means.'
                 '(b) Boundary IoU does not guarantee ≤ pixel IoU: when the prediction is slightly indented relative to GT,'
                 'The coincidence rate may be higher within the boundary zone, which is normal.'
                 '(c) The mask of the detection/open vocabulary model (FM-SAM, Grounded-SAM2, CAT-Seg) is determined by the box'
                 'The instance masks are spliced together. The boundary error is mixed with the detection frame error, and the image without detection frame is missing as a whole.'
                 'Things to note when comparing to end-to-end segmentation models.')
    lines.append("")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated -> {args.out}")


if __name__ == "__main__":
    main()
