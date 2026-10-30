"""Side-by-side ground truth (left) vs predictions (right) for chosen or automatically picked images.

    python -m src.viz --gt data/kitti/annotations/val_subset50.json --dt results/<run>/detections.json --ids 1,2,3
    python -m src.viz --gt ... --dt ... --worst 12      # images with the most missed GT boxes

Green = GT box, green dashed = ignore region (iscrowd=1, e.g. KITTI van / person_sitting),
red = prediction with score >= --score (default 0.3).
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from src.common import load_json  # noqa: E402


def iou_xywh(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def count_missed(gts, preds, iou_thr=0.5):
    """Non-crowd GT boxes with no same-class prediction at IoU >= iou_thr (greedy, highest score first)."""
    used, missed = set(), 0
    preds = sorted(preds, key=lambda d: -d["score"])
    for g in gts:
        if g["iscrowd"]:
            continue
        best, best_j = 0.0, None
        for j, p in enumerate(preds):
            if j in used or p["category_id"] != g["category_id"]:
                continue
            v = iou_xywh(g["bbox"], p["bbox"])
            if v > best:
                best, best_j = v, j
        if best >= iou_thr:
            used.add(best_j)
        else:
            missed += 1
    return missed


def draw(ax, img, boxes, names, color, title):
    ax.imshow(img)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    for b in boxes:
        x, y, w, h = b["bbox"]
        crowd = b.get("iscrowd", 0)
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=1.2,
                                       linestyle="--" if crowd else "-"))
        label = names.get(b["category_id"], str(b["category_id"]))
        if crowd:
            label = f"ignore:{b.get('kitti_class', label)}"
        elif "score" in b:
            label = f"{label} {b['score']:.2f}"
        ax.text(x, y - 2, label, color=color, fontsize=5, va="bottom")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--dt", default=None, help="detections.json (omit to draw GT only)")
    ap.add_argument("--images", required=True, help="image folder")
    ap.add_argument("--ids", default=None, help="comma-separated image ids")
    ap.add_argument("--worst", type=int, default=0, help="pick the N images with the most missed GT boxes")
    ap.add_argument("--score", type=float, default=0.3)
    ap.add_argument("--classes", default=None, help="only draw/score these category names, e.g. car,person,truck")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gt = load_json(args.gt)
    names = {c["id"]: c["name"] for c in gt["categories"]}
    keep_cats = {i for i, n in names.items() if n in args.classes.split(",")} if args.classes else set(names)
    images = {im["id"]: im for im in gt["images"]}
    gt_by_img, dt_by_img = defaultdict(list), defaultdict(list)
    for a in gt["annotations"]:
        if a["category_id"] in keep_cats:
            gt_by_img[a["image_id"]].append(a)
    for d in load_json(args.dt) if args.dt else []:
        if d["score"] >= args.score and d["category_id"] in keep_cats:
            dt_by_img[d["image_id"]].append(d)

    if args.worst:
        missed = {i: count_missed(gt_by_img[i], dt_by_img[i]) for i in images}
        ids = sorted(images, key=lambda i: -missed[i])[: args.worst]
    else:
        ids = [int(i) for i in args.ids.split(",")]
        missed = {i: count_missed(gt_by_img[i], dt_by_img[i]) for i in ids}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in ids:
        im = images[i]
        img = np.asarray(Image.open(Path(args.images) / im["file_name"]).convert("RGB"))
        aspect = im["height"] / im["width"]
        fig, axes = plt.subplots(1, 2, figsize=(14, max(2.5, 7 * aspect + 0.6)), dpi=150)
        n_gt = sum(1 for g in gt_by_img[i] if not g["iscrowd"])
        n_ign = sum(1 for g in gt_by_img[i] if g["iscrowd"])
        draw(axes[0], img, gt_by_img[i], names, "lime", f"GT: {n_gt} boxes, {n_ign} ignore  ({im['file_name']})")
        draw(axes[1], img, dt_by_img[i], names, "red",
             f"Pred (score>={args.score}): {len(dt_by_img[i])} boxes, {missed[i]} GT missed @IoU0.5")
        fig.tight_layout()
        path = out / f"img{i}_{Path(im['file_name']).stem}.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"{path}  missed={missed[i]}")


if __name__ == "__main__":
    main()
