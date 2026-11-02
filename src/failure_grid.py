"""KITTI failure grid for the report: one row per failure category, two panels per row.

Each panel is ground truth or one model's predictions (score >= 0.3) on a hand-picked val image.
Row captions also list how many GT boxes each main model misses on that image (IoU 0.5).
Images were picked from `src.viz --worst 16` per model; many worst frames are near-duplicates
from one drive, so each category uses a single frame.

    python -m src.failure_grid            # -> report/fig4_failure_grid.png
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from src.common import DATA_DIR, REPO_ROOT, load_json  # noqa: E402
from src.report import MODELS  # noqa: E402
from src.viz import count_missed, draw  # noqa: E402

# (category, image_id, left panel, right panel); a panel is "gt" or a model key
ROWS = [
    ("Crowd: overlapping pedestrians", 820, "gt", "rtdetr_l"),
    ("Small / distant: trucks missed at 640", 86, "gt", "yolo26s"),
    ("Same frame: 1280 input recovers both trucks", 86, "yolo26s_1280", "frcnn_v2"),
    ("Dense street: parked rows, distant cars", 1339, "gt", "frcnn_v2"),
    ("Edge truncation; van predicted as truck (ignore region)", 89, "gt", "rtdetr_l"),
    ("Cyclist: rider detected as person inside the ignore region", 1144, "gt", "yolo26s"),
]
MAIN = ["yolo26s", "rtdetr_l", "frcnn_v2"]
SCORE = 0.3


def detections(results, model, keep, ids):
    run = sorted(p for p in results.glob(f"{model}_kitti_full_*") if "latency" not in p.name)[-1]
    by_img = defaultdict(list)
    for d in load_json(run / "detections.json.gz"):
        if d["image_id"] in ids and d["score"] >= SCORE and d["category_id"] in keep:
            by_img[d["image_id"]].append(d)
    return by_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO_ROOT / "colab-results"))
    ap.add_argument("--gt", default=str(DATA_DIR / "kitti/annotations/val_full.json"))
    ap.add_argument("--images", default=str(DATA_DIR / "kitti/images/val"))
    ap.add_argument("--out", default=str(REPO_ROOT / "report/fig4_failure_grid.png"))
    args = ap.parse_args()

    gt = load_json(args.gt)
    names = {c["id"]: c["name"] for c in gt["categories"]}
    keep = {i for i, n in names.items() if n in ("car", "person", "truck")}
    images = {im["id"]: im for im in gt["images"]}
    ids = {r[1] for r in ROWS}
    gt_by_img = defaultdict(list)
    for a in gt["annotations"]:
        if a["image_id"] in ids and a["category_id"] in keep:
            gt_by_img[a["image_id"]].append(a)
    models = sorted({p for r in ROWS for p in r[2:] if p != "gt"} | set(MAIN))
    dets = {m: detections(Path(args.results), m, keep, ids) for m in models}

    # Size the figure to the panels (KITTI frames are ~3.3:1) so tight_layout has no slack to spread.
    panel_w, caption_h, title_h = 7.95, 0.62, 0.45
    panel_h = panel_w * 375 / 1242
    fig_h = title_h + len(ROWS) * (panel_h + caption_h)
    fig, axes = plt.subplots(len(ROWS), 2, figsize=(16, fig_h), dpi=130)
    # hspace only pads between rows, so the first row's caption is reserved in `top`
    fig.subplots_adjust(left=0.003, right=0.997, bottom=0.003, top=1 - (title_h + caption_h) / fig_h,
                        wspace=0.012, hspace=caption_h / panel_h)
    for row, (category, i, *panels) in zip(axes, ROWS):
        im = images[i]
        img = np.asarray(Image.open(Path(args.images) / im["file_name"]).convert("RGB"))
        for ax, panel in zip(row, panels):
            if panel == "gt":
                n_gt = sum(1 for g in gt_by_img[i] if not g["iscrowd"])
                draw(ax, img, gt_by_img[i], names, "lime", f"Ground truth: {n_gt} boxes (dashed = ignore)")
            else:
                missed = count_missed(gt_by_img[i], dets[panel][i])
                draw(ax, img, dets[panel][i], names, "red", f"{MODELS[panel]}: {missed} missed")
        summary = "  ·  ".join(f"{MODELS[m]} {count_missed(gt_by_img[i], dets[m][i])}" for m in MAIN)
        row[0].annotate(f"{category}   ({im['file_name']}; missed: {summary})", (0, 1.13),
                        xycoords="axes fraction", fontsize=10, fontweight="bold")
    fig.suptitle(f"KITTI val failure cases (predictions with score >= {SCORE}, missed = no match at IoU 0.5)",
                 fontsize=12, y=1 - 0.12 / fig_h, va="top")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
