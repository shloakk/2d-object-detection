"""Score COCO-format detections with pycocotools COCOeval (bbox). One evaluator for every run.

    python -m src.evaluate --gt data/coco/annotations/val2017_subset50.json --dt results/<run>/detections.json
    python -m src.evaluate --gt data/kitti/annotations/val_subset50.json --dt ... --classes car,person,truck

Writes metrics.csv and per_class.csv next to the detections and appends one row to
results/summary.csv.
"""

import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.common import COCO_CAT_NAMES, RESULTS_DIR, load_json

STAT_NAMES = ["mAP50_95", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1", "AR10", "AR100",
              "AR_small", "AR_medium", "AR_large"]


def per_class_ap(E, gt):
    """Per-class AP from E.eval['precision'], shape [IoU thr, recall, class, area, maxDets]."""
    precision = E.eval["precision"]
    rows = []
    for k, cat_id in enumerate(E.params.catIds):
        p_all = precision[:, :, k, 0, -1]  # all IoU thresholds, area=all, maxDets=100
        p_50 = precision[0, :, k, 0, -1]  # IoU=0.50
        n_gt = len(gt.getAnnIds(catIds=[cat_id], iscrowd=False))
        rows.append({
            "category_id": cat_id,
            "name": gt.cats[cat_id]["name"],
            "n_gt": n_gt,
            "AP": float(np.mean(p_all[p_all > -1])) if (p_all > -1).any() else float("nan"),
            "AP50": float(np.mean(p_50[p_50 > -1])) if (p_50 > -1).any() else float("nan"),
        })
    return pd.DataFrame(rows)


def evaluate(gt_json, dt_json, classes=None, out_dir=None, run_id=None, meta=None, summary=True):
    """Run COCOeval and write CSVs. Returns a dict of overall metrics."""
    with contextlib.redirect_stdout(io.StringIO()):  # silence pycocotools index chatter
        gt = COCO(str(gt_json))
    dets = load_json(dt_json)
    if not dets:
        raise ValueError(f"{dt_json} has no detections")
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets)

    E = COCOeval(gt, dt, "bbox")
    if classes:
        cat_ids = gt.getCatIds(catNms=classes)
        if len(cat_ids) != len(classes):
            raise ValueError(f"classes {classes} not all found in GT categories")
        E.params.catIds = sorted(cat_ids)
    E.evaluate()
    E.accumulate()
    E.summarize()  # prints the standard 12-line COCO table

    metrics = {name: float(v) for name, v in zip(STAT_NAMES, E.stats)}
    pc = per_class_ap(E, gt)

    out_dir = Path(out_dir or Path(dt_json).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or out_dir.name
    meta = meta or (load_json(out_dir / "run_meta.json") if (out_dir / "run_meta.json").exists() else {})

    row = {"run_id": run_id}
    for k in ("model", "dataset", "split", "subset", "yolo26_head", "imgsz", "device", "half", "n_images"):
        row[k] = meta.get(k)
    row["eval_classes"] = ",".join(classes) if classes else "all"
    row.update(metrics)
    # car/person/truck AP always get their own columns: they drive the COCO-vs-KITTI comparison.
    for cat_id, name in COCO_CAT_NAMES.items():
        match = pc[pc.category_id == cat_id]
        row[f"AP_{name}"] = float(match.AP.iloc[0]) if len(match) else float("nan")

    pd.DataFrame([row]).to_csv(out_dir / "metrics.csv", index=False)
    pc.to_csv(out_dir / "per_class.csv", index=False)
    if summary:
        summary_csv = RESULTS_DIR / "summary.csv"
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(summary_csv, mode="a", header=not summary_csv.exists(), index=False)
    print(f"\n{run_id}: mAP50-95={metrics['mAP50_95']:.4f}  AP50={metrics['AP50']:.4f}  -> {out_dir}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--dt", required=True)
    ap.add_argument("--classes", default=None, help="comma-separated GT category names, e.g. car,person,truck")
    ap.add_argument("--out", default=None, help="output folder (default: folder of --dt)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-summary", action="store_true", help="don't append to results/summary.csv")
    args = ap.parse_args()
    classes = args.classes.split(",") if args.classes else None
    evaluate(args.gt, args.dt, classes, args.out, args.run_id, summary=not args.no_summary)


if __name__ == "__main__":
    main()
