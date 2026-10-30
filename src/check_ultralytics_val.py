"""Acceptance check: our pipeline vs Ultralytics' own model.val() on the same images.

Builds a small Ultralytics-format dataset (symlinked images + YOLO txt labels) from a COCO GT
JSON, then compares:
  A. model.val() with its own metric (Ultralytics default: rect batches, half-stride padding)
  B. pycocotools on val()'s own predictions        -> A vs B isolates Ultralytics' AP code vs pycocotools
  C. pycocotools on predict.py's detections         -> the number our pipeline reports
  D. pycocotools on val(rect=False) predictions     -> square 640 input, same as C only when C is square
C vs B shows the effect of predict() vs val() letterboxing; the box/id/eval path is validated when
our evaluator gives the same number for the same predictions.

    python -m src.check_ultralytics_val --config configs/debug/yolo26s_coco.yaml --dt results/<run>/detections.json
"""

import argparse
import os

import yaml

from src.common import DATA_DIR, coco80_to_coco91, data_path, load_config, load_json, save_json, select_device
from src.evaluate import evaluate
from src.predict import WEIGHTS_DIR, ultralytics_kwargs


def build_yolo_dataset(gt_json, img_dir, out_dir, names):
    """COCO JSON -> images/val (symlinks) + labels/val (cls cx cy w h, normalized). Crowd boxes skipped."""
    coco91 = coco80_to_coco91()
    gt = load_json(gt_json)
    sizes = {im["id"]: (im["width"], im["height"]) for im in gt["images"]}
    (out_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    lines = {im["id"]: [] for im in gt["images"]}
    for a in gt["annotations"]:
        if a["iscrowd"]:
            continue
        x, y, w, h = a["bbox"]
        W, H = sizes[a["image_id"]]
        cls = coco91.index(a["category_id"])
        lines[a["image_id"]].append(f"{cls} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
    for im in gt["images"]:
        link = out_dir / "images" / "val" / im["file_name"]
        if not link.exists():
            os.symlink(img_dir / im["file_name"], link)
        stem = os.path.splitext(im["file_name"])[0]
        (out_dir / "labels" / "val" / f"{stem}.txt").write_text("\n".join(lines[im["id"]]))
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({"path": str(out_dir), "train": "images/val", "val": "images/val",
                                         "names": names}))
    return data_yaml


def run_val(model, data_yaml, kw, rect, out_dir, gt_json, classes):
    """model.val() -> (Ultralytics mAP50-95, pycocotools mAP50-95 on the same predictions)."""
    name = f"val_rect{rect}"
    m = model.val(data=str(data_yaml), batch=1, rect=rect, save_json=True, plots=False,
                  project=str(out_dir), name=name, exist_ok=True, **kw)
    # For a non-COCO-named dataset val() writes category_id = class index + 1; image_id = int(file stem).
    coco91 = coco80_to_coco91()
    preds = load_json(out_dir / name / "predictions.json")
    for p in preds:
        p["category_id"] = coco91[p["category_id"] - 1]
    save_json(preds, out_dir / name / "predictions_coco91.json")
    ours = evaluate(gt_json, out_dir / name / "predictions_coco91.json", classes, out_dir / name, summary=False)
    return m.box.map, ours["mAP50_95"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dt", required=True, help="detections.json from predict.py for the same config")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from ultralytics import RTDETR, YOLO

    cfg = load_config(args.config, {"device": args.device})
    device = select_device(cfg["device"])
    gt_json, img_dir = data_path(cfg["gt_json"]), data_path(cfg["images"])
    gt_images = {im["file_name"].split(".")[0]: im["id"] for im in load_json(gt_json)["images"]}
    if any(int(stem) != i for stem, i in gt_images.items()):
        raise SystemExit("this check needs image_id == int(file stem) (true for COCO)")
    model = (RTDETR if cfg["model"].startswith("rtdetr") else YOLO)(str(WEIGHTS_DIR / cfg["weights"]))

    out_dir = DATA_DIR / f"yolo_format_{cfg['run_name']}"
    data_yaml = build_yolo_dataset(gt_json, img_dir, out_dir, model.names)

    kw = ultralytics_kwargs(cfg, device)
    for k in ("verbose", "rect"):
        kw.pop(k, None)
    a, b = run_val(model, data_yaml, kw, True, out_dir, gt_json, cfg["eval_classes"])
    _, d = run_val(model, data_yaml, kw, False, out_dir, gt_json, cfg["eval_classes"])
    c = evaluate(gt_json, args.dt, cfg["eval_classes"], out_dir / "ours", summary=False)["mAP50_95"]

    print(f"\n===== {cfg['run_name']}: Ultralytics val() vs our pycocotools pipeline =====")
    print(f"settings: {kw}")
    print(f"A  val() default, Ultralytics metric        {a:.4f}")
    print(f"B  val() default preds, our pycocotools      {b:.4f}   (B-A = {100 * (b - a):+.2f} pts: metric code)")
    print(f"C  predict.py detections, our pycocotools    {c:.4f}   (C-A = {100 * (c - a):+.2f} pts: headline gap)")
    print(f"D  val(rect=False) preds, our pycocotools    {d:.4f}   (C-D = {100 * (c - d):+.2f} pts)")


if __name__ == "__main__":
    main()
