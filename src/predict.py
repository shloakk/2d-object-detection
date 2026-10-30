"""Run one model on one dataset, write COCO-format detections, then score them.

    python -m src.predict --config configs/debug/yolo26s_coco.yaml

Output folder results/<run_id>/ gets: detections.json, run_meta.json, metrics.csv,
per_class.csv and env/ (pip freeze, nvidia-smi, versions, git hash, command).

Category ids are the classic silent bug:
  * Ultralytics outputs contiguous indices 0-79 -> mapped to COCO ids 1-90 with coco80_to_coco91_class().
  * torchvision Faster R-CNN already outputs COCO ids (91-slot space) -> used as-is.
Boxes from both libraries are xyxy in original-image pixels -> converted to COCO xywh.
"""

import argparse
import time
from datetime import datetime

import torch
from tqdm import tqdm

from src.common import (RESULTS_DIR, WEIGHTS_DIR, coco80_to_coco91, data_path, load_config, load_json,
                        save_json, select_device)
from src.env_capture import capture_env
from src.evaluate import evaluate


def xyxy_to_xywh(b):
    x1, y1, x2, y2 = (float(v) for v in b)
    return [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]


# ---------------------------------------------------------------- model loading
def load_model(cfg, device):
    """Return (model, family) where family is 'ultralytics' or 'torchvision'."""
    name = cfg["model"]
    if name.startswith("yolo26") or name.startswith("rtdetr"):
        from ultralytics import RTDETR, YOLO

        weights = WEIGHTS_DIR / cfg["weights"]
        weights.parent.mkdir(parents=True, exist_ok=True)
        model = (RTDETR if name.startswith("rtdetr") else YOLO)(str(weights))  # auto-downloads if missing
        return model, "ultralytics"
    if name == "fasterrcnn_v2":
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

        model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
            box_score_thresh=cfg["conf"],
            box_nms_thresh=cfg["iou"],
            box_detections_per_img=cfg["max_det"],
        )
        model.eval().to(device)
        if cfg["half"]:
            model.half()
        return model, "torchvision"
    raise ValueError(f"unknown model {name}")


def ultralytics_kwargs(cfg, device):
    """Predict kwargs shared by predict.py, bench_latency.py and the val() cross-check."""
    kw = dict(imgsz=cfg["imgsz"], conf=cfg["conf"], iou=cfg["iou"], max_det=cfg["max_det"], device=device,
              quantize=16 if cfg["half"] else None, verbose=False)
    if cfg["model"].startswith("yolo26"):
        # nms=False selects the NMS-free one-to-one head; nms=None (library default) = one-to-many + NMS.
        kw["nms"] = False if cfg["yolo26_head"] == "one2one" else None
        kw["rect"] = cfg["rect"]
    return kw


def load_image_tensor(path, device, half=False):
    from torchvision.io import ImageReadMode, decode_image

    img = decode_image(str(path), mode=ImageReadMode.RGB).to(device)  # uint8 CHW, RGB
    img = img.half() if half else img.float()
    return img / 255.0  # Faster R-CNN normalizes and resizes internally


# ---------------------------------------------------------------- inference
def predict_ultralytics(model, cfg, device, images, img_dir):
    id_map = coco80_to_coco91()
    kw = ultralytics_kwargs(cfg, device)
    dets = []
    for im in tqdm(images):
        # One image per call = batch 1. (Passing a list makes Ultralytics batch the whole list,
        # which forces square padding and would run out of memory on full COCO.)
        r = model.predict(str(img_dir / im["file_name"]), **kw)[0]
        image_id = im["id"]
        xyxy = r.boxes.xyxy.cpu().tolist()  # original-image pixels
        cls = r.boxes.cls.int().cpu().tolist()  # contiguous 0-79
        conf = r.boxes.conf.cpu().tolist()
        for b, c, s in zip(xyxy, cls, conf):
            dets.append({"image_id": image_id, "category_id": id_map[c], "bbox": xyxy_to_xywh(b),
                         "score": round(float(s), 5)})
    return dets


@torch.inference_mode()
def predict_torchvision(model, cfg, device, images, img_dir):
    dets = []
    for im in tqdm(images):
        out = model([load_image_tensor(img_dir / im["file_name"], device, cfg["half"])])[0]
        for b, c, s in zip(out["boxes"].float().cpu().tolist(), out["labels"].cpu().tolist(),
                           out["scores"].float().cpu().tolist()):
            # labels are already COCO category ids (1-90): no remapping
            dets.append({"image_id": im["id"], "category_id": int(c), "bbox": xyxy_to_xywh(b),
                         "score": round(float(s), 5)})
    return dets


def yolo26_head_used(model, cfg):
    """Read the head actually used back from the model, not just the config."""
    if not cfg["model"].startswith("yolo26"):
        return None
    used_e2e = bool(getattr(model.predictor.model, "end2end", False))
    head = "one2one (NMS-free)" if used_e2e else "one2many + NMS"
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None, help="override config device (auto|cpu|mps|cuda)")
    ap.add_argument("--half", action="store_true", default=None, help="FP16 inference")
    ap.add_argument("--no-eval", action="store_true", help="only write detections")
    args = ap.parse_args()

    cfg = load_config(args.config, {"device": args.device, "half": args.half})
    device = select_device(cfg["device"])
    run_id = f"{cfg['run_name']}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    capture_env(run_dir, device)

    gt_json = data_path(cfg["gt_json"])
    img_dir = data_path(cfg["images"])
    images = load_json(gt_json)["images"]

    print(f"[{run_id}] {cfg['model']} on {cfg['dataset']} ({len(images)} images) device={device}")
    model, family = load_model(cfg, device)
    t0 = time.time()
    if family == "ultralytics":
        dets = predict_ultralytics(model, cfg, device, images, img_dir)
    else:
        dets = predict_torchvision(model, cfg, device, images, img_dir)
    elapsed = time.time() - t0

    save_json(dets, run_dir / "detections.json")
    meta = {
        "run_id": run_id,
        "config": str(args.config),
        "model": cfg["model"],
        "checkpoint": cfg.get("weights", "torchvision FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1"),
        "dataset": cfg["dataset"],
        "split": cfg["split"],
        "subset": cfg.get("subset"),
        "n_images": len(images),
        "gt_json": str(gt_json),
        "images": str(img_dir),
        "imgsz": cfg["imgsz"] if family == "ultralytics" else "torchvision default (min 800, max 1333)",
        "conf": cfg["conf"],
        "iou": cfg["iou"],
        "max_det": cfg["max_det"],
        "half": bool(cfg["half"]),
        "device": device,
        "yolo26_head": yolo26_head_used(model, cfg) if family == "ultralytics" else None,
        "rect": cfg.get("rect"),
        "eval_classes": cfg["eval_classes"],
        "n_detections": len(dets),
        "predict_seconds_total": round(elapsed, 2),
        "note": "predict timing is wall-clock incl. image IO; use bench_latency.py for latency",
    }
    save_json(meta, run_dir / "run_meta.json")
    print(f"{len(dets)} detections -> {run_dir / 'detections.json'}")

    if not args.no_eval:
        evaluate(gt_json, run_dir / "detections.json", cfg["eval_classes"], run_dir, run_id, meta)


if __name__ == "__main__":
    main()
