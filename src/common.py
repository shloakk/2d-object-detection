"""Shared helpers: paths, device selection, config loading, class-id mapping."""

import json
import os
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Data and results can be moved (e.g. Colab local disk / Google Drive) with env vars.
DATA_DIR = Path(os.environ.get("CMPE249_DATA", REPO_ROOT / "data")).resolve()
RESULTS_DIR = Path(os.environ.get("CMPE249_RESULTS", REPO_ROOT / "results")).resolve()
WEIGHTS_DIR = Path(os.environ.get("CMPE249_WEIGHTS", REPO_ROOT / "weights")).resolve()

# COCO category ids (the non-contiguous 1-90 "paper" ids used in instances_*.json).
COCO_PERSON, COCO_CAR, COCO_TRUCK = 1, 3, 8
COCO_CAT_NAMES = {COCO_PERSON: "person", COCO_CAR: "car", COCO_TRUCK: "truck"}


def coco80_to_coco91():
    """Ultralytics' own lookup: contiguous class index 0-79 -> COCO category id 1-90."""
    from ultralytics.data.converter import coco80_to_coco91_class

    return coco80_to_coco91_class()


def select_device(preference="auto"):
    """Pick cuda, then mps, then cpu. An explicit preference ('cpu', 'mps', 'cuda:0') wins."""
    if preference and preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device):
    """Wait for queued GPU work to finish so wall-clock timings are honest."""
    device = str(device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mps"):
        torch.mps.synchronize()


def load_config(path, overrides=None):
    """Load a run config YAML and fill in defaults. `overrides` is a dict of CLI overrides."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("run_name", Path(path).stem)
    cfg.setdefault("split", "val")
    cfg.setdefault("imgsz", 640)
    cfg.setdefault("conf", 0.001)
    cfg.setdefault("max_det", 300)
    cfg.setdefault("device", "auto")
    cfg.setdefault("eval_classes", None)
    cfg.setdefault("half", False)
    # NMS IoU: Ultralytics default 0.7. torchvision Faster R-CNN keeps its own default 0.5
    # (the value its published COCO numbers were produced with).
    cfg.setdefault("iou", 0.5 if cfg["model"] == "fasterrcnn_v2" else 0.7)
    # YOLO26 head: "one2one" = NMS-free end-to-end head (nms=False in Ultralytics),
    # "one2many" = dense head followed by NMS (Ultralytics' default, nms=None).
    if cfg["model"].startswith("yolo26"):
        cfg.setdefault("yolo26_head", "one2one")
        # rect=True: letterbox to the image's aspect ratio (as Ultralytics val() does) instead of a
        # 640x640 square. RT-DETR always uses a square input, so it ignores this.
        cfg.setdefault("rect", True)
    for k, v in (overrides or {}).items():
        if v is not None:
            cfg[k] = v
    return cfg


def data_path(rel):
    """Resolve a config path relative to DATA_DIR (absolute paths pass through)."""
    p = Path(rel)
    return p if p.is_absolute() else DATA_DIR / p


def load_json(path):
    """Read JSON, or gzipped JSON if the name ends in .gz (synced detections are .json.gz)."""
    if str(path).endswith(".gz"):
        import gzip

        with gzip.open(path, "rt") as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
