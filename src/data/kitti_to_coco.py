"""Convert Ultralytics' KITTI val labels (YOLO txt) into a COCO-format ground-truth JSON.

Ultralytics layout (verified after download):
    kitti/images/{train,val}/NNNNNN.png
    kitti/labels/{train,val}/NNNNNN.txt   one line per object: cls cx cy w h (normalized 0-1)

Class mapping into COCO category ids (so GT and predictions share one id space):
    car            -> car (3)      scored
    pedestrian     -> person (1)   scored
    truck          -> truck (8)    scored
    van            -> car (3)      iscrowd=1 ignore region
    person_sitting -> person (1)   iscrowd=1 ignore region
    cyclist        -> person (1)   iscrowd=1 ignore region (a detected rider is not a false positive)
    tram, misc                     dropped

    python -m src.data.kitti_to_coco               # full val (1496) -> kitti/annotations/val_full.json
    python -m src.data.kitti_to_coco --subset 50   # 50 images     -> kitti/annotations/val_subset50.json

Safe to re-run: the download is skipped when every image and label file is present, and the
conversion is skipped when its output JSON already exists (use --force to redo it).
"""

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

import ultralytics
import yaml
from PIL import Image

from src.common import COCO_CAR, COCO_PERSON, COCO_TRUCK, DATA_DIR, load_json, save_json

# Image counts of Ultralytics' KITTI split (from kitti.yaml), used to detect a partial download.
EXPECTED_IMAGES = {"train": 5985, "val": 1496}

# KITTI name -> (COCO category id, iscrowd). Names not listed are dropped.
KITTI_TO_COCO = {
    "car": (COCO_CAR, 0),
    "pedestrian": (COCO_PERSON, 0),
    "truck": (COCO_TRUCK, 0),
    "van": (COCO_CAR, 1),
    "person_sitting": (COCO_PERSON, 1),
    "cyclist": (COCO_PERSON, 1),
}
CATEGORIES = [
    {"id": COCO_PERSON, "name": "person", "supercategory": "person"},
    {"id": COCO_CAR, "name": "car", "supercategory": "vehicle"},
    {"id": COCO_TRUCK, "name": "truck", "supercategory": "vehicle"},
]


def kitti_complete(root):
    """True when every split has its full set of images and labels on disk."""
    for split, n in EXPECTED_IMAGES.items():
        n_img = len(list((root / "images" / split).glob("*.png")))
        n_lbl = len(list((root / "labels" / split).glob("*.txt")))
        if n_img != n or n_lbl != n:
            return False
    return True


def ensure_kitti(root):
    """Download KITTI with the URL from Ultralytics' kitti.yaml unless a complete copy is already on disk."""
    yaml_path = Path(ultralytics.__file__).parent / "cfg" / "datasets" / "kitti.yaml"
    cfg = yaml.safe_load(yaml_path.read_text())
    if kitti_complete(root):
        print(f"KITTI already on disk at {root}, skipping download")
    else:
        from ultralytics.utils.downloads import safe_download

        if root.exists():  # partial copy from an interrupted download: start clean
            print(f"KITTI at {root} is incomplete, re-downloading")
            shutil.rmtree(root)
        safe_download(cfg["download"], dir=root.parent, delete=True)
        if not kitti_complete(root):
            raise RuntimeError(f"KITTI download at {root} is still incomplete")
    return {int(k): v.lower() for k, v in cfg["names"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--subset", type=int, default=0, help="sample N images (0 = full split)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="rebuild the GT JSON even if it exists")
    args = ap.parse_args()

    root = DATA_DIR / "kitti"
    names = ensure_kitti(root)
    print("kitti.yaml class order:", names)

    tag = f"subset{args.subset}" if args.subset else "full"
    out = root / "annotations" / f"{args.split}_{tag}.json"
    n_expected = args.subset or EXPECTED_IMAGES[args.split]
    if out.exists() and not args.force and len(load_json(out)["images"]) == n_expected:
        print(f"{out} already exists ({n_expected} images), skipping conversion (--force to redo)")
        return

    img_dir = root / "images" / args.split
    lbl_dir = root / "labels" / args.split
    img_paths = sorted(img_dir.glob("*.png"))
    if args.subset:
        img_paths = sorted(random.Random(args.seed).sample(img_paths, args.subset))

    images, anns = [], []
    before, after = Counter(), Counter()
    for image_id, img_path in enumerate(img_paths, start=1):
        w, h = Image.open(img_path).size  # sizes differ across KITTI sequences
        images.append({"id": image_id, "file_name": img_path.name, "width": w, "height": h})

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        lines = lbl_path.read_text().split("\n") if lbl_path.exists() else []
        for line in lines:
            if not line.strip():
                continue
            cls, cx, cy, bw, bh = line.split()[:5]
            name = names[int(cls)]
            before[name] += 1
            if name not in KITTI_TO_COCO:
                continue
            cat_id, crowd = KITTI_TO_COCO[name]
            # normalized center xywh -> pixel top-left xywh
            bw, bh = float(bw) * w, float(bh) * h
            x, y = float(cx) * w - bw / 2, float(cy) * h - bh / 2
            anns.append({
                "id": len(anns) + 1, "image_id": image_id, "category_id": cat_id,
                "bbox": [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2), "iscrowd": crowd, "kitti_class": name,
            })
            after[f"{name} -> {'ignore ' if crowd else ''}{CATEGORIES[[c['id'] for c in CATEGORIES].index(cat_id)]['name']}"] += 1

    save_json({"images": images, "annotations": anns, "categories": CATEGORIES}, out)

    print(f"\n{len(images)} images from {img_dir}")
    print("Instances before mapping (KITTI classes):")
    for name in names.values():
        print(f"  {name:15s} {before[name]:6d}")
    print("Instances after mapping (COCO ids):")
    for k, v in sorted(after.items()):
        print(f"  {k:28s} {v:6d}")
    print(f"  dropped (tram/misc)          {sum(before.values()) - sum(after.values()):6d}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
