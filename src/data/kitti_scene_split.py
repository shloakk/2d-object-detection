"""Stage 3a: a scene-disjoint KITTI split for the zero-shot vs fine-tuned comparison.

Ultralytics' kitti.yaml splits KITTI's 7,481 training frames at random, so consecutive frames of one
drive land in both train and val (about half of its val frames have a near-identical train frame).
Fine-tuning on it scores the model on scenes it has already seen. This split holds out whole raw
drives instead: drives are shuffled with a fixed seed and taken until they cover 20% of the frames.

The frame -> drive mapping comes from the KITTI object devkit (mapping/train_rand.txt and
mapping/train_mapping.txt). The chosen frame ids are committed under splits/kitti_scene/, so Colab
never needs the devkit:

    python -m src.data.kitti_scene_split --make --devkit /path/to/devkit/mapping   # once, locally
    python -m src.data.kitti_scene_split                                          # build data files

The second command writes, under DATA_DIR/kitti:
    splits/scene_{train,val}.txt           Ultralytics image lists (frames come from images/train and images/val)
    kitti_scene.yaml                       Ultralytics dataset yaml for the fine-tune
    annotations/scene_val_full.json        COCO GT for scene val, same class mapping as val_full.json
Safe to re-run: existing outputs are rebuilt only with --force.
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import yaml

from src.common import DATA_DIR, REPO_ROOT, load_json, save_json
from src.data.kitti_to_coco import CATEGORIES, ensure_kitti, print_mapping, to_coco

SPLIT_DIR = REPO_ROOT / "splits" / "kitti_scene"
VAL_SHARE = 0.2
SEED = 0


def make_split(devkit):
    """Assign whole drives to val until they hold VAL_SHARE of the frames; write the id lists."""
    devkit = Path(devkit)
    rand = [int(x) for x in (devkit / "train_rand.txt").read_text().replace(",", " ").split()]
    mapping = [line.split() for line in (devkit / "train_mapping.txt").read_text().splitlines() if line.strip()]
    drive = {frame: mapping[r - 1][1] for frame, r in enumerate(rand)}  # train_rand is 1-based

    frames = defaultdict(list)
    for frame, d in drive.items():
        frames[d].append(frame)
    drives = sorted(frames)
    random.Random(SEED).shuffle(drives)
    val_drives = []
    n_val = 0
    for d in drives:
        if n_val >= VAL_SHARE * len(drive):
            break
        val_drives.append(d)
        n_val += len(frames[d])

    val = sorted(f for d in val_drives for f in frames[d])
    train = sorted(set(drive) - set(val))
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    (SPLIT_DIR / "val.txt").write_text("".join(f"{f:06d}\n" for f in val))
    (SPLIT_DIR / "train.txt").write_text("".join(f"{f:06d}\n" for f in train))
    (SPLIT_DIR / "val_drives.txt").write_text("".join(f"{d}\n" for d in sorted(val_drives)))
    print(f"{len(val_drives)} of {len(drives)} drives held out: {len(val)} val / {len(train)} train frames -> {SPLIT_DIR}")


def build_data(force):
    root = DATA_DIR / "kitti"
    names = ensure_kitti(root)
    where = {p.stem: p for split in ("train", "val") for p in (root / "images" / split).glob("*.png")}
    ids = {s: (SPLIT_DIR / f"{s}.txt").read_text().split() for s in ("train", "val")}

    lists = root / "splits"
    lists.mkdir(exist_ok=True)
    for s, frames in ids.items():
        # absolute paths: Ultralytics resolves ./ against the list file's folder, not the dataset root
        (lists / f"scene_{s}.txt").write_text("".join(f"{where[f]}\n" for f in frames))

    with open(root / "kitti_scene.yaml", "w") as f:
        yaml.safe_dump({"path": str(root), "train": "splits/scene_train.txt", "val": "splits/scene_val.txt",
                        "names": names}, f, sort_keys=False)

    out = root / "annotations" / "scene_val_full.json"
    if out.exists() and not force and len(load_json(out)["images"]) == len(ids["val"]):
        print(f"{out} already exists ({len(ids['val'])} images), skipping conversion (--force to redo)")
        return
    images, anns, before, after = to_coco([where[f] for f in ids["val"]], names, root / "images")
    save_json({"images": images, "annotations": anns, "categories": CATEGORIES}, out)
    print(f"\n{len(images)} scene-val images ({len(ids['train'])} scene-train)")
    print_mapping(names, before, after)
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", action="store_true", help="choose the held-out drives and write splits/kitti_scene/")
    ap.add_argument("--devkit", help="KITTI object devkit mapping/ folder (needed with --make)")
    ap.add_argument("--force", action="store_true", help="rebuild the GT JSON even if it exists")
    args = ap.parse_args()
    if args.make:
        make_split(args.devkit)
    build_data(args.force)


if __name__ == "__main__":
    main()
