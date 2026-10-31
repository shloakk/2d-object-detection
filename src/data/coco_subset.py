"""Build a COCO val2017 subset: annotation JSON + only the images it needs.

Downloads annotations_trainval2017.zip (not the image zip), samples N val images with a fixed
seed, fetches each image from its `coco_url`, and writes a filtered annotation JSON.

    python -m src.data.coco_subset            # 50 images -> coco/annotations/val2017_subset50.json
    python -m src.data.coco_subset --full     # all 5000  -> coco/annotations/val2017_full.json

Safe to re-run: if the subset JSON and every image it lists are already on disk, nothing is
downloaded; otherwise only missing images are fetched (use --force to rebuild the JSON).
"""

import argparse
import random
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from src.common import DATA_DIR, load_json, save_json

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def download(url, dest, retries=3):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dest)
            return dest
        except Exception as e:  # network hiccup: retry
            if attempt == retries - 1:
                raise RuntimeError(f"failed to download {url}: {e}")


def get_val_annotations(coco_dir):
    """Return path to instances_val2017.json, downloading/extracting it if needed."""
    ann_json = coco_dir / "annotations" / "instances_val2017.json"
    if not ann_json.exists():
        zip_path = coco_dir / "annotations_trainval2017.zip"
        if not zip_path.exists():
            print(f"Downloading {ANN_URL} ...")
        download(ANN_URL, zip_path)  # no-op if the zip is already complete
        # Extract to a temp name first so an interrupted extract never leaves a truncated JSON.
        tmp = ann_json.with_suffix(".json.part")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z, open(tmp, "wb") as f:
            f.write(z.read("annotations/instances_val2017.json"))
        tmp.rename(ann_json)
    return ann_json


def subset_complete(out_json, img_dir):
    """True when the subset JSON exists and every image it lists is on disk."""
    if not out_json.exists():
        return False
    images = load_json(out_json)["images"]
    return all((img_dir / im["file_name"]).is_file() and (img_dir / im["file_name"]).stat().st_size > 0
               for im in images)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="number of val images to sample")
    ap.add_argument("--full", action="store_true", help="use all 5000 val2017 images")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16, help="parallel image downloads")
    ap.add_argument("--force", action="store_true", help="rebuild the subset JSON even if it exists")
    args = ap.parse_args()

    coco_dir = DATA_DIR / "coco"
    tag = "full" if args.full else f"subset{args.n}"
    out_json = coco_dir / "annotations" / f"val2017_{tag}.json"
    img_dir = coco_dir / "val2017"
    if not args.force and subset_complete(out_json, img_dir):
        n = len(load_json(out_json)["images"])
        print(f"{out_json} and all {n} images already on disk, skipping (--force to rebuild)")
        return

    full = load_json(get_val_annotations(coco_dir))

    images = sorted(full["images"], key=lambda im: im["id"])
    if not args.full:
        images = sorted(random.Random(args.seed).sample(images, args.n), key=lambda im: im["id"])
    keep = {im["id"] for im in images}
    anns = [a for a in full["annotations"] if a["image_id"] in keep]

    save_json(
        {"info": full.get("info", {}), "licenses": full.get("licenses", []), "images": images,
         "annotations": anns, "categories": full["categories"]},
        out_json,
    )

    jobs = [(im["coco_url"], img_dir / im["file_name"]) for im in images]
    with ThreadPoolExecutor(args.workers) as pool:
        list(tqdm(pool.map(lambda j: download(*j), jobs), total=len(jobs), desc="images"))

    print(f"{len(images)} images, {len(anns)} annotations -> {out_json}")
    print(f"images in {img_dir}")


if __name__ == "__main__":
    main()
