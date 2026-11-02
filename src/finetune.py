"""Stage 3a: fine-tune COCO-pretrained YOLO26s on KITTI train (5,985 images, all 8 KITTI classes).

    python -m src.finetune --split scene                      # Colab: 50 epochs, imgsz 640, seed 0
    python -m src.finetune --split scene --epochs 1 --fraction 0.02 --no-cache --batch 8   # debug on the Mac

--split scene (the one to report) trains on the scene-disjoint split from src.data.kitti_scene_split:
whole drives held out, so no val scene is seen in training. --split ultralytics uses kitti.yaml's
random frame split, where about half of the val frames have a near-identical train frame; its
scores are inflated and kept only to show that.

Writes results/<run_id>/ (train curves, args.yaml, env/, the final weights) and copies the final
weights to weights/yolo26s_kitti_scene_ft.pt (or yolo26s_kitti_ft.pt for --split ultralytics),
which configs/*/yolo26s_ft*_kitti.yaml load.

Model selection: Ultralytics validates on the split's val every epoch, and that is also our test set, so
picking best.pt would tune on the test images. We keep last.pt (fixed 50-epoch schedule) instead;
the per-epoch val curve is only plotted.
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import ultralytics
import yaml

from src.common import DATA_DIR, RESULTS_DIR, WEIGHTS_DIR, save_json, select_device
from src.env_capture import capture_env

FT_WEIGHTS = {"scene": "yolo26s_kitti_scene_ft.pt", "ultralytics": "yolo26s_kitti_ft.pt"}
# Small files from the Ultralytics run folder worth keeping next to the run (weights are copied separately).
KEEP = ["args.yaml", "results.csv", "results.png", "confusion_matrix.png", "confusion_matrix_normalized.png",
        "BoxPR_curve.png", "labels.jpg"]


def dataset_yaml(split):
    if split == "scene":
        out = DATA_DIR / "kitti" / "kitti_scene.yaml"
        if not out.exists():
            raise SystemExit(f"{out} missing: run python -m src.data.kitti_scene_split first")
        return out
    return ultralytics_yaml()


def ultralytics_yaml():
    """Ultralytics' kitti.yaml with `path` pointed at our copy, so nothing is downloaded twice."""
    src = Path(ultralytics.__file__).parent / "cfg" / "datasets" / "kitti.yaml"
    cfg = yaml.safe_load(src.read_text())
    cfg["path"] = str(DATA_DIR / "kitti")
    cfg.pop("download", None)
    out = DATA_DIR / "kitti" / "kitti_train.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["scene", "ultralytics"], default="scene")
    ap.add_argument("--weights", default="yolo26s.pt", help="COCO-pretrained starting point")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fraction", type=float, default=1.0, help="share of KITTI train to use (debug)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-cache", action="store_true", help="do not cache decoded images in RAM")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO

    device = select_device(args.device or "auto")
    tag = "kitti_scene" if args.split == "scene" else "kitti"
    out_weights = WEIGHTS_DIR / FT_WEIGHTS[args.split]
    run_id = f"yolo26s_{tag}_finetune_e{args.epochs}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)  # never overwrite another run
    capture_env(run_dir, device)

    data = dataset_yaml(args.split)
    model = YOLO(str(WEIGHTS_DIR / args.weights))  # auto-downloads if missing
    print(f"[{run_id}] fine-tuning {args.weights} on {data} for {args.epochs} epochs, device={device}")
    model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, seed=args.seed,
                deterministic=True, fraction=args.fraction, workers=args.workers,
                cache=False if args.no_cache else "ram", device=device,
                project=str(WEIGHTS_DIR / "train"), name=run_id, exist_ok=False)

    train_dir = Path(model.trainer.save_dir)
    for name in KEEP:
        if (train_dir / name).exists():
            shutil.copy2(train_dir / name, run_dir / name)
    last = train_dir / "weights" / "last.pt"
    shutil.copy2(last, run_dir / "last.pt")  # pushed with the results so a runtime reset does not lose it
    shutil.copy2(last, out_weights)

    meta = {
        "run_id": run_id,
        "start_weights": args.weights,
        "split": args.split,
        "dataset": f"{data} (8 KITTI classes)",
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "fraction": args.fraction,
        "cache": not args.no_cache,
        "device": device,
        "kept_weights": "last.pt (fixed schedule; best.pt would select on KITTI val = our test set)",
        "train_dir": str(train_dir),
        "output_weights": str(out_weights),
    }
    save_json(meta, run_dir / "finetune_meta.json")
    print(f"final weights -> {out_weights} (copy in {run_dir})")


if __name__ == "__main__":
    main()
