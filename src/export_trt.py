"""Stage 3a: export the KITTI fine-tuned YOLO26s to a TensorRT FP16 engine (CUDA only).

    python -m src.export_trt          # weights/yolo26s_kitti_scene_ft.pt -> weights/yolo26s_kitti_scene_ft.engine

Static 1x3x640x640 input, NMS-free one-to-one head (nms=False), as the PyTorch runs use. A static
engine always letterboxes to the 640x640 square, so its PyTorch reference is the square config
(configs/full/yolo26s_ft_scene_sq_kitti.yaml), not the rect one. The engine is tied to this GPU and
TensorRT version, so only its metadata goes to results/; rebuild it after a runtime reset.

Then score and time it like any other run:
    python -m src.predict --config configs/full/yolo26s_ft_scene_trt_kitti.yaml
    python -m src.bench_latency --config configs/full/yolo26s_ft_scene_trt_kitti.yaml
"""

import argparse
from datetime import datetime
from pathlib import Path

from src.common import RESULTS_DIR, WEIGHTS_DIR, save_json, select_device
from src.env_capture import capture_env
from src.finetune import FT_WEIGHTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=FT_WEIGHTS["scene"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workspace", type=float, default=8, help="TensorRT builder workspace (GiB)")
    args = ap.parse_args()

    from ultralytics import YOLO

    device = select_device("auto")
    if not device.startswith("cuda"):
        raise SystemExit("TensorRT export needs a CUDA GPU")
    run_id = f"yolo26s_ft_trt_fp16_export_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    capture_env(run_dir, device)

    pt = WEIGHTS_DIR / args.weights
    engine = Path(YOLO(str(pt)).export(format="engine", quantize=16, imgsz=args.imgsz, batch=1, dynamic=False,
                                       nms=False, workspace=args.workspace, device=0))
    import tensorrt as trt  # installed by the export above when missing

    meta = {
        "run_id": run_id,
        "source_weights": str(pt),
        "engine": str(engine),
        "precision": "fp16",
        "input": f"1x3x{args.imgsz}x{args.imgsz} (static)",
        "head": "one2one (NMS-free), nms=False",
        "tensorrt": trt.__version__,
        "pt_size_mb": round(pt.stat().st_size / 2**20, 2),
        "engine_size_mb": round(engine.stat().st_size / 2**20, 2),
    }
    save_json(meta, run_dir / "export_meta.json")
    print(f"{engine} ({meta['engine_size_mb']} MB, from {meta['pt_size_mb']} MB .pt), TensorRT {trt.__version__}")


if __name__ == "__main__":
    main()
