"""Batch-1 latency benchmark: 50 warm-up + 200 timed runs, GPU synchronized around every timing.

    python -m src.bench_latency --config configs/full/yolo26s_kitti.yaml          # FP32
    python -m src.bench_latency --config configs/full/yolo26s_kitti.yaml --half   # FP16

Every model starts from the same input: a decoded uint8 image already in memory (disk read and
decode are excluded for all models). Reported per run:
  pre      Ultralytics: letterbox + to-tensor (its own `speed` dict)
           torchvision: to-tensor + GeneralizedRCNNTransform (normalize + resize)
  infer    Ultralytics: forward pass
           torchvision: backbone + RPN + ROI heads (per-class NMS sits inside ROI heads)
  post     Ultralytics: NMS / top-k decode + scale boxes back
           torchvision: scale boxes back to the original image + copy to CPU
  e2e      wall clock around the whole call, synchronized: p50 and p95
Peak memory is torch.cuda.max_memory_allocated (CUDA only). Only CUDA numbers are valid for the
report; anything measured elsewhere is labeled debug only.
"""

import argparse
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import torch

from src.common import RESULTS_DIR, data_path, load_config, load_json, save_json, select_device, sync
from src.env_capture import capture_env
from src.predict import load_model, ultralytics_kwargs


def bench_ultralytics(model, cfg, device, frames, n_warmup, n_runs):
    kw = ultralytics_kwargs(cfg, device)
    rows = []
    for i in range(n_warmup + n_runs):
        img = frames[i % len(frames)]
        sync(device)
        t0 = time.perf_counter()
        r = model.predict(img, **kw)[0]
        sync(device)
        e2e = (time.perf_counter() - t0) * 1000
        if i >= n_warmup:
            rows.append({"pre_ms": r.speed["preprocess"], "infer_ms": r.speed["inference"],
                         "post_ms": r.speed["postprocess"], "e2e_ms": e2e})
    return rows


@torch.inference_mode()
def bench_torchvision(model, cfg, device, frames, n_warmup, n_runs):
    dtype = torch.float16 if cfg["half"] else torch.float32
    rows = []

    def timed(fn):
        sync(device)
        t = time.perf_counter()
        out = fn()
        sync(device)
        return out, (time.perf_counter() - t) * 1000

    for i in range(n_warmup + n_runs):
        rgb = frames[i % len(frames)][:, :, ::-1].copy()  # BGR -> RGB
        orig_size = [rgb.shape[:2]]
        t0 = time.perf_counter()

        def pre():
            x = torch.from_numpy(rgb).permute(2, 0, 1).to(device).to(dtype) / 255.0
            return model.transform([x])[0]

        def infer():
            feats = model.backbone(images.tensors)
            proposals, _ = model.rpn(images, feats)
            dets, _ = model.roi_heads(feats, proposals, images.image_sizes)
            return dets

        def post():
            out = model.transform.postprocess(dets, images.image_sizes, orig_size)[0]
            return {k: v.cpu() for k, v in out.items()}

        images, t_pre = timed(pre)
        dets, t_inf = timed(infer)
        _, t_post = timed(post)
        e2e = (time.perf_counter() - t0) * 1000
        if i >= n_warmup:
            rows.append({"pre_ms": t_pre, "infer_ms": t_inf, "post_ms": t_post, "e2e_ms": e2e})
    return rows


def model_cost(model, family, imgsz):
    if family == "ultralytics":
        from ultralytics.utils.torch_utils import get_flops

        net = model.model
        return sum(p.numel() for p in net.parameters()), get_flops(net, imgsz)
    return sum(p.numel() for p in model.parameters()), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--half", action="store_true", default=None, help="FP16")
    ap.add_argument("--conf", type=float, default=None, help="score threshold (default: config, 0.001)")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--n-images", type=int, default=200, help="distinct images cycled through")
    args = ap.parse_args()

    cfg = load_config(args.config, {"device": args.device, "half": args.half, "conf": args.conf})
    device = select_device(cfg["device"])
    if cfg["half"] and device == "cpu":
        raise SystemExit("FP16 is not supported for this benchmark on CPU; use --device cuda or mps")
    precision = "fp16" if cfg["half"] else "fp32"
    valid = device.startswith("cuda")

    run_id = f"{cfg['run_name']}_latency_{precision}_conf{cfg['conf']}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)  # never overwrite another run
    capture_env(run_dir, device)

    images = load_json(data_path(cfg["gt_json"]))["images"][: args.n_images]
    img_dir = data_path(cfg["images"])
    frames = [cv2.imread(str(img_dir / im["file_name"])) for im in images]  # decoded once, BGR uint8

    model, family = load_model(cfg, device)
    if valid:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    bench = bench_ultralytics if family == "ultralytics" else bench_torchvision
    print(f"[{run_id}] {cfg['model']} {precision} on {device}: {args.warmup} warm-up + {args.runs} timed runs")
    rows = bench(model, cfg, device, frames, args.warmup, args.runs)
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if valid else None

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "latency_runs.csv", index=False)
    params, gflops = model_cost(model, family, cfg["imgsz"] or 640)
    summary = {
        "run_id": run_id,
        "model": cfg["model"],
        "dataset": cfg["dataset"],
        "precision": precision,
        "conf": cfg["conf"],
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if valid else None,
        "batch": 1,
        "warmup": args.warmup,
        "runs": args.runs,
        "n_images": len(frames),
        "imgsz": cfg["imgsz"] if family == "ultralytics" else "torchvision default (min 800, max 1333)",
        "yolo26_head": cfg.get("yolo26_head"),
        "pre_ms_mean": df.pre_ms.mean(),
        "infer_ms_mean": df.infer_ms.mean(),
        "post_ms_mean": df.post_ms.mean(),
        "e2e_ms_p50": float(np.percentile(df.e2e_ms, 50)),
        "e2e_ms_p95": float(np.percentile(df.e2e_ms, 95)),
        "e2e_ms_mean": df.e2e_ms.mean(),
        "peak_mem_mb": peak_mb if valid else "not available (CUDA only)",
        "params_M": params / 1e6,
        "gflops_640": gflops,
        "valid_for_report": valid,
        "note": None if valid else "DEBUG ONLY, NOT FOR REPORT (measured off-CUDA)",
    }
    save_json(summary, run_dir / "latency.json")
    summary_csv = RESULTS_DIR / "latency_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, mode="a", header=not summary_csv.exists(), index=False)

    if not valid:
        print("*** DEBUG ONLY, NOT FOR REPORT ***")
    print(f"pre {summary['pre_ms_mean']:.2f}  infer {summary['infer_ms_mean']:.2f}  post {summary['post_ms_mean']:.2f} ms"
          f" | e2e p50 {summary['e2e_ms_p50']:.2f}  p95 {summary['e2e_ms_p95']:.2f} ms | peak mem {summary['peak_mem_mb']}"
          f" -> {run_dir}")


if __name__ == "__main__":
    main()
