"""Merge Colab results into one CSV and draw the Stage 2 report figures.

Reads summary.csv and latency_summary.csv from the results-branch checkout
(default: ./colab-results) and writes to ./report:
  all_runs.csv              one row per accuracy run, with KITTI latency joined on
  fig1_per_class_ap.png     car / person / truck AP, COCO vs KITTI, per model
  fig2_ap_by_size.png       AP small / medium / large on both datasets
  fig3_pareto.png           KITTI mAP vs end-to-end latency, point size = peak memory (zero-shot models)
  fig5_finetune.png         Stage 3a: zero-shot vs KITTI fine-tuned YOLO26s on the scene-disjoint val
  stage3a_deploy.csv        Stage 3a: fine-tuned YOLO26s, PyTorch FP32/FP16 vs TensorRT FP16 (square 640)

When a config was run more than once, the latest run (by timestamp suffix) wins.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.common import REPO_ROOT

# Display order and labels. Color follows the model family, never its rank.
MODELS = {
    "yolo26s": "YOLO26s (NMS-free)",
    "yolo26s_o2m": "YOLO26s o2m + NMS",
    "yolo26s_1280": "YOLO26s @1280",
    "rtdetr_l": "RT-DETR-L",
    "frcnn_v2": "Faster R-CNN v2",
}
FAMILY_COLOR = {"yolo26s": "#2a78d6", "rtdetr_l": "#eb6834", "frcnn_v2": "#1baf7a"}
YOLO_MARKER = {"yolo26s": "o", "yolo26s_o2m": "s", "yolo26s_1280": "D"}
DATASET_COLOR = {"coco": "#2a78d6", "kitti": "#eb6834"}
SIZE_COLOR = {"small": "#a9c8ef", "medium": "#5b97e0", "large": "#1f5aa6"}  # one hue, light -> dark
CLASSES = ["car", "person", "truck"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def family(key):
    return "yolo26s" if key.startswith("yolo26s") else key


def config_of(run_id, subset):
    """'yolo26s_o2m_kitti_full_2026...' -> 'yolo26s_o2m_kitti'."""
    return run_id.split(f"_{subset}_")[0]


def latest(df, subset):
    df = df[df["run_id"].str.contains(f"_{subset}_", regex=False)].copy()
    df["config"] = df["run_id"].map(lambda r: config_of(r, subset))
    return df.sort_values("run_id").groupby("config", as_index=False).last()


def load(results, subset):
    acc = latest(pd.read_csv(results / "summary.csv"), subset)
    acc["model_key"] = acc["config"].str.rsplit("_", n=1).str[0]

    # Latency runs are only valid on CUDA; keep the latest per (config, precision).
    lat_all = pd.read_csv(results / "latency_summary.csv")
    lat_all = lat_all[(lat_all["valid_for_report"].astype(str) == "True")
                      & lat_all["run_id"].str.contains(f"_{subset}_latency_", regex=False)].copy()
    lat_all["config"] = lat_all["run_id"].str.split(f"_{subset}_latency_").str[0]
    lat_all = lat_all.sort_values("run_id").groupby(["config", "precision"], as_index=False).last()

    wide = lat_all.pivot(index="config", columns="precision",
                         values=["e2e_ms_p50", "e2e_ms_p95", "pre_ms_mean", "infer_ms_mean",
                                 "post_ms_mean", "peak_mem_mb"])
    wide.columns = [f"{m}_{p}" for m, p in wide.columns]
    static = lat_all.groupby("config")[["gpu", "params_M", "gflops_640"]].first()
    lat_wide = wide.join(static).reset_index()

    return acc.merge(lat_wide, on="config", how="left")


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def label_bars(ax, bars):
    for b in bars:
        ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                    fontsize=7, color=MUTED)


def fig_per_class(df, out):
    keys = [k for k in MODELS if {"coco", "kitti"} <= set(df.loc[df.model_key == k, "dataset"])]
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(12, 3.8), sharey=True)
    width = 0.38
    for ax, cls in zip(axes, CLASSES):
        for j, ds in enumerate(["coco", "kitti"]):
            vals = [100 * df.loc[(df.model_key == k) & (df.dataset == ds), f"AP_{cls}"].iloc[0] for k in keys]
            xs = [i + (j - 0.5) * (width + 0.02) for i in range(len(keys))]
            bars = ax.bar(xs, vals, width, color=DATASET_COLOR[ds], label=ds.upper(),
                          edgecolor="white", linewidth=1)
            label_bars(ax, bars)
        ax.set_title(cls, color=INK)
        ax.set_xticks(range(len(keys)), [MODELS[k] for k in keys], rotation=20, ha="right", fontsize=8)
        style(ax)
    axes[0].set_ylabel("AP@[.5:.95] (%)", color=INK)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=2, frameon=False, fontsize=8)
    fig.suptitle("Per-class AP, COCO val2017 vs KITTI val (zero-shot)", color=INK)
    fig.tight_layout()
    fig.savefig(out / "fig1_per_class_ap.png", dpi=200)
    plt.close(fig)


def fig_by_size(df, out):
    keys = [k for k in MODELS if {"coco", "kitti"} <= set(df.loc[df.model_key == k, "dataset"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    width = 0.27
    for ax, ds in zip(axes, ["coco", "kitti"]):
        for j, size in enumerate(SIZE_COLOR):
            vals = [100 * df.loc[(df.model_key == k) & (df.dataset == ds), f"AP_{size}"].iloc[0] for k in keys]
            xs = [i + (j - 1) * (width + 0.02) for i in range(len(keys))]
            bars = ax.bar(xs, vals, width, color=SIZE_COLOR[size], label=size,
                          edgecolor="white", linewidth=1)
            label_bars(ax, bars)
        ax.set_title(f"{ds.upper()} val", color=INK)
        ax.set_xticks(range(len(keys)), [MODELS[k] for k in keys], rotation=20, ha="right", fontsize=8)
        style(ax)
    axes[0].set_ylabel("AP@[.5:.95] (%)", color=INK)
    fig.legend(*axes[0].get_legend_handles_labels(), title="object size", loc="upper right", ncol=3,
               frameon=False, fontsize=8, title_fontsize=8)
    fig.suptitle("AP by object size (COCO area ranges)", color=INK)
    fig.tight_layout()
    fig.savefig(out / "fig2_ap_by_size.png", dpi=200)
    plt.close(fig)


def fig_pareto(df, out):
    k = df[(df.dataset == "kitti") & df["e2e_ms_p50_fp16"].notna() & df.model_key.isin(MODELS)]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for _, r in k.iterrows():
        key, color = r.model_key, FAMILY_COLOR[family(r.model_key)]
        marker = YOLO_MARKER.get(key, "o")
        y = 100 * r.mAP50_95
        # FP32 as a hollow marker joined to the FP16 point, so the precision effect is visible
        ax.plot([r.e2e_ms_p50_fp32, r.e2e_ms_p50_fp16], [y, y], color=color, linewidth=1, alpha=0.6, zorder=1)
        ax.scatter(r.e2e_ms_p50_fp32, y, s=60, marker=marker, facecolors="white", edgecolors=color,
                   linewidths=1.5, zorder=4)
        ax.scatter(r.e2e_ms_p50_fp16, y, s=20 + 1.2 * r.peak_mem_mb_fp16, marker=marker, color=color,
                   edgecolors="white", linewidths=2, zorder=3)
        ax.annotate(MODELS[key], (max(r.e2e_ms_p50_fp16, r.e2e_ms_p50_fp32), y), xytext=(10, -3),
                    textcoords="offset points", fontsize=8, color=INK)
    style(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_xlabel("End-to-end latency, p50, batch 1 (ms)", color=INK)
    ax.set_ylabel("KITTI mAP@[.5:.95] (%)", color=INK)
    gpu = k["gpu"].dropna().iloc[0] if k["gpu"].notna().any() else "GPU"
    ax.set_title(f"Speed vs accuracy on {gpu}\nfilled = FP16 (size = peak memory), hollow = FP32",
                 color=INK, fontsize=10)
    xmax = k[["e2e_ms_p50_fp16", "e2e_ms_p50_fp32"]].max().max()
    ax.set_xlim(0, xmax * 1.35)
    fig.tight_layout()
    fig.savefig(out / "fig3_pareto.png", dpi=200)
    plt.close(fig)


# Stage 3a: the fine-tuned model next to its zero-shot starting point, both on the scene-disjoint
# KITTI val (whole drives held out of training), same imgsz 640, rect, NMS-free head
FT_PAIR = {"yolo26s_scene": "YOLO26s zero-shot (COCO)", "yolo26s_ft_scene": "YOLO26s fine-tuned on KITTI"}
FT_COLOR = {"yolo26s_scene": "#a9c8ef", "yolo26s_ft_scene": "#1f5aa6"}
# Deployment rows: all use the square 640x640 input the static TensorRT engine needs
DEPLOY = [("yolo26s_ft_scene_sq", "fp32", "PyTorch FP32"), ("yolo26s_ft_scene_sq", "fp16", "PyTorch FP16"),
          ("yolo26s_ft_scene_trt16", "fp16", "TensorRT FP16")]


def fig_finetune(df, out):
    k = df[df.dataset == "kitti"].set_index("model_key")
    if not set(FT_PAIR) <= set(k.index):
        print("fig5 skipped: no fine-tuned KITTI run yet")
        return
    metrics = ["mAP50_95", "AP50", *[f"AP_{c}" for c in CLASSES], "AP_small", "AP_medium", "AP_large"]
    labels = ["mAP", "AP50", *CLASSES, "small", "medium", "large"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    width = 0.38
    top = 0
    for j, key in enumerate(FT_PAIR):
        vals = [100 * k.loc[key, m] for m in metrics]
        top = max(top, *vals)
        xs = [i + (j - 0.5) * (width + 0.02) for i in range(len(metrics))]
        label_bars(ax, ax.bar(xs, vals, width, color=FT_COLOR[key], label=FT_PAIR[key], edgecolor="white", linewidth=1))
    ax.set_xticks(range(len(metrics)), labels, fontsize=9)
    ax.axvline(1.5, color=GRID, linewidth=1)
    ax.axvline(4.5, color=GRID, linewidth=1)
    style(ax)
    ax.set_ylabel("AP@[.5:.95] (%)  (AP50 at IoU .5)", color=INK)
    ax.set_ylim(0, top * 1.25)
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    ax.set_title("Scene-disjoint KITTI val (20 held-out drives): zero-shot vs fine-tuned YOLO26s, imgsz 640",
                 color=INK)
    fig.tight_layout()
    fig.savefig(out / "fig5_finetune.png", dpi=200)
    plt.close(fig)


def deploy_table(df, out):
    k = df[df.dataset == "kitti"].set_index("model_key")
    rows = []
    for key, prec, label in DEPLOY:
        if key not in k.index or pd.isna(k.loc[key].get(f"e2e_ms_p50_{prec}")):
            continue
        r = k.loc[key]
        rows.append({"variant": label, "mAP50_95": r.mAP50_95, "AP50": r.AP50, "AP_small": r.AP_small,
                     "e2e_ms_p50": r[f"e2e_ms_p50_{prec}"], "e2e_ms_p95": r[f"e2e_ms_p95_{prec}"],
                     "infer_ms_mean": r[f"infer_ms_mean_{prec}"], "gpu": r.gpu})
    if not rows:
        print("stage3a_deploy.csv skipped: no fine-tuned latency runs yet")
        return
    t = pd.DataFrame(rows)
    # mAP of PyTorch FP16 is not measured separately (predict runs are FP32), so the deltas use FP32
    t["mAP_delta_vs_fp32"] = t.mAP50_95 - t.mAP50_95.iloc[0]
    t["speedup_vs_fp32"] = t.e2e_ms_p50.iloc[0] / t.e2e_ms_p50
    t.to_csv(out / "stage3a_deploy.csv", index=False)
    print(t.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(REPO_ROOT / "colab-results"), help="results-branch checkout")
    ap.add_argument("--subset", default="full", help="'full' or e.g. 'subset50' for debug runs")
    ap.add_argument("--out", default=str(REPO_ROOT / "report"))
    args = ap.parse_args()

    results, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load(results, args.subset)
    df.to_csv(out / "all_runs.csv", index=False)
    fig_per_class(df, out)
    fig_by_size(df, out)
    fig_pareto(df, out)
    fig_finetune(df, out)
    deploy_table(df, out)

    cols = ["config", "mAP50_95", "AP50", "AP_small", "e2e_ms_p50_fp32", "e2e_ms_p50_fp16", "peak_mem_mb_fp16"]
    print(df[cols].round(4).to_string(index=False))
    print(f"\n{len(df)} accuracy runs, {df['e2e_ms_p50_fp16'].notna().sum()} with latency -> {out}")


if __name__ == "__main__":
    main()
