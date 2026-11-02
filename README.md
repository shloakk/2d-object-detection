# 2d-object-detection: CMPE 249, COCO → KITTI zero-shot

Three COCO-pretrained detector families on COCO val2017 and on the KITTI val split, with no
retraining. Every run is scored by one evaluator, pycocotools COCOeval.

| Model | Library / weights | Family |
| --- | --- | --- |
| YOLO26s | Ultralytics `yolo26s.pt` | one-stage CNN, NMS-free one-to-one head |
| RT-DETR-L | Ultralytics `rtdetr-l.pt` | query-based transformer |
| Faster R-CNN R50-FPN v2 | torchvision `fasterrcnn_resnet50_fpn_v2`, `COCO_V1` | two-stage, anchor-based |

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Paths can be moved with env vars: `CMPE249_DATA` (default `./data`), `CMPE249_RESULTS`
(default `./results`) and `CMPE249_WEIGHTS` (default `./weights`). The device is picked
automatically (cuda, then mps, then cpu); override it with `--device`.

## One command per step

```bash
# 1. Data
python -m src.data.coco_subset                 # 50-image COCO val subset (debug)
python -m src.data.coco_subset --full          # all 5000 (Colab)
python -m src.data.kitti_to_coco --subset 50   # 50-image KITTI val subset (debug)
python -m src.data.kitti_to_coco               # full KITTI val, 1496 images (Colab)

# 2. Predict + evaluate (writes results/<run_id>/ and appends to results/summary.csv)
python -m src.predict --config configs/debug/yolo26s_coco.yaml     # configs/full/... on Colab

# 3. Evaluate standalone
python -m src.evaluate --gt data/kitti/annotations/val_subset50.json \
    --dt results/<run_id>/detections.json --classes car,person,truck

# 4. Latency (batch 1, 50 warm-up + 200 timed runs; only CUDA numbers are valid)
python -m src.bench_latency --config configs/full/yolo26s_kitti.yaml [--half] [--conf 0.25]

# 5. Visualize: GT (green, dashed = ignore) vs predictions (red)
python -m src.viz --gt data/kitti/annotations/val_subset50.json --dt results/<run_id>/detections.json \
    --images data/kitti/images/val --classes car,person,truck --worst 12 --out results/viz/<run_id>

# 6. Pipeline check against Ultralytics' own val()
python -m src.check_ultralytics_val --config configs/debug/yolo26s_coco.yaml --dt results/<run_id>/detections.json

# 7. Report: report/all_runs.csv + per-class, AP-by-size and Pareto figures from colab-results/
python -m src.report                     # --subset subset50 for debug runs
python -m src.failure_grid               # report/fig4_failure_grid.png (hand-picked KITTI val frames)
```

## Stage 3a: KITTI fine-tune + TensorRT FP16 (Colab, CUDA)

Ultralytics' `kitti.yaml` splits KITTI's frames at random, so consecutive frames of one drive land in
both train and val: about half of its val frames have a near-identical train frame. A fine-tune scored on
it reaches 72.0 mAP mostly on memorised scenes. Stage 3a therefore uses a **scene-disjoint split**: 20 of
the 141 raw drives (1,559 frames) are held out as val, the other 5,922 frames are train. The drive of each
frame comes from the KITTI object devkit; the chosen frame ids are committed in `splits/kitti_scene/`.

```bash
python -m src.data.kitti_scene_split                                       # lists, dataset yaml, scene_val GT
python -m src.finetune --split scene                                       # 50 epochs, imgsz 640, seed 0
python -m src.predict --config configs/full/yolo26s_scene_kitti.yaml          # zero-shot on scene val
python -m src.predict --config configs/full/yolo26s_ft_scene_kitti.yaml       # fine-tuned, rect 640
python -m src.predict --config configs/full/yolo26s_ft_scene_sq_kitti.yaml    # fine-tuned, square 640 (engine reference)
pip install "nvidia-modelopt[onnx]>=0.44"                                  # TensorRT 11 FP16 export needs it
python -m src.export_trt                                                   # -> weights/yolo26s_kitti_scene_ft.engine
python -m src.predict --config configs/full/yolo26s_ft_scene_trt_kitti.yaml
python -m src.bench_latency --config configs/full/yolo26s_ft_scene_trt_kitti.yaml
python -m src.bench_latency --config configs/full/yolo26s_ft_scene_sq_kitti.yaml [--half]
```

- The fine-tune keeps `last.pt`, not `best.pt`: Ultralytics picks `best.pt` on the val split, which is our test set.
- The fine-tuned model predicts the 8 KITTI classes (`class_map: kitti`). Only car, pedestrian and truck are
  kept, mapped to COCO ids 3, 1 and 8. Van, person_sitting and cyclist are ignore regions in the GT, so their
  predictions are dropped.
- `configs/full/yolo26s_ft_kitti.yaml` (Ultralytics split, `--split ultralytics`) is kept only as the leakage reference.
- `python -m src.report` adds `fig5_finetune.png` (zero-shot vs fine-tuned on scene val) and `stage3a_deploy.csv`
  (PyTorch FP32/FP16 vs TensorRT FP16, all on the same square 640 input).

## Colab results (no Google Drive)

On Colab, `CMPE249_RESULTS` points at a clone of the `results` branch, and
`python -m src.sync_results` commits and pushes after each run (detections are stored as
`detections.json.gz`; `evaluate` and `viz` read them directly). To get them on your laptop:

```bash
git fetch origin results && git worktree add colab-results results   # first time
git -C colab-results pull                                             # afterwards
```

Each run folder holds `detections.json` (COCO format), `run_meta.json`, `metrics.csv`,
`per_class.csv` and `env/` (pip freeze, nvidia-smi, versions, git commit, exact command).

## Protocol

- conf=0.001, max_det=300 and batch 1 for every model. COCOeval uses the standard maxDets=100.
- NMS IoU: Ultralytics' default of 0.7 (used only by YOLO26's one2many head). Faster R-CNN uses
  torchvision's default `box_nms_thresh=0.5`. RT-DETR has no NMS.
- **YOLO26 head.** In Ultralytics 8.4.193 the library default (`nms=None`) runs the one-to-many
  head plus NMS. The configs use `yolo26_head: one2one`, which passes `nms=False` and selects the
  NMS-free end-to-end head. The head actually used is read back from the model into `run_meta.json`.
- Input size: Ultralytics models use imgsz 640. YOLO26 letterboxes to the image's aspect ratio
  (`rect: true`). RT-DETR always resizes to a 640×640 square. Faster R-CNN uses torchvision's own
  resize (shorter side 800, longer side at most 1333).
- Class ids: Ultralytics outputs 0–79, which are mapped to COCO ids 1–90 with `coco80_to_coco91_class()`.
  torchvision already outputs COCO ids.
- KITTI → COCO: car→car, pedestrian→person and truck→truck are scored. Van→car,
  person_sitting→person and cyclist→person become `iscrowd=1` ignore regions (a detected rider
  is not a false positive). Tram and misc are dropped.
- Extra YOLO26s configs: `yolo26s_o2m_*` (one-to-many head + NMS, `nms=True`) and
  `yolo26s_1280_kitti` (NMS-free at imgsz 1280, to separate resolution from architecture).
- `run_meta.json` records each model's effective input: the tensor size the network sees and how
  much objects are scaled on each axis (e.g. KITTI: YOLO26s 224×640 at 0.52×, RT-DETR 640×640
  stretched 0.52× wide / 1.71× tall, Faster R-CNN 416×1344 at 1.07×).
  KITTI is scored on `car, person, truck` only.
