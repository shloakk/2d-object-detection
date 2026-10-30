# Results branch

Run outputs pushed from Colab by `src/sync_results.py`, one folder per run_id, plus
`summary.csv` (accuracy) and `latency_summary.csv`. Code lives on `main`.

Detections are stored gzipped (`detections.json.gz`); `src.evaluate` and `src.viz` read them directly.
