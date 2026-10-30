"""Commit and push the results folder to the repo's `results` branch (replaces Google Drive).

On Colab, CMPE249_RESULTS points at a clone of the `results` branch. After each run:

    python -m src.sync_results

gzips every detections.json to detections.json.gz (raw JSON stays local and git-ignored, since
full-COCO files can pass 100 MB), then commits and pushes. Pull results on your laptop with:

    git fetch origin results && git worktree add colab-results results     # first time
    git -C colab-results pull                                               # afterwards
"""

import gzip
import shutil
import subprocess
from pathlib import Path

from src.common import RESULTS_DIR


def git(*args):
    return subprocess.run(["git", "-C", str(RESULTS_DIR), *args], check=True, capture_output=True, text=True).stdout


def main():
    if not (RESULTS_DIR / ".git").exists():
        raise SystemExit(f"{RESULTS_DIR} is not a git clone of the results branch")

    for raw in RESULTS_DIR.rglob("detections.json"):
        gz = raw.with_suffix(".json.gz")
        if not gz.exists() or gz.stat().st_mtime < raw.stat().st_mtime:
            with open(raw, "rb") as src, gzip.open(gz, "wb") as dst:
                shutil.copyfileobj(src, dst)

    git("add", "-A")
    if not git("status", "--porcelain").strip():
        print("nothing new to sync")
        return
    changed = len(git("diff", "--cached", "--name-only").split())
    git("commit", "-q", "-m", f"Results sync ({changed} files)")
    git("pull", "-q", "--rebase")
    git("push", "-q")
    print(f"pushed {changed} files to {git('remote', 'get-url', 'origin').strip().split('@')[-1]} (results branch)")


if __name__ == "__main__":
    main()
