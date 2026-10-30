"""Write an environment snapshot into a run folder (called by predict.py and bench_latency.py)."""

import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from src.common import REPO_ROOT


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def capture_env(run_dir, device=None):
    import torch
    import torchvision
    import ultralytics

    env_dir = Path(run_dir) / "env"
    env_dir.mkdir(parents=True, exist_ok=True)

    (env_dir / "pip_freeze.txt").write_text(_run([sys.executable, "-m", "pip", "freeze"]) or "")
    smi = _run(["nvidia-smi"])
    (env_dir / "nvidia_smi.txt").write_text(smi or "nvidia-smi not available\n")

    commit = _run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    dirty = _run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"])
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(shlex.quote(a) for a in [sys.executable] + sys.argv),
        "git_commit": commit.strip() if commit else None,
        "git_dirty": bool(dirty and dirty.strip()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mps_available": torch.backends.mps.is_available(),
        "device": str(device) if device else None,
    }
    (env_dir / "env.json").write_text(json.dumps(info, indent=2))
    (env_dir / "command.txt").write_text(info["command"] + "\n")
    return info


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "results/env_check"
    print(json.dumps(capture_env(out), indent=2))
