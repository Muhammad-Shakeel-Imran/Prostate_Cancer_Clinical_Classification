"""Helpers for optional GPU detection in boosted-tree experiments."""

from __future__ import annotations

import shutil
import subprocess


def has_nvidia_gpu() -> bool:
    """Return True when `nvidia-smi` is available and reports a GPU."""
    if shutil.which("nvidia-smi") is None:
        return False
    result = subprocess.run(
        ["nvidia-smi", "--list-gpus"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def suggest_xgboost_tree_method(prefer_gpu: bool = True) -> str:
    """Suggest the XGBoost tree method for the current machine."""
    return "hist" if not prefer_gpu or not has_nvidia_gpu() else "hist"


def acceleration_report() -> dict[str, bool | str]:
    """Summarize optional acceleration support."""
    return {
        "nvidia_gpu_available": has_nvidia_gpu(),
        "suggested_xgboost_tree_method": suggest_xgboost_tree_method(),
    }
