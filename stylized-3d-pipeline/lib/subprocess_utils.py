from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from pathlib import Path


DEFAULT_HF_CACHE_ROOT = Path("/root/autodl-tmp/hf-cache")


def huggingface_cache_env(cache_root: Path = DEFAULT_HF_CACHE_ROOT) -> dict[str, str]:
    root = Path(cache_root)
    return {
        "HF_HOME": str(root),
        "HUGGINGFACE_HUB_CACHE": str(root / "hub"),
        "TRANSFORMERS_CACHE": str(root / "transformers"),
        "DIFFUSERS_CACHE": str(root / "hub"),
    }


def run_checked(cmd: Sequence[str], env: dict[str, str] | None = None) -> None:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(cmd, check=True, env=merged_env)
