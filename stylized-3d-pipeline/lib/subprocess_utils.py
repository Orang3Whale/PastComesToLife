from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence


def run_checked(cmd: Sequence[str], env: dict[str, str] | None = None) -> None:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(cmd, check=True, env=merged_env)
