from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.io_paths import create_run_tree, resolve_run_dir
from lib.pipeline_runner import ordered_steps, run_pipeline, write_run_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--style-image", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--runs-root", default=Path("runs"), type=Path)
    parser.add_argument("--sf3d-python", required=True, type=Path)
    parser.add_argument("--instantstyle-python", required=True, type=Path)
    parser.add_argument("--foreground-ratio", default=0.85, type=float)
    parser.add_argument("--texture-resolution", default=1024, type=int)
    parser.add_argument("--view-resolution", default=512, type=int)
    parser.add_argument("--camera-distance", default=1.8, type=float)
    parser.add_argument("--camera-fovy-deg", default=40.0, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--remesh-option",
        default="none",
        choices=["none", "triangle", "quad"],
    )
    parser.add_argument(
        "--resume-from",
        choices=ordered_steps(),
    )
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.runs_root, args.run_name)
    args.run_dir = run_dir
    create_run_tree(run_dir)
    write_run_config(run_dir, args)
    run_pipeline(args)


if __name__ == "__main__":
    main()
