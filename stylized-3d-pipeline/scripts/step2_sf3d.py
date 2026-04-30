from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from lib.io_paths import create_run_tree, write_json
from lib.subprocess_utils import run_checked


def build_sf3d_command(
    sf3d_python: Path,
    worker_script: Path,
    run_dir: Path,
    texture_resolution: int,
    remesh_option: str,
) -> list[str]:
    return [
        str(sf3d_python),
        str(worker_script),
        "--run-dir",
        str(run_dir),
        "--texture-resolution",
        str(texture_resolution),
        "--remesh-option",
        remesh_option,
    ]


def run_step(
    run_dir: Path,
    sf3d_python: Path,
    texture_resolution: int = 1024,
    remesh_option: str = "none",
    runner: Callable[..., None] = run_checked,
) -> dict:
    paths = create_run_tree(run_dir)
    worker_script = Path(__file__).resolve().parent / "workers" / "sf3d_worker.py"
    cmd = build_sf3d_command(
        sf3d_python=sf3d_python,
        worker_script=worker_script,
        run_dir=paths.root,
        texture_resolution=texture_resolution,
        remesh_option=remesh_option,
    )
    runner(cmd, env={"HF_ENDPOINT": "https://hf-mirror.com"})

    mesh_path = paths.sf3d / "mesh_raw.glb"
    if not mesh_path.is_file():
        raise FileNotFoundError(f"missing SF3D output: {mesh_path}")

    result = {
        "mesh_path": str(mesh_path),
        "texture_resolution": texture_resolution,
        "remesh_option": remesh_option,
    }
    write_json(paths.sf3d / "sf3d_meta.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--sf3d-python", required=True, type=Path)
    parser.add_argument("--texture-resolution", default=1024, type=int)
    parser.add_argument(
        "--remesh-option",
        default="none",
        choices=["none", "triangle", "quad"],
    )
    args = parser.parse_args()
    run_step(
        run_dir=args.run_dir,
        sf3d_python=args.sf3d_python,
        texture_resolution=args.texture_resolution,
        remesh_option=args.remesh_option,
    )


if __name__ == "__main__":
    main()
