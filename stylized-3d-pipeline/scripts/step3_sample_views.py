from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.camera_views import build_six_view_spec
from lib.io_paths import create_run_tree, write_json
from lib.view_sampling import render_view_assets, write_view_assets


def _load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(mesh_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"no geometry found in {mesh_path}")
        loaded = loaded.to_geometry()
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"unsupported mesh type: {type(loaded)!r}")
    return loaded


def run_step(
    run_dir: Path,
    view_resolution: int,
    camera_distance: float,
    camera_fovy_deg: float,
    renderer: Callable[..., dict[str, dict[str, object]]] = render_view_assets,
) -> dict:
    paths = create_run_tree(run_dir)
    mesh = _load_mesh(paths.sf3d / "mesh_raw.glb")
    views = build_six_view_spec(mesh, camera_distance, camera_fovy_deg)
    assets = renderer(mesh, views, view_resolution)

    manifest = {
        "render_mode": "mesh_textured_offscreen",
        "view_resolution": view_resolution,
        "camera_distance": camera_distance,
        "camera_fovy_deg": camera_fovy_deg,
        "views": [],
    }
    for view in views:
        entry = write_view_assets(paths.views / view.name, assets[view.name])
        manifest["views"].append({"name": view.name, **entry})

    write_json(paths.views / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--view-resolution", default=512, type=int)
    parser.add_argument("--camera-distance", default=1.8, type=float)
    parser.add_argument("--camera-fovy-deg", default=40.0, type=float)
    args = parser.parse_args()
    run_step(args.run_dir, args.view_resolution, args.camera_distance, args.camera_fovy_deg)


if __name__ == "__main__":
    main()
