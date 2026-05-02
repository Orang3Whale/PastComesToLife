from __future__ import annotations

import json
from itertools import count
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    root: Path
    inputs: Path
    preprocess: Path
    sf3d: Path
    views: Path
    stylize: Path
    retexture: Path
    viewer: Path


_anonymous_run_counter = count()


def resolve_run_dir(base_dir: Path, run_name: str | None) -> Path:
    if run_name is not None:
        if (
            not run_name
            or Path(run_name).is_absolute()
            or run_name in {".", ".."}
            or "/" in run_name
            or "\\" in run_name
        ):
            raise ValueError("run_name must be a non-empty relative leaf name")
        return base_dir / run_name
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    while True:
        candidate = base_dir / f"{stamp}-{next(_anonymous_run_counter):04d}"
        if not candidate.exists():
            return candidate


def create_run_tree(run_dir: Path) -> RunPaths:
    inputs = run_dir / "inputs"
    preprocess = run_dir / "preprocess"
    sf3d = run_dir / "sf3d"
    views = run_dir / "views"
    stylize = run_dir / "stylize"
    retexture = run_dir / "retexture"
    viewer = run_dir / "viewer"

    for path in (run_dir, inputs, preprocess, sf3d, views, stylize, retexture, viewer):
        path.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        root=run_dir,
        inputs=inputs,
        preprocess=preprocess,
        sf3d=sf3d,
        views=views,
        stylize=stylize,
        retexture=retexture,
        viewer=viewer,
    )


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
