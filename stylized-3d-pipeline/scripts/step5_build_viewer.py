from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.io_paths import create_run_tree, write_json
from lib.viewer_utils import write_viewer


EXPECTED_VIEW_ORDER = ["front", "back", "left", "right", "top", "bottom"]


def _load_view_names(view_manifest_path: Path) -> list[str]:
    view_manifest = json.loads(view_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(view_manifest, dict):
        raise ValueError(f"invalid view manifest: {view_manifest_path}")
    views = view_manifest.get("views")
    if not isinstance(views, list) or not views:
        raise ValueError(f"view manifest must contain views: {view_manifest_path}")

    seen_view_order: list[str] = []
    for view in views:
        if not isinstance(view, dict):
            raise ValueError(f"invalid view entry in {view_manifest_path}")
        view_name = view.get("name")
        if not isinstance(view_name, str):
            raise ValueError(f"invalid view entry in {view_manifest_path}")
        seen_view_order.append(view_name)

    if seen_view_order != EXPECTED_VIEW_ORDER:
        raise ValueError(
            f"view manifest must match canonical order: expected {EXPECTED_VIEW_ORDER}, got {seen_view_order}",
        )
    return seen_view_order


def run_step(run_dir: Path) -> dict:
    paths = create_run_tree(run_dir)
    view_names = _load_view_names(paths.views / "manifest.json")
    viewer_html = paths.viewer / "index.html"
    viewer_meta = {
        "viewer_html": str(viewer_html),
        "view_names": view_names,
    }
    write_viewer(viewer_html, view_names)
    write_json(paths.viewer / "viewer_meta.json", viewer_meta)
    return viewer_meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_step(args.run_dir)


if __name__ == "__main__":
    main()
