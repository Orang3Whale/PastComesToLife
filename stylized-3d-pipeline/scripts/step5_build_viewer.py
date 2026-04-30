from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.io_paths import create_run_tree, write_json
from lib.viewer_utils import write_viewer


def run_step(run_dir: Path) -> dict:
    paths = create_run_tree(run_dir)
    viewer_html = paths.viewer / "index.html"
    viewer_meta = {
        "viewer_html": str(viewer_html),
    }
    write_viewer(viewer_html)
    write_json(paths.viewer / "viewer_meta.json", viewer_meta)
    return viewer_meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_step(args.run_dir)


if __name__ == "__main__":
    main()
