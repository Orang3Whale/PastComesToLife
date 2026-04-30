from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from PIL import Image
import rembg

from lib.image_utils import resize_foreground_rgba, save_mask
from lib.io_paths import create_run_tree, write_json


def _default_remove_background(image: Image.Image) -> Image.Image:
    session = rembg.new_session()
    return rembg.remove(image, session=session)


def run_step(
    input_path: Path,
    run_dir: Path,
    foreground_ratio: float = 0.85,
    remove_background_fn: Callable[[Image.Image], Image.Image] | None = None,
) -> dict:
    paths = create_run_tree(run_dir)
    remove_fn = remove_background_fn or _default_remove_background
    with Image.open(input_path) as source_image:
        source_image.save(paths.inputs / "content.png", format="PNG")
        src = source_image.convert("RGBA")
    rgba = remove_fn(src)
    rgba = resize_foreground_rgba(rgba, foreground_ratio)

    rgba_path = paths.preprocess / "rgba.png"
    mask_path = paths.preprocess / "mask.png"
    rgba.save(rgba_path)
    save_mask(rgba, mask_path)

    result = {
        "input_path": str(input_path),
        "rgba_path": str(rgba_path),
        "mask_path": str(mask_path),
        "foreground_ratio": foreground_ratio,
    }
    write_json(paths.preprocess / "meta.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--foreground-ratio", default=0.85, type=float)
    args = parser.parse_args()
    run_step(args.input, args.run_dir, args.foreground_ratio)


if __name__ == "__main__":
    main()
