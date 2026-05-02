from __future__ import annotations

import argparse
import json
import sys
import shutil
from pathlib import Path
from typing import Callable

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.io_paths import create_run_tree, write_json
from lib.subprocess_utils import run_checked


def build_instantstyle_command(
    instantstyle_python: Path,
    worker_script: Path,
    run_dir: Path,
    control_image: Path,
    style_image: Path,
    prompt: str,
    output_image: Path,
    seed: int,
) -> list[str]:
    return [
        str(instantstyle_python),
        str(worker_script),
        "--run-dir",
        str(run_dir),
        "--control-image",
        str(control_image),
        "--style-image",
        str(style_image),
        "--prompt",
        prompt,
        "--output-image",
        str(output_image),
        "--seed",
        str(seed),
    ]


def run_step(
    run_dir: Path,
    instantstyle_python: Path,
    style_image: Path,
    prompt: str,
    seed: int = 42,
    runner: Callable[..., None] = run_checked,
) -> dict:
    paths = create_run_tree(run_dir)
    style_copy = paths.inputs / "style.png"
    prompt_file = paths.inputs / "prompt.txt"
    style_copy.write_bytes(style_image.read_bytes())
    prompt_file.write_text(prompt, encoding="utf-8")

    view_manifest_path = paths.views / "manifest.json"
    view_manifest = json.loads(view_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(view_manifest, dict):
        raise ValueError(f"invalid view manifest: {view_manifest_path}")
    views = view_manifest.get("views")
    if not isinstance(views, list) or not views:
        raise ValueError(f"view manifest must contain views: {view_manifest_path}")

    expected_view_order = ["front", "back", "left", "right", "top", "bottom"]
    seen_view_order: list[str] = []
    for view in views:
        if not isinstance(view, dict):
            raise ValueError(f"invalid view entry in {view_manifest_path}")
        view_name = view.get("name")
        control_path = view.get("control_path")
        if not isinstance(view_name, str) or not isinstance(control_path, str):
            raise ValueError(f"invalid view entry in {view_manifest_path}")
        seen_view_order.append(view_name)

    if seen_view_order != expected_view_order:
        raise ValueError(
            f"view manifest must match canonical order: expected {expected_view_order}, got {seen_view_order}",
        )

    worker_script = Path(__file__).resolve().parent / "workers" / "instantstyle_worker.py"
    result: dict[str, dict[str, dict[str, str]] | str] = {
        "style_image": str(style_copy),
        "prompt": prompt,
        "views": {},
    }

    for view in views:
        view_name = view["name"]
        control_image = Path(view["control_path"])
        mask_path_value = view.get("mask_path")
        if not isinstance(mask_path_value, str):
            raise ValueError(f"invalid view entry in {view_manifest_path}")
        mask_image = Path(mask_path_value)
        output_image = paths.stylize / view_name / "stylized.png"
        output_image.parent.mkdir(parents=True, exist_ok=True)
        cmd = build_instantstyle_command(
            instantstyle_python=instantstyle_python,
            worker_script=worker_script,
            run_dir=paths.root,
            control_image=control_image,
            style_image=style_copy,
            prompt=prompt,
            output_image=output_image,
            seed=seed,
        )
        runner(cmd, env={"HF_ENDPOINT": "https://hf-mirror.com"})
        if not output_image.is_file():
            raise FileNotFoundError(f"missing stylized output: {output_image}")
        with Image.open(output_image) as stylized_image, Image.open(mask_image) as mask_source:
            stylized = stylized_image.convert("RGBA")
            mask = mask_source.convert("L").resize(stylized.size, Image.Resampling.NEAREST)
            stylized.putalpha(mask)
            stylized.save(output_image)
        result["views"][view_name] = {
            "control_path": str(control_image),
            "stylized_path": str(output_image),
        }

    front_output = paths.stylize / "front" / "stylized.png"
    legacy_output = paths.stylize / "stylized.png"
    if front_output.is_file():
        shutil.copyfile(front_output, legacy_output)

    write_json(paths.stylize / "manifest.json", result)
    write_json(paths.stylize / "stylize_meta.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--instantstyle-python", required=True, type=Path)
    parser.add_argument("--style-image", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    run_step(
        args.run_dir,
        args.instantstyle_python,
        args.style_image,
        args.prompt,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
