from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from lib.io_paths import create_run_tree, write_json
from lib.subprocess_utils import run_checked


def build_instantstyle_command(
    instantstyle_python: Path,
    worker_script: Path,
    run_dir: Path,
    style_image: Path,
    prompt: str,
) -> list[str]:
    return [
        str(instantstyle_python),
        str(worker_script),
        "--run-dir",
        str(run_dir),
        "--style-image",
        str(style_image),
        "--prompt",
        prompt,
    ]


def run_step(
    run_dir: Path,
    instantstyle_python: Path,
    style_image: Path,
    prompt: str,
    runner: Callable[..., None] = run_checked,
) -> dict:
    paths = create_run_tree(run_dir)
    style_copy = paths.inputs / "style.png"
    prompt_file = paths.inputs / "prompt.txt"
    style_copy.write_bytes(style_image.read_bytes())
    prompt_file.write_text(prompt, encoding="utf-8")

    worker_script = Path(__file__).resolve().parent / "workers" / "instantstyle_worker.py"
    cmd = build_instantstyle_command(
        instantstyle_python=instantstyle_python,
        worker_script=worker_script,
        run_dir=paths.root,
        style_image=style_copy,
        prompt=prompt,
    )
    runner(cmd, env={"HF_ENDPOINT": "https://hf-mirror.com"})

    stylized_path = paths.stylize / "stylized.png"
    if not stylized_path.is_file():
        raise FileNotFoundError(f"missing stylized output: {stylized_path}")

    result = {
        "stylized_path": str(stylized_path),
        "style_image": str(style_copy),
        "prompt": prompt,
    }
    write_json(paths.stylize / "stylize_meta.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--instantstyle-python", required=True, type=Path)
    parser.add_argument("--style-image", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()
    run_step(args.run_dir, args.instantstyle_python, args.style_image, args.prompt)


if __name__ == "__main__":
    main()
