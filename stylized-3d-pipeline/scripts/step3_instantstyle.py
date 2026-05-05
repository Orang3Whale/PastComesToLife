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
from lib.subprocess_utils import huggingface_cache_env, run_checked

DEFAULT_STYLE_SCALE = 1.0
DEFAULT_GUIDANCE_SCALE = 5.0
DEFAULT_NUM_INFERENCE_STEPS = 30
DEFAULT_CONTROLNET_CONDITIONING_SCALE = 0.6


def build_instantstyle_command(
    instantstyle_python: Path,
    worker_script: Path,
    run_dir: Path,
    rgb_image: Path,
    control_image: Path,
    style_image: Path,
    prompt: str,
    output_image: Path,
    seed: int,
    strength: float,
) -> list[str]:
    return [
        str(instantstyle_python),
        str(worker_script),
        "--run-dir",
        str(run_dir),
        "--rgb-image",
        str(rgb_image),
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
        "--strength",
        str(strength),
    ]


def build_instantstyle_batch_command(
    instantstyle_python: Path,
    worker_script: Path,
    run_dir: Path,
    jobs_manifest: Path,
) -> list[str]:
    return [
        str(instantstyle_python),
        str(worker_script),
        "--run-dir",
        str(run_dir),
        "--jobs-manifest",
        str(jobs_manifest),
    ]


def _build_stylize_jobs(
    views: list[dict[str, str]],
    style_image: Path,
    prompt: str,
    seed: int,
    strength: float,
    style_scale: float,
    guidance_scale: float,
    num_inference_steps: int,
    controlnet_conditioning_scale: float,
    stylize_root: Path,
) -> list[dict[str, object]]:
    jobs: list[dict[str, object]] = []
    for view in views:
        view_name = view["name"]
        output_image = stylize_root / view_name / "stylized.png"
        output_image.parent.mkdir(parents=True, exist_ok=True)
        jobs.append(
            {
                "name": view_name,
                "rgb_image": view["rgb_path"],
                "control_image": view["control_path"],
                "style_image": str(style_image),
                "prompt": prompt,
                "output_image": str(output_image),
                "seed": seed,
                "strength": strength,
                "style_scale": style_scale,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "controlnet_conditioning_scale": controlnet_conditioning_scale,
            }
        )
    return jobs


def run_step(
    run_dir: Path,
    instantstyle_python: Path,
    style_image: Path,
    prompt: str,
    seed: int = 42,
    strength: float = 0.45,
    style_scale: float = DEFAULT_STYLE_SCALE,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    controlnet_conditioning_scale: float = DEFAULT_CONTROLNET_CONDITIONING_SCALE,
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
        if not isinstance(view_name, str):
            raise ValueError(f"invalid view entry in {view_manifest_path}")
        seen_view_order.append(view_name)

    if seen_view_order != expected_view_order:
        raise ValueError(
            f"view manifest must match canonical order: expected {expected_view_order}, got {seen_view_order}",
        )

    for view in views:
        if not all(
            isinstance(view.get(key), str)
            for key in ("rgb_path", "control_path", "mask_path")
        ):
            raise ValueError(f"invalid view entry in {view_manifest_path}")
        for key in ("rgb_path", "control_path", "mask_path"):
            path = Path(view[key])
            if not path.is_file():
                raise FileNotFoundError(f"missing view asset: {path}")

    worker_script = Path(__file__).resolve().parent / "workers" / "instantstyle_worker.py"
    result: dict[str, dict[str, dict[str, str]] | float | str] = {
        "style_image": str(style_copy),
        "prompt": prompt,
        "strength": strength,
        "style_scale": style_scale,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "controlnet_conditioning_scale": controlnet_conditioning_scale,
        "views": {},
    }

    jobs = _build_stylize_jobs(
        views=views,
        style_image=style_copy,
        prompt=prompt,
        seed=seed,
        strength=strength,
        style_scale=style_scale,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        stylize_root=paths.stylize,
    )
    jobs_manifest = paths.stylize / "worker_jobs.json"
    write_json(jobs_manifest, {"jobs": jobs})

    cmd = build_instantstyle_batch_command(
        instantstyle_python=instantstyle_python,
        worker_script=worker_script,
        run_dir=paths.root,
        jobs_manifest=jobs_manifest,
    )
    worker_env = huggingface_cache_env()
    worker_env.update({"HF_ENDPOINT": "https://hf-mirror.com", "OMP_NUM_THREADS": "1"})
    runner(cmd, env=worker_env)

    for view in views:
        view_name = view["name"]
        rgb_image = Path(view["rgb_path"])
        control_image = Path(view["control_path"])
        mask_image = Path(view["mask_path"])
        output_image = paths.stylize / view_name / "stylized.png"
        if not output_image.is_file():
            raise FileNotFoundError(f"missing stylized output: {output_image}")
        with Image.open(output_image) as stylized_image, Image.open(mask_image) as mask_source:
            stylized = stylized_image.convert("RGBA")
            mask = mask_source.convert("L").resize(stylized.size, Image.Resampling.NEAREST)
            stylized.putalpha(mask)
            stylized.save(output_image)
        result["views"][view_name] = {
            "rgb_path": str(rgb_image),
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
    parser.add_argument("--strength", default=0.45, type=float)
    parser.add_argument("--style-scale", default=DEFAULT_STYLE_SCALE, type=float)
    parser.add_argument("--guidance-scale", default=DEFAULT_GUIDANCE_SCALE, type=float)
    parser.add_argument("--num-inference-steps", default=DEFAULT_NUM_INFERENCE_STEPS, type=int)
    parser.add_argument(
        "--controlnet-conditioning-scale",
        default=DEFAULT_CONTROLNET_CONDITIONING_SCALE,
        type=float,
    )
    args = parser.parse_args()
    run_step(
        args.run_dir,
        args.instantstyle_python,
        args.style_image,
        args.prompt,
        seed=args.seed,
        strength=args.strength,
        style_scale=args.style_scale,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
    )


if __name__ == "__main__":
    main()
