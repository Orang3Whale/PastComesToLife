from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Callable

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.io_paths import create_run_tree, resolve_run_dir, write_json
from lib.subprocess_utils import huggingface_cache_env, run_checked
from scripts.step1_preprocess import run_step as run_preprocess_step
from scripts.step2_sf3d import run_step as run_sf3d_step
from scripts.step3_sample_views import run_step as run_sample_views_step
from scripts.step4_retexture import run_step as run_retexture_step
from scripts.step5_build_viewer import run_step as run_viewer_step

EXPECTED_VIEW_ORDER = ["front", "back", "left", "right", "top", "bottom"]
VIEW_FILE_NAMES = {
    "rgb_path": "rgb.png",
    "control_path": "control.png",
    "depth_path": "depth.npy",
    "depth_preview_path": "depth.png",
    "normal_path": "normal.png",
    "mask_path": "mask.png",
    "camera_path": "camera.json",
}
DEFAULT_PROMPT = "a wooden chair, blue starry night, crescent moon, golden stars, decorative painted textile style, bold blue and gold folk art"
DEFAULT_STYLE_SCALE = 1.8
DEFAULT_GUIDANCE_SCALE = 6.5
DEFAULT_NUM_INFERENCE_STEPS = 35
DEFAULT_CONTROLNET_CONDITIONING_SCALE = 0.45
DEFAULT_STRENGTH = 0.72
DEFAULT_SF3D_PYTHON = Path("/root/autodl-tmp/envs/sf3d/bin/python")
DEFAULT_INSTANTSTYLE_PYTHON = Path("/root/autodl-tmp/envs/instantstyle/bin/python")


def build_worker_command(
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
    style_scale: float,
    guidance_scale: float,
    num_inference_steps: int,
    controlnet_conditioning_scale: float,
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
        "--style-scale",
        str(style_scale),
        "--guidance-scale",
        str(guidance_scale),
        "--num-inference-steps",
        str(num_inference_steps),
        "--controlnet-conditioning-scale",
        str(controlnet_conditioning_scale),
    ]


def _validate_view_manifest(view_manifest_path: Path) -> list[dict[str, str]]:
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
        for key in ("rgb_path", "control_path", "mask_path"):
            path_value = view.get(key)
            if not isinstance(path_value, str) or not Path(path_value).is_file():
                raise FileNotFoundError(f"missing view asset: {path_value}")

    if seen_view_order != EXPECTED_VIEW_ORDER:
        raise ValueError(
            f"view manifest must match canonical order: expected {EXPECTED_VIEW_ORDER}, got {seen_view_order}",
        )
    return views


def run_stylize_step(
    run_dir: Path,
    instantstyle_python: Path,
    style_image: Path,
    prompt: str,
    seed: int = 42,
    strength: float = DEFAULT_STRENGTH,
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

    views = _validate_view_manifest(paths.views / "manifest.json")
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

    for view in views:
        view_name = view["name"]
        rgb_image = Path(view["rgb_path"])
        control_image = Path(view["control_path"])
        mask_image = Path(view["mask_path"])
        output_image = paths.stylize / view_name / "stylized.png"
        output_image.parent.mkdir(parents=True, exist_ok=True)
        cmd = build_worker_command(
            instantstyle_python=instantstyle_python,
            worker_script=worker_script,
            run_dir=paths.root,
            rgb_image=rgb_image,
            control_image=control_image,
            style_image=style_copy,
            prompt=prompt,
            output_image=output_image,
            seed=seed,
            strength=strength,
            style_scale=style_scale,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        )
        worker_env = huggingface_cache_env()
        worker_env.update({"HF_ENDPOINT": "https://hf-mirror.com", "OMP_NUM_THREADS": "1"})
        runner(cmd, env=worker_env)
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


def _copy_stage_trees(source_run: Path, target_run: Path) -> None:
    for stage in ("inputs", "preprocess", "sf3d", "views", "stylize"):
        source_stage = source_run / stage
        target_stage = target_run / stage
        if source_stage.exists():
            shutil.copytree(source_stage, target_stage, dirs_exist_ok=True)


def _rewrite_views_manifest(run_dir: Path) -> None:
    manifest_path = run_dir / "views" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest.get("views", []):
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str):
            continue
        view_dir = run_dir / "views" / name
        for key, filename in VIEW_FILE_NAMES.items():
            if key in entry:
                entry[key] = str(view_dir / filename)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _rewrite_stylize_manifest(run_dir: Path) -> None:
    manifest_path = run_dir / "stylize" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["style_image"] = str(run_dir / "inputs" / "style.png")
    views = manifest.get("views", {})
    if isinstance(views, dict):
        for name, entry in views.items():
            if not isinstance(entry, dict):
                continue
            view_dir = run_dir / "views" / name
            stylize_dir = run_dir / "stylize" / name
            if "rgb_path" in entry:
                entry["rgb_path"] = str(view_dir / "rgb.png")
            if "control_path" in entry:
                entry["control_path"] = str(view_dir / "control.png")
            if "stylized_path" in entry:
                entry["stylized_path"] = str(stylize_dir / "stylized.png")
    payload = json.dumps(manifest, indent=2, sort_keys=True)
    manifest_path.write_text(payload, encoding="utf-8")
    (run_dir / "stylize" / "stylize_meta.json").write_text(payload, encoding="utf-8")


def run_rebake_experiment(
    args: object,
    retexture_runner: Callable[[Path], dict] = run_retexture_step,
    viewer_runner: Callable[[Path], dict] = run_viewer_step,
) -> dict[str, object]:
    source_run = Path(getattr(args, "source_run"))
    run_dir = resolve_run_dir(Path(getattr(args, "runs_root")), getattr(args, "run_name", None))
    create_run_tree(run_dir)
    _copy_stage_trees(source_run, run_dir)
    _rewrite_views_manifest(run_dir)
    _rewrite_stylize_manifest(run_dir)

    config = {
        "mode": "rebake",
        "source_run": str(source_run),
        "run_name": getattr(args, "run_name", None),
        "runs_root": str(getattr(args, "runs_root")),
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "run_config.json", config)

    retexture_result = retexture_runner(run_dir)
    viewer_result = viewer_runner(run_dir)
    return {
        "run_dir": str(run_dir),
        "retexture": retexture_result,
        "viewer": viewer_result,
    }


def run_full_experiment(
    args: object,
    preprocess_runner: Callable[..., dict] = run_preprocess_step,
    sf3d_runner: Callable[..., dict] = run_sf3d_step,
    sample_views_runner: Callable[..., dict] = run_sample_views_step,
    stylize_runner: Callable[..., dict] = run_stylize_step,
    retexture_runner: Callable[..., dict] = run_retexture_step,
    viewer_runner: Callable[..., dict] = run_viewer_step,
) -> dict[str, object]:
    run_dir = resolve_run_dir(Path(getattr(args, "runs_root")), getattr(args, "run_name", None))
    create_run_tree(run_dir)

    config = {
        "mode": "full",
        "input": str(getattr(args, "input")),
        "style_image": str(getattr(args, "style_image")),
        "prompt": getattr(args, "prompt"),
        "run_name": getattr(args, "run_name", None),
        "runs_root": str(getattr(args, "runs_root")),
        "run_dir": str(run_dir),
        "sf3d_python": str(getattr(args, "sf3d_python")),
        "instantstyle_python": str(getattr(args, "instantstyle_python")),
        "foreground_ratio": getattr(args, "foreground_ratio"),
        "texture_resolution": getattr(args, "texture_resolution"),
        "remesh_option": getattr(args, "remesh_option"),
        "view_resolution": getattr(args, "view_resolution"),
        "camera_distance": getattr(args, "camera_distance"),
        "camera_fovy_deg": getattr(args, "camera_fovy_deg"),
        "seed": getattr(args, "seed"),
        "strength": getattr(args, "strength"),
        "style_scale": getattr(args, "style_scale"),
        "guidance_scale": getattr(args, "guidance_scale"),
        "num_inference_steps": getattr(args, "num_inference_steps"),
        "controlnet_conditioning_scale": getattr(args, "controlnet_conditioning_scale"),
    }
    write_json(run_dir / "run_config.json", config)

    preprocess_result = preprocess_runner(
        input_path=Path(getattr(args, "input")),
        run_dir=run_dir,
        foreground_ratio=getattr(args, "foreground_ratio"),
    )
    sf3d_result = sf3d_runner(
        run_dir=run_dir,
        sf3d_python=Path(getattr(args, "sf3d_python")),
        texture_resolution=getattr(args, "texture_resolution"),
        remesh_option=getattr(args, "remesh_option"),
    )
    sample_views_result = sample_views_runner(
        run_dir=run_dir,
        view_resolution=getattr(args, "view_resolution"),
        camera_distance=getattr(args, "camera_distance"),
        camera_fovy_deg=getattr(args, "camera_fovy_deg"),
    )
    stylize_result = stylize_runner(
        run_dir=run_dir,
        instantstyle_python=Path(getattr(args, "instantstyle_python")),
        style_image=Path(getattr(args, "style_image")),
        prompt=getattr(args, "prompt"),
        seed=getattr(args, "seed"),
        strength=getattr(args, "strength"),
        style_scale=getattr(args, "style_scale"),
        guidance_scale=getattr(args, "guidance_scale"),
        num_inference_steps=getattr(args, "num_inference_steps"),
        controlnet_conditioning_scale=getattr(args, "controlnet_conditioning_scale"),
    )
    retexture_result = retexture_runner(run_dir)
    viewer_result = viewer_runner(run_dir)
    return {
        "run_dir": str(run_dir),
        "preprocess": preprocess_result,
        "sf3d": sf3d_result,
        "sample_views": sample_views_result,
        "stylize": stylize_result,
        "retexture": retexture_result,
        "viewer": viewer_result,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the current multiview stylization experiment or rebake an existing run.",
    )
    parser.add_argument("--run-name")
    parser.add_argument("--runs-root", default=Path("runs"), type=Path)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--style-image", type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--sf3d-python", default=DEFAULT_SF3D_PYTHON, type=Path)
    parser.add_argument("--instantstyle-python", default=DEFAULT_INSTANTSTYLE_PYTHON, type=Path)
    parser.add_argument("--foreground-ratio", default=0.85, type=float)
    parser.add_argument("--texture-resolution", default=1024, type=int)
    parser.add_argument("--view-resolution", default=512, type=int)
    parser.add_argument("--camera-distance", default=1.8, type=float)
    parser.add_argument("--camera-fovy-deg", default=40.0, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--strength", default=DEFAULT_STRENGTH, type=float)
    parser.add_argument("--style-scale", default=DEFAULT_STYLE_SCALE, type=float)
    parser.add_argument("--guidance-scale", default=DEFAULT_GUIDANCE_SCALE, type=float)
    parser.add_argument("--num-inference-steps", default=DEFAULT_NUM_INFERENCE_STEPS, type=int)
    parser.add_argument(
        "--controlnet-conditioning-scale",
        default=DEFAULT_CONTROLNET_CONDITIONING_SCALE,
        type=float,
    )
    parser.add_argument(
        "--remesh-option",
        default="none",
        choices=["none", "triangle", "quad"],
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.source_run is not None:
        if args.run_name is None:
            args.run_name = f"{Path(args.source_run).name}-rebake"
        run_rebake_experiment(args)
        return

    required = ["input", "style_image", "prompt", "sf3d_python", "instantstyle_python"]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        joined = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise ValueError(f"missing required full-mode arguments: {joined}")
    run_full_experiment(args)


if __name__ == "__main__":
    main()
