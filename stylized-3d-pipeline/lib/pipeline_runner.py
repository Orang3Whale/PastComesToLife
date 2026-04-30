from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from lib.io_paths import create_run_tree, write_json
from scripts.step1_preprocess import run_step as run_preprocess_step
from scripts.step2_sf3d import run_step as run_sf3d_step
from scripts.step3_instantstyle import run_step as run_instantstyle_step
from scripts.step4_retexture import run_step as run_retexture_step
from scripts.step5_build_viewer import run_step as run_viewer_step


@dataclass(frozen=True)
class StepMeta:
    name: str
    meta_path: Callable[[Path], Path]
    runner: Callable[..., dict]


def _preprocess_meta(run_dir: Path) -> Path:
    return run_dir / "preprocess" / "meta.json"


def _sf3d_meta(run_dir: Path) -> Path:
    return run_dir / "sf3d" / "sf3d_meta.json"


def _instantstyle_meta(run_dir: Path) -> Path:
    return run_dir / "stylize" / "stylize_meta.json"


def _retexture_meta(run_dir: Path) -> Path:
    return run_dir / "retexture" / "retexture_meta.json"


def _viewer_meta(run_dir: Path) -> Path:
    return run_dir / "viewer" / "viewer_meta.json"


STEP_META: Mapping[str, StepMeta] = {
    "preprocess": StepMeta("preprocess", _preprocess_meta, run_preprocess_step),
    "sf3d": StepMeta("sf3d", _sf3d_meta, run_sf3d_step),
    "instantstyle": StepMeta("instantstyle", _instantstyle_meta, run_instantstyle_step),
    "retexture": StepMeta("retexture", _retexture_meta, run_retexture_step),
    "viewer": StepMeta("viewer", _viewer_meta, run_viewer_step),
}


def ordered_steps() -> list[str]:
    return ["preprocess", "sf3d", "instantstyle", "retexture", "viewer"]


def should_run_step(
    step_name: str,
    *,
    resume_from: str | None,
    skip_existing: bool,
    run_dir: Path,
) -> bool:
    if step_name not in STEP_META:
        raise KeyError(f"unknown step: {step_name}")

    ordered = ordered_steps()
    if resume_from is not None:
        if resume_from not in STEP_META:
            raise KeyError(f"unknown resume_from step: {resume_from}")
        if ordered.index(step_name) < ordered.index(resume_from):
            return False

    if skip_existing and STEP_META[step_name].meta_path(run_dir).is_file():
        return False

    return True


def _step_kwargs(step_name: str, args: object, run_dir: Path) -> dict[str, object]:
    if step_name == "preprocess":
        return {
            "input_path": getattr(args, "input"),
            "run_dir": run_dir,
            "foreground_ratio": getattr(args, "foreground_ratio"),
        }
    if step_name == "sf3d":
        return {
            "run_dir": run_dir,
            "sf3d_python": getattr(args, "sf3d_python"),
            "texture_resolution": getattr(args, "texture_resolution"),
            "remesh_option": getattr(args, "remesh_option"),
        }
    if step_name == "instantstyle":
        return {
            "run_dir": run_dir,
            "instantstyle_python": getattr(args, "instantstyle_python"),
            "style_image": getattr(args, "style_image"),
            "prompt": getattr(args, "prompt"),
        }
    if step_name == "retexture":
        return {"run_dir": run_dir}
    if step_name == "viewer":
        return {"run_dir": run_dir}
    raise KeyError(f"unknown step: {step_name}")


def run_pipeline(args: object) -> dict[str, dict]:
    run_dir = Path(getattr(args, "run_dir"))
    create_run_tree(run_dir)

    results: dict[str, dict] = {}
    for step_name in ordered_steps():
        if not should_run_step(
            step_name,
            resume_from=getattr(args, "resume_from", None),
            skip_existing=bool(getattr(args, "skip_existing", False)),
            run_dir=run_dir,
        ):
            continue

        step_runner = STEP_META[step_name].runner
        results[step_name] = step_runner(**_step_kwargs(step_name, args, run_dir))

    return results


def write_run_config(run_dir: Path, args: object) -> Path:
    payload = {
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
        "resume_from": getattr(args, "resume_from", None),
        "skip_existing": bool(getattr(args, "skip_existing", False)),
    }
    config_path = run_dir / "run_config.json"
    write_json(config_path, payload)
    return config_path
