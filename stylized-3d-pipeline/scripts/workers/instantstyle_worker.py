from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

from PIL import Image

CONTROLNET_MODEL_ID = "diffusers/controlnet-canny-sdxl-1.0"

try:  # pragma: no cover - exercised through monkeypatched tests
    from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline
except Exception:  # pragma: no cover - dependency is provided in the worker env
    class _MissingDependency:
        pass

    ControlNetModel = _MissingDependency  # type: ignore[assignment]
    StableDiffusionXLControlNetImg2ImgPipeline = _MissingDependency  # type: ignore[assignment]

try:  # pragma: no cover - exercised through monkeypatched tests
    from ip_adapter import IPAdapterXL
except Exception:  # pragma: no cover - dependency is provided in the worker env
    class _MissingIPAdapter:
        pass

    IPAdapterXL = _MissingIPAdapter  # type: ignore[assignment]

SDXL_MODELS_ENV_VARS = ("INSTANTSTYLE_SDXL_MODELS", "IP_ADAPTER_SDXL_MODELS")


def find_upstream_repo_root(anchor: Path | None = None) -> Path:
    current = (anchor or Path(__file__)).resolve()
    for parent in (current.parent, *current.parents):
        candidate = parent / "InstantStyle"
        if candidate.is_dir():
            return parent
    raise FileNotFoundError(f"could not locate InstantStyle relative to {current}")


def _is_valid_sdxl_models_root(candidate: Path) -> bool:
    return (candidate / "image_encoder").is_dir() and (candidate / "ip-adapter_sdxl.bin").is_file()


def resolve_sdxl_models_root(project_root: Path | None = None) -> Path:
    candidates: list[Path] = []
    for env_var in SDXL_MODELS_ENV_VARS:
        env_value = os.environ.get(env_var)
        if env_value:
            candidates.append(Path(env_value))
    if project_root is not None:
        candidates.append(project_root / "sdxl_models")
    candidates.append(Path("/root/autodl-tmp/models/IP-Adapter/sdxl_models"))

    checked: list[str] = []
    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.expanduser()
        if normalized in seen:
            continue
        seen.add(normalized)
        checked.append(str(normalized))
        if _is_valid_sdxl_models_root(normalized):
            return normalized

    joined = ", ".join(checked)
    raise FileNotFoundError(f"could not locate InstantStyle SDXL models; checked: {joined}")


def build_worker_meta(
    rgb_image: Path,
    control_image: Path,
    style_image: Path,
    prompt: str,
    seed: int,
    strength: float,
    output_image: Path,
    style_scale: float = 1.0,
    guidance_scale: float = 5.0,
    num_inference_steps: int = 30,
    controlnet_conditioning_scale: float = 0.6,
) -> dict[str, str | int | float]:
    return {
        "rgb_image": str(rgb_image),
        "control_image": str(control_image),
        "style_image": str(style_image),
        "prompt": prompt,
        "seed": seed,
        "strength": strength,
        "style_scale": style_scale,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "controlnet_conditioning_scale": controlnet_conditioning_scale,
        "output_image": str(output_image),
    }


def prepare_base_image(rgb: Image.Image) -> Image.Image:
    base = rgb.convert("RGBA")
    background = Image.new("RGBA", base.size, (235, 235, 235, 255))
    return Image.alpha_composite(background, base).convert("RGB")


def prepare_control_image(control: Image.Image) -> Image.Image:
    geometry = control.convert("RGBA")
    background = Image.new("RGBA", geometry.size, (0, 0, 0, 255))
    return Image.alpha_composite(background, geometry).convert("RGB")


def build_pipeline_and_adapter(
    device: str = "cuda",
    ip_adapter_cls: Callable[..., object] | None = None,
) -> tuple[object, object]:
    import torch  # noqa: WPS433
    project_root = Path(__file__).resolve().parents[2]
    model_root = resolve_sdxl_models_root(project_root)

    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL_ID,
        use_safetensors=False,
        torch_dtype=torch.float16,
    ).to(device)
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        add_watermarker=False,
    ).to(device)
    pipe.enable_vae_tiling()
    adapter_cls = ip_adapter_cls or IPAdapterXL
    ip_model = adapter_cls(
        pipe,
        str(model_root / "image_encoder"),
        str(model_root / "ip-adapter_sdxl.bin"),
        device,
        target_blocks=["up_blocks.0.attentions.1"],
    )
    return pipe, ip_model


def generate_stylized_images(
    ip_model: object,
    style: Image.Image,
    prompt: str,
    base_image: Image.Image,
    control_image: Image.Image,
    seed: int,
    strength: float,
    style_scale: float = 1.0,
    guidance_scale: float = 5.0,
    num_inference_steps: int = 30,
    controlnet_conditioning_scale: float = 0.6,
) -> object:
    return ip_model.generate(
        pil_image=style,
        prompt=prompt,
        negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, blurry",
        scale=style_scale,
        guidance_scale=guidance_scale,
        num_samples=1,
        num_inference_steps=num_inference_steps,
        seed=seed,
        image=base_image,
        control_image=control_image,
        strength=strength,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
    )


def write_worker_outputs(
    output_image: Path,
    stylized_image: Image.Image,
    rgb_image: Path,
    control_image: Path,
    style_image: Path,
    prompt: str,
    seed: int,
    strength: float,
    style_scale: float = 1.0,
    guidance_scale: float = 5.0,
    num_inference_steps: int = 30,
    controlnet_conditioning_scale: float = 0.6,
) -> dict[str, str | int | float]:
    output_image.parent.mkdir(parents=True, exist_ok=True)
    stylized_image.save(output_image)
    metadata = build_worker_meta(
        rgb_image=rgb_image,
        control_image=control_image,
        style_image=style_image,
        prompt=prompt,
        seed=seed,
        strength=strength,
        style_scale=style_scale,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        output_image=output_image,
    )
    (output_image.parent / "worker_meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata


def _read_jobs_manifest(jobs_manifest: Path) -> list[dict[str, object]]:
    payload = json.loads(jobs_manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid jobs manifest: {jobs_manifest}")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError(f"jobs manifest must contain jobs: {jobs_manifest}")
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError(f"invalid job entry in {jobs_manifest}")
        for key in ("rgb_image", "control_image", "style_image", "prompt", "output_image"):
            if key not in job:
                raise ValueError(f"missing job field {key}: {jobs_manifest}")
        for key in ("rgb_image", "control_image", "style_image"):
            path = Path(str(job[key]))
            if not path.is_file():
                raise FileNotFoundError(f"missing job asset: {path}")
    return jobs


def _job_from_args(args: argparse.Namespace) -> dict[str, object]:
    required = {
        "rgb_image": args.rgb_image,
        "control_image": args.control_image,
        "style_image": args.style_image,
        "prompt": args.prompt,
        "output_image": args.output_image,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        joined = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise ValueError(f"missing required single-view arguments: {joined}")
    return {
        "rgb_image": str(args.rgb_image),
        "control_image": str(args.control_image),
        "style_image": str(args.style_image),
        "prompt": args.prompt,
        "output_image": str(args.output_image),
        "seed": args.seed,
        "strength": args.strength,
        "style_scale": args.style_scale,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "controlnet_conditioning_scale": args.controlnet_conditioning_scale,
    }


def run_stylization_job(
    job: dict[str, object],
    ip_model: object,
    style_cache: dict[Path, Image.Image] | None = None,
) -> dict[str, str | int | float]:
    rgb_image = Path(str(job["rgb_image"]))
    control_image = Path(str(job["control_image"]))
    style_image = Path(str(job["style_image"]))
    output_image = Path(str(job["output_image"]))
    prompt = str(job["prompt"])
    seed = int(job.get("seed", 42))
    strength = float(job.get("strength", 0.45))
    style_scale = float(job.get("style_scale", 1.0))
    guidance_scale = float(job.get("guidance_scale", 5.0))
    num_inference_steps = int(job.get("num_inference_steps", 30))
    controlnet_conditioning_scale = float(job.get("controlnet_conditioning_scale", 0.6))

    with Image.open(rgb_image) as rgb_source:
        base = prepare_base_image(rgb_source)
    with Image.open(control_image) as control_source:
        control = prepare_control_image(control_source)

    cache = style_cache if style_cache is not None else {}
    style = cache.get(style_image)
    if style is None:
        with Image.open(style_image) as style_source:
            style = style_source.convert("RGB")
        cache[style_image] = style

    images = generate_stylized_images(
        ip_model=ip_model,
        style=style,
        prompt=prompt,
        base_image=base,
        control_image=control,
        seed=seed,
        strength=strength,
        style_scale=style_scale,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
    )

    stylized = images[0].convert("RGBA").resize(control.size)
    return write_worker_outputs(
        output_image=output_image,
        stylized_image=stylized,
        rgb_image=rgb_image,
        control_image=control_image,
        style_image=style_image,
        prompt=prompt,
        seed=seed,
        strength=strength,
        style_scale=style_scale,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
    )


def run_stylization_jobs(jobs: list[dict[str, object]], ip_model: object) -> list[dict[str, str | int | float]]:
    style_cache: dict[Path, Image.Image] = {}
    return [run_stylization_job(job, ip_model=ip_model, style_cache=style_cache) for job in jobs]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--jobs-manifest", type=Path)
    parser.add_argument("--rgb-image", type=Path)
    parser.add_argument("--control-image", type=Path)
    parser.add_argument("--style-image", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--output-image", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--strength", default=0.45, type=float)
    parser.add_argument("--style-scale", default=1.0, type=float)
    parser.add_argument("--guidance-scale", default=5.0, type=float)
    parser.add_argument("--num-inference-steps", default=30, type=int)
    parser.add_argument("--controlnet-conditioning-scale", default=0.6, type=float)
    args = parser.parse_args()

    root = find_upstream_repo_root(Path(__file__))
    instantstyle_root = root / "InstantStyle"
    if str(instantstyle_root) not in sys.path:
        sys.path.insert(0, str(instantstyle_root))

    try:
        from ip_adapter import IPAdapterXL as runtime_ip_adapter
    except Exception as exc:  # pragma: no cover - depends on worker environment
        raise RuntimeError(
            f"could not import ip_adapter after adding {instantstyle_root} to sys.path"
        ) from exc

    _, ip_model = build_pipeline_and_adapter(device="cuda", ip_adapter_cls=runtime_ip_adapter)

    jobs = _read_jobs_manifest(args.jobs_manifest) if args.jobs_manifest else [_job_from_args(args)]
    run_stylization_jobs(jobs, ip_model=ip_model)


if __name__ == "__main__":
    main()
