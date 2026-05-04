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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--rgb-image", required=True, type=Path)
    parser.add_argument("--control-image", required=True, type=Path)
    parser.add_argument("--style-image", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-image", required=True, type=Path)
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

    with Image.open(args.rgb_image) as rgb_image:
        base = prepare_base_image(rgb_image)
    with Image.open(args.control_image) as control_image:
        control = prepare_control_image(control_image)
    with Image.open(args.style_image) as style_image:
        style = style_image.convert("RGB")

    images = generate_stylized_images(
        ip_model=ip_model,
        style=style,
        prompt=args.prompt,
        base_image=base,
        control_image=control,
        seed=args.seed,
        strength=args.strength,
        style_scale=args.style_scale,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
    )

    stylized = images[0].convert("RGBA").resize(control.size)
    write_worker_outputs(
        output_image=args.output_image,
        stylized_image=stylized,
        rgb_image=args.rgb_image,
        control_image=args.control_image,
        style_image=args.style_image,
        prompt=args.prompt,
        seed=args.seed,
        strength=args.strength,
        style_scale=args.style_scale,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
    )


if __name__ == "__main__":
    main()
