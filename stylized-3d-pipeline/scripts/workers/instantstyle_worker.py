from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def find_upstream_repo_root(anchor: Path | None = None) -> Path:
    current = (anchor or Path(__file__)).resolve()
    for parent in (current.parent, *current.parents):
        candidate = parent / "InstantStyle"
        if candidate.is_dir():
            return parent
    raise FileNotFoundError(f"could not locate InstantStyle relative to {current}")


def build_worker_meta(
    control_image: Path,
    style_image: Path,
    prompt: str,
    seed: int,
    output_image: Path,
) -> dict[str, str | int]:
    return {
        "control_image": str(control_image),
        "style_image": str(style_image),
        "prompt": prompt,
        "seed": seed,
        "output_image": str(output_image),
    }


def generate_stylized_images(
    ip_model: object,
    style: Image.Image,
    prompt: str,
    canny_map: Image.Image,
    seed: int,
) -> object:
    return ip_model.generate(
        pil_image=style,
        prompt=prompt,
        negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, blurry",
        scale=1.0,
        guidance_scale=5.0,
        num_samples=1,
        num_inference_steps=30,
        seed=seed,
        image=canny_map,
        controlnet_conditioning_scale=0.6,
    )


def build_canny_control_map(control: Image.Image, cv2_module: object) -> Image.Image:
    control_rgba = control.convert("RGBA")
    control_array = np.asarray(control_rgba)
    canny = cv2_module.Canny(control_array[:, :, :3], 50, 200)
    canny = np.asarray(canny, dtype=np.uint8)
    canny[control_array[:, :, 3] < 1] = 0
    return Image.fromarray(canny).convert("RGB")


def write_worker_outputs(
    output_image: Path,
    stylized_image: Image.Image,
    control_image: Path,
    style_image: Path,
    prompt: str,
    seed: int,
) -> dict[str, str | int]:
    output_image.parent.mkdir(parents=True, exist_ok=True)
    stylized_image.save(output_image)
    metadata = build_worker_meta(control_image, style_image, prompt, seed, output_image)
    (output_image.parent / "worker_meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--control-image", required=True, type=Path)
    parser.add_argument("--style-image", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-image", required=True, type=Path)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    root = find_upstream_repo_root(Path(__file__))
    instantstyle_root = root / "InstantStyle"
    if str(instantstyle_root) not in sys.path:
        sys.path.insert(0, str(instantstyle_root))

    import cv2  # noqa: WPS433
    import torch  # noqa: WPS433
    from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline  # noqa: WPS433
    from ip_adapter import IPAdapterXL  # noqa: WPS433,E402

    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        use_safetensors=False,
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        add_watermarker=False,
    ).to("cuda")
    pipe.enable_vae_tiling()
    ip_model = IPAdapterXL(
        pipe,
        "sdxl_models/image_encoder",
        "sdxl_models/ip-adapter_sdxl.bin",
        "cuda",
        target_blocks=["up_blocks.0.attentions.1"],
    )

    with Image.open(args.control_image) as control_image:
        control = control_image.convert("RGBA")
    with Image.open(args.style_image) as style_image:
        style = style_image.convert("RGB")

    canny_map = build_canny_control_map(control, cv2)

    images = generate_stylized_images(
        ip_model=ip_model,
        style=style,
        prompt=args.prompt,
        canny_map=canny_map,
        seed=args.seed,
    )

    stylized = images[0].convert("RGBA").resize(control.size)
    write_worker_outputs(
        output_image=args.output_image,
        stylized_image=stylized,
        control_image=args.control_image,
        style_image=args.style_image,
        prompt=args.prompt,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
