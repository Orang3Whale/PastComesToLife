from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from PIL import Image


def find_upstream_repo_root(anchor: Path | None = None) -> Path:
    current = (anchor or Path(__file__)).resolve()
    for parent in (current.parent, *current.parents):
        candidate = parent / "InstantStyle"
        if candidate.is_dir():
            return parent
    raise FileNotFoundError(f"could not locate InstantStyle relative to {current}")


ROOT = find_upstream_repo_root(Path(__file__))
INSTANTSTYLE_ROOT = ROOT / "InstantStyle"
if str(INSTANTSTYLE_ROOT) not in sys.path:
    sys.path.insert(0, str(INSTANTSTYLE_ROOT))

from ip_adapter import IPAdapterXL  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--style-image", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    preprocess_image = args.run_dir / "preprocess" / "rgba.png"
    stylize_dir = args.run_dir / "stylize"
    stylize_dir.mkdir(parents=True, exist_ok=True)

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

    with Image.open(preprocess_image) as content_image:
        content = content_image.convert("RGBA")
        content_rgb = np.array(content.convert("RGB"))
    with Image.open(args.style_image) as style_image:
        style = style_image.convert("RGB")

    canny = cv2.Canny(content_rgb, 50, 200)
    canny_map = Image.fromarray(canny).convert("RGB")

    images = ip_model.generate(
        pil_image=style,
        prompt=args.prompt,
        negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, blurry",
        scale=1.0,
        guidance_scale=5.0,
        num_samples=1,
        num_inference_steps=30,
        seed=42,
        image=canny_map,
        controlnet_conditioning_scale=0.6,
    )

    stylized = images[0].convert("RGBA").resize(content.size)
    stylized.putalpha(content.getchannel("A"))
    stylized.save(stylize_dir / "stylized.png")
    (stylize_dir / "worker_meta.json").write_text(
        json.dumps(
            {
                "preprocess_image": str(preprocess_image),
                "style_image": str(args.style_image),
                "prompt": args.prompt,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
