from pathlib import Path
from types import ModuleType
import sys

import numpy as np
from PIL import Image

from scripts.workers import instantstyle_worker as worker
from scripts.workers.instantstyle_worker import (
    build_pipeline_and_adapter,
    generate_stylized_images,
    prepare_base_image,
    prepare_control_image,
    resolve_sdxl_models_root,
    write_worker_outputs,
)


def test_prepare_base_image_preserves_texture_and_fills_transparency() -> None:
    base = Image.new("RGBA", (2, 2), (0, 0, 0, 0))
    base.putpixel((1, 1), (10, 20, 30, 255))

    prepared = prepare_base_image(base)
    prepared_array = np.asarray(prepared)

    assert prepared.mode == "RGB"
    assert tuple(prepared_array[0, 0]) == (235, 235, 235)
    assert tuple(prepared_array[1, 1]) == (10, 20, 30)


def test_prepare_control_image_masks_transparency_without_canny() -> None:
    control = Image.new("RGBA", (2, 2), (10, 20, 30, 0))
    control.putpixel((1, 1), (10, 20, 30, 255))

    prepared = prepare_control_image(control)
    prepared_array = np.asarray(prepared)

    assert prepared.mode == "RGB"
    assert np.all(prepared_array[0, 0] == 0)
    assert tuple(prepared_array[1, 1]) == (10, 20, 30)
    assert not hasattr(worker, "build_canny_control_map")


def test_build_pipeline_and_adapter_uses_img2img_pipeline(monkeypatch) -> None:
    seen = {}

    class FakeControlNet:
        def to(self, device):  # noqa: ANN001
            seen["controlnet_device"] = device
            return self

    class FakePipeline:
        def to(self, device):  # noqa: ANN001
            seen["pipeline_device"] = device
            return self

        def enable_vae_tiling(self) -> None:
            seen["vae_tiling"] = True

    def fake_controlnet_from_pretrained(model_id, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        seen["controlnet_model_id"] = model_id
        return FakeControlNet()

    monkeypatch.setattr(worker.ControlNetModel, "from_pretrained", fake_controlnet_from_pretrained)
    monkeypatch.setattr(
        worker.StableDiffusionXLControlNetImg2ImgPipeline,
        "from_pretrained",
        lambda *args, **kwargs: FakePipeline(),
    )
    def fake_ip_adapter(pipe, image_encoder_path, ip_ckpt, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        seen["image_encoder_path"] = image_encoder_path
        seen["ip_ckpt"] = ip_ckpt
        return pipe

    monkeypatch.setattr(worker, "IPAdapterXL", fake_ip_adapter)

    pipe, ip_model = build_pipeline_and_adapter(device="cpu")

    assert seen["pipeline_device"] == "cpu"
    assert seen["controlnet_device"] == "cpu"
    assert seen["controlnet_model_id"] == "diffusers/controlnet-canny-sdxl-1.0"
    assert seen["image_encoder_path"].endswith("/sdxl_models/image_encoder")
    assert seen["ip_ckpt"].endswith("/sdxl_models/ip-adapter_sdxl.bin")
    assert seen["vae_tiling"] is True
    assert pipe is ip_model


def test_resolve_sdxl_models_root_prefers_env_override(tmp_path: Path, monkeypatch) -> None:
    model_root = tmp_path / "cache" / "sdxl_models"
    (model_root / "image_encoder").mkdir(parents=True)
    (model_root / "ip-adapter_sdxl.bin").write_bytes(b"model")
    monkeypatch.setenv("INSTANTSTYLE_SDXL_MODELS", str(model_root))
    monkeypatch.delenv("IP_ADAPTER_SDXL_MODELS", raising=False)

    assert resolve_sdxl_models_root(project_root=tmp_path / "project") == model_root


def test_generate_stylized_images_passes_base_image_control_image_and_strength() -> None:
    seen = {}

    class FakeIPAdapter:
        def generate(self, **kwargs):  # noqa: ANN003
            seen.update(kwargs)
            return [Image.new("RGBA", (16, 16), (0, 255, 0, 255))]

    output = generate_stylized_images(
        ip_model=FakeIPAdapter(),
        style=Image.new("RGB", (16, 16), "blue"),
        prompt="ceramic mug",
        base_image=Image.new("RGB", (16, 16), "white"),
        control_image=Image.new("RGB", (16, 16), "black"),
        seed=123,
        strength=0.45,
    )

    assert len(output) == 1
    assert seen["image"].mode == "RGB"
    assert seen["control_image"].mode == "RGB"
    assert seen["strength"] == 0.45


def test_generate_stylized_images_passes_custom_style_parameters() -> None:
    seen = {}

    class FakeIPAdapter:
        def generate(self, **kwargs):  # noqa: ANN003
            seen.update(kwargs)
            return [Image.new("RGBA", (16, 16), (0, 255, 0, 255))]

    output = generate_stylized_images(
        ip_model=FakeIPAdapter(),
        style=Image.new("RGB", (16, 16), "blue"),
        prompt="ceramic mug",
        base_image=Image.new("RGB", (16, 16), "white"),
        control_image=Image.new("RGB", (16, 16), "black"),
        seed=123,
        strength=0.72,
        style_scale=1.8,
        guidance_scale=6.5,
        num_inference_steps=35,
        controlnet_conditioning_scale=0.45,
    )

    assert len(output) == 1
    assert seen["scale"] == 1.8
    assert seen["guidance_scale"] == 6.5
    assert seen["num_inference_steps"] == 35
    assert seen["controlnet_conditioning_scale"] == 0.45


def test_write_worker_outputs_records_dual_inputs_and_strength(tmp_path: Path) -> None:
    output_image = tmp_path / "stylize" / "front" / "stylized.png"
    metadata = write_worker_outputs(
        output_image=output_image,
        stylized_image=Image.new("RGBA", (16, 16), (255, 0, 0, 255)),
        rgb_image=Path("/run/views/front/rgb.png"),
        control_image=Path("/run/views/front/control.png"),
        style_image=Path("/run/inputs/style.png"),
        prompt="ceramic mug",
        seed=7,
        strength=0.45,
    )

    assert output_image.is_file()
    assert metadata["rgb_image"] == "/run/views/front/rgb.png"
    assert metadata["control_image"] == "/run/views/front/control.png"
    assert metadata["strength"] == 0.45


def test_write_worker_outputs_records_custom_style_parameters(tmp_path: Path) -> None:
    output_image = tmp_path / "stylize" / "front" / "stylized.png"
    metadata = write_worker_outputs(
        output_image=output_image,
        stylized_image=Image.new("RGBA", (16, 16), (255, 0, 0, 255)),
        rgb_image=Path("/run/views/front/rgb.png"),
        control_image=Path("/run/views/front/control.png"),
        style_image=Path("/run/inputs/style.png"),
        prompt="ceramic mug",
        seed=7,
        strength=0.72,
        style_scale=1.8,
        guidance_scale=6.5,
        num_inference_steps=35,
        controlnet_conditioning_scale=0.45,
    )

    assert metadata["style_scale"] == 1.8
    assert metadata["guidance_scale"] == 6.5
    assert metadata["num_inference_steps"] == 35
    assert metadata["controlnet_conditioning_scale"] == 0.45


def test_main_parses_custom_style_parameters(tmp_path: Path, monkeypatch) -> None:
    rgb = tmp_path / "rgb.png"
    control = tmp_path / "control.png"
    style = tmp_path / "style.png"
    output = tmp_path / "out.png"
    Image.new("RGBA", (4, 4), (10, 20, 30, 255)).save(rgb)
    Image.new("RGBA", (4, 4), (0, 0, 0, 255)).save(control)
    Image.new("RGB", (4, 4), "blue").save(style)

    captured = {}
    fake_ip_adapter_module = ModuleType("ip_adapter")
    fake_ip_adapter_module.IPAdapterXL = object()
    monkeypatch.setitem(sys.modules, "ip_adapter", fake_ip_adapter_module)
    monkeypatch.setattr(worker, "build_pipeline_and_adapter", lambda device="cuda", ip_adapter_cls=None: (None, object()))

    def fake_generate(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return [Image.new("RGBA", (4, 4), (255, 0, 0, 255))]

    def fake_write_worker_outputs(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return {"output_image": str(output)}

    monkeypatch.setattr(worker, "generate_stylized_images", fake_generate)
    monkeypatch.setattr(worker, "write_worker_outputs", fake_write_worker_outputs)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "instantstyle_worker.py",
            "--run-dir",
            str(tmp_path / "run"),
            "--rgb-image",
            str(rgb),
            "--control-image",
            str(control),
            "--style-image",
            str(style),
            "--prompt",
            "ceramic mug",
            "--output-image",
            str(output),
            "--seed",
            "123",
            "--strength",
            "0.72",
            "--style-scale",
            "1.8",
            "--guidance-scale",
            "6.5",
            "--num-inference-steps",
            "35",
            "--controlnet-conditioning-scale",
            "0.45",
        ],
    )

    worker.main()

    assert captured["style_scale"] == 1.8
    assert captured["guidance_scale"] == 6.5
    assert captured["num_inference_steps"] == 35
    assert captured["controlnet_conditioning_scale"] == 0.45
