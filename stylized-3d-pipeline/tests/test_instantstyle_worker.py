from pathlib import Path

import numpy as np
from PIL import Image

from scripts.workers import instantstyle_worker as worker
from scripts.workers.instantstyle_worker import (
    build_pipeline_and_adapter,
    generate_stylized_images,
    prepare_base_image,
    prepare_control_image,
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

    monkeypatch.setattr(worker.ControlNetModel, "from_pretrained", lambda *args, **kwargs: FakeControlNet())
    monkeypatch.setattr(
        worker.StableDiffusionXLControlNetImg2ImgPipeline,
        "from_pretrained",
        lambda *args, **kwargs: FakePipeline(),
    )
    monkeypatch.setattr(worker, "IPAdapterXL", lambda pipe, *args, **kwargs: pipe)

    pipe, ip_model = build_pipeline_and_adapter(device="cpu")

    assert seen["pipeline_device"] == "cpu"
    assert seen["controlnet_device"] == "cpu"
    assert seen["vae_tiling"] is True
    assert pipe is ip_model


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
