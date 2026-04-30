from pathlib import Path
import json

from PIL import Image, ImageDraw

from lib.io_paths import create_run_tree
from scripts.step1_preprocess import run_step


def _fake_remove_background(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = Image.new("L", rgba.size, 0)
    draw = ImageDraw.Draw(alpha)
    draw.rectangle((16, 16, 47, 47), fill=255)
    rgba.putalpha(alpha)
    return rgba


def test_run_step_writes_rgba_mask_and_metadata(tmp_path: Path) -> None:
    src = tmp_path / "content.jpg"
    Image.new("RGB", (64, 64), "white").save(src)

    paths = create_run_tree(tmp_path / "run")
    result = run_step(
        input_path=src,
        run_dir=paths.root,
        foreground_ratio=0.85,
        remove_background_fn=_fake_remove_background,
    )

    assert result["rgba_path"].endswith("preprocess/rgba.png")

    content_image = Image.open(paths.inputs / "content.png")
    rgba_image = Image.open(paths.preprocess / "rgba.png")
    mask_image = Image.open(paths.preprocess / "mask.png")
    meta = json.loads((paths.preprocess / "meta.json").read_text(encoding="utf-8"))

    assert content_image.format == "PNG"
    assert content_image.size == (64, 64)
    assert content_image.getpixel((0, 0)) == (255, 255, 255)

    assert rgba_image.format == "PNG"
    assert rgba_image.mode == "RGBA"
    assert rgba_image.size == (64, 64)
    assert rgba_image.getchannel("A").getbbox() is not None

    assert mask_image.format == "PNG"
    assert mask_image.mode == "L"
    assert mask_image.size == rgba_image.size
    assert mask_image.getbbox() is not None

    assert meta["input_path"] == str(src)
    assert meta["rgba_path"] == str(paths.preprocess / "rgba.png")
    assert meta["mask_path"] == str(paths.preprocess / "mask.png")
    assert meta["foreground_ratio"] == 0.85
