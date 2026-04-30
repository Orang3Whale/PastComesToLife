from pathlib import Path

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
    assert (paths.preprocess / "rgba.png").is_file()
    assert (paths.preprocess / "mask.png").is_file()
    assert (paths.preprocess / "meta.json").is_file()
