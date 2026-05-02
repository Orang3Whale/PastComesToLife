import json
from pathlib import Path

import numpy as np
from PIL import Image

from lib.io_paths import create_run_tree, write_json
from scripts.step3_instantstyle import build_instantstyle_command, run_step
from scripts.workers.instantstyle_worker import build_canny_control_map, write_worker_outputs


def test_build_instantstyle_command_includes_control_output_and_seed(tmp_path: Path) -> None:
    cmd = build_instantstyle_command(
        instantstyle_python=Path("/envs/instantstyle/bin/python"),
        worker_script=Path("/repo/stylized-3d-pipeline/scripts/workers/instantstyle_worker.py"),
        run_dir=tmp_path / "run",
        control_image=Path("/run/views/front/control.png"),
        style_image=Path("/tmp/style.jpg"),
        prompt="ceramic mug",
        output_image=Path("/run/stylize/front/stylized.png"),
        seed=123,
    )
    assert cmd[0] == "/envs/instantstyle/bin/python"
    assert "--control-image" in cmd
    assert "/run/views/front/control.png" in cmd
    assert "--output-image" in cmd
    assert "/run/stylize/front/stylized.png" in cmd
    assert "ceramic mug" in cmd
    assert "--seed" in cmd
    assert "123" in cmd


def test_run_step_writes_per_view_stylized_outputs(tmp_path: Path) -> None:
    paths = create_run_tree(tmp_path / "run")
    write_json(
        paths.views / "manifest.json",
        {
            "views": [
                {
                    "name": "front",
                    "control_path": str(paths.views / "front" / "control.png"),
                    "mask_path": str(paths.views / "front" / "mask.png"),
                },
                {
                    "name": "back",
                    "control_path": str(paths.views / "back" / "control.png"),
                    "mask_path": str(paths.views / "back" / "mask.png"),
                },
                {
                    "name": "left",
                    "control_path": str(paths.views / "left" / "control.png"),
                    "mask_path": str(paths.views / "left" / "mask.png"),
                },
                {
                    "name": "right",
                    "control_path": str(paths.views / "right" / "control.png"),
                    "mask_path": str(paths.views / "right" / "mask.png"),
                },
                {
                    "name": "top",
                    "control_path": str(paths.views / "top" / "control.png"),
                    "mask_path": str(paths.views / "top" / "mask.png"),
                },
                {
                    "name": "bottom",
                    "control_path": str(paths.views / "bottom" / "control.png"),
                    "mask_path": str(paths.views / "bottom" / "mask.png"),
                },
            ]
        },
    )
    for name in ("front", "back", "left", "right", "top", "bottom"):
        view_dir = paths.views / name
        view_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (8, 8), (0, 0, 0, 255)).save(view_dir / "control.png")
        mask = Image.new("L", (8, 8), 0)
        for y in range(2, 6):
            for x in range(2, 6):
                mask.putpixel((x, y), 255)
        mask.save(view_dir / "mask.png")
    style_image = tmp_path / "style.jpg"
    Image.new("RGB", (8, 8), "blue").save(style_image)

    seen: dict[str, object] = {}

    def fake_runner(cmd, env=None):  # noqa: ANN001
        seen["cmd"] = cmd
        seen["env"] = dict(env or {})
        out_index = cmd.index("--output-image") + 1
        Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(Path(cmd[out_index]))

    result = run_step(
        run_dir=paths.root,
        instantstyle_python=Path("/envs/instantstyle/bin/python"),
        style_image=style_image,
        prompt="ceramic mug",
        seed=123,
        runner=fake_runner,
    )

    copied_style = paths.inputs / "style.png"
    prompt_file = paths.inputs / "prompt.txt"
    stylize_manifest = json.loads((paths.stylize / "manifest.json").read_text(encoding="utf-8"))

    assert copied_style.is_file()
    assert copied_style.read_bytes() == style_image.read_bytes()
    assert prompt_file.read_text(encoding="utf-8") == "ceramic mug"
    assert seen["env"] == {"HF_ENDPOINT": "https://hf-mirror.com", "OMP_NUM_THREADS": "1"}
    assert "--seed" in seen["cmd"]
    assert "123" in seen["cmd"]
    assert (paths.stylize / "front" / "stylized.png").is_file()
    assert (paths.stylize / "left" / "stylized.png").is_file()
    assert (paths.stylize / "stylized.png").is_file()
    assert stylize_manifest == result
    assert [name for name in result["views"]] == ["front", "back", "left", "right", "top", "bottom"]
    assert result["views"]["front"]["stylized_path"] == str(paths.stylize / "front" / "stylized.png")
    with Image.open(paths.stylize / "front" / "stylized.png") as stylized:
        assert stylized.mode == "RGBA"
        assert stylized.getpixel((0, 0))[3] == 0
        assert stylized.getpixel((3, 3))[3] == 255


def test_run_step_rejects_empty_view_manifest(tmp_path: Path) -> None:
    paths = create_run_tree(tmp_path / "run")
    write_json(paths.views / "manifest.json", {"views": []})
    style_image = tmp_path / "style.jpg"
    Image.new("RGB", (8, 8), "blue").save(style_image)

    try:
        run_step(
            run_dir=paths.root,
            instantstyle_python=Path("/envs/instantstyle/bin/python"),
            style_image=style_image,
            prompt="ceramic mug",
            seed=123,
            runner=lambda *args, **kwargs: None,  # noqa: ANN001
        )
    except ValueError as exc:
        assert "view" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for empty view manifest")


def test_run_step_rejects_extra_view_in_manifest(tmp_path: Path) -> None:
    paths = create_run_tree(tmp_path / "run")
    write_json(
        paths.views / "manifest.json",
        {
            "views": [
                {
                    "name": "front",
                    "control_path": str(paths.views / "front" / "control.png"),
                    "mask_path": str(paths.views / "front" / "mask.png"),
                },
                {
                    "name": "back",
                    "control_path": str(paths.views / "back" / "control.png"),
                    "mask_path": str(paths.views / "back" / "mask.png"),
                },
                {
                    "name": "left",
                    "control_path": str(paths.views / "left" / "control.png"),
                    "mask_path": str(paths.views / "left" / "mask.png"),
                },
                {
                    "name": "right",
                    "control_path": str(paths.views / "right" / "control.png"),
                    "mask_path": str(paths.views / "right" / "mask.png"),
                },
                {
                    "name": "top",
                    "control_path": str(paths.views / "top" / "control.png"),
                    "mask_path": str(paths.views / "top" / "mask.png"),
                },
                {
                    "name": "bottom",
                    "control_path": str(paths.views / "bottom" / "control.png"),
                    "mask_path": str(paths.views / "bottom" / "mask.png"),
                },
                {
                    "name": "diagonal",
                    "control_path": str(paths.views / "diagonal" / "control.png"),
                    "mask_path": str(paths.views / "diagonal" / "mask.png"),
                },
            ]
        },
    )
    for name in ("front", "back", "left", "right", "top", "bottom", "diagonal"):
        view_dir = paths.views / name
        view_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (8, 8), (0, 0, 0, 255)).save(view_dir / "control.png")
        mask = Image.new("L", (8, 8), 0)
        for y in range(2, 6):
            for x in range(2, 6):
                mask.putpixel((x, y), 255)
        mask.save(view_dir / "mask.png")
    style_image = tmp_path / "style.jpg"
    Image.new("RGB", (8, 8), "blue").save(style_image)

    try:
        run_step(
            run_dir=paths.root,
            instantstyle_python=Path("/envs/instantstyle/bin/python"),
            style_image=style_image,
            prompt="ceramic mug",
            seed=123,
            runner=lambda *args, **kwargs: None,  # noqa: ANN001
        )
    except ValueError as exc:
        assert "canonical" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for extra view in manifest")


def test_generate_stylized_images_uses_seed() -> None:
    from scripts.workers import instantstyle_worker

    seen: dict[str, object] = {}

    class FakeIPAdapter:
        def generate(self, **kwargs):  # noqa: ANN003
            seen.update(kwargs)
            return [Image.new("RGBA", (16, 16), (0, 255, 0, 255))]

    helper = getattr(instantstyle_worker, "generate_stylized_images")
    output = helper(
        ip_model=FakeIPAdapter(),
        style=Image.new("RGB", (16, 16), "blue"),
        prompt="ceramic mug",
        canny_map=Image.new("RGB", (16, 16), "black"),
        seed=123,
    )

    assert len(output) == 1
    assert seen["seed"] == 123
    assert seen["prompt"] == "ceramic mug"


def test_build_canny_control_map_masks_transparent_pixels() -> None:
    class FakeCv2:
        @staticmethod
        def Canny(control_array, low, high):  # noqa: ANN001,ANN003
            return np.full(control_array.shape[:2], 255, dtype=np.uint8)

    control = Image.new("RGBA", (2, 2), (10, 20, 30, 0))
    control.putpixel((1, 1), (10, 20, 30, 255))

    canny = build_canny_control_map(control, FakeCv2())
    canny_array = np.asarray(canny)

    assert np.all(canny_array[0, 0] == 0)
    assert np.all(canny_array[1, 1] == 255)


def test_write_worker_outputs_writes_stylized_image_and_meta(tmp_path: Path) -> None:
    output_image = tmp_path / "stylize" / "front" / "stylized.png"
    stylized_image = Image.new("RGBA", (16, 16), (255, 0, 0, 255))

    metadata = write_worker_outputs(
        output_image=output_image,
        stylized_image=stylized_image,
        control_image=Path("/run/views/front/control.png"),
        style_image=Path("/run/inputs/style.png"),
        prompt="ceramic mug",
        seed=7,
    )

    stylized_path = output_image
    worker_meta_path = output_image.parent / "worker_meta.json"

    assert stylized_path.is_file()
    with Image.open(stylized_path) as saved_image:
        assert saved_image.mode == "RGBA"
        assert saved_image.size == (16, 16)
    assert worker_meta_path.is_file()
    assert json.loads(worker_meta_path.read_text(encoding="utf-8")) == metadata
    assert metadata == {
        "control_image": "/run/views/front/control.png",
        "style_image": "/run/inputs/style.png",
        "prompt": "ceramic mug",
        "seed": 7,
        "output_image": str(output_image),
    }
