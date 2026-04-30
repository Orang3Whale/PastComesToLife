import json
from pathlib import Path

from PIL import Image

from lib.io_paths import create_run_tree
from scripts.step3_instantstyle import build_instantstyle_command, run_step
from scripts.workers.instantstyle_worker import write_worker_outputs


def test_build_instantstyle_command_includes_prompt_and_style(tmp_path: Path) -> None:
    cmd = build_instantstyle_command(
        instantstyle_python=Path("/envs/instantstyle/bin/python"),
        worker_script=Path("/repo/stylized-3d-pipeline/scripts/workers/instantstyle_worker.py"),
        run_dir=tmp_path / "run",
        style_image=Path("/tmp/style.jpg"),
        prompt="ceramic mug",
    )
    assert cmd[0] == "/envs/instantstyle/bin/python"
    assert "/tmp/style.jpg" in cmd
    assert "ceramic mug" in cmd


def test_run_step_copies_inputs_and_writes_metadata(tmp_path: Path) -> None:
    paths = create_run_tree(tmp_path / "run")
    Image.new("RGBA", (32, 32), (255, 255, 255, 255)).save(paths.preprocess / "rgba.png")
    style_image = tmp_path / "style.jpg"
    Image.new("RGB", (32, 32), "blue").save(style_image)

    seen: dict[str, object] = {}

    def fake_runner(cmd, env=None):  # noqa: ANN001
        seen["cmd"] = cmd
        seen["env"] = dict(env or {})
        assert not (paths.stylize / "stylize_meta.json").exists()
        Image.new("RGBA", (32, 32), (255, 0, 0, 255)).save(paths.stylize / "stylized.png")

    result = run_step(
        run_dir=paths.root,
        instantstyle_python=Path("/envs/instantstyle/bin/python"),
        style_image=style_image,
        prompt="ceramic mug",
        runner=fake_runner,
    )

    copied_style = paths.inputs / "style.png"
    prompt_file = paths.inputs / "prompt.txt"
    stylize_meta = json.loads((paths.stylize / "stylize_meta.json").read_text(encoding="utf-8"))

    assert copied_style.is_file()
    assert copied_style.read_bytes() == style_image.read_bytes()
    assert prompt_file.read_text(encoding="utf-8") == "ceramic mug"
    assert seen["env"] == {"HF_ENDPOINT": "https://hf-mirror.com"}
    assert stylize_meta == result
    assert result == {
        "stylized_path": str(paths.stylize / "stylized.png"),
        "style_image": str(copied_style),
        "prompt": "ceramic mug",
    }


def test_write_worker_outputs_writes_stylized_image_and_meta(tmp_path: Path) -> None:
    stylize_dir = tmp_path / "stylize"
    stylize_dir.mkdir()
    stylized_image = Image.new("RGBA", (16, 16), (255, 0, 0, 255))

    metadata = write_worker_outputs(
        stylize_dir=stylize_dir,
        stylized_image=stylized_image,
        preprocess_image=Path("/run/preprocess/rgba.png"),
        style_image=Path("/run/inputs/style.png"),
        prompt="ceramic mug",
    )

    stylized_path = stylize_dir / "stylized.png"
    worker_meta_path = stylize_dir / "worker_meta.json"

    assert stylized_path.is_file()
    with Image.open(stylized_path) as saved_image:
        assert saved_image.mode == "RGBA"
        assert saved_image.size == (16, 16)
    assert worker_meta_path.is_file()
    assert json.loads(worker_meta_path.read_text(encoding="utf-8")) == metadata
    assert metadata == {
        "preprocess_image": "/run/preprocess/rgba.png",
        "style_image": "/run/inputs/style.png",
        "prompt": "ceramic mug",
    }
