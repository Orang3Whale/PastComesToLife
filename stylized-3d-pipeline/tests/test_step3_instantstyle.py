from pathlib import Path

from PIL import Image

from lib.io_paths import create_run_tree, write_json
from scripts.step3_instantstyle import build_instantstyle_command, run_step


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


def test_run_step_requires_stylized_output(tmp_path: Path) -> None:
    paths = create_run_tree(tmp_path / "run")
    Image.new("RGBA", (32, 32), (255, 255, 255, 255)).save(paths.preprocess / "rgba.png")
    style_image = tmp_path / "style.jpg"
    Image.new("RGB", (32, 32), "blue").save(style_image)

    def fake_runner(cmd, env=None):  # noqa: ANN001
        Image.new("RGBA", (32, 32), (255, 0, 0, 255)).save(paths.stylize / "stylized.png")
        write_json(paths.stylize / "stylize_meta.json", {"cmd": cmd})

    result = run_step(
        run_dir=paths.root,
        instantstyle_python=Path("/envs/instantstyle/bin/python"),
        style_image=style_image,
        prompt="ceramic mug",
        runner=fake_runner,
    )
    assert result["stylized_path"].endswith("stylize/stylized.png")
