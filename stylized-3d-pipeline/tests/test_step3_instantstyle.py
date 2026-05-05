import json
from pathlib import Path

from PIL import Image

from lib.io_paths import create_run_tree, write_json
from lib.subprocess_utils import huggingface_cache_env
from scripts.step3_instantstyle import build_instantstyle_command, run_step


def test_build_instantstyle_command_includes_rgb_control_and_strength(tmp_path: Path) -> None:
    cmd = build_instantstyle_command(
        instantstyle_python=Path("/envs/instantstyle/bin/python"),
        worker_script=Path("/repo/stylized-3d-pipeline/scripts/workers/instantstyle_worker.py"),
        run_dir=tmp_path / "run",
        rgb_image=Path("/run/views/front/rgb.png"),
        control_image=Path("/run/views/front/control.png"),
        style_image=Path("/tmp/style.jpg"),
        prompt="ceramic mug",
        output_image=Path("/run/stylize/front/stylized.png"),
        seed=123,
        strength=0.45,
    )
    assert cmd[0] == "/envs/instantstyle/bin/python"
    assert "--rgb-image" in cmd
    assert "/run/views/front/rgb.png" in cmd
    assert "--control-image" in cmd
    assert "/run/views/front/control.png" in cmd
    assert "--strength" in cmd
    assert "0.45" in cmd
    assert "--output-image" in cmd
    assert "/run/stylize/front/stylized.png" in cmd
    assert "ceramic mug" in cmd
    assert "--seed" in cmd
    assert "123" in cmd


def test_run_step_writes_rgb_control_and_strength_into_manifest(tmp_path: Path) -> None:
    paths = create_run_tree(tmp_path / "run")
    write_json(
        paths.views / "manifest.json",
        {
            "views": [
                {
                    "name": "front",
                    "rgb_path": str(paths.views / "front" / "rgb.png"),
                    "control_path": str(paths.views / "front" / "control.png"),
                    "mask_path": str(paths.views / "front" / "mask.png"),
                },
                {
                    "name": "back",
                    "rgb_path": str(paths.views / "back" / "rgb.png"),
                    "control_path": str(paths.views / "back" / "control.png"),
                    "mask_path": str(paths.views / "back" / "mask.png"),
                },
                {
                    "name": "left",
                    "rgb_path": str(paths.views / "left" / "rgb.png"),
                    "control_path": str(paths.views / "left" / "control.png"),
                    "mask_path": str(paths.views / "left" / "mask.png"),
                },
                {
                    "name": "right",
                    "rgb_path": str(paths.views / "right" / "rgb.png"),
                    "control_path": str(paths.views / "right" / "control.png"),
                    "mask_path": str(paths.views / "right" / "mask.png"),
                },
                {
                    "name": "top",
                    "rgb_path": str(paths.views / "top" / "rgb.png"),
                    "control_path": str(paths.views / "top" / "control.png"),
                    "mask_path": str(paths.views / "top" / "mask.png"),
                },
                {
                    "name": "bottom",
                    "rgb_path": str(paths.views / "bottom" / "rgb.png"),
                    "control_path": str(paths.views / "bottom" / "control.png"),
                    "mask_path": str(paths.views / "bottom" / "mask.png"),
                },
            ]
        },
    )
    for name in ("front", "back", "left", "right", "top", "bottom"):
        view_dir = paths.views / name
        view_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (8, 8), (64, 128, 192, 255)).save(view_dir / "rgb.png")
        Image.new("RGBA", (8, 8), (0, 0, 0, 255)).save(view_dir / "control.png")
        mask = Image.new("L", (8, 8), 0)
        for y in range(2, 6):
            for x in range(2, 6):
                mask.putpixel((x, y), 255)
        mask.save(view_dir / "mask.png")
    style_image = tmp_path / "style.jpg"
    Image.new("RGB", (8, 8), "blue").save(style_image)

    seen: dict[str, object] = {}
    call_count = 0

    def fake_runner(cmd, env=None):  # noqa: ANN001
        nonlocal call_count
        call_count += 1
        seen["cmd"] = cmd
        seen["env"] = dict(env or {})
        jobs_manifest = Path(cmd[cmd.index("--jobs-manifest") + 1])
        payload = json.loads(jobs_manifest.read_text(encoding="utf-8"))
        for job in payload["jobs"]:
            Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(Path(job["output_image"]))

    result = run_step(
        run_dir=paths.root,
        instantstyle_python=Path("/envs/instantstyle/bin/python"),
        style_image=style_image,
        prompt="ceramic mug",
        seed=123,
        strength=0.45,
        runner=fake_runner,
    )

    copied_style = paths.inputs / "style.png"
    prompt_file = paths.inputs / "prompt.txt"
    stylize_manifest = json.loads((paths.stylize / "manifest.json").read_text(encoding="utf-8"))

    assert copied_style.is_file()
    assert copied_style.read_bytes() == style_image.read_bytes()
    assert prompt_file.read_text(encoding="utf-8") == "ceramic mug"
    assert seen["env"] == {
        **huggingface_cache_env(),
        "HF_ENDPOINT": "https://hf-mirror.com",
        "OMP_NUM_THREADS": "1",
    }
    assert call_count == 1
    assert "--jobs-manifest" in seen["cmd"]
    jobs_manifest = json.loads((paths.stylize / "worker_jobs.json").read_text(encoding="utf-8"))
    assert len(jobs_manifest["jobs"]) == 6
    assert (paths.stylize / "front" / "stylized.png").is_file()
    assert (paths.stylize / "left" / "stylized.png").is_file()
    assert (paths.stylize / "stylized.png").is_file()
    assert stylize_manifest == result
    assert [name for name in result["views"]] == ["front", "back", "left", "right", "top", "bottom"]
    assert result["strength"] == 0.45
    assert result["views"]["front"]["rgb_path"] == str(paths.views / "front" / "rgb.png")
    assert result["views"]["front"]["control_path"] == str(paths.views / "front" / "control.png")
    assert result["views"]["front"]["stylized_path"] == str(paths.stylize / "front" / "stylized.png")
    with Image.open(paths.stylize / "front" / "stylized.png") as stylized:
        assert stylized.mode == "RGBA"
        assert stylized.getpixel((0, 0))[3] == 0
        assert stylized.getpixel((3, 3))[3] == 255


def test_run_step_rejects_missing_view_asset_before_worker(tmp_path: Path) -> None:
    paths = create_run_tree(tmp_path / "run")
    write_json(
        paths.views / "manifest.json",
        {
            "views": [
                {
                    "name": name,
                    "rgb_path": str(paths.views / name / "rgb.png"),
                    "control_path": str(paths.views / name / "control.png"),
                    "mask_path": str(paths.views / name / "mask.png"),
                }
                for name in ("front", "back", "left", "right", "top", "bottom")
            ]
        },
    )
    style_image = tmp_path / "style.jpg"
    Image.new("RGB", (8, 8), "blue").save(style_image)
    called = False

    def fake_runner(*args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        nonlocal called
        called = True

    try:
        run_step(
            run_dir=paths.root,
            instantstyle_python=Path("/envs/instantstyle/bin/python"),
            style_image=style_image,
            prompt="ceramic mug",
            seed=123,
            runner=fake_runner,
        )
    except FileNotFoundError as exc:
        assert "missing view asset" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError for missing view asset")

    assert called is False


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
