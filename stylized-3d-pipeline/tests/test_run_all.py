from pathlib import Path
import sys

from scripts import run_all

from lib.io_paths import create_run_tree
from lib.pipeline_runner import ordered_steps, should_run_step


def test_ordered_steps_matches_pipeline_contract() -> None:
    assert ordered_steps() == [
        "preprocess",
        "sf3d",
        "instantstyle",
        "retexture",
        "viewer",
    ]


def test_should_run_step_respects_resume_and_skip_existing(tmp_path: Path) -> None:
    paths = create_run_tree(tmp_path / "run")
    (paths.preprocess / "meta.json").write_text("{}", encoding="utf-8")

    assert should_run_step("preprocess", resume_from=None, skip_existing=True, run_dir=paths.root) is False
    assert should_run_step("sf3d", resume_from="sf3d", skip_existing=False, run_dir=paths.root) is True
    assert should_run_step("preprocess", resume_from="sf3d", skip_existing=False, run_dir=paths.root) is False


def test_run_all_main_parses_cli_and_orchestrates(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "content.jpg"
    style_path = tmp_path / "style.jpg"
    input_path.write_text("content", encoding="utf-8")
    style_path.write_text("style", encoding="utf-8")
    runs_root = tmp_path / "runs"
    resolved_run_dir = runs_root / "demo-mug"
    call_log: list[tuple[str, object]] = []

    def fake_resolve_run_dir(base_dir: Path, run_name: str | None) -> Path:
        call_log.append(("resolve_run_dir", base_dir, run_name))
        return resolved_run_dir

    def fake_create_run_tree(run_dir: Path) -> object:
        call_log.append(("create_run_tree", run_dir))
        return object()

    def fake_write_run_config(run_dir: Path, args: object) -> Path:
        call_log.append(
            (
                "write_run_config",
                run_dir,
                getattr(args, "run_dir"),
                getattr(args, "input"),
                getattr(args, "style_image"),
                getattr(args, "prompt"),
                getattr(args, "sf3d_python"),
                getattr(args, "instantstyle_python"),
                getattr(args, "foreground_ratio"),
                getattr(args, "texture_resolution"),
                getattr(args, "remesh_option"),
                getattr(args, "resume_from"),
                getattr(args, "skip_existing"),
            )
        )
        config_path = run_dir / "run_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("{}", encoding="utf-8")
        return config_path

    def fake_run_pipeline(args: object) -> dict[str, dict]:
        call_log.append(("run_pipeline", getattr(args, "run_dir"), getattr(args, "prompt")))
        return {"ok": True}

    monkeypatch.setattr(run_all, "resolve_run_dir", fake_resolve_run_dir)
    monkeypatch.setattr(run_all, "create_run_tree", fake_create_run_tree)
    monkeypatch.setattr(run_all, "write_run_config", fake_write_run_config)
    monkeypatch.setattr(run_all, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_all.py",
            "--input",
            str(input_path),
            "--style-image",
            str(style_path),
            "--prompt",
            "ceramic mug",
            "--run-name",
            "demo-mug",
            "--runs-root",
            str(runs_root),
            "--sf3d-python",
            "/opt/sf3d/bin/python",
            "--instantstyle-python",
            "/opt/instantstyle/bin/python",
            "--foreground-ratio",
            "0.75",
            "--texture-resolution",
            "2048",
            "--remesh-option",
            "triangle",
            "--resume-from",
            "sf3d",
            "--skip-existing",
        ],
    )

    run_all.main()

    assert call_log == [
        ("resolve_run_dir", runs_root, "demo-mug"),
        ("create_run_tree", resolved_run_dir),
        (
            "write_run_config",
            resolved_run_dir,
            resolved_run_dir,
            input_path,
            style_path,
            "ceramic mug",
            Path("/opt/sf3d/bin/python"),
            Path("/opt/instantstyle/bin/python"),
            0.75,
            2048,
            "triangle",
            "sf3d",
            True,
        ),
        ("run_pipeline", resolved_run_dir, "ceramic mug"),
    ]
    assert (resolved_run_dir / "run_config.json").is_file()
