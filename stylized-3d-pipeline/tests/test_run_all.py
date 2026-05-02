from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace

from scripts import run_all

from lib import pipeline_runner
from lib.io_paths import create_run_tree
from lib.pipeline_runner import StepMeta, ordered_steps, should_run_step


def test_ordered_steps_matches_pipeline_contract() -> None:
    assert ordered_steps() == [
        "preprocess",
        "sf3d",
        "sample_views",
        "instantstyle",
        "retexture",
        "viewer",
    ]


def test_should_run_step_respects_resume_and_skip_existing(tmp_path: Path) -> None:
    paths = create_run_tree(tmp_path / "run")
    (paths.preprocess / "meta.json").write_text("{}", encoding="utf-8")
    (paths.views / "manifest.json").write_text("{}", encoding="utf-8")

    assert should_run_step("preprocess", resume_from=None, skip_existing=True, run_dir=paths.root) is False
    assert should_run_step("sf3d", resume_from="sf3d", skip_existing=False, run_dir=paths.root) is True
    assert should_run_step("sample_views", resume_from=None, skip_existing=True, run_dir=paths.root) is False
    assert should_run_step("preprocess", resume_from="sf3d", skip_existing=False, run_dir=paths.root) is False


def test_sample_views_meta_points_to_manifest_json(tmp_path: Path) -> None:
    assert pipeline_runner.STEP_META["sample_views"].meta_path(tmp_path / "run") == tmp_path / "run" / "views" / "manifest.json"


def test_step_kwargs_for_instantstyle_include_seed(tmp_path: Path) -> None:
    args = type(
        "Args",
        (),
        {
            "instantstyle_python": Path("/opt/instantstyle/bin/python"),
            "style_image": Path("/tmp/style.jpg"),
            "prompt": "ceramic mug",
            "seed": 42,
        },
    )()

    assert pipeline_runner._step_kwargs("instantstyle", args, tmp_path / "run") == {
        "run_dir": tmp_path / "run",
        "instantstyle_python": Path("/opt/instantstyle/bin/python"),
        "style_image": Path("/tmp/style.jpg"),
        "prompt": "ceramic mug",
        "seed": 42,
    }


def test_run_pipeline_forwards_seed_and_sample_views_in_order(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_runner(name: str):
        def _runner(**kwargs):  # noqa: ANN003
            calls.append((name, dict(kwargs)))
            return {"step": name}

        return _runner

    custom_step_meta = {
        "preprocess": StepMeta("preprocess", lambda path: path / "preprocess" / "meta.json", fake_runner("preprocess")),
        "sf3d": StepMeta("sf3d", lambda path: path / "sf3d" / "sf3d_meta.json", fake_runner("sf3d")),
        "sample_views": StepMeta("sample_views", lambda path: path / "views" / "manifest.json", fake_runner("sample_views")),
        "instantstyle": StepMeta("instantstyle", lambda path: path / "stylize" / "stylize_meta.json", fake_runner("instantstyle")),
        "retexture": StepMeta("retexture", lambda path: path / "retexture" / "retexture_meta.json", fake_runner("retexture")),
        "viewer": StepMeta("viewer", lambda path: path / "viewer" / "viewer_meta.json", fake_runner("viewer")),
    }
    monkeypatch.setattr(pipeline_runner, "STEP_META", custom_step_meta)

    args = SimpleNamespace(
        run_dir=run_dir,
        input=tmp_path / "content.jpg",
        style_image=tmp_path / "style.jpg",
        prompt="ceramic mug",
        run_name="demo",
        runs_root=tmp_path / "runs",
        sf3d_python=Path("/opt/sf3d/bin/python"),
        instantstyle_python=Path("/opt/instantstyle/bin/python"),
        foreground_ratio=0.75,
        texture_resolution=2048,
        remesh_option="triangle",
        view_resolution=512,
        camera_distance=1.8,
        camera_fovy_deg=40.0,
        seed=123,
        resume_from=None,
        skip_existing=False,
    )

    results = pipeline_runner.run_pipeline(args)

    assert [name for name, _ in calls] == [
        "preprocess",
        "sf3d",
        "sample_views",
        "instantstyle",
        "retexture",
        "viewer",
    ]
    assert calls[2][1] == {
        "run_dir": run_dir,
        "view_resolution": 512,
        "camera_distance": 1.8,
        "camera_fovy_deg": 40.0,
    }
    assert calls[3][1]["seed"] == 123
    assert results["sample_views"] == {"step": "sample_views"}


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
                getattr(args, "view_resolution"),
                getattr(args, "camera_distance"),
                getattr(args, "camera_fovy_deg"),
                getattr(args, "seed"),
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
            "--view-resolution",
            "768",
            "--camera-distance",
            "2.25",
            "--camera-fovy-deg",
            "55.0",
            "--seed",
            "123",
            "--resume-from",
            "sample_views",
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
            768,
            2.25,
            55.0,
            123,
            "sample_views",
            True,
        ),
        ("run_pipeline", resolved_run_dir, "ceramic mug"),
    ]
    assert (resolved_run_dir / "run_config.json").is_file()


def test_documented_cli_entrypoints_support_help_from_project_root() -> None:
    project_root = Path(__file__).resolve().parents[1]
    scripts = [
        "scripts/run_all.py",
        "scripts/step1_preprocess.py",
        "scripts/step2_sf3d.py",
        "scripts/step3_sample_views.py",
        "scripts/step3_instantstyle.py",
        "scripts/step4_retexture.py",
        "scripts/step5_build_viewer.py",
    ]

    for script in scripts:
        completed = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout.lower()
