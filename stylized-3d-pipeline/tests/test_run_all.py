from pathlib import Path

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
