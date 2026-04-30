from pathlib import Path

from lib.io_paths import create_run_tree, resolve_run_dir, write_json


def test_create_run_tree_creates_expected_directories(tmp_path: Path) -> None:
    run_dir = resolve_run_dir(tmp_path, "demo-mug")
    paths = create_run_tree(run_dir)

    assert paths.root == run_dir
    assert paths.inputs.is_dir()
    assert paths.preprocess.is_dir()
    assert paths.sf3d.is_dir()
    assert paths.stylize.is_dir()
    assert paths.retexture.is_dir()
    assert paths.viewer.is_dir()


def test_write_json_persists_payload(tmp_path: Path) -> None:
    out_path = tmp_path / "meta.json"
    write_json(out_path, {"step": "preprocess", "ok": True})
    assert out_path.read_text(encoding="utf-8") == '{\n  "ok": true,\n  "step": "preprocess"\n}'
