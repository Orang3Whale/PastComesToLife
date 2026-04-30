from pathlib import Path

from lib.io_paths import create_run_tree, write_json
from scripts.step2_sf3d import build_sf3d_command, run_step


def test_build_sf3d_command_targets_worker(tmp_path: Path) -> None:
    cmd = build_sf3d_command(
        sf3d_python=Path("/envs/sf3d/bin/python"),
        worker_script=Path("/repo/stylized-3d-pipeline/scripts/workers/sf3d_worker.py"),
        run_dir=tmp_path / "run",
        texture_resolution=1024,
        remesh_option="none",
    )
    assert cmd[0] == "/envs/sf3d/bin/python"
    assert cmd[1].endswith("sf3d_worker.py")
    assert "--texture-resolution" in cmd


def test_run_step_validates_mesh_output(tmp_path: Path) -> None:
    paths = create_run_tree(tmp_path / "run")
    (paths.preprocess / "rgba.png").write_bytes(b"rgba")
    fake_mesh = paths.sf3d / "mesh_raw.glb"

    def fake_runner(cmd, env=None):  # noqa: ANN001
        fake_mesh.write_bytes(b"glb")
        write_json(paths.sf3d / "sf3d_meta.json", {"cmd": cmd})

    result = run_step(
        run_dir=paths.root,
        sf3d_python=Path("/envs/sf3d/bin/python"),
        runner=fake_runner,
    )
    assert result["mesh_path"].endswith("sf3d/mesh_raw.glb")
