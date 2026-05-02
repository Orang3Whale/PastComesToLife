import json
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

from lib.camera_views import build_six_view_spec
from lib.io_paths import create_run_tree
from lib.view_sampling import render_view_assets
from scripts.step3_sample_views import _load_mesh, run_step


def test_build_six_view_spec_uses_canonical_axes() -> None:
    mesh = trimesh.creation.box(bounds=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    views = build_six_view_spec(mesh, camera_distance=2.0, fovy_deg=40.0)

    assert [view.name for view in views] == [
        "front",
        "back",
        "left",
        "right",
        "top",
        "bottom",
    ]
    assert views[0].pose.shape == (4, 4)
    assert np.isfinite(views[0].pose).all()


def test_build_six_view_spec_fits_elongated_mesh() -> None:
    mesh = trimesh.creation.box(bounds=np.array([[-2.0, -1.0, -1.0], [2.0, 1.0, 1.0]]))
    views = build_six_view_spec(mesh, camera_distance=0.5, fovy_deg=40.0)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)

    for view in views:
        world_to_camera = np.linalg.inv(view.pose)
        homogenous = np.concatenate([vertices, np.ones((len(vertices), 1), dtype=np.float32)], axis=1)
        camera = (world_to_camera @ homogenous.T).T[:, :3]
        z = -camera[:, 2]
        scale = max(float(np.tan(np.deg2rad(view.fovy_deg) * 0.5)), 1e-6)
        x_ndc = camera[:, 0] / (scale * z)
        y_ndc = camera[:, 1] / (scale * z)

        assert np.all(z > 0.0)
        assert np.max(np.abs(x_ndc)) <= 1.0 + 1e-5
        assert np.max(np.abs(y_ndc)) <= 1.0 + 1e-5


def test_run_step_writes_view_manifest_and_assets(tmp_path: Path) -> None:
    paths = create_run_tree(tmp_path / "run")
    mesh = trimesh.creation.box()
    mesh.export(paths.sf3d / "mesh_raw.glb", include_normals=True)

    def fake_renderer(mesh, views, resolution):  # noqa: ANN001
        payload = {}
        for view in views:
            payload[view.name] = {
                "rgb": Image.new("RGBA", (16, 16), (10, 20, 30, 255)),
                "depth": np.ones((16, 16), dtype=np.float32),
                "depth_preview": Image.new("RGBA", (16, 16), (40, 50, 60, 255)),
                "normal": Image.new("RGBA", (16, 16), (120, 130, 140, 255)),
                "mask": Image.new("L", (16, 16), 255),
                "control": Image.new("RGBA", (16, 16), (70, 80, 90, 255)),
                "camera": {"name": view.name, "pose": view.pose.tolist(), "fovy_deg": view.fovy_deg},
            }
        return payload

    result = run_step(paths.root, 16, 2.0, 40.0, renderer=fake_renderer)

    manifest_path = paths.views / "manifest.json"
    front_rgb_path = paths.views / "front" / "rgb.png"

    assert manifest_path.is_file()
    assert front_rgb_path.is_file()
    assert result["view_resolution"] == 16
    assert [entry["name"] for entry in result["views"]] == [
        "front",
        "back",
        "left",
        "right",
        "top",
        "bottom",
    ]
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == result


def test_load_mesh_preserves_scene_transforms(tmp_path: Path) -> None:
    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.creation.box(),
        transform=trimesh.transformations.translation_matrix([2.0, 0.0, 0.0]),
    )
    mesh_path = tmp_path / "mesh.glb"
    scene.export(mesh_path)

    loaded = _load_mesh(mesh_path)

    assert np.allclose(loaded.bounds, np.array([[1.5, -0.5, -0.5], [2.5, 0.5, 0.5]]))


def test_render_view_assets_produces_nonempty_mask_and_depth() -> None:
    mesh = trimesh.creation.box()
    views = build_six_view_spec(mesh, camera_distance=2.0, fovy_deg=40.0)

    assets = render_view_assets(mesh, views, resolution=32)

    front = assets["front"]
    mask = np.asarray(front["mask"], dtype=np.uint8)
    depth = front["depth"]

    assert set(assets) == {"front", "back", "left", "right", "top", "bottom"}
    assert mask.max() == 255
    assert np.any(depth > 0.0)
    assert front["camera"]["name"] == "front"
