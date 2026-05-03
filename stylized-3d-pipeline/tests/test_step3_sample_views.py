import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
import trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from lib.camera_views import CameraView, build_six_view_spec, look_at
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


def test_render_view_assets_derives_outputs_from_offscreen_alpha(monkeypatch: pytest.MonkeyPatch) -> None:
    vertices = np.array(
        [
            [0.0, -0.5, -0.5],
            [0.0, 0.5, -0.5],
            [0.0, -0.5, 0.5],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = TextureVisuals(uv=uv, material=PBRMaterial(baseColorTexture=Image.new("RGBA", (4, 4), (0, 0, 0, 255))))

    view = CameraView(
        name="front",
        pose=look_at(
            np.array([2.0, 0.0, 0.0], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        ),
        fovy_deg=40.0,
    )
    def fake_render_offscreen_views(mesh, views, resolution):  # noqa: ANN001
        rgb = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        for y in range(8, 24):
            for x in range(8, 24):
                rgb.putpixel((x, y), (220, 220, 220, 255))
        depth = np.zeros((32, 32), dtype=np.float32)
        depth[8:24, 8:24] = 1.0
        return {
            view.name: {
                "rgb": rgb,
                "depth": depth,
                "camera": {"name": view.name, "pose": view.pose.tolist(), "fovy_deg": view.fovy_deg},
            }
            for view in views
        }

    monkeypatch.setattr("lib.view_sampling.render_offscreen_views", fake_render_offscreen_views)

    assets = render_view_assets(mesh, [view], resolution=32)

    rgb = np.asarray(assets["front"]["rgb"].convert("RGBA"))
    depth_preview = np.asarray(assets["front"]["depth_preview"].convert("RGBA"))
    normal = np.asarray(assets["front"]["normal"].convert("RGBA"))
    control = np.asarray(assets["front"]["control"].convert("RGBA"))
    mask = np.asarray(assets["front"]["mask"])
    visible = mask > 0

    assert visible.any()
    assert np.all(rgb[visible, 3] == 255)
    assert np.any(rgb[visible, :3].sum(axis=1) > 0)
    assert np.all(rgb[~visible, 3] == 0)
    assert np.all(depth_preview[~visible, 3] == 0)
    assert np.all(normal[~visible, 3] == 0)
    assert np.all(control[~visible, :3] == 0)
    assert np.all(control[~visible, 3] == 0)


def test_run_step_marks_mesh_offscreen_render_mode(tmp_path: Path) -> None:
    paths = create_run_tree(tmp_path / "run")
    mesh = trimesh.creation.box()
    mesh.export(paths.sf3d / "mesh_raw.glb", include_normals=True)

    def fake_renderer(mesh, views, resolution):  # noqa: ANN001
        payload = {}
        for view in views:
            rgb = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
            for y in range(4, 12):
                for x in range(4, 12):
                    rgb.putpixel((x, y), (220, 220, 220, 255))
            payload[view.name] = {
                "rgb": rgb,
                "depth": np.ones((16, 16), dtype=np.float32),
                "depth_preview": Image.new("RGBA", (16, 16), (40, 50, 60, 255)),
                "normal": Image.new("RGBA", (16, 16), (120, 130, 140, 255)),
                "mask": Image.new("L", (16, 16), 255),
                "control": Image.new("RGBA", (16, 16), (70, 80, 90, 255)),
                "camera": {"name": view.name, "pose": view.pose.tolist(), "fovy_deg": view.fovy_deg},
            }
        return payload

    result = run_step(paths.root, 16, 2.0, 40.0, renderer=fake_renderer)

    manifest = json.loads((paths.views / "manifest.json").read_text(encoding="utf-8"))

    assert result["render_mode"] == "mesh_offscreen"
    assert manifest["render_mode"] == "mesh_offscreen"
    assert manifest == result
    with Image.open(paths.views / "front" / "rgb.png") as front_rgb:
        assert front_rgb.mode == "RGBA"
        assert front_rgb.getpixel((0, 0))[3] == 0
        assert front_rgb.getpixel((6, 6))[3] == 255


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


def test_render_view_assets_produces_nonempty_mask_and_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    mesh = trimesh.creation.box()
    views = build_six_view_spec(mesh, camera_distance=2.0, fovy_deg=40.0)

    def fake_render_offscreen_views(mesh, views, resolution):  # noqa: ANN001
        payload = {}
        for view in views:
            rgb = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
            for y in range(8, 24):
                for x in range(8, 24):
                    rgb.putpixel((x, y), (220, 220, 220, 255))
            depth = np.zeros((32, 32), dtype=np.float32)
            depth[8:24, 8:24] = 1.0
            payload[view.name] = {
                "rgb": rgb,
                "depth": depth,
                "camera": {"name": view.name, "pose": view.pose.tolist(), "fovy_deg": view.fovy_deg},
            }
        return payload

    monkeypatch.setattr("lib.view_sampling.render_offscreen_views", fake_render_offscreen_views)

    assets = render_view_assets(mesh, views, resolution=32)

    rgb = np.asarray(assets["front"]["rgb"].convert("RGBA"))
    control = np.asarray(assets["front"]["control"].convert("RGBA"))
    mask = np.asarray(assets["front"]["mask"])
    visible = mask > 0

    assert visible.any()
    assert np.all(rgb[visible, 3] == 255)
    assert np.all(rgb[~visible, 3] == 0)
    assert np.all(control[~visible, :3] == 0)
    assert np.all(control[~visible, 3] == 0)
