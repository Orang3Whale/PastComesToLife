from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from lib.camera_views import CameraView, look_at
from lib.offscreen_renderer import build_neutral_render_mesh, render_offscreen_view


def _textured_triangle() -> trimesh.Trimesh:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = TextureVisuals(
        uv=uv,
        material=PBRMaterial(baseColorTexture=Image.new("RGBA", (4, 4), (0, 0, 0, 255))),
    )
    return mesh


def test_build_neutral_render_mesh_drops_source_texture() -> None:
    neutral = build_neutral_render_mesh(_textured_triangle())

    assert neutral.visual.kind != "texture"
    assert neutral.visual.vertex_colors.shape[1] == 4
    assert np.all(neutral.visual.vertex_colors[:, :3] == 235)


def test_render_offscreen_view_returns_rgba_color_and_depth() -> None:
    class FakeRenderer:
        def __init__(self, viewport_width: int, viewport_height: int) -> None:
            self.viewport_width = viewport_width
            self.viewport_height = viewport_height
            self.deleted = False

        def render(self, scene, flags):  # noqa: ANN001
            color = np.zeros((self.viewport_height, self.viewport_width, 4), dtype=np.uint8)
            color[2:6, 2:6, :3] = 220
            color[2:6, 2:6, 3] = 255
            depth = np.zeros((self.viewport_height, self.viewport_width), dtype=np.float32)
            depth[2:6, 2:6] = 1.0
            return color, depth

        def delete(self) -> None:
            self.deleted = True

    view = CameraView(
        name="front",
        pose=look_at(
            np.array([2.0, 0.0, 0.0], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        ),
        fovy_deg=40.0,
    )
    assets = render_offscreen_view(
        _textured_triangle(),
        view,
        resolution=8,
        renderer_factory=lambda width, height: FakeRenderer(width, height),
    )

    rgb = np.asarray(assets["rgb"].convert("RGBA"))
    depth = assets["depth"]

    assert rgb.shape == (8, 8, 4)
    assert depth.shape == (8, 8)
    assert rgb[0, 0, 3] == 0
    assert rgb[3, 3, 3] == 255
    assert rgb[3, 3, 0] == 220


def test_render_offscreen_view_raises_clear_error_when_backend_missing() -> None:
    view = CameraView(
        name="front",
        pose=look_at(
            np.array([2.0, 0.0, 0.0], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        ),
        fovy_deg=40.0,
    )

    def bad_factory(*args, **kwargs):  # noqa: ANN001, ANN003
        raise ValueError("no backend")

    with pytest.raises(RuntimeError, match="PYOPENGL_PLATFORM|OSMesa|EGL"):
        render_offscreen_view(
            _textured_triangle(),
            view,
            resolution=8,
            renderer_factory=bad_factory,
        )
