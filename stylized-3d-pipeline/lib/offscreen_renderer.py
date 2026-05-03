from __future__ import annotations

from typing import Callable

import numpy as np
import pyrender
import trimesh
from PIL import Image

from lib.camera_views import CameraView

_BACKEND_ERROR = "offscreen rendering failed; configure PYOPENGL_PLATFORM=egl or OSMesa"


def build_neutral_render_mesh(
    mesh: trimesh.Trimesh,
    base_color: tuple[int, int, int, int] = (235, 235, 235, 255),
) -> trimesh.Trimesh:
    neutral = mesh.copy()
    vertex_colors = np.tile(np.asarray(base_color, dtype=np.uint8), (len(neutral.vertices), 1))
    neutral.visual = trimesh.visual.color.ColorVisuals(mesh=neutral, vertex_colors=vertex_colors)
    return neutral


def _make_scene(mesh: trimesh.Trimesh, view: CameraView) -> pyrender.Scene:
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.18, 0.18, 0.18])
    scene.add(pyrender.Mesh.from_trimesh(build_neutral_render_mesh(mesh), smooth=False))
    scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(view.fovy_deg)), pose=view.pose)

    light_transforms = [
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.92, -0.38, 0.0],
                [0.0, 0.38, 0.92, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [0.92, 0.0, 0.38, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-0.38, 0.0, 0.92, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    ]
    for light_transform in light_transforms:
        scene.add(
            pyrender.DirectionalLight(color=np.ones(3, dtype=np.float32), intensity=2.2),
            pose=view.pose @ light_transform,
        )

    return scene


def render_offscreen_view(
    mesh: trimesh.Trimesh,
    view: CameraView,
    resolution: int,
    renderer_factory: Callable[[int, int], object] = pyrender.OffscreenRenderer,
) -> dict[str, object]:
    scene = _make_scene(mesh, view)
    try:
        renderer = renderer_factory(resolution, resolution)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(_BACKEND_ERROR) from exc

    try:
        color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(_BACKEND_ERROR) from exc
    finally:
        delete = getattr(renderer, "delete", None)
        if callable(delete):
            delete()

    color_rgba = np.asarray(color, dtype=np.uint8)
    depth_array = np.asarray(depth, dtype=np.float32)
    alpha = np.where(depth_array > 0.0, 255, 0).astype(np.uint8)
    if color_rgba.ndim == 3 and color_rgba.shape[2] == 3:
        color_rgba = np.dstack([color_rgba, alpha])
    elif color_rgba.ndim == 3 and color_rgba.shape[2] >= 4:
        color_rgba = color_rgba[:, :, :4].copy()
        color_rgba[:, :, 3] = alpha
    else:
        raise RuntimeError("unexpected color buffer shape from offscreen renderer")

    return {
        "rgb": Image.fromarray(color_rgba, mode="RGBA"),
        "depth": depth_array,
        "camera": {
            "name": view.name,
            "pose": view.pose.tolist(),
            "fovy_deg": view.fovy_deg,
        },
    }


def render_offscreen_views(
    mesh: trimesh.Trimesh,
    views: list[CameraView],
    resolution: int,
    renderer_factory: Callable[[int, int], object] = pyrender.OffscreenRenderer,
) -> dict[str, dict[str, object]]:
    names = [view.name for view in views]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        joined = ", ".join(duplicate_names)
        raise ValueError(f"duplicate view names are not supported: {joined}")

    return {
        view.name: render_offscreen_view(
            mesh,
            view,
            resolution=resolution,
            renderer_factory=renderer_factory,
        )
        for view in views
    }
