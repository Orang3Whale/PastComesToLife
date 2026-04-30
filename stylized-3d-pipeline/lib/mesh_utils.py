from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import trimesh
from PIL import Image


def _triangle_pixels(width: int, height: int, tri_uv: np.ndarray) -> Iterator[tuple[int, int, np.ndarray]]:
    if width <= 0 or height <= 0:
        return

    tri = np.asarray(tri_uv, dtype=np.float32)
    if tri.shape != (3, 2):
        raise ValueError("triangle_uv must have shape (3, 2)")

    px = np.empty_like(tri)
    px[:, 0] = tri[:, 0] * max(width - 1, 0)
    px[:, 1] = tri[:, 1] * max(height - 1, 0)

    min_x = max(int(np.floor(px[:, 0].min())), 0)
    max_x = min(int(np.ceil(px[:, 0].max())), width - 1)
    min_y = max(int(np.floor(px[:, 1].min())), 0)
    max_y = min(int(np.ceil(px[:, 1].max())), height - 1)

    a, b, c = px
    matrix = np.array(
        [
            [a[0], b[0], c[0]],
            [a[1], b[1], c[1]],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            point = np.array([x + 0.5, y + 0.5, 1.0], dtype=np.float32)
            try:
                bary = np.linalg.solve(matrix, point)
            except np.linalg.LinAlgError:
                continue
            if np.all(bary >= -1e-5):
                yield x, y, bary


def _as_rgba(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGBA")
    return Image.fromarray(np.asarray(image)).convert("RGBA")


def _extract_texture_image(mesh: trimesh.Trimesh) -> Image.Image:
    visual = getattr(mesh, "visual", None)
    if visual is None or getattr(visual, "kind", None) != "texture":
        raise ValueError("mesh must have textured visuals")

    material = getattr(visual, "material", None)
    if material is None or not hasattr(material, "baseColorTexture"):
        raise ValueError("mesh must have a texture image")

    texture = material.baseColorTexture
    if texture is None:
        raise ValueError("mesh must have a texture image")
    return _as_rgba(texture)


def load_trimesh_with_texture(mesh_path: Path) -> tuple[trimesh.Trimesh, Image.Image]:
    loaded = trimesh.load(mesh_path, process=False)
    if isinstance(loaded, trimesh.Scene):
        if len(loaded.geometry) != 1:
            raise ValueError("expected a single mesh in the scene")
        loaded = next(iter(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"unsupported mesh type: {type(loaded)!r}")
    return loaded, _extract_texture_image(loaded)


def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    face_vertices = vertices[faces]
    normals = np.cross(
        face_vertices[:, 1] - face_vertices[:, 0],
        face_vertices[:, 2] - face_vertices[:, 0],
    )
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.where(lengths == 0.0, 1.0, lengths)
    return normals / lengths


def bake_visible_texels(
    base: Image.Image,
    stylized: Image.Image,
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    projector: Callable[[np.ndarray, np.ndarray], tuple[bool, tuple[int, int]]],
) -> Image.Image:
    base_rgba = base.convert("RGBA")
    stylized_rgba = stylized.convert("RGBA")
    baked = base_rgba.copy()

    vertices_arr = np.asarray(vertices, dtype=np.float32)
    faces_arr = np.asarray(faces, dtype=np.int32)
    uv_arr = np.asarray(uv, dtype=np.float32)
    normals = _face_normals(vertices_arr, faces_arr)

    stylized_width, stylized_height = stylized_rgba.size
    stylized_pixels = stylized_rgba.load()
    baked_pixels = baked.load()

    for face_index, face in enumerate(faces_arr):
        triangle_uv = uv_arr[face]
        triangle_vertices = vertices_arr[face]
        face_normal = normals[face_index]
        for x, y, bary in _triangle_pixels(baked.width, baked.height, triangle_uv):
            position = (
                triangle_vertices[0] * bary[0]
                + triangle_vertices[1] * bary[1]
                + triangle_vertices[2] * bary[2]
            )
            visible, source_xy = projector(position, face_normal)
            if not visible:
                continue
            source_x, source_y = source_xy
            if not (0 <= source_x < stylized_width and 0 <= source_y < stylized_height):
                continue
            baked_pixels[x, y] = stylized_pixels[source_x, source_y]

    return baked
