from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import trimesh
from PIL import Image

from lib.camera_views import CameraView


def _triangle_pixels(width: int, height: int, tri_xy: np.ndarray) -> Iterable[tuple[int, int, np.ndarray]]:
    tri = np.asarray(tri_xy, dtype=np.float32)
    if tri.shape != (3, 2):
        raise ValueError("triangle coordinates must have shape (3, 2)")

    min_x = max(int(np.floor(tri[:, 0].min())), 0)
    max_x = min(int(np.ceil(tri[:, 0].max())), width - 1)
    min_y = max(int(np.floor(tri[:, 1].min())), 0)
    max_y = min(int(np.ceil(tri[:, 1].max())), height - 1)

    matrix = np.array(
        [
            [tri[0, 0], tri[1, 0], tri[2, 0]],
            [tri[0, 1], tri[1, 1], tri[2, 1]],
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


def _extract_texture_image(mesh: trimesh.Trimesh) -> Image.Image | None:
    visual = getattr(mesh, "visual", None)
    if visual is None:
        return None
    material = getattr(visual, "material", None)
    if material is None or not hasattr(material, "baseColorTexture"):
        return None
    texture = getattr(material, "baseColorTexture", None)
    if texture is None:
        return None
    return _as_rgba(texture)


def _sample_texture(texture: Image.Image | None, uv: np.ndarray) -> tuple[int, int, int, int] | None:
    if texture is None:
        return None
    width, height = texture.size
    u = float(np.clip(uv[0], 0.0, 1.0))
    v = float(np.clip(uv[1], 0.0, 1.0))
    x = min(max(int(round(u * max(width - 1, 0))), 0), width - 1)
    y = min(max(int(round(v * max(height - 1, 0))), 0), height - 1)
    return texture.getpixel((x, y))


def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    face_vertices = vertices[faces]
    normals = np.cross(
        face_vertices[:, 1] - face_vertices[:, 0],
        face_vertices[:, 2] - face_vertices[:, 0],
    )
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.where(lengths == 0.0, 1.0, lengths)
    return normals / lengths


def _is_missing_texture_sample(pixel: tuple[int, int, int, int] | None) -> bool:
    if pixel is None:
        return True
    if pixel[3] < 16:
        return True
    return int(pixel[0]) + int(pixel[1]) + int(pixel[2]) < 18


def _fallback_view_color(normal: np.ndarray, center: np.ndarray, camera_position: np.ndarray) -> tuple[int, int, int, int]:
    view_dir = np.asarray(camera_position, dtype=np.float32) - np.asarray(center, dtype=np.float32)
    view_norm = float(np.linalg.norm(view_dir))
    normal_norm = float(np.linalg.norm(normal))
    if view_norm <= 1e-6 or normal_norm <= 1e-6:
        shade = 210
    else:
        facing = abs(float(np.dot(normal / normal_norm, view_dir / view_norm)))
        shade = int(np.clip(round(172.0 + 68.0 * facing), 0, 255))
    return shade, shade, shade, 255


def _view_pixel_from_texture(
    sampled_pixel: tuple[int, int, int, int] | None,
    fallback_pixel: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    if _is_missing_texture_sample(sampled_pixel):
        return fallback_pixel
    if sampled_pixel is None:
        return fallback_pixel
    return int(sampled_pixel[0]), int(sampled_pixel[1]), int(sampled_pixel[2]), 255


def _project_vertices(vertices: np.ndarray, view: CameraView) -> tuple[np.ndarray, np.ndarray]:
    world_to_camera = np.linalg.inv(view.pose)
    homogenous = np.concatenate([vertices, np.ones((len(vertices), 1), dtype=np.float32)], axis=1)
    camera = (world_to_camera @ homogenous.T).T[:, :3].astype(np.float32)

    z = -camera[:, 2]
    scale = np.tan(np.deg2rad(view.fovy_deg) * 0.5)
    scale = max(float(scale), 1e-6)
    x_ndc = camera[:, 0] / (scale * np.maximum(z, 1e-6))
    y_ndc = camera[:, 1] / (scale * np.maximum(z, 1e-6))
    return np.stack([x_ndc, y_ndc, z], axis=1), camera


def _to_screen(projected: np.ndarray, resolution: int) -> np.ndarray:
    coords = np.empty((len(projected), 2), dtype=np.float32)
    coords[:, 0] = (projected[:, 0] + 1.0) * 0.5 * (resolution - 1)
    coords[:, 1] = (1.0 - (projected[:, 1] + 1.0) * 0.5) * (resolution - 1)
    return coords


def _make_control_image(normal_rgb: np.ndarray, depth_preview: np.ndarray, mask: np.ndarray) -> Image.Image:
    depth_rgb = np.repeat(depth_preview[:, :, None], 3, axis=2)
    control_rgb = np.uint8(np.clip(np.round(normal_rgb.astype(np.float32) * 0.65 + depth_rgb.astype(np.float32) * 0.35), 0, 255))
    control_rgb = np.where(mask[:, :, None] > 0, control_rgb, 0).astype(np.uint8)
    return Image.fromarray(np.dstack([control_rgb, mask]), mode="RGBA")


def _render_single_view(mesh: trimesh.Trimesh, view: CameraView, resolution: int) -> dict[str, object]:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    normals = _face_normals(vertices, faces)
    uv = np.asarray(getattr(mesh.visual, "uv", None), dtype=np.float32) if getattr(mesh.visual, "uv", None) is not None else None
    texture = _extract_texture_image(mesh)

    projected, _camera_vertices = _project_vertices(vertices, view)
    screen = _to_screen(projected, resolution)

    rgb = Image.new("RGBA", (resolution, resolution), (0, 0, 0, 0))
    depth = np.full((resolution, resolution), np.inf, dtype=np.float32)
    pixels = rgb.load()
    mask = np.zeros((resolution, resolution), dtype=np.uint8)

    for face_index, face in enumerate(faces):
        face_projected = projected[face]
        face_screen = screen[face]
        face_uv = uv[face] if uv is not None else None
        face_center = vertices[face].mean(axis=0)
        fallback_pixel = _fallback_view_color(normals[face_index], face_center, view.pose[:3, 3])

        if np.all(face_projected[:, 2] <= 1e-6):
            continue

        for x, y, bary in _triangle_pixels(resolution, resolution, face_screen):
            clipped_depth = np.maximum(face_projected[:, 2], 1e-6)
            depth_weights = bary * (1.0 / clipped_depth)
            weight_total = float(depth_weights.sum())
            if weight_total <= 0.0:
                continue
            depth_weights = depth_weights / weight_total
            depth_value = float(np.dot(depth_weights, face_projected[:, 2]))
            if depth_value >= depth[y, x]:
                continue

            if face_uv is not None:
                tex_uv = np.dot(depth_weights, face_uv)
                pixels[x, y] = _view_pixel_from_texture(_sample_texture(texture, tex_uv), fallback_pixel)
            else:
                pixels[x, y] = fallback_pixel
            depth[y, x] = depth_value
            mask[y, x] = 255

    valid_depth = np.where(mask > 0, depth, 0.0).astype(np.float32)
    if np.any(mask > 0):
        max_depth = float(valid_depth[mask > 0].max())
        safe_depth = np.where(mask > 0, valid_depth, max_depth).astype(np.float32)
    else:
        max_depth = 1.0
        safe_depth = np.zeros_like(valid_depth)

    depth_preview = np.uint8(np.clip(safe_depth / max(max_depth, 1e-6), 0.0, 1.0) * 255)
    gy, gx = np.gradient(safe_depth)
    normal_xyz = np.dstack((-gx, -gy, np.ones_like(safe_depth)))
    denom = np.linalg.norm(normal_xyz, axis=2, keepdims=True)
    denom = np.where(denom == 0.0, 1.0, denom)
    normal_rgb = np.uint8(np.clip((normal_xyz / denom + 1.0) * 127.5, 0, 255))
    normal_rgb = np.where(mask[:, :, None] > 0, normal_rgb, 0).astype(np.uint8)
    normal_rgba = np.dstack([normal_rgb, mask])
    control = _make_control_image(normal_rgb, depth_preview, mask)

    return {
        "rgb": rgb,
        "depth": valid_depth,
        "depth_preview": Image.fromarray(depth_preview, mode="L").convert("RGBA"),
        "normal": Image.fromarray(normal_rgba, mode="RGBA"),
        "mask": Image.fromarray(mask, mode="L"),
        "control": control,
        "camera": {"name": view.name, "pose": view.pose.tolist(), "fovy_deg": view.fovy_deg},
    }


def render_view_assets(mesh: trimesh.Trimesh, views: list[CameraView], resolution: int) -> dict[str, dict[str, object]]:
    return {view.name: _render_single_view(mesh, view, resolution) for view in views}


def write_view_assets(view_root: Path, assets: dict[str, object]) -> dict[str, str]:
    view_root.mkdir(parents=True, exist_ok=True)
    rgb_path = view_root / "rgb.png"
    depth_path = view_root / "depth.npy"
    depth_preview_path = view_root / "depth.png"
    normal_path = view_root / "normal.png"
    mask_path = view_root / "mask.png"
    control_path = view_root / "control.png"
    camera_path = view_root / "camera.json"

    cast_image = _as_rgba(assets["rgb"])
    cast_image.save(rgb_path)
    np.save(depth_path, assets["depth"])
    _as_rgba(assets["depth_preview"]).save(depth_preview_path)
    _as_rgba(assets["normal"]).save(normal_path)
    Image.fromarray(np.asarray(assets["mask"]), mode="L").save(mask_path)
    _as_rgba(assets["control"]).save(control_path)
    camera_path.write_text(json.dumps(assets["camera"], indent=2, sort_keys=True), encoding="utf-8")
    return {
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "depth_preview_path": str(depth_preview_path),
        "normal_path": str(normal_path),
        "mask_path": str(mask_path),
        "control_path": str(control_path),
        "camera_path": str(camera_path),
    }
