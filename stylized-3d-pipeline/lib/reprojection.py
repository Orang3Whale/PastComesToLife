from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import trimesh
from PIL import Image


@dataclass(frozen=True)
class ViewSample:
    name: str
    pose: np.ndarray
    intrinsic: np.ndarray
    depth: np.ndarray
    stylized: Image.Image


@dataclass(frozen=True)
class ProjectedPoint:
    x: int
    y: int
    depth: float


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


def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    face_vertices = vertices[faces]
    normals = np.cross(
        face_vertices[:, 1] - face_vertices[:, 0],
        face_vertices[:, 2] - face_vertices[:, 0],
    )
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.where(lengths == 0.0, 1.0, lengths)
    return normals / lengths


def _intrinsic_from_size(size: tuple[int, int], fovy_deg: float) -> np.ndarray:
    width, height = size
    scale = max(np.tan(np.deg2rad(fovy_deg) * 0.5), 1e-6)
    focal_x = 0.5 * max(width - 1, 0) / scale
    focal_y = -0.5 * max(height - 1, 0) / scale
    intrinsic = np.array(
        [
            [focal_x, 0.0, max(width - 1, 0) * 0.5],
            [0.0, focal_y, max(height - 1, 0) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return intrinsic


def _resolve_relative_path(path: Path, primary_root: Path | None, fallback_root: Path | None = None) -> Path:
    if path.is_absolute():
        return path

    cwd_candidate = Path.cwd() / path
    if cwd_candidate.is_file():
        return cwd_candidate

    if primary_root is not None:
        primary_candidate = primary_root / path
        if primary_candidate.is_file():
            return primary_candidate

    if fallback_root is not None:
        fallback_candidate = fallback_root / path
        if fallback_candidate.is_file():
            return fallback_candidate

    if primary_root is not None:
        return primary_root / path
    if fallback_root is not None:
        return fallback_root / path
    return cwd_candidate


def blend_samples(samples: list[tuple[float, np.ndarray]], fallback: np.ndarray) -> np.ndarray:
    if not samples:
        return np.asarray(fallback, dtype=np.uint8)

    weights = np.asarray([weight for weight, _ in samples], dtype=np.float32)
    colors = np.asarray([color for _, color in samples], dtype=np.float32)
    total = float(weights.sum())
    if total <= 1e-6:
        return np.asarray(fallback, dtype=np.uint8)
    rgb = (colors * weights[:, None]).sum(axis=0) / total
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def project_point_to_view(point: np.ndarray, view: ViewSample) -> ProjectedPoint | None:
    world_to_camera = np.linalg.inv(view.pose)
    homogenous = np.concatenate([np.asarray(point, dtype=np.float32), np.array([1.0], dtype=np.float32)])
    camera_point = world_to_camera @ homogenous
    depth = float(-camera_point[2])
    if depth <= 1e-6:
        return None

    pixel = view.intrinsic @ np.array([camera_point[0], camera_point[1], depth], dtype=np.float32)
    denom = float(pixel[2])
    if abs(denom) <= 1e-6:
        return None

    x = int(round(float(pixel[0]) / denom))
    y = int(round(float(pixel[1]) / denom))
    return ProjectedPoint(x=x, y=y, depth=depth)


def _sample_view_pixel_array(
    view: ViewSample,
    stylized_array: np.ndarray,
    point: np.ndarray,
    normal: np.ndarray,
    depth_epsilon: float = 0.02,
) -> tuple[float, np.ndarray] | None:
    projected = project_point_to_view(point, view)
    if projected is None:
        return None

    height, width = view.depth.shape[:2]
    if not (0 <= projected.x < width and 0 <= projected.y < height):
        return None

    depth_value = float(view.depth[projected.y, projected.x])
    if depth_value <= 1e-6:
        return None
    tolerance = max(depth_epsilon, 0.02 * projected.depth)
    if abs(depth_value - projected.depth) > tolerance:
        return None

    source_pixel = stylized_array[projected.y, projected.x]
    if source_pixel.shape[0] < 4 or source_pixel[3] < 1:
        return None

    view_dir = np.asarray(view.pose[:3, 3], dtype=np.float32) - np.asarray(point, dtype=np.float32)
    view_norm = float(np.linalg.norm(view_dir))
    normal_norm = float(np.linalg.norm(normal))
    if view_norm <= 1e-6 or normal_norm <= 1e-6:
        return None
    view_dir = view_dir / view_norm
    normal_dir = np.asarray(normal, dtype=np.float32) / normal_norm
    facing = float(np.clip(np.dot(normal_dir, view_dir), 0.0, 1.0))
    if facing <= 0.0:
        return None
    weight = facing**4
    return weight, np.asarray(source_pixel[:3], dtype=np.uint8)


def sample_view_pixel(
    point: np.ndarray,
    normal: np.ndarray,
    view: ViewSample,
    depth_epsilon: float = 0.02,
) -> tuple[float, np.ndarray] | None:
    stylized_array = np.asarray(view.stylized.convert("RGBA"))
    return _sample_view_pixel_array(view, stylized_array, point, normal, depth_epsilon=depth_epsilon)


def load_view_samples(
    view_manifest: dict[str, object],
    stylize_manifest: dict[str, object],
    views_root: Path | None = None,
    stylize_root: Path | None = None,
) -> list[ViewSample]:
    view_entries = view_manifest.get("views")
    stylized_views = stylize_manifest.get("views")
    if not isinstance(view_entries, list) or not isinstance(stylized_views, dict):
        raise ValueError("invalid multiview manifests")

    expected_view_order = ["front", "back", "left", "right", "top", "bottom"]
    seen_view_order: list[str] = []
    samples: list[ViewSample] = []
    for entry in view_entries:
        if not isinstance(entry, dict):
            raise ValueError("invalid view manifest entry")
        name = entry.get("name")
        camera_path = entry.get("camera_path")
        depth_path = entry.get("depth_path")
        if not isinstance(name, str) or not isinstance(camera_path, str) or not isinstance(depth_path, str):
            raise ValueError("invalid view manifest entry")
        seen_view_order.append(name)
        stylized_entry = stylized_views.get(name)
        if not isinstance(stylized_entry, dict):
            raise ValueError(f"missing stylized view: {name}")
        control_path = entry.get("control_path")
        stylized_path = stylized_entry.get("stylized_path")
        if not isinstance(control_path, str) or not isinstance(stylized_path, str):
            raise ValueError("invalid view manifest entry")

        camera_path = _resolve_relative_path(Path(camera_path), views_root)
        depth_path = _resolve_relative_path(Path(depth_path), views_root)
        control_path = _resolve_relative_path(Path(control_path), views_root)
        stylized_path = _resolve_relative_path(Path(stylized_path), stylize_root)
        if not camera_path.is_file() or not depth_path.is_file() or not stylized_path.is_file() or not control_path.is_file():
            raise FileNotFoundError(f"missing multiview asset for {name}")

        camera_data = json.loads(Path(camera_path).read_text(encoding="utf-8"))
        pose = np.asarray(camera_data.get("pose"), dtype=np.float32)
        fovy_deg = float(camera_data.get("fovy_deg", view_manifest.get("camera_fovy_deg", 40.0)))
        depth = np.asarray(np.load(depth_path), dtype=np.float32)
        with Image.open(stylized_path) as stylized_image:
            stylized = stylized_image.convert("RGBA")
        intrinsic = _intrinsic_from_size(stylized.size, fovy_deg)
        samples.append(
            ViewSample(
                name=name,
                pose=pose,
                intrinsic=intrinsic,
                depth=depth,
                stylized=stylized,
            )
        )

    if seen_view_order != expected_view_order:
        raise ValueError(
            f"view manifest must match canonical order: expected {expected_view_order}, got {seen_view_order}",
        )

    return samples


def bake_texture(
    mesh: trimesh.Trimesh,
    base_texture: Image.Image,
    samples: Sequence[ViewSample],
) -> Image.Image:
    base_rgba = base_texture.convert("RGBA")
    baked = base_rgba.copy()

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    uv = np.asarray(mesh.visual.uv, dtype=np.float32)
    normals = _face_normals(vertices, faces)

    stylized_arrays = {sample.name: np.asarray(sample.stylized.convert("RGBA")) for sample in samples}
    baked_pixels = baked.load()
    base_pixels = base_rgba.load()

    for face_index, face in enumerate(faces):
        triangle_uv = uv[face]
        triangle_vertices = vertices[face]
        face_normal = normals[face_index]
        for x, y, bary in _triangle_pixels(baked.width, baked.height, triangle_uv):
            position = (
                triangle_vertices[0] * bary[0]
                + triangle_vertices[1] * bary[1]
                + triangle_vertices[2] * bary[2]
            )
            fallback_pixel = np.asarray(base_pixels[x, y][:3], dtype=np.uint8)
            samples_for_texel: list[tuple[float, np.ndarray]] = []
            for sample in samples:
                sample_result = _sample_view_pixel_array(sample, stylized_arrays[sample.name], position, face_normal)
                if sample_result is not None:
                    samples_for_texel.append(sample_result)
            color = blend_samples(samples_for_texel, fallback_pixel)
            baked_pixels[x, y] = (int(color[0]), int(color[1]), int(color[2]), 255)

    return baked
