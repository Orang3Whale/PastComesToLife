from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from lib.camera_views import CameraView
from lib.offscreen_renderer import render_offscreen_views


def _as_rgba(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGBA")
    return Image.fromarray(np.asarray(image)).convert("RGBA")


def _make_control_image(normal_rgb: np.ndarray, depth_preview: np.ndarray, mask: np.ndarray) -> Image.Image:
    depth_rgb = np.repeat(depth_preview[:, :, None], 3, axis=2)
    control_rgb = np.uint8(np.clip(np.round(normal_rgb.astype(np.float32) * 0.65 + depth_rgb.astype(np.float32) * 0.35), 0, 255))
    control_rgb = np.where(mask[:, :, None] > 0, control_rgb, 0).astype(np.uint8)
    return Image.fromarray(np.dstack([control_rgb, mask]), mode="RGBA")


def _depth_for_normal_estimation(valid_depth: np.ndarray, visible: np.ndarray) -> np.ndarray:
    if not np.any(visible):
        return np.zeros_like(valid_depth, dtype=np.float32)

    try:
        from scipy import ndimage
    except ImportError:
        fill_value = float(valid_depth[visible].max())
        return np.where(visible, valid_depth, fill_value).astype(np.float32)

    _, indices = ndimage.distance_transform_edt(~visible, return_indices=True)
    filled = np.asarray(valid_depth, dtype=np.float32).copy()
    filled[~visible] = filled[indices[0][~visible], indices[1][~visible]]
    return filled


def _derive_secondary_maps(rgb: Image.Image | np.ndarray, depth: np.ndarray, fovy_deg: float) -> dict[str, object]:
    del fovy_deg
    rgba = _as_rgba(rgb)
    rgba_array = np.asarray(rgba, dtype=np.uint8).copy()
    mask = rgba_array[:, :, 3].copy()
    visible = mask > 0

    depth_array = np.asarray(depth, dtype=np.float32)
    valid_depth = np.where(visible, depth_array, 0.0).astype(np.float32)
    if np.any(visible):
        visible_depth = valid_depth[visible]
        min_depth = float(visible_depth.min())
        max_depth = float(visible_depth.max())
        if max_depth > min_depth:
            depth_preview = np.zeros_like(valid_depth, dtype=np.float32)
            depth_preview[visible] = (visible_depth - min_depth) / (max_depth - min_depth)
        else:
            depth_preview = np.where(visible, 1.0, 0.0).astype(np.float32)
    else:
        depth_preview = np.zeros_like(valid_depth, dtype=np.float32)

    depth_preview_u8 = np.uint8(np.clip(depth_preview, 0.0, 1.0) * 255)
    normal_depth = _depth_for_normal_estimation(valid_depth, visible)
    gy, gx = np.gradient(normal_depth)
    normal_xyz = np.dstack((-gx, -gy, np.ones_like(normal_depth)))
    denom = np.linalg.norm(normal_xyz, axis=2, keepdims=True)
    denom = np.where(denom == 0.0, 1.0, denom)
    normal_rgb = np.uint8(np.clip((normal_xyz / denom + 1.0) * 127.5, 0, 255))
    normal_rgb = np.where(mask[:, :, None] > 0, normal_rgb, 0).astype(np.uint8)
    normal = Image.fromarray(np.dstack([normal_rgb, mask]), mode="RGBA")
    depth_rgba = np.dstack([np.repeat(depth_preview_u8[:, :, None], 3, axis=2), mask])

    return {
        "rgb": Image.fromarray(rgba_array, mode="RGBA"),
        "depth": valid_depth,
        "depth_preview": Image.fromarray(depth_rgba, mode="RGBA"),
        "normal": normal,
        "mask": Image.fromarray(mask, mode="L"),
        "control": _make_control_image(normal_rgb, depth_preview_u8, mask),
    }


def render_view_assets(mesh: trimesh.Trimesh, views: list[CameraView], resolution: int) -> dict[str, dict[str, object]]:
    rendered = render_offscreen_views(mesh, views, resolution)
    assets = {}
    for view in views:
        view_assets = rendered[view.name]
        derived = _derive_secondary_maps(view_assets["rgb"], view_assets["depth"], view.fovy_deg)
        assets[view.name] = {**derived, "camera": view_assets["camera"]}
    return assets


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
