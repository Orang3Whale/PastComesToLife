from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass(frozen=True)
class CameraView:
    name: str
    pose: np.ndarray
    fovy_deg: float


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    eye_vec = np.asarray(eye, dtype=np.float32)
    target_vec = np.asarray(target, dtype=np.float32)
    up_vec = np.asarray(up, dtype=np.float32)

    forward = target_vec - eye_vec
    forward_norm = np.linalg.norm(forward)
    if forward_norm == 0.0:
        raise ValueError("eye and target must differ")
    forward = forward / forward_norm

    right = np.cross(forward, up_vec)
    right_norm = np.linalg.norm(right)
    if right_norm == 0.0:
        raise ValueError("up vector must not be parallel to the view direction")
    right = right / right_norm

    true_up = np.cross(right, forward)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye_vec
    return pose


def _view_fits(
    vertices: np.ndarray,
    center: np.ndarray,
    direction: np.ndarray,
    up: np.ndarray,
    distance: float,
    fovy_deg: float,
) -> bool:
    eye = center + direction * distance
    pose = look_at(eye, center, up)
    world_to_camera = np.linalg.inv(pose)
    homogenous = np.concatenate([vertices, np.ones((len(vertices), 1), dtype=np.float32)], axis=1)
    camera = (world_to_camera @ homogenous.T).T[:, :3]

    z = -camera[:, 2]
    if np.any(z <= 1e-6):
        return False

    scale = max(float(np.tan(np.deg2rad(fovy_deg) * 0.5)), 1e-6)
    x_ndc = camera[:, 0] / (scale * z)
    y_ndc = camera[:, 1] / (scale * z)
    return bool(np.all(np.abs(x_ndc) <= 1.0) and np.all(np.abs(y_ndc) <= 1.0))


def _fit_distance(
    vertices: np.ndarray,
    center: np.ndarray,
    direction: np.ndarray,
    up: np.ndarray,
    fovy_deg: float,
    distance_hint: float,
    padding: float = 1.05,
) -> float:
    radius = float(np.max(np.linalg.norm(vertices - center, axis=1)))
    low = max(radius, 1e-3)
    high = max(distance_hint, low)
    while not _view_fits(vertices, center, direction, up, high, fovy_deg):
        high *= 2.0

    for _ in range(32):
        mid = 0.5 * (low + high)
        if _view_fits(vertices, center, direction, up, mid, fovy_deg):
            high = mid
        else:
            low = mid

    return max(high * padding, radius + 1e-3)


def build_six_view_spec(
    mesh: trimesh.Trimesh,
    camera_distance: float,
    fovy_deg: float,
) -> list[CameraView]:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    bounds = np.asarray(mesh.bounds, dtype=np.float32)
    center = bounds.mean(axis=0)
    radius = float(np.max(np.linalg.norm(vertices - center, axis=1)))

    specs = [
        ("front", np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        ("back", np.array([-1.0, 0.0, 0.0], dtype=np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        ("left", np.array([0.0, -1.0, 0.0], dtype=np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        ("right", np.array([0.0, 1.0, 0.0], dtype=np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        ("top", np.array([0.0, 0.0, 1.0], dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        ("bottom", np.array([0.0, 0.0, -1.0], dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    ]
    return [
        CameraView(
            name=name,
            pose=look_at(
                center + direction * _fit_distance(
                    vertices=vertices,
                    center=center,
                    direction=direction,
                    up=up,
                    fovy_deg=fovy_deg,
                    distance_hint=max(camera_distance * radius, radius + 1e-3),
                ),
                center,
                up,
            ),
            fovy_deg=fovy_deg,
        )
        for name, direction, up in specs
    ]
