import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from lib.camera_views import look_at
from lib.io_paths import create_run_tree, write_json
from lib.reprojection import (
    ViewSample,
    _fill_texture_gaps,
    _intrinsic_from_size,
    bake_texture,
    blend_samples,
    load_view_samples,
    project_point_to_view,
)
from lib.mesh_utils import bake_visible_texels, load_trimesh_with_texture
from scripts.step4_retexture import _build_projector, run_step


def test_blend_samples_weights_colors() -> None:
    fallback = np.array([255, 255, 255], dtype=np.uint8)
    blended = blend_samples(
        [
            (0.75, np.array([255, 0, 0], dtype=np.uint8)),
            (0.25, np.array([0, 0, 255], dtype=np.uint8)),
        ],
        fallback,
    )

    assert np.array_equal(blended, np.array([191, 0, 64], dtype=np.uint8))


def test_fill_texture_gaps_expands_painted_colors_into_uv_holes_and_background() -> None:
    texture = Image.new("RGBA", (5, 5), (0, 0, 0, 255))
    texture.putpixel((2, 2), (255, 0, 0, 255))
    painted_mask = np.zeros((5, 5), dtype=bool)
    painted_mask[2, 2] = True
    uv_mask = np.zeros((5, 5), dtype=bool)
    uv_mask[1:4, 1:4] = True

    filled = _fill_texture_gaps(texture, painted_mask, uv_mask)
    filled_array = np.asarray(filled)

    assert np.all(filled_array[1:4, 1:4, :3] == np.array([255, 0, 0], dtype=np.uint8))
    assert np.all(filled_array[0, 0, :3] == np.array([255, 0, 0], dtype=np.uint8))
    assert np.all(filled_array[:, :, 3] == 255)


def _front_facing_uv_plane() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [0.0, -0.5, -0.5],
            [0.0, 0.5, -0.5],
            [0.0, 0.5, 0.5],
            [0.0, -0.5, 0.5],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    uv = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return vertices, faces, uv


def test_bake_texture_maps_uv_v_zero_to_texture_bottom() -> None:
    vertices, faces, uv = _front_facing_uv_plane()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = TextureVisuals(
        uv=uv,
        material=PBRMaterial(baseColorTexture=Image.new("RGBA", (8, 8), (128, 128, 128, 255))),
    )

    stylized = Image.new("RGBA", (9, 9), (0, 0, 255, 255))
    for y in range(4):
        for x in range(9):
            stylized.putpixel((x, y), (255, 0, 0, 255))

    view = ViewSample(
        name="front",
        pose=look_at(
            np.array([2.0, 0.0, 0.0], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        ),
        intrinsic=_intrinsic_from_size(stylized.size, 90.0),
        depth=np.full((9, 9), 2.0, dtype=np.float32),
        stylized=stylized,
    )

    baked = bake_texture(mesh, Image.new("RGBA", (8, 8), (128, 128, 128, 255)), [view])

    assert baked.getpixel((4, 0))[:3] == (255, 0, 0)
    assert baked.getpixel((4, 7))[:3] == (0, 0, 255)


def test_bake_visible_texels_maps_uv_v_zero_to_texture_bottom() -> None:
    vertices, faces, uv = _front_facing_uv_plane()
    stylized = Image.new("RGBA", (2, 2), (0, 0, 255, 255))
    stylized.putpixel((0, 0), (255, 0, 0, 255))
    stylized.putpixel((1, 0), (255, 0, 0, 255))

    def projector(position: np.ndarray, normal: np.ndarray) -> tuple[bool, tuple[int, int]]:
        source_y = 0 if float(position[2]) > 0.0 else 1
        return float(normal[0]) > 0.0, (0, source_y)

    baked = bake_visible_texels(
        base=Image.new("RGBA", (8, 8), (128, 128, 128, 255)),
        stylized=stylized,
        vertices=vertices,
        faces=faces,
        uv=uv,
        projector=projector,
    )

    assert baked.getpixel((4, 0))[:3] == (255, 0, 0)
    assert baked.getpixel((4, 6))[:3] == (0, 0, 255)


def test_project_point_to_view_returns_screen_coordinates() -> None:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    intrinsic = np.array(
        [
            [100.0, 0.0, 4.0],
            [0.0, 100.0, 4.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    view = ViewSample(
        name="front",
        pose=pose,
        intrinsic=intrinsic,
        depth=np.ones((8, 8), dtype=np.float32),
        stylized=Image.new("RGBA", (8, 8), (255, 0, 0, 255)),
    )

    projected = project_point_to_view(np.array([0.0, 0.0, 0.0], dtype=np.float32), view)

    assert projected is not None
    assert projected.x == 4
    assert projected.y == 4
    assert projected.depth > 0


def test_project_point_to_view_maps_positive_y_upward() -> None:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    intrinsic = _intrinsic_from_size((9, 9), 90.0)
    view = ViewSample(
        name="front",
        pose=pose,
        intrinsic=intrinsic,
        depth=np.ones((9, 9), dtype=np.float32),
        stylized=Image.new("RGBA", (9, 9), (255, 0, 0, 255)),
    )

    projected = project_point_to_view(np.array([0.0, 1.0, 0.0], dtype=np.float32), view)

    assert projected is not None
    assert projected.x == 4
    assert projected.y == 2


def test_load_view_samples_reads_manifests_and_assets(tmp_path: Path) -> None:
    paths = create_run_tree(tmp_path / "run")
    write_json(
        paths.views / "manifest.json",
        {
            "views": [
                {
                    "name": name,
                    "control_path": f"{name}/control.png",
                    "camera_path": f"{name}/camera.json",
                    "depth_path": f"{name}/depth.npy",
                }
                for name in ("front", "back", "left", "right", "top", "bottom")
            ],
            "camera_fovy_deg": 40.0,
        },
    )
    write_json(
        paths.stylize / "manifest.json",
        {
            "views": {
                name: {
                    "control_path": f"{name}/control.png",
                    "stylized_path": f"{name}/stylized.png",
                }
                for name in ("front", "back", "left", "right", "top", "bottom")
            }
        },
    )
    for name in ("front", "back", "left", "right", "top", "bottom"):
        (paths.views / name).mkdir(parents=True, exist_ok=True)
        (paths.stylize / name).mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (8, 8), (0, 0, 0, 255)).save(paths.views / name / "control.png")
        Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(paths.stylize / name / "stylized.png")
        np.save(paths.views / name / "depth.npy", np.ones((8, 8), dtype=np.float32))
        (paths.views / name / "camera.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "pose": np.eye(4, dtype=np.float32).tolist(),
                    "fovy_deg": 40.0,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    samples = load_view_samples(
        json.loads((paths.views / "manifest.json").read_text(encoding="utf-8")),
        json.loads((paths.stylize / "manifest.json").read_text(encoding="utf-8")),
        views_root=paths.views,
        stylize_root=paths.stylize,
    )

    assert [sample.name for sample in samples] == ["front", "back", "left", "right", "top", "bottom"]
    assert samples[0].stylized.size == (8, 8)
    assert samples[0].depth.shape == (8, 8)


def test_load_view_samples_ignores_render_mode_field(tmp_path: Path) -> None:
    paths = create_run_tree(tmp_path / "run")
    write_json(
        paths.views / "manifest.json",
        {
            "render_mode": "mesh_offscreen",
            "views": [
                {
                    "name": name,
                    "control_path": f"{name}/control.png",
                    "camera_path": f"{name}/camera.json",
                    "depth_path": f"{name}/depth.npy",
                }
                for name in ("front", "back", "left", "right", "top", "bottom")
            ],
            "camera_fovy_deg": 40.0,
        },
    )
    write_json(
        paths.stylize / "manifest.json",
        {
            "views": {
                name: {
                    "control_path": f"{name}/control.png",
                    "stylized_path": f"{name}/stylized.png",
                }
                for name in ("front", "back", "left", "right", "top", "bottom")
            },
        },
    )
    for name in ("front", "back", "left", "right", "top", "bottom"):
        (paths.views / name).mkdir(parents=True, exist_ok=True)
        (paths.stylize / name).mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (8, 8), (0, 0, 0, 255)).save(paths.views / name / "control.png")
        Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(paths.stylize / name / "stylized.png")
        np.save(paths.views / name / "depth.npy", np.ones((8, 8), dtype=np.float32))
        (paths.views / name / "camera.json").write_text(
            json.dumps({"name": name, "pose": np.eye(4, dtype=np.float32).tolist(), "fovy_deg": 40.0}, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    samples = load_view_samples(
        json.loads((paths.views / "manifest.json").read_text(encoding="utf-8")),
        json.loads((paths.stylize / "manifest.json").read_text(encoding="utf-8")),
        views_root=paths.views,
        stylize_root=paths.stylize,
    )

    assert [sample.name for sample in samples] == ["front", "back", "left", "right", "top", "bottom"]
    assert samples[0].stylized.size == (8, 8)


def test_load_view_samples_resolves_relative_run_paths_against_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    paths = create_run_tree(Path("runs/run"))
    write_json(
        paths.views / "manifest.json",
        {
            "views": [
                {
                    "name": name,
                    "control_path": str(paths.views / name / "control.png"),
                    "camera_path": str(paths.views / name / "camera.json"),
                    "depth_path": str(paths.views / name / "depth.npy"),
                }
                for name in ("front", "back", "left", "right", "top", "bottom")
            ],
            "camera_fovy_deg": 40.0,
        },
    )
    write_json(
        paths.stylize / "manifest.json",
        {
            "views": {
                name: {
                    "control_path": str(paths.views / name / "control.png"),
                    "stylized_path": str(paths.stylize / name / "stylized.png"),
                }
                for name in ("front", "back", "left", "right", "top", "bottom")
            }
        },
    )
    for name in ("front", "back", "left", "right", "top", "bottom"):
        (paths.views / name).mkdir(parents=True, exist_ok=True)
        (paths.stylize / name).mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (8, 8), (0, 0, 0, 255)).save(paths.views / name / "control.png")
        Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(paths.stylize / name / "stylized.png")
        np.save(paths.views / name / "depth.npy", np.ones((8, 8), dtype=np.float32))
        (paths.views / name / "camera.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "pose": np.eye(4, dtype=np.float32).tolist(),
                    "fovy_deg": 40.0,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    samples = load_view_samples(
        json.loads((paths.views / "manifest.json").read_text(encoding="utf-8")),
        json.loads((paths.stylize / "manifest.json").read_text(encoding="utf-8")),
        views_root=paths.views,
        stylize_root=paths.stylize,
    )

    assert [sample.name for sample in samples] == ["front", "back", "left", "right", "top", "bottom"]


def test_run_step_uses_multiview_manifests(tmp_path: Path) -> None:
    paths = create_run_tree(tmp_path / "run")
    _write_textured_triangle(paths.sf3d / "mesh_raw.glb", Image.new("RGBA", (8, 8), (255, 255, 255, 255)))
    write_json(
        paths.views / "manifest.json",
        {
            "views": [
                {
                    "name": "front",
                    "control_path": str(paths.views / "front" / "control.png"),
                    "camera_path": str(paths.views / "front" / "camera.json"),
                    "depth_path": str(paths.views / "front" / "depth.npy"),
                },
                {
                    "name": "back",
                    "control_path": str(paths.views / "back" / "control.png"),
                    "camera_path": str(paths.views / "back" / "camera.json"),
                    "depth_path": str(paths.views / "back" / "depth.npy"),
                },
                {
                    "name": "left",
                    "control_path": str(paths.views / "left" / "control.png"),
                    "camera_path": str(paths.views / "left" / "camera.json"),
                    "depth_path": str(paths.views / "left" / "depth.npy"),
                },
                {
                    "name": "right",
                    "control_path": str(paths.views / "right" / "control.png"),
                    "camera_path": str(paths.views / "right" / "camera.json"),
                    "depth_path": str(paths.views / "right" / "depth.npy"),
                },
                {
                    "name": "top",
                    "control_path": str(paths.views / "top" / "control.png"),
                    "camera_path": str(paths.views / "top" / "camera.json"),
                    "depth_path": str(paths.views / "top" / "depth.npy"),
                },
                {
                    "name": "bottom",
                    "control_path": str(paths.views / "bottom" / "control.png"),
                    "camera_path": str(paths.views / "bottom" / "camera.json"),
                    "depth_path": str(paths.views / "bottom" / "depth.npy"),
                },
            ]
        },
    )
    write_json(
        paths.stylize / "manifest.json",
        {
            "views": {
                name: {
                    "control_path": str(paths.views / name / "control.png"),
                    "stylized_path": str(paths.stylize / name / "stylized.png"),
                }
                for name in ("front", "back", "left", "right", "top", "bottom")
            }
        },
    )
    for name in ("front", "back", "left", "right", "top", "bottom"):
        (paths.views / name).mkdir(parents=True, exist_ok=True)
        (paths.stylize / name).mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (8, 8), (0, 0, 0, 255)).save(paths.views / name / "control.png")
        Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(paths.stylize / name / "stylized.png")
        np.save(paths.views / name / "depth.npy", np.ones((8, 8), dtype=np.float32))
        (paths.views / name / "camera.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "pose": np.eye(4, dtype=np.float32).tolist(),
                    "fovy_deg": 40.0,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    seen: dict[str, object] = {}

    def fake_sample_loader(view_manifest, stylize_manifest, **kwargs):  # noqa: ANN001
        seen["view_manifest"] = view_manifest
        seen["stylize_manifest"] = stylize_manifest
        seen["loader_kwargs"] = kwargs
        return []

    def fake_baker(mesh, base_texture, samples):  # noqa: ANN001
        seen["mesh"] = mesh
        seen["base_texture"] = base_texture
        seen["samples"] = samples
        return Image.new("RGBA", (8, 8), (255, 0, 0, 255))

    result = run_step(paths.root, sample_loader=fake_sample_loader, baker=fake_baker)

    assert "view_manifest" in seen
    assert "stylize_manifest" in seen
    assert seen["loader_kwargs"]["views_root"] == paths.views
    assert seen["loader_kwargs"]["stylize_root"] == paths.stylize
    assert seen["samples"] == []
    assert result["mesh_path"].endswith("mesh_stylized.glb")
    assert (paths.retexture / "texture_preview.png").is_file()


def test_run_step_rejects_partial_multiview_state(tmp_path: Path) -> None:
    paths = create_run_tree(tmp_path / "run")
    _write_textured_triangle(paths.sf3d / "mesh_raw.glb", Image.new("RGBA", (8, 8), (255, 255, 255, 255)))
    write_json(paths.views / "manifest.json", {"views": []})
    Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(paths.stylize / "stylized.png")

    with pytest.raises(FileNotFoundError, match="multiview"):
        run_step(paths.root)


def test_bake_visible_texels_updates_visible_pixels() -> None:
    base = Image.new("RGBA", (4, 4), (255, 255, 255, 255))
    stylized = Image.new("RGBA", (4, 4), (255, 0, 0, 255))
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    def projector(position, normal):  # noqa: ANN001
        return True, (0, 0)

    baked = bake_visible_texels(base, stylized, vertices, faces, uv, projector)
    assert baked.getpixel((0, 0))[:3] == (255, 0, 0)


def test_bake_visible_texels_preserves_hidden_pixels() -> None:
    base = Image.new("RGBA", (4, 4), (255, 255, 255, 255))
    stylized = Image.new("RGBA", (4, 4), (255, 0, 0, 255))
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    def projector(position, normal):  # noqa: ANN001
        return False, (0, 0)

    baked = bake_visible_texels(base, stylized, vertices, faces, uv, projector)
    assert baked.getpixel((0, 0))[:3] == (255, 255, 255)


def test_bake_visible_texels_skips_transparent_source_pixels() -> None:
    base = Image.new("RGBA", (4, 4), (255, 255, 255, 255))
    stylized = Image.new("RGBA", (4, 4), (255, 0, 0, 0))
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    def projector(position, normal):  # noqa: ANN001
        return True, (0, 0)

    baked = bake_visible_texels(base, stylized, vertices, faces, uv, projector)
    assert baked.getpixel((0, 0))[:3] == (255, 255, 255)


def test_build_projector_uses_yz_front_axes_and_x_visibility() -> None:
    mesh = trimesh.creation.box(bounds=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    projector = _build_projector(mesh, (10, 10), (2, 2, 7, 7))

    visible, left_xy = projector(
        np.array([0.5, 0.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    _, right_xy = projector(
        np.array([0.5, 1.0, 1.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    hidden, _ = projector(
        np.array([0.5, 0.5, 0.5], dtype=np.float32),
        np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    )

    assert visible
    assert 2 <= left_xy[0] <= 7
    assert 2 <= right_xy[0] <= 7
    assert left_xy[0] < right_xy[0]
    assert left_xy[1] > right_xy[1]
    assert hidden is False


def test_run_step_retextures_front_facing_x_positive_faces(tmp_path: Path) -> None:
    paths = create_run_tree(tmp_path / "run")
    stylized_path = paths.stylize / "stylized.png"
    Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(stylized_path)

    vertices = np.array(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    uv = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = TextureVisuals(uv=uv, material=PBRMaterial(baseColorTexture=Image.new("RGBA", (4, 4), (255, 255, 255, 255))))
    mesh.export(paths.sf3d / "mesh_raw.glb", include_normals=True)

    result = run_step(paths.root)

    preview = Image.open(paths.retexture / "texture_preview.png").convert("RGBA")
    assert preview.getpixel((0, 0))[:3] == (255, 0, 0)
    assert result["mesh_path"].endswith("mesh_stylized.glb")


def _write_textured_triangle(mesh_path: Path, texture: Image.Image) -> None:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = TextureVisuals(uv=uv, material=PBRMaterial(baseColorTexture=texture))
    mesh.export(mesh_path, include_normals=True)


def test_load_trimesh_with_texture_rejects_non_textured_mesh(tmp_path: Path) -> None:
    mesh_path = tmp_path / "mesh.glb"
    trimesh.creation.box().export(mesh_path)

    with pytest.raises(ValueError, match="textured visuals"):
        load_trimesh_with_texture(mesh_path)


def test_load_trimesh_with_texture_loads_textured_mesh(tmp_path: Path) -> None:
    mesh_path = tmp_path / "mesh.glb"
    texture = Image.new("RGBA", (4, 4), (12, 34, 56, 255))
    _write_textured_triangle(mesh_path, texture)

    mesh, loaded_texture = load_trimesh_with_texture(mesh_path)

    assert isinstance(mesh, trimesh.Trimesh)
    assert mesh.visual.kind == "texture"
    assert loaded_texture.getpixel((0, 0)) == (12, 34, 56, 255)


def test_run_step_writes_retexture_outputs_and_metadata(tmp_path: Path) -> None:
    paths = create_run_tree(tmp_path / "run")
    stylized_path = paths.stylize / "stylized.png"
    Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(stylized_path)
    _write_textured_triangle(paths.sf3d / "mesh_raw.glb", Image.new("RGBA", (4, 4), (255, 255, 255, 255)))

    result = run_step(paths.root)

    meta_path = paths.retexture / "retexture_meta.json"
    preview_path = paths.retexture / "texture_preview.png"
    mesh_stylized_path = paths.retexture / "mesh_stylized.glb"

    assert result == {
        "mesh_path": str(mesh_stylized_path),
        "texture_preview": str(preview_path),
    }
    assert json.loads(meta_path.read_text(encoding="utf-8")) == result
    assert preview_path.is_file()
    assert mesh_stylized_path.is_file()
