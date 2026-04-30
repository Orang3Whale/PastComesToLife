import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from lib.mesh_utils import bake_visible_texels, load_trimesh_with_texture
from lib.io_paths import create_run_tree
from scripts.step4_retexture import run_step


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
