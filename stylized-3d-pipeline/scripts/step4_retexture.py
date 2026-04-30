from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from lib.io_paths import create_run_tree, write_json
from lib.mesh_utils import bake_visible_texels, load_trimesh_with_texture


def _build_front_projector(mesh: object, image_size: tuple[int, int]) -> Callable[[np.ndarray, np.ndarray], tuple[bool, tuple[int, int]]]:
    bounds = np.asarray(getattr(mesh, "bounds"), dtype=np.float32)
    min_bounds = bounds[0]
    max_bounds = bounds[1]
    extents = np.maximum(max_bounds - min_bounds, 1e-6)
    width, height = image_size

    def projector(position: np.ndarray, normal: np.ndarray) -> tuple[bool, tuple[int, int]]:
        if float(normal[2]) <= 0.0:
            return False, (0, 0)
        u = (float(position[0]) - float(min_bounds[0])) / float(extents[0])
        v = (float(position[1]) - float(min_bounds[1])) / float(extents[1])
        x = int(np.clip(round(u * (width - 1)), 0, width - 1))
        y = int(np.clip(round((1.0 - v) * (height - 1)), 0, height - 1))
        return True, (x, y)

    return projector


def run_step(run_dir: Path) -> dict:
    paths = create_run_tree(run_dir)
    mesh_path = paths.sf3d / "mesh_raw.glb"
    stylized_path = paths.stylize / "stylized.png"
    mesh, base_texture = load_trimesh_with_texture(mesh_path)

    with Image.open(stylized_path) as stylized_image:
        stylized = stylized_image.convert("RGBA")

    projector = _build_front_projector(mesh, stylized.size)
    baked_texture = bake_visible_texels(
        base=base_texture,
        stylized=stylized,
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.faces),
        uv=np.asarray(mesh.visual.uv),
        projector=projector,
    )

    retexture_dir = paths.retexture
    texture_preview_path = retexture_dir / "texture_preview.png"
    mesh_stylized_path = retexture_dir / "mesh_stylized.glb"
    baked_texture.save(texture_preview_path)

    mesh.visual.material.image = baked_texture
    mesh.export(mesh_stylized_path, include_normals=True)

    result = {
        "mesh_path": str(mesh_path),
        "stylized_path": str(stylized_path),
        "texture_preview_path": str(texture_preview_path),
        "mesh_stylized_path": str(mesh_stylized_path),
    }
    write_json(retexture_dir / "retexture_meta.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_step(args.run_dir)


if __name__ == "__main__":
    main()
