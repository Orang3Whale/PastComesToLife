from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

from lib.io_paths import create_run_tree, write_json
from lib.mesh_utils import bake_visible_texels, load_trimesh_with_texture
from lib.reprojection import bake_texture, load_view_samples


def _foreground_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    alpha = np.asarray(image.convert("RGBA"))[..., 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0 or len(ys) == 0:
        width, height = image.size
        return 0, 0, width - 1, height - 1
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _build_projector(
    mesh: object,
    image_size: tuple[int, int],
    image_bbox: tuple[int, int, int, int],
) -> Callable[[np.ndarray, np.ndarray], tuple[bool, tuple[int, int]]]:
    bounds = np.asarray(getattr(mesh, "bounds"), dtype=np.float32)
    min_bounds = bounds[0]
    max_bounds = bounds[1]
    extents = np.maximum(max_bounds - min_bounds, 1e-6)
    width, height = image_size
    x0, y0, x1, y1 = image_bbox

    def projector(position: np.ndarray, normal: np.ndarray) -> tuple[bool, tuple[int, int]]:
        if float(normal[0]) <= 0.0:
            return False, (0, 0)
        u = (float(position[1]) - float(min_bounds[1])) / float(extents[1])
        v = (float(position[2]) - float(min_bounds[2])) / float(extents[2])
        x = int(np.clip(round(x0 + u * max((x1 - x0), 1)), 0, width - 1))
        y = int(np.clip(round(y0 + (1.0 - v) * max((y1 - y0), 1)), 0, height - 1))
        return bool(np.isfinite(normal).all()), (x, y)

    return projector


def run_step(
    run_dir: Path,
    sample_loader: Callable[..., list] = load_view_samples,
    baker: Callable[..., Image.Image] = bake_texture,
) -> dict:
    paths = create_run_tree(run_dir)
    mesh_path = paths.sf3d / "mesh_raw.glb"
    mesh, base_texture = load_trimesh_with_texture(mesh_path)
    view_manifest_path = paths.views / "manifest.json"
    stylize_manifest_path = paths.stylize / "manifest.json"

    has_view_manifest = view_manifest_path.is_file()
    has_stylize_manifest = stylize_manifest_path.is_file()
    if has_view_manifest != has_stylize_manifest:
        raise FileNotFoundError("multiview manifests must appear together")

    if has_view_manifest and has_stylize_manifest:
        view_manifest = json.loads(view_manifest_path.read_text(encoding="utf-8"))
        stylize_manifest = json.loads(stylize_manifest_path.read_text(encoding="utf-8"))
        samples = sample_loader(
            view_manifest,
            stylize_manifest,
            views_root=paths.views,
            stylize_root=paths.stylize,
        )
        baked_texture = baker(mesh, base_texture, samples)
    else:
        stylized_path = paths.stylize / "stylized.png"
        with Image.open(stylized_path) as stylized_image:
            stylized = stylized_image.convert("RGBA")

        projector = _build_projector(mesh, stylized.size, _foreground_bbox(stylized))
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

    mesh.visual.material.baseColorTexture = baked_texture
    mesh.export(mesh_stylized_path, include_normals=True)

    result = {
        "mesh_path": str(mesh_stylized_path),
        "texture_preview": str(texture_preview_path),
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
