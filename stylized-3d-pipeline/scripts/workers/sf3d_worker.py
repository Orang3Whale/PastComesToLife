from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image


def _unwrap_mesh(output: object) -> object:
    if isinstance(output, tuple):
        return output[0]
    return output


def build_model(device: str, sf3d_cls: type[object]) -> object:
    model = sf3d_cls.from_pretrained(
        "stabilityai/stable-fast-3d",
        config_name="config.yaml",
        weight_name="model.safetensors",
    )
    return model.to(device)


def find_upstream_repo_root(anchor: Path | None = None) -> Path:
    current = (anchor or Path(__file__)).resolve()
    for parent in (current.parent, *current.parents):
        candidate = parent / "stable-fast-3d"
        if candidate.is_dir():
            return parent
    raise FileNotFoundError(
        f"could not locate stable-fast-3d relative to {current}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--texture-resolution", default=1024, type=int)
    parser.add_argument(
        "--remesh-option",
        default="none",
        choices=["none", "triangle", "quad"],
    )
    args = parser.parse_args()

    input_path = args.run_dir / "preprocess" / "rgba.png"
    out_dir = args.run_dir / "sf3d"
    out_dir.mkdir(parents=True, exist_ok=True)

    root = find_upstream_repo_root(Path(__file__))
    sf3d_root = root / "stable-fast-3d"
    if str(sf3d_root) not in sys.path:
        sys.path.insert(0, str(sf3d_root))

    from sf3d.system import SF3D  # noqa: E402
    import torch  # noqa: E402

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(device, SF3D)
    with Image.open(input_path) as image:
        mesh = _unwrap_mesh(
            model.run_image(
                image.convert("RGBA"),
                bake_resolution=args.texture_resolution,
                remesh=args.remesh_option,
                vertex_count=-1,
            )
        )

    mesh.export(out_dir / "mesh_raw.glb", include_normals=True)
    (out_dir / "input.png").write_bytes(input_path.read_bytes())
    (out_dir / "worker_meta.json").write_text(
        json.dumps(
            {
                "input_path": str(input_path),
                "mesh_path": str(out_dir / "mesh_raw.glb"),
                "texture_resolution": args.texture_resolution,
                "remesh_option": args.remesh_option,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
