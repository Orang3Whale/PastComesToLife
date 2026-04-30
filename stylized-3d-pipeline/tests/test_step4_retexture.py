from pathlib import Path

import numpy as np
from PIL import Image

from lib.mesh_utils import bake_visible_texels


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
