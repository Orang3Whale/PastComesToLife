from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def alpha_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("foreground mask is empty")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def resize_foreground_rgba(image: Image.Image, foreground_ratio: float) -> Image.Image:
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    x0, y0, x1, y1 = alpha_bbox(arr[..., 3])
    crop = rgba.crop((x0, y0, x1, y1))
    side = max(crop.size)
    target = max(64, int(side / max(foreground_ratio, 1e-3)))
    canvas = Image.new("RGBA", (target, target), (0, 0, 0, 0))
    scale = min(target * foreground_ratio / crop.width, target * foreground_ratio / crop.height)
    resized = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))))
    left = (target - resized.width) // 2
    top = (target - resized.height) // 2
    canvas.paste(resized, (left, top), resized)
    return canvas


def save_mask(rgba: Image.Image, out_path: Path) -> None:
    rgba.getchannel("A").save(out_path)
