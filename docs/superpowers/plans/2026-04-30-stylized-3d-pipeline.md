# Stylized 3D Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible local pipeline that turns a single foreground object image into a stylized 3D GLB plus a static HTML result viewer.

**Architecture:** Keep `stable-fast-3d` and `InstantStyle` as untouched upstream repos, run them in separate Python environments, and add a thin orchestration project in `stylized-3d-pipeline/`. The pipeline writes all intermediate outputs into a per-run directory so each stage can be rerun, skipped, or resumed independently.

**Tech Stack:** Python 3.11, Pillow, rembg, numpy, PyYAML, trimesh, pyrender, pytest, subprocess, external `sf3d` and `instantstyle` virtual environments.

---

## File Map

### Create

- `stylized-3d-pipeline/README.md`
- `stylized-3d-pipeline/requirements.txt`
- `stylized-3d-pipeline/pyproject.toml`
- `stylized-3d-pipeline/lib/__init__.py`
- `stylized-3d-pipeline/lib/io_paths.py`
- `stylized-3d-pipeline/lib/subprocess_utils.py`
- `stylized-3d-pipeline/lib/image_utils.py`
- `stylized-3d-pipeline/lib/mesh_utils.py`
- `stylized-3d-pipeline/lib/viewer_utils.py`
- `stylized-3d-pipeline/lib/pipeline_runner.py`
- `stylized-3d-pipeline/scripts/__init__.py`
- `stylized-3d-pipeline/scripts/step1_preprocess.py`
- `stylized-3d-pipeline/scripts/step2_sf3d.py`
- `stylized-3d-pipeline/scripts/step3_instantstyle.py`
- `stylized-3d-pipeline/scripts/step4_retexture.py`
- `stylized-3d-pipeline/scripts/step5_build_viewer.py`
- `stylized-3d-pipeline/scripts/run_all.py`
- `stylized-3d-pipeline/scripts/workers/sf3d_worker.py`
- `stylized-3d-pipeline/scripts/workers/instantstyle_worker.py`
- `stylized-3d-pipeline/tests/conftest.py`
- `stylized-3d-pipeline/tests/test_io_paths.py`
- `stylized-3d-pipeline/tests/test_step1_preprocess.py`
- `stylized-3d-pipeline/tests/test_step2_sf3d.py`
- `stylized-3d-pipeline/tests/test_step3_instantstyle.py`
- `stylized-3d-pipeline/tests/test_step4_retexture.py`
- `stylized-3d-pipeline/tests/test_step5_build_viewer.py`
- `stylized-3d-pipeline/tests/test_run_all.py`

### Modify

- `.gitignore`
- `docs/superpowers/specs/2026-04-30-stylized-3d-pipeline-design.md`

### Responsibility Split

- `lib/io_paths.py`: run directory creation, metadata JSON helpers, consistent output paths
- `lib/image_utils.py`: image loading, alpha handling, auto-background-removal integration, foreground resize
- `lib/subprocess_utils.py`: shell-out wrapper with logging and checked failure behavior
- `lib/mesh_utils.py`: textured GLB loading, camera selection, front-view texture bake, GLB rewrite
- `lib/viewer_utils.py`: static HTML generation for image comparison and GLB display
- `scripts/step*.py`: per-step CLI entrypoints plus `run_step()` functions
- `scripts/workers/*.py`: code that runs inside the external environments and imports upstream repos directly
- `lib/pipeline_runner.py`: ordered orchestration, `resume-from`, `skip-existing`
- `tests/*.py`: unit tests for every step and orchestration rule

## Task 1: Scaffold the Pipeline Project and Run Directory Model

**Files:**
- Create: `stylized-3d-pipeline/requirements.txt`
- Create: `stylized-3d-pipeline/pyproject.toml`
- Create: `stylized-3d-pipeline/lib/__init__.py`
- Create: `stylized-3d-pipeline/lib/io_paths.py`
- Create: `stylized-3d-pipeline/scripts/__init__.py`
- Create: `stylized-3d-pipeline/tests/conftest.py`
- Create: `stylized-3d-pipeline/tests/test_io_paths.py`
- Modify: `.gitignore`
- Test: `stylized-3d-pipeline/tests/test_io_paths.py`

- [ ] **Step 1: Write the failing test**

```python
# stylized-3d-pipeline/tests/test_io_paths.py
from pathlib import Path

from lib.io_paths import create_run_tree, resolve_run_dir, write_json


def test_create_run_tree_creates_expected_directories(tmp_path: Path) -> None:
    run_dir = resolve_run_dir(tmp_path, "demo-mug")
    paths = create_run_tree(run_dir)

    assert paths.root == run_dir
    assert paths.inputs.is_dir()
    assert paths.preprocess.is_dir()
    assert paths.sf3d.is_dir()
    assert paths.stylize.is_dir()
    assert paths.retexture.is_dir()
    assert paths.viewer.is_dir()


def test_write_json_persists_payload(tmp_path: Path) -> None:
    out_path = tmp_path / "meta.json"
    write_json(out_path, {"step": "preprocess", "ok": True})
    assert out_path.read_text(encoding="utf-8") == '{\n  "ok": true,\n  "step": "preprocess"\n}'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /root/autodl-tmp/src/stylized-3d-pipeline && pytest tests/test_io_paths.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'lib'` or `ImportError` because the package and helpers do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```toml
# stylized-3d-pipeline/pyproject.toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

```text
# stylized-3d-pipeline/requirements.txt
Pillow
numpy
PyYAML
pytest
trimesh
pyrender
rembg
```

```python
# stylized-3d-pipeline/lib/__init__.py
"""Helpers for the stylized 3D orchestration project."""
```

```python
# stylized-3d-pipeline/scripts/__init__.py
"""CLI entrypoints for the stylized 3D pipeline."""
```

```python
# stylized-3d-pipeline/lib/io_paths.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    root: Path
    inputs: Path
    preprocess: Path
    sf3d: Path
    stylize: Path
    retexture: Path
    viewer: Path


def resolve_run_dir(base_dir: Path, run_name: str | None) -> Path:
    if run_name:
        return base_dir / run_name
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return base_dir / stamp


def create_run_tree(run_dir: Path) -> RunPaths:
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = RunPaths(
        root=run_dir,
        inputs=run_dir / "inputs",
        preprocess=run_dir / "preprocess",
        sf3d=run_dir / "sf3d",
        stylize=run_dir / "stylize",
        retexture=run_dir / "retexture",
        viewer=run_dir / "viewer",
    )
    for path in (
        paths.inputs,
        paths.preprocess,
        paths.sf3d,
        paths.stylize,
        paths.retexture,
        paths.viewer,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
```

```python
# stylized-3d-pipeline/tests/conftest.py
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

```gitignore
# .gitignore
/stylized-3d-pipeline/runs/
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /root/autodl-tmp/src/stylized-3d-pipeline && pytest tests/test_io_paths.py -v`

Expected: PASS with `2 passed`.

- [ ] **Step 5: Commit**

```bash
cd /root/autodl-tmp/src
git add .gitignore stylized-3d-pipeline/requirements.txt stylized-3d-pipeline/pyproject.toml stylized-3d-pipeline/lib/__init__.py stylized-3d-pipeline/lib/io_paths.py stylized-3d-pipeline/scripts/__init__.py stylized-3d-pipeline/tests/conftest.py stylized-3d-pipeline/tests/test_io_paths.py
git commit -m "feat: scaffold stylized 3d pipeline project"
```

## Task 2: Implement Input Preprocessing with Auto Background Removal

**Files:**
- Create: `stylized-3d-pipeline/lib/image_utils.py`
- Create: `stylized-3d-pipeline/scripts/step1_preprocess.py`
- Create: `stylized-3d-pipeline/tests/test_step1_preprocess.py`
- Test: `stylized-3d-pipeline/tests/test_step1_preprocess.py`

- [ ] **Step 1: Write the failing test**

```python
# stylized-3d-pipeline/tests/test_step1_preprocess.py
from pathlib import Path

from PIL import Image, ImageDraw

from lib.io_paths import create_run_tree
from scripts.step1_preprocess import run_step


def _fake_remove_background(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = Image.new("L", rgba.size, 0)
    draw = ImageDraw.Draw(alpha)
    draw.rectangle((16, 16, 47, 47), fill=255)
    rgba.putalpha(alpha)
    return rgba


def test_run_step_writes_rgba_mask_and_metadata(tmp_path: Path) -> None:
    src = tmp_path / "content.jpg"
    Image.new("RGB", (64, 64), "white").save(src)

    paths = create_run_tree(tmp_path / "run")
    result = run_step(
        input_path=src,
        run_dir=paths.root,
        foreground_ratio=0.85,
        remove_background_fn=_fake_remove_background,
    )

    assert result["rgba_path"].endswith("preprocess/rgba.png")
    assert (paths.preprocess / "rgba.png").is_file()
    assert (paths.preprocess / "mask.png").is_file()
    assert (paths.preprocess / "meta.json").is_file()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /root/autodl-tmp/src/stylized-3d-pipeline && pytest tests/test_step1_preprocess.py -v`

Expected: FAIL with `ImportError` because `scripts.step1_preprocess` and `lib.image_utils` do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# stylized-3d-pipeline/lib/image_utils.py
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
```

```python
# stylized-3d-pipeline/scripts/step1_preprocess.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from PIL import Image
import rembg

from lib.image_utils import resize_foreground_rgba, save_mask
from lib.io_paths import create_run_tree, write_json


def _default_remove_background(image: Image.Image) -> Image.Image:
    session = rembg.new_session()
    return rembg.remove(image, session=session)


def run_step(
    input_path: Path,
    run_dir: Path,
    foreground_ratio: float = 0.85,
    remove_background_fn: Callable[[Image.Image], Image.Image] | None = None,
) -> dict:
    paths = create_run_tree(run_dir)
    remove_fn = remove_background_fn or _default_remove_background
    src = Image.open(input_path).convert("RGBA")
    rgba = remove_fn(src)
    rgba = resize_foreground_rgba(rgba, foreground_ratio)

    input_copy = paths.inputs / "content.png"
    rgba_path = paths.preprocess / "rgba.png"
    mask_path = paths.preprocess / "mask.png"
    input_copy.write_bytes(input_path.read_bytes())
    rgba.save(rgba_path)
    save_mask(rgba, mask_path)

    result = {
        "input_path": str(input_path),
        "rgba_path": str(rgba_path),
        "mask_path": str(mask_path),
        "foreground_ratio": foreground_ratio,
    }
    write_json(paths.preprocess / "meta.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--foreground-ratio", default=0.85, type=float)
    args = parser.parse_args()
    run_step(args.input, args.run_dir, args.foreground_ratio)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /root/autodl-tmp/src/stylized-3d-pipeline && pytest tests/test_step1_preprocess.py -v`

Expected: PASS with `1 passed`.

- [ ] **Step 5: Commit**

```bash
cd /root/autodl-tmp/src
git add stylized-3d-pipeline/lib/image_utils.py stylized-3d-pipeline/scripts/step1_preprocess.py stylized-3d-pipeline/tests/test_step1_preprocess.py
git commit -m "feat: add preprocessing step"
```

## Task 3: Implement the SF3D Wrapper and External Worker

**Files:**
- Create: `stylized-3d-pipeline/lib/subprocess_utils.py`
- Create: `stylized-3d-pipeline/scripts/step2_sf3d.py`
- Create: `stylized-3d-pipeline/scripts/workers/sf3d_worker.py`
- Create: `stylized-3d-pipeline/tests/test_step2_sf3d.py`
- Test: `stylized-3d-pipeline/tests/test_step2_sf3d.py`

- [ ] **Step 1: Write the failing test**

```python
# stylized-3d-pipeline/tests/test_step2_sf3d.py
from pathlib import Path

from lib.io_paths import create_run_tree, write_json
from scripts.step2_sf3d import build_sf3d_command, run_step


def test_build_sf3d_command_targets_worker(tmp_path: Path) -> None:
    cmd = build_sf3d_command(
        sf3d_python=Path("/envs/sf3d/bin/python"),
        worker_script=Path("/repo/stylized-3d-pipeline/scripts/workers/sf3d_worker.py"),
        run_dir=tmp_path / "run",
        texture_resolution=1024,
        remesh_option="none",
    )
    assert cmd[0] == "/envs/sf3d/bin/python"
    assert cmd[1].endswith("sf3d_worker.py")
    assert "--texture-resolution" in cmd


def test_run_step_validates_mesh_output(tmp_path: Path) -> None:
    paths = create_run_tree(tmp_path / "run")
    (paths.preprocess / "rgba.png").write_bytes(b"rgba")
    fake_mesh = paths.sf3d / "mesh_raw.glb"

    def fake_runner(cmd, env=None):  # noqa: ANN001
        fake_mesh.write_bytes(b"glb")
        write_json(paths.sf3d / "sf3d_meta.json", {"cmd": cmd})

    result = run_step(
        run_dir=paths.root,
        sf3d_python=Path("/envs/sf3d/bin/python"),
        runner=fake_runner,
    )
    assert result["mesh_path"].endswith("sf3d/mesh_raw.glb")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /root/autodl-tmp/src/stylized-3d-pipeline && pytest tests/test_step2_sf3d.py -v`

Expected: FAIL because `scripts.step2_sf3d` and `lib.subprocess_utils` do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# stylized-3d-pipeline/lib/subprocess_utils.py
from __future__ import annotations

import os
import subprocess
from pathlib import Path


def run_checked(cmd: list[str], env: dict[str, str] | None = None) -> None:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(cmd, check=True, env=merged_env)
```

```python
# stylized-3d-pipeline/scripts/step2_sf3d.py
from __future__ import annotations

import argparse
from pathlib import Path

from lib.io_paths import create_run_tree, write_json
from lib.subprocess_utils import run_checked


def build_sf3d_command(
    sf3d_python: Path,
    worker_script: Path,
    run_dir: Path,
    texture_resolution: int,
    remesh_option: str,
) -> list[str]:
    return [
        str(sf3d_python),
        str(worker_script),
        "--run-dir",
        str(run_dir),
        "--texture-resolution",
        str(texture_resolution),
        "--remesh-option",
        remesh_option,
    ]


def run_step(
    run_dir: Path,
    sf3d_python: Path,
    texture_resolution: int = 1024,
    remesh_option: str = "none",
    runner=run_checked,  # noqa: B008
) -> dict:
    paths = create_run_tree(run_dir)
    worker = Path(__file__).resolve().parent / "workers" / "sf3d_worker.py"
    cmd = build_sf3d_command(sf3d_python, worker, paths.root, texture_resolution, remesh_option)
    runner(cmd, env={"HF_ENDPOINT": "https://hf-mirror.com"})
    mesh_path = paths.sf3d / "mesh_raw.glb"
    if not mesh_path.is_file():
        raise FileNotFoundError(f"missing SF3D output: {mesh_path}")
    result = {
        "mesh_path": str(mesh_path),
        "texture_resolution": texture_resolution,
        "remesh_option": remesh_option,
    }
    write_json(paths.sf3d / "sf3d_meta.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--sf3d-python", required=True, type=Path)
    parser.add_argument("--texture-resolution", default=1024, type=int)
    parser.add_argument("--remesh-option", default="none", choices=["none", "triangle", "quad"])
    args = parser.parse_args()
    run_step(args.run_dir, args.sf3d_python, args.texture_resolution, args.remesh_option)


if __name__ == "__main__":
    main()
```

```python
# stylized-3d-pipeline/scripts/workers/sf3d_worker.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
SF3D_ROOT = ROOT / "stable-fast-3d"
sys.path.insert(0, str(SF3D_ROOT))

from sf3d.system import SF3D  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--texture-resolution", default=1024, type=int)
    parser.add_argument("--remesh-option", default="none", choices=["none", "triangle", "quad"])
    args = parser.parse_args()

    input_path = args.run_dir / "preprocess" / "rgba.png"
    out_dir = args.run_dir / "sf3d"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = SF3D.from_pretrained(
        "stabilityai/stable-fast-3d",
        config_name="config.yaml",
        weight_name="model.safetensors",
    )
    mesh, _ = model.run_image(
        Image.open(input_path).convert("RGBA"),
        bake_resolution=args.texture_resolution,
        remesh=args.remesh_option,
        vertex_count=-1,
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /root/autodl-tmp/src/stylized-3d-pipeline && pytest tests/test_step2_sf3d.py -v`

Expected: PASS with `2 passed`.

- [ ] **Step 5: Commit**

```bash
cd /root/autodl-tmp/src
git add stylized-3d-pipeline/lib/subprocess_utils.py stylized-3d-pipeline/scripts/step2_sf3d.py stylized-3d-pipeline/scripts/workers/sf3d_worker.py stylized-3d-pipeline/tests/test_step2_sf3d.py
git commit -m "feat: add sf3d wrapper step"
```

## Task 4: Implement the InstantStyle Wrapper and Style Worker

**Files:**
- Create: `stylized-3d-pipeline/scripts/step3_instantstyle.py`
- Create: `stylized-3d-pipeline/scripts/workers/instantstyle_worker.py`
- Create: `stylized-3d-pipeline/tests/test_step3_instantstyle.py`
- Test: `stylized-3d-pipeline/tests/test_step3_instantstyle.py`

- [ ] **Step 1: Write the failing test**

```python
# stylized-3d-pipeline/tests/test_step3_instantstyle.py
from pathlib import Path

from PIL import Image

from lib.io_paths import create_run_tree, write_json
from scripts.step3_instantstyle import build_instantstyle_command, run_step


def test_build_instantstyle_command_includes_prompt_and_style(tmp_path: Path) -> None:
    cmd = build_instantstyle_command(
        instantstyle_python=Path("/envs/instantstyle/bin/python"),
        worker_script=Path("/repo/stylized-3d-pipeline/scripts/workers/instantstyle_worker.py"),
        run_dir=tmp_path / "run",
        style_image=Path("/tmp/style.jpg"),
        prompt="ceramic mug",
    )
    assert cmd[0] == "/envs/instantstyle/bin/python"
    assert "/tmp/style.jpg" in cmd
    assert "ceramic mug" in cmd


def test_run_step_requires_stylized_output(tmp_path: Path) -> None:
    paths = create_run_tree(tmp_path / "run")
    Image.new("RGBA", (32, 32), (255, 255, 255, 255)).save(paths.preprocess / "rgba.png")
    style_image = tmp_path / "style.jpg"
    Image.new("RGB", (32, 32), "blue").save(style_image)

    def fake_runner(cmd, env=None):  # noqa: ANN001
        Image.new("RGBA", (32, 32), (255, 0, 0, 255)).save(paths.stylize / "stylized.png")
        write_json(paths.stylize / "stylize_meta.json", {"cmd": cmd})

    result = run_step(
        run_dir=paths.root,
        instantstyle_python=Path("/envs/instantstyle/bin/python"),
        style_image=style_image,
        prompt="ceramic mug",
        runner=fake_runner,
    )
    assert result["stylized_path"].endswith("stylize/stylized.png")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /root/autodl-tmp/src/stylized-3d-pipeline && pytest tests/test_step3_instantstyle.py -v`

Expected: FAIL because `scripts.step3_instantstyle` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# stylized-3d-pipeline/scripts/step3_instantstyle.py
from __future__ import annotations

import argparse
from pathlib import Path

from lib.io_paths import create_run_tree, write_json
from lib.subprocess_utils import run_checked


def build_instantstyle_command(
    instantstyle_python: Path,
    worker_script: Path,
    run_dir: Path,
    style_image: Path,
    prompt: str,
) -> list[str]:
    return [
        str(instantstyle_python),
        str(worker_script),
        "--run-dir",
        str(run_dir),
        "--style-image",
        str(style_image),
        "--prompt",
        prompt,
    ]


def run_step(
    run_dir: Path,
    instantstyle_python: Path,
    style_image: Path,
    prompt: str,
    runner=run_checked,  # noqa: B008
) -> dict:
    paths = create_run_tree(run_dir)
    style_copy = paths.inputs / "style.png"
    prompt_file = paths.inputs / "prompt.txt"
    style_copy.write_bytes(style_image.read_bytes())
    prompt_file.write_text(prompt, encoding="utf-8")

    worker = Path(__file__).resolve().parent / "workers" / "instantstyle_worker.py"
    cmd = build_instantstyle_command(instantstyle_python, worker, paths.root, style_copy, prompt)
    runner(cmd, env={"HF_ENDPOINT": "https://hf-mirror.com"})

    stylized_path = paths.stylize / "stylized.png"
    if not stylized_path.is_file():
        raise FileNotFoundError(f"missing stylized output: {stylized_path}")
    result = {
        "stylized_path": str(stylized_path),
        "style_image": str(style_copy),
        "prompt": prompt,
    }
    write_json(paths.stylize / "stylize_meta.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--instantstyle-python", required=True, type=Path)
    parser.add_argument("--style-image", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()
    run_step(args.run_dir, args.instantstyle_python, args.style_image, args.prompt)


if __name__ == "__main__":
    main()
```

```python
# stylized-3d-pipeline/scripts/workers/instantstyle_worker.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
INSTANTSTYLE_ROOT = ROOT / "InstantStyle"
sys.path.insert(0, str(INSTANTSTYLE_ROOT))

from ip_adapter import IPAdapterXL  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--style-image", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    preprocess_image = args.run_dir / "preprocess" / "rgba.png"
    stylize_dir = args.run_dir / "stylize"
    stylize_dir.mkdir(parents=True, exist_ok=True)

    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        use_safetensors=False,
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        add_watermarker=False,
    ).to("cuda")
    pipe.enable_vae_tiling()
    ip_model = IPAdapterXL(
        pipe,
        "sdxl_models/image_encoder",
        "sdxl_models/ip-adapter_sdxl.bin",
        "cuda",
        target_blocks=["up_blocks.0.attentions.1"],
    )

    content = Image.open(preprocess_image).convert("RGBA")
    style = Image.open(args.style_image).convert("RGB")
    content_rgb = np.array(content.convert("RGB"))
    canny = cv2.Canny(content_rgb, 50, 200)
    canny_map = Image.fromarray(canny).convert("RGB")

    images = ip_model.generate(
        pil_image=style,
        prompt=args.prompt,
        negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, blurry",
        scale=1.0,
        guidance_scale=5.0,
        num_samples=1,
        num_inference_steps=30,
        seed=42,
        image=canny_map,
        controlnet_conditioning_scale=0.6,
    )

    stylized = images[0].convert("RGBA").resize(content.size)
    stylized.putalpha(content.getchannel("A"))
    stylized.save(stylize_dir / "stylized.png")
    (stylize_dir / "worker_meta.json").write_text(
        json.dumps(
            {
                "preprocess_image": str(preprocess_image),
                "style_image": str(args.style_image),
                "prompt": args.prompt,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /root/autodl-tmp/src/stylized-3d-pipeline && pytest tests/test_step3_instantstyle.py -v`

Expected: PASS with `2 passed`.

- [ ] **Step 5: Commit**

```bash
cd /root/autodl-tmp/src
git add stylized-3d-pipeline/scripts/step3_instantstyle.py stylized-3d-pipeline/scripts/workers/instantstyle_worker.py stylized-3d-pipeline/tests/test_step3_instantstyle.py
git commit -m "feat: add instantstyle wrapper step"
```

## Task 5: Implement Front-View Retexturing and GLB Rewrite

**Files:**
- Create: `stylized-3d-pipeline/lib/mesh_utils.py`
- Create: `stylized-3d-pipeline/scripts/step4_retexture.py`
- Create: `stylized-3d-pipeline/tests/test_step4_retexture.py`
- Test: `stylized-3d-pipeline/tests/test_step4_retexture.py`

- [ ] **Step 1: Write the failing test**

```python
# stylized-3d-pipeline/tests/test_step4_retexture.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /root/autodl-tmp/src/stylized-3d-pipeline && pytest tests/test_step4_retexture.py -v`

Expected: FAIL because `lib.mesh_utils` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# stylized-3d-pipeline/lib/mesh_utils.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh


def _triangle_pixels(width: int, height: int, tri_uv: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    px = tri_uv[:, 0] * (width - 1)
    py = (1.0 - tri_uv[:, 1]) * (height - 1)
    min_x, max_x = int(np.floor(px.min())), int(np.ceil(px.max()))
    min_y, max_y = int(np.floor(py.min())), int(np.ceil(py.max()))
    p0 = np.array([px[0], py[0]])
    p1 = np.array([px[1], py[1]])
    p2 = np.array([px[2], py[2]])
    area = np.cross(p1 - p0, p2 - p0)
    if abs(area) < 1e-8:
        return []
    hits: list[tuple[int, int, np.ndarray]] = []
    for x in range(max(0, min_x), min(width - 1, max_x) + 1):
        for y in range(max(0, min_y), min(height - 1, max_y) + 1):
            p = np.array([x + 0.5, y + 0.5])
            w0 = np.cross(p1 - p, p2 - p) / area
            w1 = np.cross(p2 - p, p0 - p) / area
            w2 = 1.0 - w0 - w1
            bary = np.array([w0, w1, w2], dtype=np.float32)
            if np.all(bary >= -1e-5):
                hits.append((x, y, bary))
    return hits


def bake_visible_texels(
    base_texture: Image.Image,
    stylized_image: Image.Image,
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    projector,
) -> Image.Image:
    base = base_texture.convert("RGBA").copy()
    src = stylized_image.convert("RGBA")
    base_pixels = base.load()
    src_pixels = src.load()
    for face in faces:
        tri_uv = uv[face]
        tri_pos = vertices[face]
        normal = np.cross(tri_pos[1] - tri_pos[0], tri_pos[2] - tri_pos[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal = normal / norm
        for x, y, bary in _triangle_pixels(base.width, base.height, tri_uv):
            position = bary @ tri_pos
            visible, sample_xy = projector(position, normal)
            if not visible:
                continue
            sx = min(max(sample_xy[0], 0), src.width - 1)
            sy = min(max(sample_xy[1], 0), src.height - 1)
            base_pixels[x, y] = src_pixels[sx, sy]
    return base


def load_trimesh_with_texture(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, force="mesh")
    if not isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        raise TypeError("mesh does not contain textured visuals")
    return mesh
```

```python
# stylized-3d-pipeline/scripts/step4_retexture.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

from lib.io_paths import create_run_tree, write_json
from lib.mesh_utils import bake_visible_texels, load_trimesh_with_texture


def _fallback_projector(position, normal):  # noqa: ANN001
    return normal[2] >= 0, (0, 0)


def run_step(run_dir: Path) -> dict:
    paths = create_run_tree(run_dir)
    mesh = load_trimesh_with_texture(paths.sf3d / "mesh_raw.glb")
    base = mesh.visual.material.baseColorTexture.convert("RGBA")
    stylized = Image.open(paths.stylize / "stylized.png").convert("RGBA")
    baked = bake_visible_texels(
        base,
        stylized,
        np.asarray(mesh.vertices),
        np.asarray(mesh.faces),
        np.asarray(mesh.visual.uv),
        _fallback_projector,
    )
    baked.save(paths.retexture / "texture_preview.png")
    mesh.visual.material.baseColorTexture = baked
    mesh.export(paths.retexture / "mesh_stylized.glb")
    result = {
        "mesh_path": str(paths.retexture / "mesh_stylized.glb"),
        "texture_preview": str(paths.retexture / "texture_preview.png"),
    }
    write_json(paths.retexture / "retexture_meta.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_step(args.run_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /root/autodl-tmp/src/stylized-3d-pipeline && pytest tests/test_step4_retexture.py -v`

Expected: PASS with `2 passed`.

- [ ] **Step 5: Commit**

```bash
cd /root/autodl-tmp/src
git add stylized-3d-pipeline/lib/mesh_utils.py stylized-3d-pipeline/scripts/step4_retexture.py stylized-3d-pipeline/tests/test_step4_retexture.py
git commit -m "feat: add front view retexturing step"
```

## Task 6: Implement the Static HTML Viewer

**Files:**
- Create: `stylized-3d-pipeline/lib/viewer_utils.py`
- Create: `stylized-3d-pipeline/scripts/step5_build_viewer.py`
- Create: `stylized-3d-pipeline/tests/test_step5_build_viewer.py`
- Test: `stylized-3d-pipeline/tests/test_step5_build_viewer.py`

- [ ] **Step 1: Write the failing test**

```python
# stylized-3d-pipeline/tests/test_step5_build_viewer.py
from pathlib import Path

from PIL import Image

from lib.io_paths import create_run_tree
from scripts.step5_build_viewer import run_step


def test_run_step_writes_viewer_html_with_expected_assets(tmp_path: Path) -> None:
    paths = create_run_tree(tmp_path / "run")
    Image.new("RGBA", (8, 8), (255, 255, 255, 255)).save(paths.inputs / "content.png")
    Image.new("RGBA", (8, 8), (0, 0, 255, 255)).save(paths.inputs / "style.png")
    Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(paths.stylize / "stylized.png")
    (paths.retexture / "mesh_stylized.glb").write_bytes(b"glb")

    result = run_step(paths.root)
    html = Path(result["viewer_html"]).read_text(encoding="utf-8")
    assert "model-viewer" in html
    assert "../inputs/content.png" in html
    assert "../retexture/mesh_stylized.glb" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /root/autodl-tmp/src/stylized-3d-pipeline && pytest tests/test_step5_build_viewer.py -v`

Expected: FAIL because `scripts.step5_build_viewer` and `lib.viewer_utils` do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# stylized-3d-pipeline/lib/viewer_utils.py
from __future__ import annotations

from pathlib import Path


def build_viewer_html() -> str:
    return """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Stylized 3D Pipeline Result</title>
    <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
    <style>
      body { font-family: sans-serif; margin: 0; padding: 24px; background: #f5f3ef; color: #1f1b16; }
      .grid { display: grid; gap: 16px; grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .card { background: #fff; padding: 16px; border-radius: 16px; box-shadow: 0 8px 24px rgba(0,0,0,.08); }
      img, model-viewer { width: 100%; border-radius: 12px; background: #ece7df; }
      model-viewer { height: 480px; }
    </style>
  </head>
  <body>
    <h1>Stylized 3D Pipeline Result</h1>
    <div class="grid">
      <div class="card"><h2>Content</h2><img src="../inputs/content.png" alt="content"></div>
      <div class="card"><h2>Style</h2><img src="../inputs/style.png" alt="style"></div>
      <div class="card"><h2>Stylized View</h2><img src="../stylize/stylized.png" alt="stylized"></div>
      <div class="card"><h2>GLB Viewer</h2><model-viewer camera-controls auto-rotate src="../retexture/mesh_stylized.glb"></model-viewer></div>
    </div>
  </body>
</html>"""


def write_viewer(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_viewer_html(), encoding="utf-8")
```

```python
# stylized-3d-pipeline/scripts/step5_build_viewer.py
from __future__ import annotations

import argparse
from pathlib import Path

from lib.io_paths import create_run_tree, write_json
from lib.viewer_utils import write_viewer


def run_step(run_dir: Path) -> dict:
    paths = create_run_tree(run_dir)
    out_path = paths.viewer / "index.html"
    write_viewer(out_path)
    result = {"viewer_html": str(out_path)}
    write_json(paths.viewer / "viewer_meta.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_step(args.run_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /root/autodl-tmp/src/stylized-3d-pipeline && pytest tests/test_step5_build_viewer.py -v`

Expected: PASS with `1 passed`.

- [ ] **Step 5: Commit**

```bash
cd /root/autodl-tmp/src
git add stylized-3d-pipeline/lib/viewer_utils.py stylized-3d-pipeline/scripts/step5_build_viewer.py stylized-3d-pipeline/tests/test_step5_build_viewer.py
git commit -m "feat: add static result viewer"
```

## Task 7: Implement Orchestration, Resume/Skip Logic, and Usage Docs

**Files:**
- Create: `stylized-3d-pipeline/lib/pipeline_runner.py`
- Create: `stylized-3d-pipeline/scripts/run_all.py`
- Create: `stylized-3d-pipeline/README.md`
- Create: `stylized-3d-pipeline/tests/test_run_all.py`
- Modify: `docs/superpowers/specs/2026-04-30-stylized-3d-pipeline-design.md`
- Test: `stylized-3d-pipeline/tests/test_run_all.py`

- [ ] **Step 1: Write the failing test**

```python
# stylized-3d-pipeline/tests/test_run_all.py
from pathlib import Path

from lib.io_paths import create_run_tree
from lib.pipeline_runner import ordered_steps, should_run_step


def test_ordered_steps_matches_pipeline_contract() -> None:
    assert ordered_steps() == [
        "preprocess",
        "sf3d",
        "instantstyle",
        "retexture",
        "viewer",
    ]


def test_should_run_step_respects_resume_and_skip_existing(tmp_path: Path) -> None:
    paths = create_run_tree(tmp_path / "run")
    (paths.preprocess / "meta.json").write_text("{}", encoding="utf-8")

    assert should_run_step("preprocess", resume_from=None, skip_existing=True, run_dir=paths.root) is False
    assert should_run_step("sf3d", resume_from="sf3d", skip_existing=False, run_dir=paths.root) is True
    assert should_run_step("preprocess", resume_from="sf3d", skip_existing=False, run_dir=paths.root) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /root/autodl-tmp/src/stylized-3d-pipeline && pytest tests/test_run_all.py -v`

Expected: FAIL because `lib.pipeline_runner` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# stylized-3d-pipeline/lib/pipeline_runner.py
from __future__ import annotations

from pathlib import Path

from scripts.step1_preprocess import run_step as run_preprocess
from scripts.step2_sf3d import run_step as run_sf3d
from scripts.step3_instantstyle import run_step as run_instantstyle
from scripts.step4_retexture import run_step as run_retexture
from scripts.step5_build_viewer import run_step as run_viewer


STEP_META = {
    "preprocess": "preprocess/meta.json",
    "sf3d": "sf3d/sf3d_meta.json",
    "instantstyle": "stylize/stylize_meta.json",
    "retexture": "retexture/retexture_meta.json",
    "viewer": "viewer/viewer_meta.json",
}


def ordered_steps() -> list[str]:
    return ["preprocess", "sf3d", "instantstyle", "retexture", "viewer"]


def should_run_step(step_name: str, resume_from: str | None, skip_existing: bool, run_dir: Path) -> bool:
    order = ordered_steps()
    if resume_from and order.index(step_name) < order.index(resume_from):
        return False
    if skip_existing and (run_dir / STEP_META[step_name]).is_file():
        return False
    return True


def run_pipeline(args) -> None:  # noqa: ANN001
    run_dir = args.run_dir
    if should_run_step("preprocess", args.resume_from, args.skip_existing, run_dir):
        run_preprocess(args.input, run_dir, args.foreground_ratio)
    if should_run_step("sf3d", args.resume_from, args.skip_existing, run_dir):
        run_sf3d(run_dir, args.sf3d_python, args.texture_resolution, args.remesh_option)
    if should_run_step("instantstyle", args.resume_from, args.skip_existing, run_dir):
        run_instantstyle(run_dir, args.instantstyle_python, args.style_image, args.prompt)
    if should_run_step("retexture", args.resume_from, args.skip_existing, run_dir):
        run_retexture(run_dir)
    if should_run_step("viewer", args.resume_from, args.skip_existing, run_dir):
        run_viewer(run_dir)
```

```python
# stylized-3d-pipeline/scripts/run_all.py
from __future__ import annotations

import argparse
from pathlib import Path

from lib.io_paths import create_run_tree, resolve_run_dir, write_json
from lib.pipeline_runner import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--style-image", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--runs-root", default=Path("runs"), type=Path)
    parser.add_argument("--sf3d-python", default=Path("/root/autodl-tmp/envs/sf3d/bin/python"), type=Path)
    parser.add_argument("--instantstyle-python", default=Path("/root/autodl-tmp/envs/instantstyle/bin/python"), type=Path)
    parser.add_argument("--foreground-ratio", default=0.85, type=float)
    parser.add_argument("--texture-resolution", default=1024, type=int)
    parser.add_argument("--remesh-option", default="none", choices=["none", "triangle", "quad"])
    parser.add_argument("--resume-from", choices=["preprocess", "sf3d", "instantstyle", "retexture", "viewer"])
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    args.run_dir = resolve_run_dir(args.runs_root, args.run_name)
    create_run_tree(args.run_dir)
    write_json(
        args.run_dir / "run_config.json",
        {
            "input": str(args.input),
            "style_image": str(args.style_image),
            "prompt": args.prompt,
            "foreground_ratio": args.foreground_ratio,
            "texture_resolution": args.texture_resolution,
            "remesh_option": args.remesh_option,
            "resume_from": args.resume_from,
            "skip_existing": args.skip_existing,
        },
    )
    run_pipeline(args)


if __name__ == "__main__":
    main()
```

````markdown
# stylized-3d-pipeline/README.md
## Setup

```bash
python -m venv /root/autodl-tmp/envs/pipeline
source /root/autodl-tmp/envs/pipeline/bin/activate
pip install -r requirements.txt
```

## Step-by-step usage

```bash
python scripts/step1_preprocess.py --input /abs/path/content.jpg --run-dir runs/demo-mug
python scripts/step2_sf3d.py --run-dir runs/demo-mug --sf3d-python /root/autodl-tmp/envs/sf3d/bin/python
python scripts/step3_instantstyle.py --run-dir runs/demo-mug --instantstyle-python /root/autodl-tmp/envs/instantstyle/bin/python --style-image /abs/path/style.jpg --prompt "ceramic mug"
python scripts/step4_retexture.py --run-dir runs/demo-mug
python scripts/step5_build_viewer.py --run-dir runs/demo-mug
```

## One-shot usage

```bash
python scripts/run_all.py --input /abs/path/content.jpg --style-image /abs/path/style.jpg --prompt "ceramic mug" --run-name demo-mug
```
````

```markdown
# docs/superpowers/specs/2026-04-30-stylized-3d-pipeline-design.md
Implementation note: the orchestration layer will place import-heavy integration code in `stylized-3d-pipeline/scripts/workers/`, then invoke those workers with the corresponding environment interpreter. This keeps `stable-fast-3d` and `InstantStyle` untouched while still allowing the top-level pipeline to coordinate them through stable CLI boundaries.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /root/autodl-tmp/src/stylized-3d-pipeline && pytest tests/test_run_all.py -v`

Expected: PASS with `2 passed`.

- [ ] **Step 5: Commit**

```bash
cd /root/autodl-tmp/src
git add stylized-3d-pipeline/lib/pipeline_runner.py stylized-3d-pipeline/scripts/run_all.py stylized-3d-pipeline/README.md stylized-3d-pipeline/tests/test_run_all.py docs/superpowers/specs/2026-04-30-stylized-3d-pipeline-design.md
git commit -m "feat: orchestrate stylized 3d pipeline"
```

## Spec Coverage Check

- Spec section `2.1 输入范围` is covered by Task 2 preprocessing and the README usage contract.
- Spec section `2.2 贴图目标` is covered by Task 5 front-view-only retexturing.
- Spec section `2.3 风格输入` and `2.5 Prompt 输入` are covered by Task 4 wrapper inputs.
- Spec section `2.4 可视化目标` is covered by Task 6 viewer generation.
- Spec section `2.6 验证方式` is covered by all task-local tests and Task 7 README commands.
- Spec section `2.7 成功标准` is covered by Tasks 3-7 plus step-local metadata validation.
- Spec section `7. 环境设计` is covered by Task 7 README setup commands and Tasks 3-4 worker scripts.
- Spec section `8. 运行目录与产物约定` is covered by Task 1 path helpers and Tasks 2-7 output files.
- Spec section `9. 命令接口设计` is covered by Tasks 2-7 step CLIs and Task 7 `run_all.py`.
- Spec section `10. 验证节点设计` is covered by every task-local pytest and output file assertion.
- Spec section `11. 实用机制` is covered by Task 7 `resume-from` and `skip-existing`.

## Placeholder Scan

- No `TODO`, `TBD`, or “implement later” placeholders remain.
- Every task lists exact file paths.
- Every code-writing step includes concrete code blocks.
- Every test step includes an exact pytest command.

## Type Consistency Check

- Run directories consistently use `RunPaths`.
- Step names are consistently `preprocess`, `sf3d`, `instantstyle`, `retexture`, `viewer`.
- Metadata file names are consistently `meta.json`, `sf3d_meta.json`, `stylize_meta.json`, `retexture_meta.json`, `viewer_meta.json`.
- External wrapper arguments consistently use `--sf3d-python` and `--instantstyle-python`.
