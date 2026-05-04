# Offscreen Multiview Render Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace UV atlas sampling in `sample_views` with direct offscreen mesh rendering so the exported multiview images match the clean viewer appearance instead of exposing SF3D texture holes.

**Architecture:** Add a focused `pyrender` backend that renders a neutral mesh into RGBA color + depth buffers. Keep `lib/view_sampling.py` as the adaptor that turns those raw buffers into the existing asset bundle (`rgb.png`, `depth.npy`, `depth.png`, `normal.png`, `mask.png`, `control.png`). Camera fitting remains delegated to the existing `build_six_view_spec()` path; this plan preserves that auto-fit behavior while replacing only the brittle UV-atlas view generation. `scripts/step3_sample_views.py` continues to orchestrate the step and writes a manifest that marks the render mode as `mesh_offscreen`; downstream `instantstyle` and `retexture` keep the same entrypoints.

**Tech Stack:** Python 3.11, `pyrender`, `trimesh`, `Pillow`, `numpy`, `pytest`, existing run-tree helpers.

---

## File Map

### Create

- `stylized-3d-pipeline/lib/offscreen_renderer.py`
- `stylized-3d-pipeline/tests/test_offscreen_renderer.py`

### Modify

- `stylized-3d-pipeline/lib/view_sampling.py`
- `stylized-3d-pipeline/scripts/step3_sample_views.py`
- `stylized-3d-pipeline/tests/test_step3_sample_views.py`
- `stylized-3d-pipeline/tests/test_step4_retexture.py`

### Responsibility Split

- `lib/offscreen_renderer.py`: convert a textured SF3D mesh into a neutral renderable mesh, render color/depth with `pyrender`, and wrap renderer failures with a clear headless-backend error.
- `lib/view_sampling.py`: derive `mask`, `depth_preview`, `normal`, and `control` from the offscreen depth/color outputs and keep the write-path stable.
- `scripts/step3_sample_views.py`: keep CLI and manifest orchestration, add `render_mode: "mesh_offscreen"` to the manifest payload.
- `tests/test_offscreen_renderer.py`: validate the new renderer helper in isolation with injected fake renderers.
- `tests/test_step3_sample_views.py`: validate the step-level manifest and asset wiring.
- `tests/test_step4_retexture.py`: keep a regression that downstream manifest parsing still ignores the new top-level `render_mode` field.

## Task 1: Add a Neutral Offscreen Renderer Backend

**Files:**
- Create: `stylized-3d-pipeline/lib/offscreen_renderer.py`
- Create: `stylized-3d-pipeline/tests/test_offscreen_renderer.py`

- [ ] **Step 1: Write the failing tests**

```python
# stylized-3d-pipeline/tests/test_offscreen_renderer.py
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from lib.camera_views import CameraView, look_at
from lib.offscreen_renderer import build_neutral_render_mesh, render_offscreen_view


def _textured_triangle() -> trimesh.Trimesh:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = TextureVisuals(
        uv=uv,
        material=PBRMaterial(baseColorTexture=Image.new("RGBA", (4, 4), (0, 0, 0, 255))),
    )
    return mesh


def test_build_neutral_render_mesh_drops_source_texture() -> None:
    neutral = build_neutral_render_mesh(_textured_triangle())

    assert neutral.visual.kind != "texture"
    assert neutral.visual.vertex_colors.shape[1] == 4
    assert np.all(neutral.visual.vertex_colors[:, :3] == 235)


def test_render_offscreen_view_returns_rgba_color_and_depth() -> None:
    class FakeRenderer:
        def __init__(self, viewport_width: int, viewport_height: int) -> None:
            self.viewport_width = viewport_width
            self.viewport_height = viewport_height
            self.deleted = False

        def render(self, scene, flags):  # noqa: ANN001
            color = np.zeros((self.viewport_height, self.viewport_width, 4), dtype=np.uint8)
            color[2:6, 2:6, :3] = 220
            color[2:6, 2:6, 3] = 255
            depth = np.zeros((self.viewport_height, self.viewport_width), dtype=np.float32)
            depth[2:6, 2:6] = 1.0
            return color, depth

        def delete(self) -> None:
            self.deleted = True

    view = CameraView(
        name="front",
        pose=look_at(
            np.array([2.0, 0.0, 0.0], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        ),
        fovy_deg=40.0,
    )
    assets = render_offscreen_view(
        _textured_triangle(),
        view,
        resolution=8,
        renderer_factory=lambda width, height: FakeRenderer(width, height),
    )

    rgb = np.asarray(assets["rgb"].convert("RGBA"))
    depth = assets["depth"]

    assert rgb.shape == (8, 8, 4)
    assert depth.shape == (8, 8)
    assert rgb[0, 0, 3] == 0
    assert rgb[3, 3, 3] == 255
    assert rgb[3, 3, 0] == 220


def test_render_offscreen_view_raises_clear_error_when_backend_missing() -> None:
    view = CameraView(
        name="front",
        pose=look_at(
            np.array([2.0, 0.0, 0.0], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        ),
        fovy_deg=40.0,
    )

    def bad_factory(*args, **kwargs):  # noqa: ANN001, ANN003
        raise ValueError("no backend")

    with pytest.raises(RuntimeError, match="PYOPENGL_PLATFORM|OSMesa|EGL"):
        render_offscreen_view(
            _textured_triangle(),
            view,
            resolution=8,
            renderer_factory=bad_factory,
        )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /root/autodl-tmp/src/.worktrees/stylized-3d-pipeline/stylized-3d-pipeline && pytest tests/test_offscreen_renderer.py::test_build_neutral_render_mesh_drops_source_texture -v`

Expected: fail with `ModuleNotFoundError` or missing symbol errors because `lib/offscreen_renderer.py` does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
# stylized-3d-pipeline/lib/offscreen_renderer.py
from __future__ import annotations

from typing import Callable

import numpy as np
import pyrender
import trimesh
from PIL import Image

from lib.camera_views import CameraView


def build_neutral_render_mesh(
    mesh: trimesh.Trimesh,
    base_color: tuple[int, int, int, int] = (235, 235, 235, 255),
) -> trimesh.Trimesh:
    neutral = mesh.copy()
    vertex_colors = np.tile(np.asarray(base_color, dtype=np.uint8), (len(neutral.vertices), 1))
    neutral.visual = trimesh.visual.color.ColorVisuals(mesh=neutral, vertex_colors=vertex_colors)
    return neutral


def _make_scene(mesh: trimesh.Trimesh, view: CameraView) -> pyrender.Scene:
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.18, 0.18, 0.18])
    render_mesh = pyrender.Mesh.from_trimesh(build_neutral_render_mesh(mesh), smooth=False)
    scene.add(render_mesh, pose=np.eye(4, dtype=np.float32))
    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(view.fovy_deg))
    scene.add(camera, pose=view.pose)

    light_poses = [
        view.pose @ np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.92, -0.38, 0.0], [0.0, 0.38, 0.92, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        view.pose @ np.array([[0.92, 0.0, 0.38, 0.0], [0.0, 1.0, 0.0, 0.0], [-0.38, 0.0, 0.92, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    ]
    for light_pose in light_poses:
        scene.add(pyrender.DirectionalLight(color=np.ones(3, dtype=np.float32), intensity=2.2), pose=light_pose)
    return scene


def render_offscreen_view(
    mesh: trimesh.Trimesh,
    view: CameraView,
    resolution: int,
    renderer_factory: Callable[[int, int], object] = pyrender.OffscreenRenderer,
) -> dict[str, object]:
    scene = _make_scene(mesh, view)
    renderer = renderer_factory(resolution, resolution)
    try:
        color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "offscreen rendering failed; configure a headless OpenGL backend such as PYOPENGL_PLATFORM=egl or OSMesa",
        ) from exc
    finally:
        delete = getattr(renderer, "delete", None)
        if callable(delete):
            delete()

    color_rgba = np.asarray(color, dtype=np.uint8)
    if color_rgba.ndim == 3 and color_rgba.shape[2] == 3:
        alpha = np.where(np.asarray(depth) > 0.0, 255, 0).astype(np.uint8)
        color_rgba = np.dstack([color_rgba, alpha])

    return {
        "rgb": Image.fromarray(color_rgba, mode="RGBA"),
        "depth": np.asarray(depth, dtype=np.float32),
        "camera": {"name": view.name, "pose": view.pose.tolist(), "fovy_deg": view.fovy_deg},
    }


def render_offscreen_views(
    mesh: trimesh.Trimesh,
    views: list[CameraView],
    resolution: int,
    renderer_factory: Callable[[int, int], object] = pyrender.OffscreenRenderer,
) -> dict[str, dict[str, object]]:
    return {
        view.name: render_offscreen_view(mesh, view, resolution, renderer_factory=renderer_factory)
        for view in views
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /root/autodl-tmp/src/.worktrees/stylized-3d-pipeline/stylized-3d-pipeline && pytest tests/test_offscreen_renderer.py -v`

Expected: pass with all three tests green.

- [ ] **Step 5: Commit**

```bash
cd /root/autodl-tmp/src/.worktrees/stylized-3d-pipeline
git add stylized-3d-pipeline/lib/offscreen_renderer.py stylized-3d-pipeline/tests/test_offscreen_renderer.py
git commit -m "feat: add offscreen multiview renderer backend"
```

## Task 2: Wire `sample_views` to the Offscreen Backend

**Files:**
- Modify: `stylized-3d-pipeline/lib/view_sampling.py`
- Modify: `stylized-3d-pipeline/scripts/step3_sample_views.py`
- Modify: `stylized-3d-pipeline/tests/test_step3_sample_views.py`
- Modify: `stylized-3d-pipeline/tests/test_step4_retexture.py`

- [ ] **Step 1: Write the failing tests**

```python
# stylized-3d-pipeline/tests/test_step3_sample_views.py
def test_run_step_marks_mesh_offscreen_render_mode(tmp_path: Path) -> None:
    paths = create_run_tree(tmp_path / "run")
    mesh = trimesh.creation.box()
    mesh.export(paths.sf3d / "mesh_raw.glb", include_normals=True)

    def fake_renderer(mesh, views, resolution):  # noqa: ANN001
        payload = {}
        for view in views:
            rgb = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
            for y in range(4, 12):
                for x in range(4, 12):
                    rgb.putpixel((x, y), (220, 220, 220, 255))
            payload[view.name] = {
                "rgb": rgb,
                "depth": np.ones((16, 16), dtype=np.float32),
                "depth_preview": Image.new("RGBA", (16, 16), (40, 50, 60, 255)),
                "normal": Image.new("RGBA", (16, 16), (120, 130, 140, 255)),
                "mask": Image.new("L", (16, 16), 255),
                "control": Image.new("RGBA", (16, 16), (70, 80, 90, 255)),
                "camera": {"name": view.name, "pose": view.pose.tolist(), "fovy_deg": view.fovy_deg},
            }
        return payload

    result = run_step(paths.root, 16, 2.0, 40.0, renderer=fake_renderer)

    manifest = json.loads((paths.views / "manifest.json").read_text(encoding="utf-8"))

    assert result["render_mode"] == "mesh_offscreen"
    assert manifest["render_mode"] == "mesh_offscreen"
    assert manifest == result
    with Image.open(paths.views / "front" / "rgb.png") as front_rgb:
        assert front_rgb.mode == "RGBA"
        assert front_rgb.getpixel((0, 0))[3] == 0
        assert front_rgb.getpixel((6, 6))[3] == 255
```

```python
# stylized-3d-pipeline/tests/test_step4_retexture.py
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /root/autodl-tmp/src/.worktrees/stylized-3d-pipeline/stylized-3d-pipeline && pytest tests/test_step3_sample_views.py::test_run_step_marks_mesh_offscreen_render_mode -v`

Expected: fail because `run_step()` does not yet include `render_mode`.

- [ ] **Step 3: Write the minimal implementation**

```python
# stylized-3d-pipeline/lib/view_sampling.py
from lib.offscreen_renderer import render_offscreen_views


def _derive_secondary_maps(rgb: Image.Image, depth: np.ndarray, fovy_deg: float) -> dict[str, object]:
    mask = np.where(depth > 0.0, 255, 0).astype(np.uint8)
    valid_depth = np.where(mask > 0, depth, 0.0).astype(np.float32)
    if np.any(mask > 0):
        max_depth = float(valid_depth[mask > 0].max())
        safe_depth = np.where(mask > 0, valid_depth, max_depth).astype(np.float32)
    else:
        safe_depth = np.zeros_like(valid_depth)
        max_depth = 1.0

    depth_preview = np.uint8(np.clip(safe_depth / max(max_depth, 1e-6), 0.0, 1.0) * 255)
    gy, gx = np.gradient(safe_depth)
    normal_xyz = np.dstack((-gx, -gy, np.ones_like(safe_depth)))
    denom = np.linalg.norm(normal_xyz, axis=2, keepdims=True)
    denom = np.where(denom == 0.0, 1.0, denom)
    normal_rgb = np.uint8(np.clip((normal_xyz / denom + 1.0) * 127.5, 0, 255))
    normal_rgb = np.where(mask[:, :, None] > 0, normal_rgb, 0).astype(np.uint8)
    normal_rgba = np.dstack([normal_rgb, mask])
    control = _make_control_image(normal_rgb, depth_preview, mask)
    rgba = rgb.convert("RGBA")
    rgba.putalpha(Image.fromarray(mask, mode="L"))
    return {
        "rgb": rgba,
        "depth": valid_depth,
        "depth_preview": Image.fromarray(depth_preview, mode="L").convert("RGBA"),
        "normal": Image.fromarray(normal_rgba, mode="RGBA"),
        "mask": Image.fromarray(mask, mode="L"),
        "control": control,
    }


def render_view_assets(mesh: trimesh.Trimesh, views: list[CameraView], resolution: int) -> dict[str, dict[str, object]]:
    rendered_views = render_offscreen_views(mesh, views, resolution)
    assets: dict[str, dict[str, object]] = {}
    for view in views:
        raw = rendered_views[view.name]
        derived = _derive_secondary_maps(raw["rgb"], raw["depth"], view.fovy_deg)
        assets[view.name] = {
            **derived,
            "camera": raw["camera"],
        }
    return assets
```

```python
# stylized-3d-pipeline/scripts/step3_sample_views.py
manifest = {
    "render_mode": "mesh_offscreen",
    "view_resolution": view_resolution,
    "camera_distance": camera_distance,
    "camera_fovy_deg": camera_fovy_deg,
    "views": [],
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cd /root/autodl-tmp/src/.worktrees/stylized-3d-pipeline/stylized-3d-pipeline
pytest tests/test_step3_sample_views.py -v
pytest tests/test_step4_retexture.py -v
```

Expected:
- `test_step3_sample_views.py` passes with the manifest containing `render_mode: mesh_offscreen`
- `test_step4_retexture.py` still passes because downstream manifest parsing ignores the new top-level field

- [ ] **Step 5: Commit**

```bash
cd /root/autodl-tmp/src/.worktrees/stylized-3d-pipeline
git add stylized-3d-pipeline/lib/view_sampling.py stylized-3d-pipeline/scripts/step3_sample_views.py stylized-3d-pipeline/tests/test_step3_sample_views.py stylized-3d-pipeline/tests/test_step4_retexture.py
git commit -m "feat: render multiviews directly from mesh"
```

## Task 3: Validate on the Real Run and Preserve the Existing Outputs

**Files:**
- Modify: `stylized-3d-pipeline/runs/real-chair-starry-multiview-v2/*` only by rerunning steps in place

- [ ] **Step 1: Run the focused test suite**

Run:

```bash
cd /root/autodl-tmp/src/.worktrees/stylized-3d-pipeline
NUMBA_CACHE_DIR=/root/autodl-tmp/numba-cache pytest stylized-3d-pipeline/tests/test_offscreen_renderer.py -v
NUMBA_CACHE_DIR=/root/autodl-tmp/numba-cache pytest stylized-3d-pipeline/tests/test_step3_sample_views.py -v
NUMBA_CACHE_DIR=/root/autodl-tmp/numba-cache pytest stylized-3d-pipeline/tests/test_step3_instantstyle.py -v
NUMBA_CACHE_DIR=/root/autodl-tmp/numba-cache pytest stylized-3d-pipeline/tests/test_step4_retexture.py -v
```

Expected: all tests pass before touching the real run again.

- [ ] **Step 2: Rebuild the real sample with the offscreen renderer**

Run:

```bash
cd /root/autodl-tmp/src/.worktrees/stylized-3d-pipeline/stylized-3d-pipeline
mkdir -p /root/autodl-tmp/tmp /root/autodl-tmp/numba-cache /root/autodl-tmp/hf-cache
TMPDIR=/root/autodl-tmp/tmp NUMBA_CACHE_DIR=/root/autodl-tmp/numba-cache PYOPENGL_PLATFORM=egl \
python scripts/step3_sample_views.py \
  --run-dir runs/real-chair-starry-multiview-v2 \
  --view-resolution 512 \
  --camera-distance 1.8 \
  --camera-fovy-deg 40.0

TMPDIR=/root/autodl-tmp/tmp NUMBA_CACHE_DIR=/root/autodl-tmp/numba-cache HF_HOME=/root/autodl-tmp/hf-cache \
python scripts/step3_instantstyle.py \
  --run-dir runs/real-chair-starry-multiview-v2 \
  --instantstyle-python /root/miniconda3/bin/python \
  --style-image /root/autodl-tmp/src/InstantStyle/assets/4.jpg \
  --prompt "a wooden chair" \
  --seed 42

TMPDIR=/root/autodl-tmp/tmp NUMBA_CACHE_DIR=/root/autodl-tmp/numba-cache \
python scripts/step4_retexture.py --run-dir runs/real-chair-starry-multiview-v2

python scripts/step5_build_viewer.py --run-dir runs/real-chair-starry-multiview-v2
```

Expected:
- `views/*/rgb.png` no longer carries the sparse SF3D black patches.
- `views/*/control.png` background alpha stays at 0.
- `retexture/texture_preview.png` gets materially cleaner because the stylized views are cleaner.

- [ ] **Step 3: Measure the outcome**

Run a small metrics script against `runs/real-chair-starry-multiview-v2`:

```bash
python - <<'PY'
from pathlib import Path
from PIL import Image
import numpy as np

run = Path("runs/real-chair-starry-multiview-v2")
for name in ["front", "back", "left", "right", "top", "bottom"]:
    rgb = np.asarray(Image.open(run / "views" / name / "rgb.png").convert("RGBA"))
    mask = rgb[:, :, 3] > 0
    dark_ratio = float((rgb[:, :, :3].sum(axis=2) < 18)[mask].mean()) if mask.any() else 1.0
    print(name, "dark_ratio", round(dark_ratio, 4))
PY
```

Expected: each visible region dark ratio stays low and should be far below the old sparse-atlas baseline.

- [ ] **Step 4: Commit only if the validation is clean**

If the validation passes and no new cleanup is required, leave `runs/` intact and keep the code commits from Tasks 1 and 2. If validation reveals a renderer-specific issue, fix it in `lib/offscreen_renderer.py` before any broader refactor.

## Self-Review Checklist

- Spec coverage:
  - direct offscreen render backend: Task 1
  - `sample_views` integration: Task 2
  - manifest compatibility with `render_mode`: Task 2
  - real run verification: Task 3
- Placeholder scan:
  - no `TBD`, `TODO`, or vague filler text
  - every code step names concrete files and concrete code
- Type consistency:
  - `render_offscreen_view()` and `render_offscreen_views()` are used consistently across tests and integration steps
  - `render_mode: "mesh_offscreen"` is the only new manifest marker
  - downstream consumers continue to ignore unknown manifest fields
