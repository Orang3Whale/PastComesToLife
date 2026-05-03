# Step3 Dual-Input Stylization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use textured offscreen RGB as the SDXL img2img base while keeping `control.png` geometry-only, so step3 stops relying on Canny over the control image and the downstream pipeline stays compatible.

**Architecture:** Restore textured mesh rendering at the renderer boundary, then keep `step3_sample_views.py` and `view_sampling.py` as thin asset writers. `step3_instantstyle.py` becomes a dual-input coordinator that reads both `rgb_path` and `control_path` from the multiview manifest and passes them to the worker. The worker switches from plain ControlNet txt2img to `StableDiffusionXLControlNetImg2ImgPipeline`, flattens the RGB base onto a neutral canvas, flattens the control image to a pure structural RGB map, and uses a fixed default redraw strength around `0.45` without changing the top-level step3 CLI.

**Tech Stack:** Python 3.11, `pyrender`, `trimesh`, `Pillow`, `numpy`, `diffusers`, `IP-Adapter`, `pytest`.

---

## File Map

### Create

- `stylized-3d-pipeline/tests/test_instantstyle_worker.py`

### Modify

- `stylized-3d-pipeline/lib/offscreen_renderer.py`
- `stylized-3d-pipeline/scripts/step3_sample_views.py`
- `stylized-3d-pipeline/scripts/step3_instantstyle.py`
- `stylized-3d-pipeline/scripts/workers/instantstyle_worker.py`
- `stylized-3d-pipeline/tests/test_offscreen_renderer.py`
- `stylized-3d-pipeline/tests/test_step3_sample_views.py`
- `stylized-3d-pipeline/tests/test_step3_instantstyle.py`

### Responsibility Split

- `lib/offscreen_renderer.py`: stop converting the source mesh to neutral vertex colors in the default render path; keep RGBA color plus depth output intact.
- `scripts/step3_sample_views.py`: keep orchestration stable, but rename the manifest `render_mode` to `mesh_textured_offscreen` so the exported views describe the actual render path.
- `scripts/step3_instantstyle.py`: read `rgb_path`, `control_path`, and `mask_path` from the manifest, pass both images to the worker, and record the redraw strength in the stylize manifest.
- `scripts/workers/instantstyle_worker.py`: flatten `rgb.png` and `control.png` separately, remove the Canny preprocessing path, and call SDXL img2img with ControlNet.
- `tests/test_offscreen_renderer.py`: prove the renderer preserves source mesh visuals instead of feeding pyrender a neutral surrogate.
- `tests/test_step3_sample_views.py`: prove the step manifest still contains `rgb_path` and now advertises the textured render mode.
- `tests/test_step3_instantstyle.py`: prove the command line and manifest wiring carry both input images and the redraw strength.
- `tests/test_instantstyle_worker.py`: prove the worker uses img2img, preserves the two roles of base image vs control image, and no longer depends on `cv2.Canny`.

## Task 1: Restore Textured Offscreen RGB

**Files:**
- Modify: `stylized-3d-pipeline/lib/offscreen_renderer.py:15-99`
- Modify: `stylized-3d-pipeline/scripts/step3_sample_views.py:30-54`
- Modify: `stylized-3d-pipeline/tests/test_offscreen_renderer.py`
- Modify: `stylized-3d-pipeline/tests/test_step3_sample_views.py:113-146`

- [ ] **Step 1: Write the failing tests**

```python
# stylized-3d-pipeline/tests/test_offscreen_renderer.py
import pyrender


def test_render_offscreen_view_keeps_source_texture(monkeypatch) -> None:
    captured = {}

    original_from_trimesh = pyrender.Mesh.from_trimesh

    def fake_from_trimesh(mesh, smooth=False):  # noqa: ANN001,ANN003
        captured["visual_kind"] = mesh.visual.kind
        return original_from_trimesh(mesh, smooth=smooth)

    monkeypatch.setattr(pyrender.Mesh, "from_trimesh", fake_from_trimesh)

    class FakeRenderer:
        def __init__(self, viewport_width: int, viewport_height: int) -> None:
            self.viewport_width = viewport_width
            self.viewport_height = viewport_height

        def render(self, scene, flags):  # noqa: ANN001
            color = np.zeros((self.viewport_height, self.viewport_width, 4), dtype=np.uint8)
            depth = np.zeros((self.viewport_height, self.viewport_width), dtype=np.float32)
            return color, depth

        def delete(self) -> None:
            pass

    view = CameraView(
        name="front",
        pose=look_at(
            np.array([2.0, 0.0, 0.0], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        ),
        fovy_deg=40.0,
    )
    render_offscreen_view(
        _textured_triangle(),
        view,
        resolution=8,
        renderer_factory=lambda w, h: FakeRenderer(w, h),
    )

    assert captured["visual_kind"] == "texture"
```

```python
# stylized-3d-pipeline/tests/test_step3_sample_views.py
def test_run_step_marks_mesh_textured_offscreen_render_mode(tmp_path: Path) -> None:
    result = run_step(paths.root, 16, 2.0, 40.0, renderer=fake_renderer)

    manifest = json.loads((paths.views / "manifest.json").read_text(encoding="utf-8"))

    assert result["render_mode"] == "mesh_textured_offscreen"
    assert manifest["render_mode"] == "mesh_textured_offscreen"
    assert manifest["views"][0]["rgb_path"].endswith("/rgb.png")
    assert manifest["views"][0]["control_path"].endswith("/control.png")
```

- [ ] **Step 2: Run the targeted tests and confirm they fail**

Run:

```bash
cd /root/autodl-tmp/src/.worktrees/stylized-3d-pipeline/stylized-3d-pipeline
pytest tests/test_offscreen_renderer.py::test_render_offscreen_view_keeps_source_texture tests/test_step3_sample_views.py::test_run_step_marks_mesh_textured_offscreen_render_mode -v
```

Expected: fail because the renderer still feeds pyrender a neutral mesh and `step3_sample_views.py` still reports `mesh_offscreen`.

- [ ] **Step 3: Write the minimal implementation**

```python
# stylized-3d-pipeline/lib/offscreen_renderer.py
def _make_scene(mesh: trimesh.Trimesh, view: CameraView) -> pyrender.Scene:
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.18, 0.18, 0.18])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(view.fovy_deg)), pose=view.pose)
    # Keep the existing directional light setup unchanged.

# stylized-3d-pipeline/scripts/step3_sample_views.py
manifest = {
    "render_mode": "mesh_textured_offscreen",
    "view_resolution": view_resolution,
    "camera_distance": camera_distance,
    "camera_fovy_deg": camera_fovy_deg,
    "views": [],
}
```

Keep `build_neutral_render_mesh()` as an explicit helper for diagnostics and tests, but do not use it in the default render path.

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cd /root/autodl-tmp/src/.worktrees/stylized-3d-pipeline/stylized-3d-pipeline
pytest tests/test_offscreen_renderer.py tests/test_step3_sample_views.py -v
```

Expected: all tests pass and `rgb.png` remains RGBA with transparent background.

- [ ] **Step 5: Commit**

```bash
cd /root/autodl-tmp/src/.worktrees/stylized-3d-pipeline
git add stylized-3d-pipeline/lib/offscreen_renderer.py stylized-3d-pipeline/scripts/step3_sample_views.py stylized-3d-pipeline/tests/test_offscreen_renderer.py stylized-3d-pipeline/tests/test_step3_sample_views.py
git commit -m "feat: restore textured offscreen multiviews"
```

## Task 2: Wire Step3 to Dual Inputs and Redraw Strength

**Files:**
- Modify: `stylized-3d-pipeline/scripts/step3_instantstyle.py:20-132`
- Modify: `stylized-3d-pipeline/tests/test_step3_instantstyle.py`

- [ ] **Step 1: Write the failing tests**

```python
# stylized-3d-pipeline/tests/test_step3_instantstyle.py
def test_build_instantstyle_command_includes_rgb_control_and_strength(tmp_path: Path) -> None:
    cmd = build_instantstyle_command(
        instantstyle_python=Path("/envs/instantstyle/bin/python"),
        worker_script=Path("/repo/stylized-3d-pipeline/scripts/workers/instantstyle_worker.py"),
        run_dir=tmp_path / "run",
        rgb_image=Path("/run/views/front/rgb.png"),
        control_image=Path("/run/views/front/control.png"),
        style_image=Path("/tmp/style.jpg"),
        prompt="ceramic mug",
        output_image=Path("/run/stylize/front/stylized.png"),
        seed=123,
        strength=0.45,
    )

    assert "--rgb-image" in cmd
    assert "/run/views/front/rgb.png" in cmd
    assert "--control-image" in cmd
    assert "--strength" in cmd
    assert "0.45" in cmd
```

Update the existing per-view `run_step` fixture so every manifest entry has `rgb_path`, then assert the returned stylize manifest records the RGB path and redraw strength.

```python
def test_run_step_writes_rgb_control_and_strength_into_manifest(tmp_path: Path) -> None:
    write_json(
        paths.views / "manifest.json",
        {
            "views": [
                {
                    "name": "front",
                    "rgb_path": str(paths.views / "front" / "rgb.png"),
                    "control_path": str(paths.views / "front" / "control.png"),
                    "mask_path": str(paths.views / "front" / "mask.png"),
                },
                {
                    "name": "back",
                    "rgb_path": str(paths.views / "back" / "rgb.png"),
                    "control_path": str(paths.views / "back" / "control.png"),
                    "mask_path": str(paths.views / "back" / "mask.png"),
                },
                {
                    "name": "left",
                    "rgb_path": str(paths.views / "left" / "rgb.png"),
                    "control_path": str(paths.views / "left" / "control.png"),
                    "mask_path": str(paths.views / "left" / "mask.png"),
                },
                {
                    "name": "right",
                    "rgb_path": str(paths.views / "right" / "rgb.png"),
                    "control_path": str(paths.views / "right" / "control.png"),
                    "mask_path": str(paths.views / "right" / "mask.png"),
                },
                {
                    "name": "top",
                    "rgb_path": str(paths.views / "top" / "rgb.png"),
                    "control_path": str(paths.views / "top" / "control.png"),
                    "mask_path": str(paths.views / "top" / "mask.png"),
                },
                {
                    "name": "bottom",
                    "rgb_path": str(paths.views / "bottom" / "rgb.png"),
                    "control_path": str(paths.views / "bottom" / "control.png"),
                    "mask_path": str(paths.views / "bottom" / "mask.png"),
                },
            ]
        },
    )

    result = run_step(
        run_dir=paths.root,
        instantstyle_python=Path("/envs/instantstyle/bin/python"),
        style_image=style_image,
        prompt="ceramic mug",
        seed=123,
        strength=0.45,
        runner=fake_runner,
    )

    assert result["strength"] == 0.45
    assert result["views"]["front"]["rgb_path"] == str(paths.views / "front" / "rgb.png")
    assert result["views"]["front"]["control_path"] == str(paths.views / "front" / "control.png")
```

- [ ] **Step 2: Run the targeted tests and confirm they fail**

Run:

```bash
cd /root/autodl-tmp/src/.worktrees/stylized-3d-pipeline/stylized-3d-pipeline
pytest tests/test_step3_instantstyle.py::test_build_instantstyle_command_includes_rgb_control_and_strength tests/test_step3_instantstyle.py::test_run_step_writes_rgb_control_and_strength_into_manifest -v
```

Expected: fail because `build_instantstyle_command()` still only accepts `control_image`, and `run_step()` still drops `rgb_path` and `strength`.

- [ ] **Step 3: Write the minimal implementation**

```python
# stylized-3d-pipeline/scripts/step3_instantstyle.py
def build_instantstyle_command(
    instantstyle_python: Path,
    worker_script: Path,
    run_dir: Path,
    rgb_image: Path,
    control_image: Path,
    style_image: Path,
    prompt: str,
    output_image: Path,
    seed: int,
    strength: float,
) -> list[str]:
    return [
        str(instantstyle_python),
        str(worker_script),
        "--run-dir",
        str(run_dir),
        "--rgb-image",
        str(rgb_image),
        "--control-image",
        str(control_image),
        "--style-image",
        str(style_image),
        "--prompt",
        prompt,
        "--output-image",
        str(output_image),
        "--seed",
        str(seed),
        "--strength",
        str(strength),
    ]
```

```python
def run_step(
    run_dir: Path,
    instantstyle_python: Path,
    style_image: Path,
    prompt: str,
    seed: int = 42,
    strength: float = 0.45,
    runner: Callable = run_checked,
) -> dict:
    rgb_image = Path(view["rgb_path"])
    control_image = Path(view["control_path"])
    cmd = build_instantstyle_command(
        instantstyle_python=instantstyle_python,
        worker_script=worker_script,
        run_dir=paths.root,
        rgb_image=rgb_image,
        control_image=control_image,
        style_image=style_copy,
        prompt=prompt,
        output_image=output_image,
        seed=seed,
        strength=strength,
    )
    result["strength"] = strength
    result["views"][view_name] = {
        "rgb_path": str(rgb_image),
        "control_path": str(control_image),
        "stylized_path": str(output_image),
    }
```

Do not add a new top-level CLI flag to `step3_instantstyle.py`; keep the default strength internal so the outer step entrypoint stays unchanged.

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cd /root/autodl-tmp/src/.worktrees/stylized-3d-pipeline/stylized-3d-pipeline
pytest tests/test_step3_instantstyle.py -v
```

Expected: pass, and the stylize manifest should now record the dual-input wiring plus the redraw strength.

- [ ] **Step 5: Commit**

```bash
cd /root/autodl-tmp/src/.worktrees/stylized-3d-pipeline
git add stylized-3d-pipeline/scripts/step3_instantstyle.py stylized-3d-pipeline/tests/test_step3_instantstyle.py
git commit -m "feat: pass rgb base image into step3 stylization"
```

## Task 3: Replace the Worker Canny Path with SDXL Img2Img

**Files:**
- Create: `stylized-3d-pipeline/tests/test_instantstyle_worker.py`
- Modify: `stylized-3d-pipeline/scripts/workers/instantstyle_worker.py:21-148`

- [ ] **Step 1: Write the failing tests**

```python
# stylized-3d-pipeline/tests/test_instantstyle_worker.py
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.workers import instantstyle_worker as worker
from scripts.workers.instantstyle_worker import (
    build_pipeline_and_adapter,
    generate_stylized_images,
    prepare_base_image,
    prepare_control_image,
    write_worker_outputs,
)


def test_prepare_base_image_preserves_texture_and_fills_transparency() -> None:
    base = Image.new("RGBA", (2, 2), (0, 0, 0, 0))
    base.putpixel((1, 1), (10, 20, 30, 255))

    prepared = prepare_base_image(base)
    prepared_array = np.asarray(prepared)

    assert prepared.mode == "RGB"
    assert tuple(prepared_array[0, 0]) == (235, 235, 235)
    assert tuple(prepared_array[1, 1]) == (10, 20, 30)
```

```python
def test_prepare_control_image_masks_transparency_without_canny() -> None:
    control = Image.new("RGBA", (2, 2), (10, 20, 30, 0))
    control.putpixel((1, 1), (10, 20, 30, 255))

    prepared = prepare_control_image(control)
    prepared_array = np.asarray(prepared)

    assert prepared.mode == "RGB"
    assert np.all(prepared_array[0, 0] == 0)
    assert tuple(prepared_array[1, 1]) == (10, 20, 30)
    assert not hasattr(worker, "build_canny_control_map")
```

```python
def test_build_pipeline_and_adapter_uses_img2img_pipeline(monkeypatch) -> None:
    seen = {}

    class FakeControlNet:
        def to(self, device):  # noqa: ANN001
            seen["controlnet_device"] = device
            return self

    class FakePipeline:
        def to(self, device):  # noqa: ANN001
            seen["pipeline_device"] = device
            return self

        def enable_vae_tiling(self) -> None:
            seen["vae_tiling"] = True

    monkeypatch.setattr(worker.ControlNetModel, "from_pretrained", lambda *args, **kwargs: FakeControlNet())
    monkeypatch.setattr(
        worker.StableDiffusionXLControlNetImg2ImgPipeline,
        "from_pretrained",
        lambda *args, **kwargs: FakePipeline(),
    )
    monkeypatch.setattr(worker, "IPAdapterXL", lambda pipe, *args, **kwargs: pipe)

    pipe, ip_model = build_pipeline_and_adapter(device="cpu")

    assert seen["pipeline_device"] == "cpu"
    assert seen["controlnet_device"] == "cpu"
    assert seen["vae_tiling"] is True
    assert pipe is ip_model
```

```python
def test_generate_stylized_images_passes_base_image_control_image_and_strength() -> None:
    seen = {}

    class FakeIPAdapter:
        def generate(self, **kwargs):  # noqa: ANN003
            seen.update(kwargs)
            return [Image.new("RGBA", (16, 16), (0, 255, 0, 255))]

    output = generate_stylized_images(
        ip_model=FakeIPAdapter(),
        style=Image.new("RGB", (16, 16), "blue"),
        prompt="ceramic mug",
        base_image=Image.new("RGB", (16, 16), "white"),
        control_image=Image.new("RGB", (16, 16), "black"),
        seed=123,
        strength=0.45,
    )

    assert len(output) == 1
    assert seen["image"].mode == "RGB"
    assert seen["control_image"].mode == "RGB"
    assert seen["strength"] == 0.45
```

```python
def test_write_worker_outputs_records_dual_inputs_and_strength(tmp_path: Path) -> None:
    output_image = tmp_path / "stylize" / "front" / "stylized.png"
    metadata = write_worker_outputs(
        output_image=output_image,
        stylized_image=Image.new("RGBA", (16, 16), (255, 0, 0, 255)),
        rgb_image=Path("/run/views/front/rgb.png"),
        control_image=Path("/run/views/front/control.png"),
        style_image=Path("/run/inputs/style.png"),
        prompt="ceramic mug",
        seed=7,
        strength=0.45,
    )

    assert output_image.is_file()
    assert metadata["rgb_image"] == "/run/views/front/rgb.png"
    assert metadata["control_image"] == "/run/views/front/control.png"
    assert metadata["strength"] == 0.45
```

- [ ] **Step 2: Run the targeted tests and confirm they fail**

Run:

```bash
cd /root/autodl-tmp/src/.worktrees/stylized-3d-pipeline/stylized-3d-pipeline
pytest tests/test_instantstyle_worker.py -v
```

Expected: fail because the worker still imports `cv2`, still builds a Canny map, and still uses `StableDiffusionXLControlNetPipeline` instead of the img2img variant.

- [ ] **Step 3: Write the minimal implementation**

```python
# stylized-3d-pipeline/scripts/workers/instantstyle_worker.py
def prepare_base_image(rgb: Image.Image) -> Image.Image:
    rgba = rgb.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, (235, 235, 235, 255))
    return Image.alpha_composite(canvas, rgba).convert("RGB")


def prepare_control_image(control: Image.Image) -> Image.Image:
    rgba = control.convert("RGBA")
    canvas = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    return Image.alpha_composite(canvas, rgba).convert("RGB")


def build_pipeline_and_adapter(device: str = "cuda") -> tuple[object, object]:
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        use_safetensors=False,
        torch_dtype=torch.float16,
    ).to(device)
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        add_watermarker=False,
    ).to(device)
    pipe.enable_vae_tiling()
    ip_model = IPAdapterXL(
        pipe,
        "sdxl_models/image_encoder",
        "sdxl_models/ip-adapter_sdxl.bin",
        device,
        target_blocks=["up_blocks.0.attentions.1"],
    )
    return pipe, ip_model
```

```python
def generate_stylized_images(
    ip_model: object,
    style: Image.Image,
    prompt: str,
    base_image: Image.Image,
    control_image: Image.Image,
    seed: int,
    strength: float,
) -> object:
    return ip_model.generate(
        pil_image=style,
        prompt=prompt,
        negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, blurry",
        scale=1.0,
        guidance_scale=5.0,
        num_samples=1,
        num_inference_steps=30,
        seed=seed,
        image=base_image,
        control_image=control_image,
        strength=strength,
        controlnet_conditioning_scale=0.7,
    )
```

```python
parser.add_argument("--rgb-image", required=True, type=Path)
parser.add_argument("--strength", default=0.45, type=float)
with Image.open(args.rgb_image) as rgb_image:
    base_image = prepare_base_image(rgb_image)
with Image.open(args.control_image) as control_image:
    control = prepare_control_image(control_image)
with Image.open(args.style_image) as style_image:
    style = style_image.convert("RGB")

pipe, ip_model = build_pipeline_and_adapter("cuda")
images = generate_stylized_images(
    ip_model=ip_model,
    style=style,
    prompt=args.prompt,
    base_image=base_image,
    control_image=control,
    seed=args.seed,
    strength=args.strength,
)
```

Update `build_worker_meta()` and `write_worker_outputs()` so the worker metadata records `rgb_image`, `control_image`, and `strength`.

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cd /root/autodl-tmp/src/.worktrees/stylized-3d-pipeline/stylized-3d-pipeline
pytest tests/test_instantstyle_worker.py -v
```

Expected: pass, and the worker no longer depends on `cv2.Canny`.

- [ ] **Step 5: Commit**

```bash
cd /root/autodl-tmp/src/.worktrees/stylized-3d-pipeline
git add stylized-3d-pipeline/scripts/workers/instantstyle_worker.py stylized-3d-pipeline/tests/test_instantstyle_worker.py
git commit -m "feat: switch step3 worker to img2img controlnet"
```

## Final Verification

After the three implementation tasks land, run the downstream regressions and the preserved real run:

```bash
cd /root/autodl-tmp/src/.worktrees/stylized-3d-pipeline/stylized-3d-pipeline
pytest tests/test_step4_retexture.py tests/test_step5_build_viewer.py -v
```

```bash
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/root/autodl-tmp/gl-shims:/root/miniconda3/lib/python3.11/site-packages/mediapipe.libs:/usr/lib/x86_64-linux-gnu
/root/miniconda3/bin/python scripts/step3_sample_views.py --run-dir runs/real-chair-starry-multiview-v2 --view-resolution 512 --camera-distance 1.8 --camera-fovy-deg 40.0
PROMPT="$(cat runs/real-chair-starry-multiview-v2/inputs/prompt.txt)"
/root/miniconda3/bin/python scripts/step3_instantstyle.py --run-dir runs/real-chair-starry-multiview-v2 --instantstyle-python /root/miniconda3/bin/python --style-image runs/real-chair-starry-multiview-v2/inputs/style.png --prompt "$PROMPT" --seed 42
/root/miniconda3/bin/python scripts/step4_retexture.py --run-dir runs/real-chair-starry-multiview-v2
/root/miniconda3/bin/python scripts/step5_build_viewer.py --run-dir runs/real-chair-starry-multiview-v2
```

Then inspect the key outputs:

```bash
/root/miniconda3/bin/python - <<'PY'
from pathlib import Path
from PIL import Image
import numpy as np

for path in [
    Path("runs/real-chair-starry-multiview-v2/views/front/rgb.png"),
    Path("runs/real-chair-starry-multiview-v2/views/front/control.png"),
    Path("runs/real-chair-starry-multiview-v2/stylize/front/stylized.png"),
    Path("runs/real-chair-starry-multiview-v2/retexture/texture_preview.png"),
]:
    img = np.asarray(Image.open(path).convert("RGBA"))
    alpha = img[:, :, 3] > 0
    dark = (img[:, :, :3].mean(axis=2) < 20) & alpha
    print(path, "alpha_coverage=", float(alpha.mean()), "dark_ratio=", float(dark.mean()))
PY
```
