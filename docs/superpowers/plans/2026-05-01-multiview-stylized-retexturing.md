# Multiview Stylized Retexturing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the current single-view texture projector into a six-view baseline that samples fixed camera views, stylizes them with shared style constraints, and reprojects them back into UV space for a cleaner GLB export.

**Architecture:** Keep `SF3D` unchanged, insert a dedicated six-view sampling stage, expand `InstantStyle` into per-view stylization, replace the heuristic projector with UV-space multi-view reprojection, and update the viewer to surface all sampled and stylized views. Every stage writes manifests into the per-run directory so it can be resumed or replayed independently.

**Tech Stack:** Python 3.11, numpy, Pillow, trimesh, pyrender, OpenCV, rembg, pytest, subprocess, existing `stable-fast-3d` and `InstantStyle` environments.

---

## File Map

### Create

- `stylized-3d-pipeline/lib/camera_views.py`
- `stylized-3d-pipeline/lib/view_sampling.py`
- `stylized-3d-pipeline/lib/reprojection.py`
- `stylized-3d-pipeline/scripts/step3_sample_views.py`
- `stylized-3d-pipeline/tests/test_step3_sample_views.py`

### Modify

- `stylized-3d-pipeline/lib/io_paths.py`
- `stylized-3d-pipeline/lib/pipeline_runner.py`
- `stylized-3d-pipeline/lib/viewer_utils.py`
- `stylized-3d-pipeline/scripts/run_all.py`
- `stylized-3d-pipeline/scripts/step3_instantstyle.py`
- `stylized-3d-pipeline/scripts/workers/instantstyle_worker.py`
- `stylized-3d-pipeline/scripts/step4_retexture.py`
- `stylized-3d-pipeline/scripts/step5_build_viewer.py`
- `stylized-3d-pipeline/tests/test_io_paths.py`
- `stylized-3d-pipeline/tests/test_run_all.py`
- `stylized-3d-pipeline/tests/test_step3_instantstyle.py`
- `stylized-3d-pipeline/tests/test_step4_retexture.py`
- `stylized-3d-pipeline/tests/test_step5_build_viewer.py`
- `stylized-3d-pipeline/README.md`

### Responsibility Split

- `lib/camera_views.py`: canonical six-view camera ordering, pose math, and shared camera metadata
- `lib/view_sampling.py`: render RGB/depth/normal/mask/control assets for each view and persist the manifest
- `scripts/step3_sample_views.py`: CLI wrapper for Stage 2 sampling
- `scripts/step3_instantstyle.py`: CLI wrapper for Stage 3 per-view stylization
- `scripts/workers/instantstyle_worker.py`: the external `InstantStyle` process, now parameterized by control image and output path
- `lib/reprojection.py`: point projection, visibility checks, and weighted color fusion in UV space
- `scripts/step4_retexture.py`: load the six stylized views and bake them into the mesh texture atlas
- `lib/viewer_utils.py` and `scripts/step5_build_viewer.py`: static HTML viewer for sampled views, stylized views, and final GLB
- `lib/io_paths.py`, `lib/pipeline_runner.py`, `scripts/run_all.py`: run directory layout, step ordering, CLI contract, and resume/skip behavior
- `tests/*.py`: unit tests for each stage plus orchestration and CLI help contracts

## Task 1: Extend the run contract for a six-view pipeline

**Files:**
- Modify: `stylized-3d-pipeline/lib/io_paths.py`
- Modify: `stylized-3d-pipeline/lib/pipeline_runner.py`
- Modify: `stylized-3d-pipeline/scripts/run_all.py`
- Modify: `stylized-3d-pipeline/tests/test_io_paths.py`
- Modify: `stylized-3d-pipeline/tests/test_run_all.py`

- [ ] **Step 1: Write the failing test**

```python
# stylized-3d-pipeline/tests/test_io_paths.py
def test_create_run_tree_creates_expected_directories(tmp_path: Path) -> None:
    run_dir = resolve_run_dir(tmp_path, "demo-chair")
    paths = create_run_tree(run_dir)

    assert paths.root == run_dir
    assert paths.inputs.is_dir()
    assert paths.preprocess.is_dir()
    assert paths.sf3d.is_dir()
    assert paths.views.is_dir()
    assert paths.stylize.is_dir()
    assert paths.retexture.is_dir()
    assert paths.viewer.is_dir()
```

```python
# stylized-3d-pipeline/tests/test_run_all.py
def test_ordered_steps_matches_pipeline_contract() -> None:
    assert ordered_steps() == [
        "preprocess",
        "sf3d",
        "sample_views",
        "instantstyle",
        "retexture",
        "viewer",
    ]
```

```python
# stylized-3d-pipeline/tests/test_run_all.py
def test_run_all_main_parses_cli_and_orchestrates(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "content.jpg"
    style_path = tmp_path / "style.jpg"
    input_path.write_text("content", encoding="utf-8")
    style_path.write_text("style", encoding="utf-8")
    runs_root = tmp_path / "runs"
    resolved_run_dir = runs_root / "demo-chair"
    call_log: list[tuple[str, object]] = []

    def fake_resolve_run_dir(base_dir: Path, run_name: str | None) -> Path:
        call_log.append(("resolve_run_dir", base_dir, run_name))
        return resolved_run_dir

    def fake_create_run_tree(run_dir: Path) -> object:
        call_log.append(("create_run_tree", run_dir))
        return object()

    def fake_write_run_config(run_dir: Path, args: object) -> Path:
        call_log.append(
            (
                "write_run_config",
                run_dir,
                getattr(args, "view_resolution"),
                getattr(args, "camera_distance"),
                getattr(args, "camera_fovy_deg"),
                getattr(args, "seed"),
            )
        )
        config_path = run_dir / "run_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("{}", encoding="utf-8")
        return config_path

    def fake_run_pipeline(args: object) -> dict[str, dict]:
        call_log.append(("run_pipeline", getattr(args, "run_dir"), getattr(args, "prompt")))
        return {"ok": True}

    monkeypatch.setattr(run_all, "resolve_run_dir", fake_resolve_run_dir)
    monkeypatch.setattr(run_all, "create_run_tree", fake_create_run_tree)
    monkeypatch.setattr(run_all, "write_run_config", fake_write_run_config)
    monkeypatch.setattr(run_all, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_all.py",
            "--input",
            str(input_path),
            "--style-image",
            str(style_path),
            "--prompt",
            "ceramic chair",
            "--run-name",
            "demo-chair",
            "--runs-root",
            str(runs_root),
            "--sf3d-python",
            "/opt/sf3d/bin/python",
            "--instantstyle-python",
            "/opt/instantstyle/bin/python",
            "--foreground-ratio",
            "0.75",
            "--view-resolution",
            "512",
            "--camera-distance",
            "1.8",
            "--camera-fovy-deg",
            "40.0",
            "--seed",
            "42",
            "--texture-resolution",
            "2048",
            "--remesh-option",
            "triangle",
            "--resume-from",
            "sample_views",
            "--skip-existing",
        ],
    )
    run_all.main()

    assert call_log == [
        ("resolve_run_dir", runs_root, "demo-chair"),
        ("create_run_tree", resolved_run_dir),
        ("write_run_config", resolved_run_dir, 512, 1.8, 40.0, 42),
        ("run_pipeline", resolved_run_dir, "ceramic chair"),
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /root/autodl-tmp/src/.worktrees/stylized-3d-pipeline/stylized-3d-pipeline && pytest tests/test_io_paths.py tests/test_run_all.py -v`

Expected: fail because `RunPaths.views`, `sample_views`, and the new CLI args are not wired through yet.

- [ ] **Step 3: Write minimal implementation**

```python
# stylized-3d-pipeline/lib/io_paths.py
@dataclass(frozen=True)
class RunPaths:
    root: Path
    inputs: Path
    preprocess: Path
    sf3d: Path
    views: Path
    stylize: Path
    retexture: Path
    viewer: Path

def create_run_tree(run_dir: Path) -> RunPaths:
    inputs = run_dir / "inputs"
    preprocess = run_dir / "preprocess"
    sf3d = run_dir / "sf3d"
    views = run_dir / "views"
    stylize = run_dir / "stylize"
    retexture = run_dir / "retexture"
    viewer = run_dir / "viewer"
    for path in (run_dir, inputs, preprocess, sf3d, views, stylize, retexture, viewer):
        path.mkdir(parents=True, exist_ok=True)
    return RunPaths(run_dir, inputs, preprocess, sf3d, views, stylize, retexture, viewer)
```

```python
# stylized-3d-pipeline/lib/pipeline_runner.py
def ordered_steps() -> list[str]:
    return ["preprocess", "sf3d", "sample_views", "instantstyle", "retexture", "viewer"]

def _views_meta(run_dir: Path) -> Path:
    return run_dir / "views" / "manifest.json"

def _step_kwargs(step_name: str, args: object, run_dir: Path) -> dict[str, object]:
    if step_name == "preprocess":
        return {
            "input_path": getattr(args, "input"),
            "run_dir": run_dir,
            "foreground_ratio": getattr(args, "foreground_ratio"),
        }
    if step_name == "sf3d":
        return {
            "run_dir": run_dir,
            "sf3d_python": getattr(args, "sf3d_python"),
            "texture_resolution": getattr(args, "texture_resolution"),
            "remesh_option": getattr(args, "remesh_option"),
        }
    if step_name == "sample_views":
        return {
            "run_dir": run_dir,
            "view_resolution": getattr(args, "view_resolution"),
            "camera_distance": getattr(args, "camera_distance"),
            "camera_fovy_deg": getattr(args, "camera_fovy_deg"),
        }
    if step_name == "instantstyle":
        return {
            "run_dir": run_dir,
            "instantstyle_python": getattr(args, "instantstyle_python"),
            "style_image": getattr(args, "style_image"),
            "prompt": getattr(args, "prompt"),
            "seed": getattr(args, "seed"),
        }
    if step_name == "retexture":
        return {"run_dir": run_dir}
    if step_name == "viewer":
        return {"run_dir": run_dir}
    raise KeyError(f"unknown step: {step_name}")
```

```python
# stylized-3d-pipeline/scripts/run_all.py
parser.add_argument("--view-resolution", default=512, type=int)
parser.add_argument("--camera-distance", default=1.8, type=float)
parser.add_argument("--camera-fovy-deg", default=40.0, type=float)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--resume-from", choices=ordered_steps())
```

```python
# stylized-3d-pipeline/lib/pipeline_runner.py
def write_run_config(run_dir: Path, args: object) -> Path:
    payload = {
        "input": str(getattr(args, "input")),
        "style_image": str(getattr(args, "style_image")),
        "prompt": getattr(args, "prompt"),
        "run_name": getattr(args, "run_name", None),
        "runs_root": str(getattr(args, "runs_root")),
        "run_dir": str(run_dir),
        "sf3d_python": str(getattr(args, "sf3d_python")),
        "instantstyle_python": str(getattr(args, "instantstyle_python")),
        "foreground_ratio": getattr(args, "foreground_ratio"),
        "view_resolution": getattr(args, "view_resolution"),
        "camera_distance": getattr(args, "camera_distance"),
        "camera_fovy_deg": getattr(args, "camera_fovy_deg"),
        "seed": getattr(args, "seed"),
        "texture_resolution": getattr(args, "texture_resolution"),
        "remesh_option": getattr(args, "remesh_option"),
        "resume_from": getattr(args, "resume_from", None),
        "skip_existing": bool(getattr(args, "skip_existing", False)),
    }
    config_path = run_dir / "run_config.json"
    write_json(config_path, payload)
    return config_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /root/autodl-tmp/src/.worktrees/stylized-3d-pipeline/stylized-3d-pipeline && pytest tests/test_io_paths.py tests/test_run_all.py -v`

Expected: pass, including the new `sample_views` step and new CLI args.

- [ ] **Step 5: Commit**

```bash
git add stylized-3d-pipeline/lib/io_paths.py stylized-3d-pipeline/lib/pipeline_runner.py stylized-3d-pipeline/scripts/run_all.py stylized-3d-pipeline/tests/test_io_paths.py stylized-3d-pipeline/tests/test_run_all.py
git commit -m "feat: extend pipeline contract for multiview sampling"
```

## Task 2: Render six canonical views and persist a view manifest

**Files:**
- Create: `stylized-3d-pipeline/lib/camera_views.py`
- Create: `stylized-3d-pipeline/lib/view_sampling.py`
- Create: `stylized-3d-pipeline/scripts/step3_sample_views.py`
- Create: `stylized-3d-pipeline/tests/test_step3_sample_views.py`

- [ ] **Step 1: Write the failing test**

```python
# stylized-3d-pipeline/tests/test_step3_sample_views.py
def test_build_six_view_spec_uses_canonical_axes() -> None:
    mesh = trimesh.creation.box(bounds=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    views = build_six_view_spec(mesh, camera_distance=2.0, fovy_deg=40.0)

    assert [view.name for view in views] == [
        "front",
        "back",
        "left",
        "right",
        "top",
        "bottom",
    ]
    assert views[0].pose.shape == (4, 4)
    assert np.isfinite(views[0].pose).all()
```

```python
# stylized-3d-pipeline/tests/test_step3_sample_views.py
def test_run_step_writes_view_manifest_and_assets(tmp_path: Path) -> None:
    paths = create_run_tree(tmp_path / "run")
    mesh = trimesh.creation.box()
    mesh.export(paths.sf3d / "mesh_raw.glb", include_normals=True)

    def fake_renderer(mesh, views, resolution):  # noqa: ANN001
        payload = {}
        for view in views:
            payload[view.name] = {
                "rgb": Image.new("RGBA", (16, 16), (10, 20, 30, 255)),
                "depth": np.ones((16, 16), dtype=np.float32),
                "normal": Image.new("RGBA", (16, 16), (120, 130, 140, 255)),
                "mask": Image.new("L", (16, 16), 255),
                "control": Image.new("RGBA", (16, 16), (70, 80, 90, 255)),
                "camera": {"name": view.name, "pose": view.pose.tolist()},
            }
        return payload

    result = run_step(paths.root, 16, 2.0, 40.0, renderer=fake_renderer)
    assert (paths.views / "manifest.json").is_file()
    assert (paths.views / "front" / "rgb.png").is_file()
    assert result["view_resolution"] == 16
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /root/autodl-tmp/src/.worktrees/stylized-3d-pipeline/stylized-3d-pipeline && pytest tests/test_step3_sample_views.py -v`

Expected: fail because the camera spec, rendering helper, and CLI wrapper do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# stylized-3d-pipeline/lib/camera_views.py
@dataclass(frozen=True)
class CameraView:
    name: str
    pose: np.ndarray
    fovy_deg: float

def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose

def build_six_view_spec(mesh: trimesh.Trimesh, camera_distance: float, fovy_deg: float) -> list[CameraView]:
    center = mesh.bounds.mean(axis=0)
    radius = float(np.max(mesh.extents) * 0.5)
    distance = max(camera_distance * radius, 1e-3)
    specs = [
        ("front", np.array([distance, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        ("back", np.array([-distance, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        ("left", np.array([0.0, -distance, 0.0]), np.array([0.0, 0.0, 1.0])),
        ("right", np.array([0.0, distance, 0.0]), np.array([0.0, 0.0, 1.0])),
        ("top", np.array([0.0, 0.0, distance]), np.array([0.0, 1.0, 0.0])),
        ("bottom", np.array([0.0, 0.0, -distance]), np.array([0.0, 1.0, 0.0])),
    ]
    return [
        CameraView(name=name, pose=look_at(center + eye, center, up), fovy_deg=fovy_deg)
        for name, eye, up in specs
    ]
```

```python
# stylized-3d-pipeline/lib/view_sampling.py
def render_view_assets(mesh: trimesh.Trimesh, views: list[CameraView], resolution: int) -> dict[str, dict[str, object]]:
    payload: dict[str, dict[str, object]] = {}
    for view in views:
        scene = pyrender.Scene(bg_color=np.array([0, 0, 0, 0], dtype=np.uint8))
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
        camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(view.fovy_deg))
        scene.add(camera, pose=view.pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light, pose=view.pose)
        renderer = pyrender.OffscreenRenderer(resolution, resolution)
        color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        renderer.delete()

        mask_np = np.uint8(depth > 0) * 255
        safe_depth = np.where(depth > 0, depth, 0.0).astype(np.float32)
        max_depth = float(safe_depth.max()) if np.any(safe_depth > 0) else 1.0
        depth_preview = np.uint8(np.clip(safe_depth / max_depth, 0.0, 1.0) * 255)
        gy, gx = np.gradient(safe_depth)
        normal_xyz = np.dstack((-gx, -gy, np.ones_like(safe_depth)))
        denom = np.linalg.norm(normal_xyz, axis=2, keepdims=True)
        denom = np.where(denom == 0.0, 1.0, denom)
        normal_rgb = np.uint8(np.clip((normal_xyz / denom + 1.0) * 127.5, 0, 255))
        normal_rgba = np.dstack([normal_rgb, mask_np])

        payload[view.name] = {
            "rgb": Image.fromarray(color, mode="RGBA"),
            "depth": safe_depth,
            "depth_preview": Image.fromarray(depth_preview, mode="L").convert("RGBA"),
            "normal": Image.fromarray(normal_rgba, mode="RGBA"),
            "mask": Image.fromarray(mask_np, mode="L"),
            "control": Image.blend(
                Image.fromarray(normal_rgba, mode="RGBA"),
                Image.fromarray(depth_preview, mode="L").convert("RGBA"),
                0.5,
            ),
            "camera": {"name": view.name, "pose": view.pose.tolist(), "fovy_deg": view.fovy_deg},
        }
    return payload

def write_view_assets(view_root: Path, assets: dict[str, object]) -> dict[str, str]:
    view_root.mkdir(parents=True, exist_ok=True)
    rgb_path = view_root / "rgb.png"
    depth_path = view_root / "depth.npy"
    depth_preview_path = view_root / "depth.png"
    normal_path = view_root / "normal.png"
    mask_path = view_root / "mask.png"
    control_path = view_root / "control.png"
    camera_path = view_root / "camera.json"
    cast(Image.Image, assets["rgb"]).save(rgb_path)
    np.save(depth_path, assets["depth"])
    cast(Image.Image, assets["depth_preview"]).save(depth_preview_path)
    cast(Image.Image, assets["normal"]).save(normal_path)
    cast(Image.Image, assets["mask"]).save(mask_path)
    cast(Image.Image, assets["control"]).save(control_path)
    camera_path.write_text(json.dumps(assets["camera"], indent=2, sort_keys=True), encoding="utf-8")
    return {
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
        "depth_preview_path": str(depth_preview_path),
        "normal_path": str(normal_path),
        "mask_path": str(mask_path),
        "control_path": str(control_path),
        "camera_path": str(camera_path),
    }
```

```python
# stylized-3d-pipeline/scripts/step3_sample_views.py
def run_step(
    run_dir: Path,
    view_resolution: int,
    camera_distance: float,
    camera_fovy_deg: float,
    renderer: Callable[..., dict[str, dict[str, object]]] = render_view_assets,
) -> dict:
    paths = create_run_tree(run_dir)
    mesh, _ = load_trimesh_with_texture(paths.sf3d / "mesh_raw.glb")
    views = build_six_view_spec(mesh, camera_distance, camera_fovy_deg)
    assets = renderer(mesh, views, view_resolution)
    manifest = {"view_resolution": view_resolution, "views": []}
    for view in views:
        view_dir = paths.views / view.name
        entry = write_view_assets(view_dir, assets[view.name])
        manifest["views"].append({"name": view.name, **entry})
    write_json(paths.views / "manifest.json", manifest)
    return manifest
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /root/autodl-tmp/src/.worktrees/stylized-3d-pipeline/stylized-3d-pipeline && pytest tests/test_step3_sample_views.py -v`

Expected: pass, and the manifest should list all six canonical views.

- [ ] **Step 5: Commit**

```bash
git add stylized-3d-pipeline/lib/camera_views.py stylized-3d-pipeline/lib/view_sampling.py stylized-3d-pipeline/scripts/step3_sample_views.py stylized-3d-pipeline/tests/test_step3_sample_views.py
git commit -m "feat: add six-view sampling stage"
```

## Task 3: Expand InstantStyle into per-view stylization

**Files:**
- Modify: `stylized-3d-pipeline/scripts/workers/instantstyle_worker.py`
- Modify: `stylized-3d-pipeline/scripts/step3_instantstyle.py`
- Modify: `stylized-3d-pipeline/tests/test_step3_instantstyle.py`

- [ ] **Step 1: Write the failing test**

```python
# stylized-3d-pipeline/tests/test_step3_instantstyle.py
def test_build_instantstyle_command_uses_control_and_output_paths() -> None:
    cmd = build_instantstyle_command(
        instantstyle_python=Path("/envs/instantstyle/bin/python"),
        worker_script=Path("/repo/stylized-3d-pipeline/scripts/workers/instantstyle_worker.py"),
        run_dir=Path("/run"),
        control_image=Path("/run/views/front/control.png"),
        style_image=Path("/run/inputs/style.png"),
        prompt="ceramic chair",
        output_image=Path("/run/stylize/front/stylized.png"),
        seed=42,
    )
    assert cmd[0] == "/envs/instantstyle/bin/python"
    assert "--control-image" in cmd
    assert "--output-image" in cmd
```

```python
# stylized-3d-pipeline/tests/test_step3_instantstyle.py
def test_run_step_writes_per_view_stylized_outputs(tmp_path: Path) -> None:
    paths = create_run_tree(tmp_path / "run")
    write_json(
        paths.views / "manifest.json",
        {
            "views": [
                {"name": "front", "control_path": str(paths.views / "front" / "control.png")},
                {"name": "left", "control_path": str(paths.views / "left" / "control.png")},
            ]
        },
    )
    for name in ("front", "left"):
        view_dir = paths.views / name
        view_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (8, 8), (0, 0, 0, 255)).save(view_dir / "control.png")
    Image.new("RGB", (8, 8), "blue").save(paths.inputs / "style.png")

    def fake_runner(cmd, env=None):  # noqa: ANN001
        out_index = cmd.index("--output-image") + 1
        out_path = Path(cmd[out_index])
        Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(out_path)

    result = run_step(
        run_dir=paths.root,
        instantstyle_python=Path("/envs/instantstyle/bin/python"),
        style_image=paths.inputs / "style.png",
        prompt="ceramic chair",
        seed=42,
        runner=fake_runner,
    )
    assert (paths.stylize / "front" / "stylized.png").is_file()
    assert (paths.stylize / "left" / "stylized.png").is_file()
    assert result["views"]["front"]["stylized_path"].endswith("stylized.png")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /root/autodl-tmp/src/.worktrees/stylized-3d-pipeline/stylized-3d-pipeline && pytest tests/test_step3_instantstyle.py -v`

Expected: fail because the worker still only accepts one content image and one output path, and the orchestrator still assumes a single stylized PNG.

- [ ] **Step 3: Write minimal implementation**

```python
# stylized-3d-pipeline/scripts/workers/instantstyle_worker.py
parser.add_argument("--control-image", required=True, type=Path)
parser.add_argument("--style-image", required=True, type=Path)
parser.add_argument("--prompt", required=True)
parser.add_argument("--output-image", required=True, type=Path)
parser.add_argument("--seed", required=True, type=int)

with Image.open(args.control_image) as control_image:
    control = control_image.convert("RGB")
control_map = Image.fromarray(cv2.Canny(np.asarray(control), 50, 200)).convert("RGB")
images = ip_model.generate(
    pil_image=style,
    prompt=args.prompt,
    image=control_map,
    seed=args.seed,
    num_samples=1,
    num_inference_steps=30,
    controlnet_conditioning_scale=0.6,
)
images[0].convert("RGBA").save(args.output_image)
```

```python
# stylized-3d-pipeline/scripts/step3_instantstyle.py
def build_instantstyle_command(
    instantstyle_python: Path,
    worker_script: Path,
    run_dir: Path,
    control_image: Path,
    style_image: Path,
    prompt: str,
    output_image: Path,
    seed: int,
) -> list[str]:
    return [
        str(instantstyle_python),
        str(worker_script),
        "--run-dir",
        str(run_dir),
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
    ]

def run_step(run_dir: Path, instantstyle_python: Path, style_image: Path, prompt: str, seed: int, runner: Callable[..., None] = run_checked) -> dict:
    paths = create_run_tree(run_dir)
    style_copy = paths.inputs / "style.png"
    prompt_file = paths.inputs / "prompt.txt"
    style_copy.write_bytes(style_image.read_bytes())
    prompt_file.write_text(prompt, encoding="utf-8")

    view_manifest = json.loads((paths.views / "manifest.json").read_text(encoding="utf-8"))
    results: dict[str, dict[str, str]] = {"views": {}}
    for view in view_manifest["views"]:
        control_image = Path(view["control_path"])
        output_image = paths.stylize / view["name"] / "stylized.png"
        cmd = build_instantstyle_command(
            instantstyle_python=instantstyle_python,
            worker_script=Path(__file__).resolve().parent / "workers" / "instantstyle_worker.py",
            run_dir=paths.root,
            control_image=control_image,
            style_image=style_copy,
            prompt=prompt,
            output_image=output_image,
            seed=seed,
        )
        runner(cmd, env={"HF_ENDPOINT": "https://hf-mirror.com"})
        results["views"][view["name"]] = {
            "control_path": str(control_image),
            "stylized_path": str(output_image),
        }
    write_json(paths.stylize / "manifest.json", results)
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /root/autodl-tmp/src/.worktrees/stylized-3d-pipeline/stylized-3d-pipeline && pytest tests/test_step3_instantstyle.py -v`

Expected: pass, and every sampled view gets its own stylized output.

- [ ] **Step 5: Commit**

```bash
git add stylized-3d-pipeline/scripts/workers/instantstyle_worker.py stylized-3d-pipeline/scripts/step3_instantstyle.py stylized-3d-pipeline/tests/test_step3_instantstyle.py
git commit -m "feat: stylize all sampled views"
```

## Task 4: Replace the heuristic projector with UV-space multiview reprojection

**Files:**
- Create: `stylized-3d-pipeline/lib/reprojection.py`
- Modify: `stylized-3d-pipeline/scripts/step4_retexture.py`
- Modify: `stylized-3d-pipeline/tests/test_step4_retexture.py`

- [ ] **Step 1: Write the failing test**

```python
# stylized-3d-pipeline/tests/test_step4_retexture.py
def test_multiview_reprojection_prefers_front_facing_visible_view(tmp_path: Path) -> None:
    paths = create_run_tree(tmp_path / "run")
    _write_textured_triangle(paths.sf3d / "mesh_raw.glb", Image.new("RGBA", (8, 8), (255, 255, 255, 255)))
    write_json(paths.views / "manifest.json", {"views": []})
    write_json(paths.stylize / "manifest.json", {"views": {}})

    def fake_loader(view_manifest, stylize_manifest):  # noqa: ANN001
        return ["front"]

    def fake_baker(mesh, base_texture, samples):  # noqa: ANN001
        assert samples == ["front"]
        return Image.new("RGBA", base_texture.size, (255, 0, 0, 255))

    result = run_step(paths.root, sample_loader=fake_loader, baker=fake_baker)
    preview = Image.open(paths.retexture / "texture_preview.png").convert("RGBA")
    assert preview.getpixel((4, 4))[:3] == (255, 0, 0)
    assert result["mesh_path"].endswith("mesh_stylized.glb")
```

```python
# stylized-3d-pipeline/tests/test_step4_retexture.py
def test_multiview_reprojection_falls_back_when_no_view_is_valid(tmp_path: Path) -> None:
    fallback = np.array([255, 255, 255], dtype=np.uint8)
    assert np.array_equal(blend_samples([], fallback), fallback)
    assert np.array_equal(
        blend_samples(
            [
                (0.75, np.array([255, 0, 0], dtype=np.uint8)),
                (0.25, np.array([0, 0, 255], dtype=np.uint8)),
            ],
            fallback,
        ),
        np.array([191, 0, 64], dtype=np.uint8),
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /root/autodl-tmp/src/.worktrees/stylized-3d-pipeline/stylized-3d-pipeline && pytest tests/test_step4_retexture.py -v`

Expected: fail because the current code still uses the single-view heuristic projector and ignores the six-view manifests.

- [ ] **Step 3: Write minimal implementation**

```python
# stylized-3d-pipeline/lib/reprojection.py
@dataclass(frozen=True)
class ViewSample:
    name: str
    pose: np.ndarray
    intrinsic: np.ndarray
    depth: np.ndarray
    stylized: Image.Image

def blend_samples(samples: list[tuple[float, np.ndarray]], fallback: np.ndarray) -> np.ndarray:
    if not samples:
        return fallback
    weights = np.asarray([weight for weight, _ in samples], dtype=np.float32)
    colors = np.asarray([color for _, color in samples], dtype=np.float32)
    rgb = (colors * weights[:, None]).sum(axis=0) / max(weights.sum(), 1e-6)
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)

def project_point_to_view(point: np.ndarray, view: ViewSample) -> tuple[int, int, float] | None:
    world_to_camera = np.linalg.inv(view.pose)
    hom = np.concatenate([point, [1.0]], axis=0)
    camera_point = world_to_camera @ hom
    if camera_point[2] <= 0:
        return None
    pixel = view.intrinsic @ camera_point[:3]
    x = int(round(pixel[0] / pixel[2]))
    y = int(round(pixel[1] / pixel[2]))
    return x, y, float(camera_point[2])
```

```python
# stylized-3d-pipeline/scripts/step4_retexture.py
def run_step(
    run_dir: Path,
    sample_loader: Callable[..., list[ViewSample]] = load_view_samples,
    baker: Callable[[trimesh.Trimesh, Image.Image, list[ViewSample]], Image.Image] = bake_multiview_texture,
) -> dict:
    paths = create_run_tree(run_dir)
    mesh, base_texture = load_trimesh_with_texture(paths.sf3d / "mesh_raw.glb")
    view_manifest = json.loads((paths.views / "manifest.json").read_text(encoding="utf-8"))
    stylize_manifest = json.loads((paths.stylize / "manifest.json").read_text(encoding="utf-8"))
    samples = sample_loader(view_manifest, stylize_manifest)
    baked_texture = baker(mesh, base_texture, samples)
    baked_texture.save(paths.retexture / "texture_preview.png")
    mesh.visual.material.baseColorTexture = baked_texture
    mesh.export(paths.retexture / "mesh_stylized.glb", include_normals=True)
    result = {
        "mesh_path": str(paths.retexture / "mesh_stylized.glb"),
        "texture_preview": str(paths.retexture / "texture_preview.png"),
    }
    write_json(paths.retexture / "retexture_meta.json", result)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /root/autodl-tmp/src/.worktrees/stylized-3d-pipeline/stylized-3d-pipeline && pytest tests/test_step4_retexture.py -v`

Expected: pass, and the preview should show the front-facing view dominating the front texels while unobserved texels fall back to the base SF3D texture.

- [ ] **Step 5: Commit**

```bash
git add stylized-3d-pipeline/lib/reprojection.py stylized-3d-pipeline/scripts/step4_retexture.py stylized-3d-pipeline/tests/test_step4_retexture.py
git commit -m "feat: add multiview UV reprojection"
```

## Task 5: Update the viewer and docs for multiview outputs

**Files:**
- Modify: `stylized-3d-pipeline/lib/viewer_utils.py`
- Modify: `stylized-3d-pipeline/scripts/step5_build_viewer.py`
- Modify: `stylized-3d-pipeline/tests/test_step5_build_viewer.py`
- Modify: `stylized-3d-pipeline/README.md`

- [ ] **Step 1: Write the failing test**

```python
# stylized-3d-pipeline/tests/test_step5_build_viewer.py
def test_write_viewer_includes_multiview_assets(tmp_path: Path) -> None:
    paths = create_run_tree(tmp_path / "run")
    Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(paths.inputs / "content.png")
    Image.new("RGBA", (8, 8), (0, 255, 0, 255)).save(paths.inputs / "style.png")
    Image.new("RGBA", (8, 8), (0, 0, 255, 255)).save(paths.views / "front" / "rgb.png")
    Image.new("RGBA", (8, 8), (255, 255, 0, 255)).save(paths.stylize / "front" / "stylized.png")
    Image.new("RGBA", (8, 8), (255, 255, 255, 255)).save(paths.retexture / "texture_preview.png")
    write_viewer(paths.viewer / "index.html", ["front", "back", "left", "right", "top", "bottom"])

    html = (paths.viewer / "index.html").read_text(encoding="utf-8")
    assert "../views/front/rgb.png" in html
    assert "../stylize/front/stylized.png" in html
    assert "../retexture/texture_preview.png" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /root/autodl-tmp/src/.worktrees/stylized-3d-pipeline/stylized-3d-pipeline && pytest tests/test_step5_build_viewer.py -v`

Expected: fail because the viewer still only knows about one stylized image and does not surface the six sampled views.

- [ ] **Step 3: Write minimal implementation**

```python
# stylized-3d-pipeline/lib/viewer_utils.py
def build_viewer_html(view_names: Sequence[str]) -> str:
    view_cards = "\n".join(
        f"""
        <article class="view-card">
          <h3>{name}</h3>
          <img src="../views/{name}/rgb.png" alt="{name} RGB" />
          <img src="../stylize/{name}/stylized.png" alt="{name} stylized" />
        </article>
        """
        for name in view_names
    )
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Multiview Stylized 3D Viewer</title>
    <script type="module" src="model-viewer.min.js"></script>
    <style>
      :root {{ color-scheme: dark; font-family: Arial, sans-serif; }}
      body {{ margin: 0; background: #111827; color: #f9fafb; }}
      main {{ display: grid; grid-template-columns: 420px 1fr; gap: 24px; padding: 24px; }}
      .panel {{ display: grid; gap: 16px; }}
      .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
      .view-card {{ background: rgba(17, 24, 39, 0.9); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 12px; }}
      .view-card img {{ width: 100%; height: auto; border-radius: 8px; margin-bottom: 8px; }}
      model-viewer {{ width: 100%; height: calc(100vh - 48px); background: linear-gradient(180deg, #1f2937, #0f172a); border-radius: 16px; }}
    </style>
  </head>
  <body>
    <main>
      <section class="panel">
        <div class="view-card">
          <h2>Pipeline Inputs</h2>
          <img src="../inputs/content.png" alt="Content image" />
          <img src="../inputs/style.png" alt="Style image" />
          <img src="../retexture/texture_preview.png" alt="Texture preview" />
        </div>
        <div class="grid">
          {view_cards}
        </div>
      </section>
      <section>
        <model-viewer src="../retexture/mesh_stylized.glb" camera-controls auto-rotate exposure="1" shadow-intensity="1"></model-viewer>
      </section>
    </main>
  </body>
</html>"""

def write_viewer(out_path: Path, view_names: Sequence[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_viewer_html(view_names), encoding="utf-8")
    shutil.copyfile(Path(__file__).resolve().parent / "assets" / "model-viewer.min.js", out_path.parent / "model-viewer.min.js")
```

```python
# stylized-3d-pipeline/scripts/step5_build_viewer.py
def run_step(run_dir: Path) -> dict:
    paths = create_run_tree(run_dir)
    view_manifest = json.loads((paths.views / "manifest.json").read_text(encoding="utf-8"))
    view_names = [view["name"] for view in view_manifest["views"]]
    viewer_html = paths.viewer / "index.html"
    write_viewer(viewer_html, view_names)
    result = {"viewer_html": str(viewer_html), "view_names": view_names}
    write_json(paths.viewer / "viewer_meta.json", result)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /root/autodl-tmp/src/.worktrees/stylized-3d-pipeline/stylized-3d-pipeline && pytest tests/test_step5_build_viewer.py -v`

Expected: pass, and the HTML should surface the six-view gallery plus the final GLB.

- [ ] **Step 5: Update the README and commit**

```md
# stylized-3d-pipeline/README.md
python scripts/run_all.py \
  --input /abs/path/content.png \
  --style-image /abs/path/style.png \
  --prompt "ceramic chair" \
  --sf3d-python /root/autodl-tmp/envs/sf3d/bin/python \
  --instantstyle-python /root/autodl-tmp/envs/instantstyle/bin/python \
  --view-resolution 512 \
  --camera-distance 1.8 \
  --camera-fovy-deg 40.0 \
  --seed 42
```

```bash
git add stylized-3d-pipeline/lib/viewer_utils.py stylized-3d-pipeline/scripts/step5_build_viewer.py stylized-3d-pipeline/tests/test_step5_build_viewer.py stylized-3d-pipeline/README.md
git commit -m "feat: surface multiview outputs in the viewer"
```
