from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from lib.io_paths import create_run_tree, write_json
from scripts import reproduce_current_experiment as repro


def test_parser_defaults_worker_python_to_current_interpreter() -> None:
    args = repro.build_parser().parse_args([])

    assert args.sf3d_python == Path(sys.executable)
    assert args.instantstyle_python == Path(sys.executable)


def test_build_worker_command_includes_custom_stylize_params(tmp_path: Path) -> None:
    cmd = repro.build_worker_command(
        instantstyle_python=Path("/envs/instantstyle/bin/python"),
        worker_script=Path("/repo/scripts/workers/instantstyle_worker.py"),
        run_dir=tmp_path / "run",
        rgb_image=Path("/run/views/front/rgb.png"),
        control_image=Path("/run/views/front/control.png"),
        style_image=Path("/run/inputs/style.png"),
        prompt="ceramic mug",
        output_image=Path("/run/stylize/front/stylized.png"),
        seed=123,
        strength=0.72,
        style_scale=1.8,
        guidance_scale=6.5,
        num_inference_steps=35,
        controlnet_conditioning_scale=0.45,
    )

    assert cmd[0] == "/envs/instantstyle/bin/python"
    assert "--strength" in cmd and "0.72" in cmd
    assert "--style-scale" in cmd and "1.8" in cmd
    assert "--guidance-scale" in cmd and "6.5" in cmd
    assert "--num-inference-steps" in cmd and "35" in cmd
    assert "--controlnet-conditioning-scale" in cmd and "0.45" in cmd


def test_rebake_mode_rewrites_manifest_paths(tmp_path: Path) -> None:
    source_run = create_run_tree(tmp_path / "source")
    target_run = tmp_path / "runs" / "rebake"

    write_json(
        source_run.views / "manifest.json",
        {
            "views": [
                {
                    "name": name,
                    "rgb_path": f"{source_run.views}/{name}/rgb.png",
                    "control_path": f"{source_run.views}/{name}/control.png",
                    "depth_path": f"{source_run.views}/{name}/depth.npy",
                    "depth_preview_path": f"{source_run.views}/{name}/depth.png",
                    "normal_path": f"{source_run.views}/{name}/normal.png",
                    "mask_path": f"{source_run.views}/{name}/mask.png",
                    "camera_path": f"{source_run.views}/{name}/camera.json",
                }
                for name in ("front", "back", "left", "right", "top", "bottom")
            ],
        },
    )
    write_json(
        source_run.stylize / "manifest.json",
        {
            "style_image": f"{source_run.inputs}/style.png",
            "views": {
                name: {
                    "rgb_path": f"{source_run.views}/{name}/rgb.png",
                    "control_path": f"{source_run.views}/{name}/control.png",
                    "stylized_path": f"{source_run.stylize}/{name}/stylized.png",
                }
                for name in ("front", "back", "left", "right", "top", "bottom")
            },
        },
    )
    for name in ("front", "back", "left", "right", "top", "bottom"):
        (source_run.views / name).mkdir(parents=True, exist_ok=True)
        (source_run.stylize / name).mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (4, 4), (0, 0, 0, 255)).save(source_run.views / name / "rgb.png")
        Image.new("RGBA", (4, 4), (0, 0, 0, 255)).save(source_run.views / name / "control.png")
        np.save(source_run.views / name / "depth.npy", np.ones((4, 4), dtype=np.float32))
        Image.new("RGBA", (4, 4), (0, 0, 0, 255)).save(source_run.views / name / "normal.png")
        Image.new("RGBA", (4, 4), (0, 0, 0, 255)).save(source_run.views / name / "mask.png")
        (source_run.views / name / "camera.json").write_text("{}", encoding="utf-8")
        Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(source_run.stylize / name / "stylized.png")
    Image.new("RGB", (4, 4), "blue").save(source_run.inputs / "style.png")

    calls: list[tuple[str, object]] = []

    def fake_retexture(run_dir: Path) -> dict[str, str]:
        calls.append(("retexture", run_dir))
        return {"mesh_path": str(run_dir / "retexture" / "mesh_stylized.glb")}

    def fake_viewer(run_dir: Path) -> dict[str, str]:
        calls.append(("viewer", run_dir))
        return {"viewer_html": str(run_dir / "viewer" / "index.html")}

    args = SimpleNamespace(
        source_run=source_run.root,
        runs_root=tmp_path / "runs",
        run_name="rebake",
    )

    result = repro.run_rebake_experiment(args, retexture_runner=fake_retexture, viewer_runner=fake_viewer)

    views_manifest = json.loads((target_run / "views" / "manifest.json").read_text(encoding="utf-8"))
    stylize_manifest = json.loads((target_run / "stylize" / "manifest.json").read_text(encoding="utf-8"))

    assert calls == [("retexture", target_run), ("viewer", target_run)]
    assert result["run_dir"] == str(target_run)
    assert views_manifest["views"][0]["rgb_path"].startswith(str(target_run / "views"))
    assert stylize_manifest["style_image"] == str(target_run / "inputs" / "style.png")
    assert stylize_manifest["views"]["front"]["stylized_path"] == str(target_run / "stylize" / "front" / "stylized.png")
