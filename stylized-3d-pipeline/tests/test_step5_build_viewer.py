from pathlib import Path
import json

from PIL import Image

from lib.io_paths import create_run_tree, write_json
from scripts.step5_build_viewer import run_step


def test_run_step_writes_viewer_html_with_expected_assets(tmp_path: Path) -> None:
    paths = create_run_tree(tmp_path / "run")
    Image.new("RGBA", (8, 8), (255, 255, 255, 255)).save(paths.inputs / "content.png")
    Image.new("RGBA", (8, 8), (0, 0, 255, 255)).save(paths.inputs / "style.png")
    write_json(
        paths.views / "manifest.json",
        {
            "views": [{"name": name} for name in ("front", "back", "left", "right", "top", "bottom")],
        },
    )
    for name in ("front", "back", "left", "right", "top", "bottom"):
        (paths.views / name).mkdir(parents=True, exist_ok=True)
        (paths.stylize / name).mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (8, 8), (255, 255, 0, 255)).save(paths.views / name / "rgb.png")
        Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(paths.stylize / name / "stylized.png")
    Image.new("RGBA", (8, 8), (255, 255, 255, 255)).save(paths.retexture / "texture_preview.png")
    (paths.retexture / "mesh_stylized.glb").write_bytes(b"glb")

    result = run_step(paths.root)
    html = Path(result["viewer_html"]).read_text(encoding="utf-8")
    viewer_meta = json.loads((paths.viewer / "viewer_meta.json").read_text(encoding="utf-8"))
    local_asset = paths.viewer / "model-viewer.min.js"
    assert '<script type="module" src="model-viewer.min.js"></script>' in html
    assert "https://unpkg.com" not in html
    assert "../inputs/content.png" in html
    assert "../inputs/style.png" in html
    assert "../views/front/rgb.png" in html
    assert "../stylize/front/stylized.png" in html
    assert "../retexture/texture_preview.png" in html
    assert "../retexture/mesh_stylized.glb" in html
    assert local_asset.is_file()
    assert local_asset.stat().st_size > 0
    assert viewer_meta == {
        "viewer_html": str(paths.viewer / "index.html"),
        "view_names": ["front", "back", "left", "right", "top", "bottom"],
    }


def test_run_step_rejects_noncanonical_view_manifest(tmp_path: Path) -> None:
    paths = create_run_tree(tmp_path / "run")
    write_json(
        paths.views / "manifest.json",
        {
            "views": [
                {"name": "front"},
                {"name": "left"},
                {"name": "back"},
                {"name": "right"},
                {"name": "top"},
                {"name": "bottom"},
            ],
        },
    )

    try:
        run_step(paths.root)
    except ValueError as exc:
        assert "canonical" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for non-canonical view manifest")


def test_run_step_rejects_view_manifest_entry_without_name(tmp_path: Path) -> None:
    paths = create_run_tree(tmp_path / "run")
    write_json(
        paths.views / "manifest.json",
        {
            "views": [
                {"name": "front"},
                {"name": "back"},
                {"control_path": "missing-name.png"},
                {"name": "right"},
                {"name": "top"},
                {"name": "bottom"},
            ],
        },
    )

    try:
        run_step(paths.root)
    except ValueError as exc:
        assert "invalid" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for malformed view manifest entry")
