from pathlib import Path
import json

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
    viewer_meta = json.loads((paths.viewer / "viewer_meta.json").read_text(encoding="utf-8"))
    assert "model-viewer" in html
    assert "../inputs/content.png" in html
    assert "../inputs/style.png" in html
    assert "../stylize/stylized.png" in html
    assert "../retexture/mesh_stylized.glb" in html
    assert viewer_meta == {"viewer_html": str(paths.viewer / "index.html")}
