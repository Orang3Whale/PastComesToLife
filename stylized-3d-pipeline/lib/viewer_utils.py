from __future__ import annotations

from pathlib import Path


def build_viewer_html() -> str:
    return """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Stylized 3D Viewer</title>
    <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
    <style>
      :root {
        color-scheme: dark;
        font-family: Arial, sans-serif;
      }
      body {
        margin: 0;
        min-height: 100vh;
        background: #111827;
        color: #f9fafb;
      }
      main {
        display: grid;
        grid-template-columns: minmax(280px, 360px) 1fr;
        gap: 24px;
        min-height: 100vh;
        padding: 24px;
        box-sizing: border-box;
      }
      section {
        background: rgba(17, 24, 39, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 16px;
      }
      .assets {
        display: grid;
        gap: 12px;
      }
      .asset {
        display: grid;
        gap: 6px;
      }
      .asset img {
        width: 100%;
        height: auto;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: #0f172a;
      }
      model-viewer {
        width: 100%;
        height: calc(100vh - 48px);
        background: linear-gradient(180deg, #1f2937, #0f172a);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.12);
      }
      h1, h2, p {
        margin: 0 0 12px;
      }
      .caption {
        font-size: 0.875rem;
        color: #cbd5e1;
      }
    </style>
  </head>
  <body>
    <main>
      <section>
        <h1>Stylized 3D Pipeline</h1>
        <p class="caption">Inputs, stylized texture, and textured mesh preview.</p>
        <div class="assets">
          <div class="asset">
            <h2>Content</h2>
            <img src="../inputs/content.png" alt="Content image" />
          </div>
          <div class="asset">
            <h2>Style</h2>
            <img src="../inputs/style.png" alt="Style image" />
          </div>
          <div class="asset">
            <h2>Stylized</h2>
            <img src="../stylize/stylized.png" alt="Stylized texture" />
          </div>
        </div>
      </section>
      <section>
        <model-viewer
          src="../retexture/mesh_stylized.glb"
          camera-controls
          auto-rotate
          exposure="1"
          shadow-intensity="1"
          alt="Stylized 3D model"
        ></model-viewer>
      </section>
    </main>
  </body>
</html>
"""


def write_viewer(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_viewer_html(), encoding="utf-8")
