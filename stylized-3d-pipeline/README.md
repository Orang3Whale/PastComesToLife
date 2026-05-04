# Stylized 3D Pipeline

This project wires together preprocessing, SF3D reconstruction, InstantStyle stylization, texture transfer, and a local viewer.

## Setup

Install dependencies in the project environment, then make sure the upstream repos are available alongside this repository:

- `stable-fast-3d/`
- `InstantStyle/`

The orchestration layer calls the upstream worker scripts through the configured Python interpreters, so each upstream project can keep its own virtual environment.

## Step-by-step usage

Run each stage in order when you want to inspect intermediates:

```bash
python scripts/step1_preprocess.py --input /abs/path/content.jpg --run-dir runs/demo-mug
python scripts/step2_sf3d.py --run-dir runs/demo-mug --sf3d-python /root/autodl-tmp/envs/sf3d/bin/python
python scripts/step3_sample_views.py --run-dir runs/demo-mug --view-resolution 512 --camera-distance 1.8 --camera-fovy-deg 40.0
python scripts/step3_instantstyle.py --run-dir runs/demo-mug --style-image /abs/path/style.jpg --prompt "ceramic mug" --instantstyle-python /root/autodl-tmp/envs/instantstyle/bin/python
python scripts/step4_retexture.py --run-dir runs/demo-mug
python scripts/step5_build_viewer.py --run-dir runs/demo-mug
```

## One-shot usage

Use the orchestration script to create the run tree, record the config, and execute the full pipeline:

```bash
python scripts/run_all.py \
  --input /abs/path/content.jpg \
  --style-image /abs/path/style.jpg \
  --prompt "ceramic mug" \
  --run-name demo-mug \
  --runs-root runs \
  --sf3d-python /root/autodl-tmp/envs/sf3d/bin/python \
  --instantstyle-python /root/autodl-tmp/envs/instantstyle/bin/python \
  --view-resolution 512 \
  --camera-distance 1.8 \
  --camera-fovy-deg 40.0 \
  --seed 42
```

Useful flags:

- `--resume-from <step_name>` resumes from `preprocess`, `sf3d`, `sample_views`, `instantstyle`, `retexture`, or `viewer`
- `--skip-existing` skips any step whose metadata already exists
- `--foreground-ratio`, `--texture-resolution`, and `--remesh-option` forward directly into the corresponding stages
- `--view-resolution`, `--camera-distance`, `--camera-fovy-deg`, and `--seed` control the multiview sampling and stylization stages

## Current experiment repro

Use this wrapper when you want to reproduce the current strong-style multiview result with editable parameters.

Fast rebake from an existing run that already has `sf3d`, `views`, and `stylize`:

```bash
python scripts/reproduce_current_experiment.py \
  --source-run runs/real-chair-starry-strongstyle-v1 \
  --run-name real-chair-starry-strongstyle-vflip-repro
```

Full run from an input image:

```bash
python scripts/reproduce_current_experiment.py \
  --input /abs/path/content.png \
  --style-image /root/autodl-tmp/src/InstantStyle/assets/4.jpg \
  --run-name real-chair-starry-full-repro \
  --prompt "a wooden chair, blue starry night, crescent moon, golden stars, decorative painted textile style, bold blue and gold folk art" \
  --strength 0.72 \
  --style-scale 1.8 \
  --guidance-scale 6.5 \
  --num-inference-steps 35 \
  --controlnet-conditioning-scale 0.45
```

The wrapper defaults `--sf3d-python` and `--instantstyle-python` to the current Python interpreter. Override either flag only when you intentionally run SF3D or InstantStyle from a separate environment.

The default strong-style values in `reproduce_current_experiment.py` match the latest accepted run. Override any CLI flag to tune camera, SF3D, style strength, IP-Adapter scale, guidance, sampling steps, or ControlNet scale.
