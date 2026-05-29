# 万物赋新台

> 岁月器物的三维留存与跨次元重塑

单张实物照片 → SF3D 三维重建 → 六视角渲染 → InstantStyle 风格迁移 → UV 回贴 → 风格化 3D 模型。

---

## 目录结构

```
src/
├── app.py                  # Gradio 前端入口
├── stable-fast-3d/          # SF3D 开源项目（git submodule）
├── InstantStyle/            # InstantStyle 开源项目（git submodule）
├── stylized-3d-pipeline/    # Pipeline 编排层
│   ├── lib/                 # 核心库
│   ├── scripts/             # 步骤脚本 + workers
│   └── tests/               # 单元测试
└── requirements.txt
```

---

## 一、环境部署

### 1.1 硬件要求

- **GPU**：NVIDIA GPU，显存 ≥ 16 GB（SF3D + SDXL 各需约 6-8 GB）
- **CUDA**：12.4（当前环境），12.1+ 均可
- **磁盘**：约 80 GB（模型文件约 50 GB + Python 环境）

### 1.2 获取代码

```bash
git clone <repo-url> src
cd src

# 拉取子模块（SF3D + InstantStyle）
git submodule update --init --recursive

# 安装本地 C++ 扩展
pip install stable-fast-3d/texture_baker/
pip install stable-fast-3d/uv_unwrapper/
```

### 1.3 Python 环境

本项目三种模型（SF3D、InstantStyle、Gradio）的依赖可以共存在同一 Python 环境中，也可以各自使用独立环境。当前默认使用单环境方案。

```bash
# 创建环境（推荐 conda / venv）
conda create -n stylized3d python=3.11
conda activate stylized3d

# 安装依赖
pip install -r requirements.txt

# 安装本地扩展
pip install stable-fast-3d/texture_baker/
pip install stable-fast-3d/uv_unwrapper/

# 验证 PyTorch CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
```

> **跨环境方案**（可选）：如果 SF3D 或 InstantStyle 依赖冲突，可以创建独立的 conda 环境，然后通过 `--sf3d-python` / `--instantstyle-python` 参数或环境变量 `APP_SF3D_PYTHON` / `APP_INSTANTSTYLE_PYTHON` 指向各自的 Python 解释器。

### 1.4 模型文件

运行时会自动从 HuggingFace 下载以下模型。如果网络受限（国内环境），需要手动预下载并配置缓存路径。

#### SF3D 模型（约 2 GB）

| 模型 | HuggingFace 路径 |
|------|-----------------|
| 主模型 | `stabilityai/stable-fast-3d` |

自动缓存到 `HF_HOME` 指向的目录。

#### InstantStyle 模型（约 15 GB）

| 模型 | HuggingFace 路径 |
|------|-----------------|
| SDXL Base | `stabilityai/stable-diffusion-xl-base-1.0` |
| ControlNet Canny | `diffusers/controlnet-canny-sdxl-1.0` |
| IP-Adapter SDXL | `h94/IP-Adapter` (仅 `sdxl_models/ip-adapter_sdxl.bin`) |
| IP-Adapter Image Encoder | `h94/IP-Adapter` (仅 `sdxl_models/image_encoder/`) |
| CLIP ViT-B/32 | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` |

#### 模型缓存路径配置

```bash
# 模型缓存根目录（默认 /root/autodl-tmp/hf-cache）
export HF_HOME=/root/autodl-tmp/hf-cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf-cache/hub
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf-cache/transformers
export DIFFUSERS_CACHE=/root/autodl-tmp/hf-cache/hub

# 国内镜像（加速下载）
export HF_ENDPOINT=https://hf-mirror.com

# IP-Adapter 模型路径（worker 会自动搜索，也可以手动指定）
export IP_ADAPTER_SDXL_MODELS=/root/autodl-tmp/models/IP-Adapter/sdxl_models

# SDXL Base 模型（如需使用本地副本）
export INSTANTSTYLE_SDXL_MODELS=/root/autodl-tmp/models/sdxl-base
```

路径查找优先级（`instantstyle_worker.py` 中的 `resolve_sdxl_models_root`）：
1. `$INSTANTSTYLE_SDXL_MODELS` 或 `$IP_ADAPTER_SDXL_MODELS` 环境变量
2. `<project_root>/sdxl_models/`
3. `/root/autodl-tmp/models/IP-Adapter/sdxl_models`（默认路径）

#### 模型目录预期结构

```
/root/autodl-tmp/
├── hf-cache/                     # HuggingFace 缓存
│   ├── hub/                      # diffusers / SF3D 模型
│   │   ├── models--stabilityai--stable-fast-3d/
│   │   ├── models--stabilityai--stable-diffusion-xl-base-1.0/
│   │   ├── models--diffusers--controlnet-canny-sdxl-1.0/
│   │   ├── models--h94--IP-Adapter/
│   │   └── models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K/
│   └── transformers/             # transformers 缓存
└── models/                       # 手动下载的模型
    └── IP-Adapter/
        └── sdxl_models/
            ├── image_encoder/     # CLIP image encoder 权重
            └── ip-adapter_sdxl.bin
```

### 1.5 环境变量总览

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `APP_RUNS_ROOT` | `runs_manual/app` | 运行产出目录 |
| `APP_SF3D_PYTHON` | 当前 Python | SF3D 解释器路径 |
| `APP_INSTANTSTYLE_PYTHON` | 当前 Python | InstantStyle 解释器路径 |
| `APP_FOREGROUND_RATIO` | `0.85` | 前景占比 |
| `APP_TEXTURE_RESOLUTION` | `1024` | 纹理分辨率 |
| `APP_VIEW_RESOLUTION` | `512` | 渲染视角分辨率 |
| `APP_CAMERA_DISTANCE` | `1.8` | 相机距离系数 |
| `APP_CAMERA_FOVY_DEG` | `40.0` | 相机垂直 FOV |
| `APP_SEED` | `42` | 随机种子 |
| `APP_STRENGTH` | `0.72` | 风格化强度 |
| `APP_STYLE_SCALE` | `1.8` | IP-Adapter 风格注入权重 |
| `APP_GUIDANCE_SCALE` | `6.5` | 无分类器引导强度 |
| `APP_NUM_INFERENCE_STEPS` | `35` | 去噪步数 |
| `APP_CONTROLNET_CONDITIONING_SCALE` | `0.45` | ControlNet 控制强度 |
| `HF_HOME` | — | HuggingFace 缓存根目录 |
| `HF_ENDPOINT` | — | HuggingFace 镜像（国内用） |
| `PYOPENGL_PLATFORM` | `egl` | 离屏渲染后端（必须为 egl 或 osmesa） |

---

## 二、运行方式

### 2.1 Gradio 前端（推荐）

```bash
cd /root/autodl-tmp/src
python app.py
```

启动后访问 `http://<host>:7860`，上传物体照片 + 风格参考图 + 填写提示词，点击「开始生成」即可。

⚠️ **注意**：Gradio 前端会串联运行全部 6 个步骤，单次生成耗时约 3-8 分钟（取决于 GPU 性能和参数设置）。建议设置 `default_concurrency_limit=1` 避免并发导致显存溢出。

### 2.2 命令行全流程

```bash
cd /root/autodl-tmp/src/stylized-3d-pipeline

python scripts/run_all.py \
  --input /path/to/content.jpg \
  --style-image /path/to/style.jpg \
  --prompt "描述文本" \
  --run-name my-run \
  --runs-root runs \
  --sf3d-python $(which python) \
  --instantstyle-python $(which python) \
  --view-resolution 512 \
  --camera-distance 1.8 \
  --camera-fovy-deg 40.0 \
  --seed 42 \
  --foreground-ratio 0.85 \
  --texture-resolution 1024
```

### 2.3 分步执行（调试用）

```bash
cd /root/autodl-tmp/src/stylized-3d-pipeline

# 步骤1：预处理（去背景 + 缩放）
python scripts/step1_preprocess.py \
  --input /path/to/content.jpg \
  --run-dir runs/my-run \
  --foreground-ratio 0.85

# 步骤2：SF3D 三维重建
python scripts/step2_sf3d.py \
  --run-dir runs/my-run \
  --sf3d-python $(which python) \
  --texture-resolution 1024

# 步骤3a：六视角渲染采样
python scripts/step3_sample_views.py \
  --run-dir runs/my-run \
  --view-resolution 512 \
  --camera-distance 1.8 \
  --camera-fovy-deg 40.0

# 步骤3b：InstantStyle 风格化
python scripts/step3_instantstyle.py \
  --run-dir runs/my-run \
  --instantstyle-python $(which python) \
  --style-image /path/to/style.jpg \
  --prompt "描述文本" \
  --seed 42 \
  --strength 0.72

# 步骤4：UV 回贴
python scripts/step4_retexture.py --run-dir runs/my-run

# 步骤5：生成 3D 预览页
python scripts/step5_build_viewer.py --run-dir runs/my-run
```

### 2.4 断点续跑

```bash
# 从风格化步骤恢复（跳过前序步骤）
python scripts/run_all.py \
  ... \
  --resume-from instantstyle \
  --skip-existing
```

`--resume-from` 可选值：`preprocess`, `sf3d`, `sample_views`, `instantstyle`, `retexture`, `viewer`。

### 2.5 关键参数说明

| 参数 | 默认值 | 影响 |
|------|--------|------|
| `--foreground-ratio` | `0.85` | 越高前景越大，SF3D 重建更聚焦但可能裁边 |
| `--texture-resolution` | `1024` | 纹理贴图分辨率，越高细节越多但显存消耗更大 |
| `--view-resolution` | `512` | 六视角渲染分辨率，影响风格化精度 |
| `--camera-distance` | `1.8` | 相机距离系数，越大视角越远物体越小 |
| `--strength` | `0.72` | 风格化强度，0.4-1.0 之间。越高风格越强，几何保持越弱 |
| `--style-scale` | `1.8` | IP-Adapter 风格特征权重 |
| `--guidance-scale` | `6.5` | CFG 引导强度，越高越贴近 prompt，越低越自由 |
| `--controlnet-conditioning-scale` | `0.45` | ControlNet 结构保持强度，越高几何越忠实 |
| `--seed` | `42` | 随机种子，固定可复现 |

### 2.6 产出文件

每次运行在 `<runs_root>/<run_name>/` 下生成：

```
<run_name>/
├── run_config.json          # 完整参数快照
├── inputs/                  # 输入备份
├── preprocess/              # 预处理结果（rgba.png + mask.png）
├── sf3d/                    # SF3D 重建结果（mesh_raw.glb）
├── views/                   # 六视角渲染（rgb / depth / normal / control / mask）
├── stylize/                 # 六视角风格化结果（stylized.png）
├── retexture/
│   ├── mesh_stylized.glb    # ★ 最终风格化 3D 模型
│   └── texture_preview.png  # 烘焙纹理预览
└── viewer/
    └── index.html           # 离线 3D 预览页
```

`retexture/mesh_stylized.glb` 即为可下载、可在任意 glTF 查看器中打开的最终模型。

---

## 三、常见问题

**Q: 显存不足 (OOM)？**
- 降低 `--texture-resolution` 到 512
- 降低 `--view-resolution` 到 384
- 降低 `--num-inference-steps` 到 20
- 确保没有其他进程占用 GPU

**Q: 离屏渲染报错？**
- 确认 `PYOPENGL_PLATFORM=egl` 或安装 `libosmesa6`
- 无头服务器必须使用 EGL/OSMesa，不能用 GLX

**Q: HuggingFace 下载超时？**
- 设置 `HF_ENDPOINT=https://hf-mirror.com` 使用国内镜像
- 或手动下载模型后放到 `HF_HOME` 对应目录

**Q: rembg 下载模型失败？**
- rembg 首次运行会自动下载 U²-Net 模型到 `~/.u2net/`
- 也可手动下载 `u2net.pth` 放到该目录

**Q: 风格化结果太强/太弱？**
- 太强（丢失结构）：降低 `--strength`（0.4-0.6）、增大 `--controlnet-conditioning-scale`（0.6-0.8）
- 太弱（风格不明显）：增大 `--strength`（0.8-1.0）、增大 `--style-scale`（2.0-3.0）
