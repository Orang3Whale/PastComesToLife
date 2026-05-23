# 万物赋新台 — 单图到风格化3D模型技术文档

## 概述

本项目实现了一条从**单张实物照片**到**风格化三维模型**的完整处理管线。用户上传一张物体照片和一张风格参考图，系统自动完成：背景去除 → 三维重建 → 六视角渲染 → 多视角风格迁移 → UV纹理回贴 → 最终风格化模型输出。

整个管线由 **Gradio 前端**（`app.py`）和 **Pipeline 后端**（`stylized-3d-pipeline/`）组成，底层依赖两个开源项目：`stable-fast-3d`（单图3D重建）和 `InstantStyle`（图像风格迁移）。

---

## 一、整体架构

```
┌──────────────────────────────────────────────────────────┐
│  Gradio UI (app.py)                                      │
│  - 图片上传 / 提示词输入                                   │
│  - 七步流水线进度展示                                      │
│  - 3D 模型实时预览                                        │
└──────────┬───────────────────────────────────────────────┘
           │ 调用 pipeline step
           ▼
┌──────────────────────────────────────────────────────────┐
│  Pipeline Orchestration (stylized-3d-pipeline/)           │
│                                                          │
│  step1 → step2 → step3_sample → step3_instantstyle       │
│                        → step4 → step5                   │
│                                                          │
│  每个 step 通过 subprocess 调用底层 worker                 │
│  每个 step 产出 JSON meta 文件，支持断点续跑                │
└──────────┬───────────────────────────────────────────────┘
           │ 调用子进程 / 独立 Python 解释器
           ▼
┌──────────────────────┐   ┌──────────────────────────────┐
│  stable-fast-3d      │   │  InstantStyle                 │
│  (3D 重建)            │   │  (风格迁移)                    │
│  - SF3D model         │   │  - ControlNet + IP-Adapter    │
│  - Texture Baker       │   │  - SDXL img2img pipeline      │
│  - UV Unwrapper        │   │                                │
└──────────────────────┘   └──────────────────────────────┘
```

### 目录结构

```
src/
├── app.py                          # Gradio 前端入口
├── stable-fast-3d/                  # SF3D 开源项目（子模块）
│   └── sf3d/system.py              # SF3D 核心推理逻辑
├── InstantStyle/                    # InstantStyle 开源项目（子模块）
│   ├── ip_adapter/                 # IP-Adapter 核心实现
│   │   ├── ip_adapter.py           # IP-Adapter/XL 类定义
│   │   ├── attention_processor.py  # 注意力处理器（注入风格特征）
│   │   ├── resampler.py            # 特征重采样
│   │   └── utils.py
│   └── infer_style_plus.py         # 示例推理脚本
└── stylized-3d-pipeline/           # Pipeline 编排层
    ├── lib/
    │   ├── camera_views.py          # 六视角相机位姿计算
    │   ├── view_sampling.py         # 视角渲染 + Control map 生成
    │   ├── offscreen_renderer.py    # Pyrender 离屏渲染
    │   ├── reprojection.py          # UV 回贴（多视角→单纹理）
    │   ├── mesh_utils.py            # Mesh 加载 / 单视角 bake
    │   ├── image_utils.py           # 图像预处理工具
    │   ├── io_paths.py              # 目录结构与 JSON 读写
    │   ├── subprocess_utils.py      # 子进程调用 + HuggingFace 缓存
    │   ├── viewer_utils.py          # HTML viewer 生成
    │   └── pipeline_runner.py       # Pipeline 编排器（断点续跑）
    ├── scripts/
    │   ├── step1_preprocess.py      # 步骤1：预处理
    │   ├── step2_sf3d.py            # 步骤2：SF3D 重建
    │   ├── step3_sample_views.py    # 步骤3a：六视角采样
    │   ├── step3_instantstyle.py    # 步骤3b：风格化
    │   ├── step4_retexture.py       # 步骤4：UV 回贴
    │   ├── step5_build_viewer.py    # 步骤5：生成预览页
    │   ├── run_all.py               # 一键全流程
    │   ├── reproduce_current_experiment.py
    │   └── workers/
    │       ├── sf3d_worker.py       # SF3D 子进程入口
    │       └── instantstyle_worker.py  # InstantStyle 子进程入口
    └── tests/                       # 各步骤单元测试
```

---

## 二、数据流与目录约定

每个任务在 `runs_manual/app/<task_id>/` 下创建标准化的目录树：

```
<run_dir>/
├── run_config.json          # 任务参数快照
├── inputs/
│   ├── content.png          # 原始输入图
│   ├── style.png            # 风格参考图
│   └── prompt.txt           # 提示词
├── preprocess/
│   ├── rgba.png             # 去背景 + 缩放的 RGBA 图
│   ├── mask.png             # Alpha 遮罩
│   └── meta.json
├── sf3d/
│   ├── mesh_raw.glb         # SF3D 重建输出（带原始纹理）
│   ├── input.png            # 输入图备份
│   └── sf3d_meta.json
├── views/
│   ├── manifest.json        # 相机参数与文件路径索引
│   ├── front/               # (rgb.png, depth.npy, depth.png,
│   ├── back/                #  normal.png, mask.png,
│   ├── left/                #  control.png, camera.json)
│   ├── right/
│   ├── top/
│   └── bottom/
├── stylize/
│   ├── manifest.json        # 风格化参数与输出索引
│   ├── worker_jobs.json     # 批量任务描述
│   └── <view>/stylized.png  # 各视角风格化结果
├── retexture/
│   ├── mesh_stylized.glb    # 最终风格化模型
│   ├── texture_preview.png  # 烘焙纹理预览
│   └── retexture_meta.json
└── viewer/
    ├── index.html           # 3D 预览页
    ├── model-viewer.min.js
    └── viewer_meta.json
```

---

## 三、步骤详解

### 步骤 1：图像预处理 (`step1_preprocess.py`)

**目的：** 将用户上传的实物照片处理为干净的前景 RGBA 图，供 SF3D 使用。

**实现细节：**

1. **去背景 — rembg**：使用 `rembg` 库（基于 U²-Net 的显著性检测模型）自动移除图像背景，生成 RGBA 四通道图像。
   - 创建新的 `rembg.Session()` 后调用 `rembg.remove(image, session=session)`
   - 输入图像若不含 alpha 通道，先执行 `image.convert("RGBA")`

2. **前景缩放 — `resize_foreground_rgba`**（`lib/image_utils.py`）：
   - 通过 alpha 通道定位前景的包围盒（`alpha_bbox`）
   - 裁剪出前景区域，按 `foreground_ratio`（默认 0.85）计算目标画布大小
   - 前景缩放至画布的 85%，居中放置于透明画布上
   - 最终输出正方形 RGBA 图（最小 64px）

3. **输出：** `preprocess/rgba.png` 和 `preprocess/mask.png`

---

### 步骤 2：SF3D 三维重建 (`step2_sf3d.py`)

**目的：** 从单张 RGBA 图像生成带纹理的 3D Mesh。

**实现细节：**

本步骤通过子进程调用 `workers/sf3d_worker.py`（使用独立 Python 解释器以隔离 CUDA 依赖），worker 内部调用 Stable-Fast-3D 模型。

**Stable-Fast-3D 推理流程**（`stable-fast-3d/sf3d/system.py`）：

1. **模型加载**：从 HuggingFace `stabilityai/stable-fast-3d` 加载预训练权重（`model.safetensors`），模型包含以下核心组件：
   - **Image Tokenizer**（基于 DINOv2）：将输入图像编码为视觉 token 序列
   - **Camera Embedder**：将相机参数编码为条件嵌入
   - **Backbone Transformer**：融合图像 token + 相机嵌入，生成场景编码
   - **Triplane Tokenizer/Decoder**：将场景编码解码为三平面（Triplane）表示
   - **Marching Tetrahedra**：从三平面采样的 SDF 值中提取等值面生成 Mesh
   - **Image Estimator**（CLIP-based）：估计全局外观特征（albedo、roughness、metallic）

2. **图像准备**：
   - 输入 RGBA 图 resize 至 `cond_image_size`
   - 用 background_color（灰色 `[0.5, 0.5, 0.5]`）与前景做 `torch.lerp` 混合
   - 生成 `rgb_cond`（3 通道）和 `mask_cond`（1 通道）

3. **Triplane 生成与 Mesh 提取**：
   - 固定相机位姿（`default_cond_c2w`）和相机内参（由 `default_fovy_deg = 40°` 推导）
   - 通过 Backbone 获得 `scene_codes`（三平面表示）
   - 在三平面采样网格上查询 SDF 值（`query_triplane`）
   - 解码器输出 vertex offset（几何细节）和 density（有符号距离场）
   - `density - isosurface_threshold(10.0)` 得到 SDF
   - 使用 Marching Tetrahedra 算法从 SDF + deformation 提取显式 Mesh

4. **UV 展开与纹理烘焙**：
   - 调用 `mesh.unwrap_uv()` 进行 UV 展开
   - `TextureBaker` 将 Mesh 光栅化到纹理空间
   - 在纹理空间采样三平面特征，解码为 albedo、roughness、metallic、normal
   - 生成 PBR 材质，导出为 `mesh_raw.glb`

5. **输出：** `sf3d/mesh_raw.glb`（含原始纹理的 GLB 模型）

**子进程隔离机制**：
- 通过 `--sf3d-python` 指定独立 Python 解释器路径
- 环境变量注入 HuggingFace 缓存路径（`HF_HOME`、`HUGGINGFACE_HUB_CACHE` 等）与镜像端点 `HF_ENDPOINT=https://hf-mirror.com`

---

### 步骤 3a：六视角渲染采样 (`step3_sample_views.py`)

**目的：** 对原始 Mesh 从六个正交方向做离屏渲染，为后续风格化提供多视角数据。

**实现细节：**

1. **六视角相机计算 — `build_six_view_spec`**（`lib/camera_views.py`）：
   - 计算 Mesh 包围盒中心 `center` 和边界球半径 `radius`
   - 定义六个标准方向：front (+X)、back (-X)、left (-Y)、right (+Y)、top (+Z)、bottom (-Z)
   - 每个方向通过二分搜索确定最佳相机距离（`_fit_distance`）：
     - 确保所有顶点都在视锥体内
     - 使用 NDC 空间裁剪（|x_ndc| ≤ 1.0, |y_ndc| ≤ 1.0）
     - 添加 5% padding 避免边缘裁剪
   - 使用 `look_at` 函数构造 4×4 相机外参矩阵（right-up-forward-eye 标准布局）

2. **离屏渲染 — `render_offscreen_views`**（`lib/offscreen_renderer.py`）：
   - 渲染引擎：**Pyrender** + **EGL**（`PYOPENGL_PLATFORM=egl`）
   - 材质：所有顶点赋予中性灰色 `(235, 235, 235, 255)`
   - 场景配置：
     - 黑色透明背景 `[0, 0, 0, 0]`
     - 环境光 `[0.18, 0.18, 0.18]`
     - 两盏方向光（强度 2.2），相对相机旋转以提供稳定的明暗
   - 输出：RGBA 颜色缓冲 (`np.uint8`) + 深度缓冲 (`np.float32`)

3. **辅助图生成 — `_derive_secondary_maps`**（`lib/view_sampling.py`）：
   - **Depth Preview**：将深度值归一化到 [0, 1]，做灰度可视化
   - **Normal Map**：从深度图估算法线：
     - 对有效深度区域做 EDT（Euclidean Distance Transform）填充空洞
     - 计算深度梯度 `np.gradient` 得到表面法线
     - 编码为 RGB（法线 × 0.5 + 0.5）
   - **Control Map**：法线（65% 权重）+ 深度（35% 权重）混合，生成供 ControlNet 使用的控制图

4. **输出文件**（每个视角目录下）：
   - `rgb.png` — RGBA 渲染图
   - `depth.npy` — 原始深度浮点数组
   - `depth.png` — 深度可视化
   - `normal.png` — 法线图
   - `mask.png` — 前景遮罩（L 模式）
   - `control.png` — ControlNet 控制图
   - `camera.json` — 相机参数（位姿矩阵 + FOV）
   - `views/manifest.json` — 全局索引

---

### 步骤 3b：InstantStyle 风格化 (`step3_instantstyle.py`)

**目的：** 对六个视角的渲染结果执行风格迁移，使每个视角都呈现参考图的风格。

**实现细节：**

本步骤通过子进程调用 `workers/instantstyle_worker.py`。Worker 使用 **SDXL ControlNet Img2Img + IP-Adapter** 实现可控风格迁移。

1. **批量任务编排**：
   - 从 `views/manifest.json` 读取六个视角列表
   - 为每个视角生成一个 job：包含 RGB 路径、Control 路径、风格图路径、prompt、种子等参数
   - 写入 `stylize/worker_jobs.json`，一次性提交给 worker

2. **InstantStyle Worker 核心流程**：

   **a. 模型加载：**
   - **ControlNet**：`diffusers/controlnet-canny-sdxl-1.0`（fp16）
   - **SDXL Base**：`stabilityai/stable-diffusion-xl-base-1.0`（fp16，禁用 watermarker）
   - **IP-Adapter**：`IPAdapterXL`（加载 `ip-adapter_sdxl.bin` + CLIP `image_encoder`）
   - IP-Adapter 的目标层：`["up_blocks.0.attentions.1"]`（仅注入风格特征，不注入布局特征）
   - 启用 VAE Tiling 以节省显存

   **b. 图像准备：**
   - **Base Image（img2img 输入）**：将 RGB 渲染图的透明背景替换为灰色 `(235, 235, 235)`，转为 RGB
   - **Control Image**：将 Control 图的透明背景替换为黑色 `(0, 0, 0)`，转为 RGB
   - **Style Image**：直接加载为 RGB，在批量模式下缓存复用

   **c. 风格迁移（IP-Adapter 机制）：**
   - IP-Adapter 通过额外的交叉注意力层将风格图像特征注入 SDXL
   - 风格图像 → CLIP Image Encoder → Image Projection → Cross-Attention Key/Value
   - 在 `up_blocks.0.attentions.1` 位置的 Transformer 块中注入风格特征
   - ControlNet（Canny-based）通过残差连接约束结构保持

   **d. 关键参数：**
   - `strength`(0.72)：控制 img2img 的噪声强度，越高风格化越强但结构保持越弱
   - `style_scale`(1.8)：IP-Adapter 风格特征注入权重
   - `guidance_scale`(6.5)：无分类器引导强度
   - `num_inference_steps`(35)：去噪步数
   - `controlnet_conditioning_scale`(0.45)：ControlNet 结构控制强度

   **e. 后处理：**
   - 将风格化输出 resize 到 Control 图尺寸
   - 使用原始 mask 恢复 alpha 通道
   - 保存 `stylized.png`

3. **输出：** 六个视角的 `stylize/<view>/stylized.png`

---

### 步骤 4：UV 回贴 (`step4_retexture.py`)

**目的：** 将六视角风格化结果融合并烘回模型的 UV 纹理空间，生成风格化纹理贴图。

**实现细节：**

这是整个管线中最核心的技术步骤，实现在 `lib/reprojection.py` 的 `bake_texture` 函数中。

1. **数据加载**：
   - 加载原始 Mesh（`mesh_raw.glb`），提取顶点、面片、UV 坐标
   - 从 `views/manifest.json` 和 `stylize/manifest.json` 加载六个视角的相机位姿、深度图和风格化图像

2. **视角采样结构 — `ViewSample`**：
   - 每个视角封装：相机外参 `pose`(4×4)、相机内参 `intrinsic`(3×3)、深度图 `depth`、风格化图 `stylized`
   - 内参矩阵从 FOV 和图像尺寸推导：`focal = 0.5 * max(width-1, 0) / tan(fovy/2)`

3. **逐 texel 烘焙算法**：

   ```
   For each face (triangle):
       计算面法线
       For each texel in triangle (UV 光栅化):
           1. 用重心坐标插值三角形的 3D 位置
           2. 从 base_texture 取 fallback 颜色
           3. For each of 6 views:
               a. project_point_to_view: 3D点 → 世界→相机→像素坐标
               b. 深度一致性检查: |depth_map - projected_depth| < tolerance
               c. 可见性检查: 面法线与视线方向夹角 (facing > 0)
               d. 像素 alpha > 0
               e. 计算权重: weight = facing^4
           4. blend_samples: 按权重混合所有可见视角的像素颜色
           5. 写入 baked texture
   ```

4. **关键技术细节**：

   **a. 三角形光栅化**（`_triangle_pixels`）：
   - 将 UV 三角形映射到纹理像素空间
   - 对三角形包围盒内的每个像素，用 `np.linalg.solve` 求解重心坐标
   - 重心坐标全部 ≥ -1e-5 则该像素在三角形内

   **b. 3D → 2D 投影**（`project_point_to_view`）：
   - `world_to_camera = inv(pose)`
   - `camera_point = world_to_camera @ [x, y, z, 1]`
   - 深度 = `-camera_point.z`（OpenGL 坐标系，相机看向 -Z）
   - 像素坐标 = `intrinsic @ camera_point`，再做透视除法

   **c. 深度一致性检查**：
   - `tolerance = max(depth_epsilon, 0.02 * projected_depth)`
   - 确保投影点的深度与深度图记录值一致（防止被前景遮挡的 texel 采样到前景颜色）

   **d. 面朝向加权**：
   - `facing = clamp(dot(normal, view_dir), 0, 1)`
   - `weight = facing^4`（四次方衰减使斜向视角贡献急剧下降）
   - 正面朝向的视角获得最高权重，侧面视角几乎不贡献

   **e. 多视角混合**（`blend_samples`）：
   - 加权平均所有可见视角的颜色：`rgb = Σ(weight_i × color_i) / Σ(weight_i)`
   - 若所有视角都不可见，回退到 SF3D 原始纹理
   - 若原始纹理太暗（`sum(rgb) < 18`），使用中性灰 `(184,184,184)` 回退

   **f. 纹理空洞填充**（`_fill_texture_gaps`）：
   - 第一遍：用最近邻（EDT）填充已涂色的空洞到整个 UV 区域
   - 第二遍：用最近邻填充剩余空洞
   - Alpha 通道全部设为 255

5. **降级模式**：
   - 若无多视角 manifest（仅单视角风格化结果），则退化为 `bake_visible_texels`（`lib/mesh_utils.py`）：
     - 按物体包围盒做简单正交投影
     - 法线 X 分量 > 0 的面可见
     - 直接拷贝 stylized 像素到对应 texel

6. **输出**：
   - `retexture/mesh_stylized.glb` — 最终风格化模型
   - `retexture/texture_preview.png` — 新烘焙的纹理贴图预览

---

### 步骤 5：生成 3D 预览页 (`step5_build_viewer.py`)

**目的：** 生成一个离线 HTML 页面，支持旋转查看风格化模型并对比六个视角。

**实现细节：**

- 使用 Google `<model-viewer>` Web Component 做 3D 模型交互展示
- 页面包含：
  - 左侧面板：输入图、风格参考图、纹理预览
  - 六视角网格：每格显示原始 RGB 渲染 vs 风格化结果对比
  - 右侧：3D 模型查看器（自动旋转、可拖拽旋转）
- 复制 `lib/assets/model-viewer.min.js` 到 `viewer/` 目录，实现完全离线可用

---

## 四、前端应用 (`app.py`)

### Gradio UI 架构

- **布局**：左右分栏（5:7 比例），左栏为控制面板（上传 + 参数），右栏为 3D 模型预览窗口
- **主题**：自定义 CSS（东方美学配色，暖色调渐变背景，毛玻璃卡片效果）

### 七步流水线

`run_generation` 函数是核心事件处理，采用 Generator（`yield`）模式实现实时进度推送：

| 步骤 | 显示 | 实际操作 |
|------|------|----------|
| 1 | 接收任务 | 创建运行目录、写入配置、生成 Task ID |
| 2 | 图像预处理 | step1: rembg 去背景 |
| 3 | SF3D重建 | step2: 子进程调用 SF3D worker |
| 4 | 多视角采样 | step3a: 六视角离屏渲染 |
| 5 | 六视角风格化 | step3b: 子进程批量调用 InstantStyle |
| 6 | UV回贴 | step4: 多视角纹理烘焙 + step5: 生成预览页 |
| 7 | 返回模型 | 输出最终 GLB 文件路径供 3D viewer 展示 |

### 配置参数

所有 Pipeline 参数可通过环境变量覆盖（优先级高于代码默认值）：

| 参数 | 默认值 | 环境变量 |
|------|--------|----------|
| foreground_ratio | 0.85 | `APP_FOREGROUND_RATIO` |
| texture_resolution | 1024 | `APP_TEXTURE_RESOLUTION` |
| view_resolution | 512 | `APP_VIEW_RESOLUTION` |
| camera_distance | 1.8 | `APP_CAMERA_DISTANCE` |
| camera_fovy_deg | 40.0 | `APP_CAMERA_FOVY_DEG` |
| seed | 42 | `APP_SEED` |
| strength | 0.72 | `APP_STRENGTH` |
| style_scale | 1.8 | `APP_STYLE_SCALE` |
| guidance_scale | 6.5 | `APP_GUIDANCE_SCALE` |
| num_inference_steps | 35 | `APP_NUM_INFERENCE_STEPS` |
| controlnet_conditioning_scale | 0.45 | `APP_CONTROLNET_CONDITIONING_SCALE` |

### 任务管理

- **Task ID 格式**：`TASK-MMDD-HHMMSS-XXXX`（时间戳 + 4 位随机 hex）
- **历史记录**：内存中维护列表（Session 级别），最新任务置顶
- **查询功能**：按 Task ID 检索记录状态

---

## 五、关键技术决策

### 1. 子进程隔离

SF3D 和 InstantStyle 各自通过独立子进程运行（`subprocess.run`），原因：
- 两个模型的 CUDA 依赖不同，同一进程加载会导致显存溢出
- 可以通过 `--sf3d-python` / `--instantstyle-python` 指向不同 conda 环境
- 子进程崩溃不会导致整个 Pipeline 崩溃，便于错误定位

### 2. 多视角策略

选择六视角（前后左右上下）而非更多视角的原因：
- 六个标准方向覆盖模型的所有可见面
- 每个视角的渲染和风格化耗时较长，六视角是效果与速度的平衡点
- 配合 `facing^4` 权重函数，相邻视角可以平滑过渡

### 3. ControlNet + IP-Adapter 组合

- **ControlNet**（Canny 边缘）：保持渲染视角的几何结构，防止风格化后物体轮廓变形
- **IP-Adapter**（Style-only block）：仅注入风格特征到 SDXL 的 `up_blocks.0.attentions.1`，不注入布局特征，避免改变物体形态

### 4. 深度一致性验证

UV 回贴时的深度一致性检查是关键：
- 防止被自遮挡 texel 采样到错误视角的像素
- `tolerance = max(0.02, 0.02 × depth)` 适应不同尺度的模型
- 不通过深度检查的视角不参与该 texel 的颜色混合

### 5. 断点续跑

`pipeline_runner.py` 支持 `--resume-from` 和 `--skip-existing`：
- 每个步骤完成后写入 meta JSON 文件
- 通过检查 meta 文件是否存在来决定是否跳过
- 支持从任意步骤恢复执行

---

## 六、运行方式

### 启动 Gradio 前端

```bash
cd /root/autodl-tmp/src
python app.py
```

### 命令行全流程

```bash
cd /root/autodl-tmp/src/stylized-3d-pipeline
python scripts/run_all.py \
  --input /path/to/photo.jpg \
  --style-image /path/to/style.jpg \
  --prompt "描述文本" \
  --run-name my-run \
  --runs-root runs \
  --sf3d-python /root/autodl-tmp/envs/sf3d/bin/python \
  --instantstyle-python /root/autodl-tmp/envs/instantstyle/bin/python
```

### 单步执行（调试用）

```bash
python scripts/step1_preprocess.py --input /path/to/photo.jpg --run-dir runs/my-run
python scripts/step2_sf3d.py --run-dir runs/my-run --sf3d-python /root/autodl-tmp/envs/sf3d/bin/python
python scripts/step3_sample_views.py --run-dir runs/my-run
python scripts/step3_instantstyle.py --run-dir runs/my-run --style-image /path/to/style.jpg --prompt "..." --instantstyle-python /root/autodl-tmp/envs/instantstyle/bin/python
python scripts/step4_retexture.py --run-dir runs/my-run
python scripts/step5_build_viewer.py --run-dir runs/my-run
```
