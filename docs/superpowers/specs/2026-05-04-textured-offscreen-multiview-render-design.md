# Step3 双输入风格化设计

## 1. 背景

当前离屏多视角已经能直接渲染 textured mesh，`camera_views.build_six_view_spec()` 也会按主体包围盒自动把相机拉远到能装下整个物体。问题不在视角，而在 step3 的输入方式。

现在 `step3_instantstyle.py` 只把 `control.png` 送进 worker，worker 再对它做 Canny。这样纹理信息没有进入 SD 的主输入通道，`rgb.png` 只能停留在中间产物，导致风格化仍然容易碎、乱、视角间不一致。

新的路线要把责任拆开：

- `rgb.png` 负责提供 Stable Diffusion 的底图。
- `control.png` 只负责物理骨架和大结构，不承载纹理，也不再做 Canny。

## 2. 目标

- step3 支持同时读取 `rgb.png` 和 `control.png`。
- `rgb.png` 作为 SDXL img2img 的 init image，保留原始纹理和颜色分布。
- `control.png` 只表达几何骨架、轮廓和体块，不依赖纹理边缘。
- worker 不再对 `control.png` 做 Canny。
- 默认重绘幅度保持中低档，建议 `strength=0.45`，可配置到 `0.35-0.55`。
- 继续保留背景透明和 mask 约束。
- 不新增模型下载，不重做 step4 回烘逻辑。

## 3. 非目标

- 不在本阶段重写 `step4_retexture`。
- 不在本阶段更换成新的控制模型家族，先保持当前缓存可用的 SDXL ControlNet + IP-Adapter 路线。
- 不改变 step3 的顶层 CLI 入口，只扩展内部 view manifest 和 worker 参数。

## 4. 推荐方案

采用“`rgb.png` 作为 img2img 底图 + `control.png` 作为几何控制图”的双输入方案。

### 4.1 视图阶段

`lib/view_sampling.py` 继续写出 `rgb.png`、`depth.png`、`normal.png`、`mask.png` 和 `control.png`，但职责要重新划分：

- `rgb.png` 保存修补后的 textured render。
- `control.png` 只从 `depth`、`normal`、`mask` 生成结构图，不再混入纹理颜色。
- `control.png` 的目标是锁定主体姿态、外轮廓、体块转折和局部几何，不是复原材质。

推荐的 `control.png` 组成方式：

- 前景 mask 提供主体轮廓。
- depth 提供体块层次。
- normal 提供朝向变化。
- 不使用 `rgb.png` 参与控制图构造。

### 4.2 step3 输入

`views/manifest.json` 需要新增 `rgb_path`，每个视角至少包含：

- `rgb_path`
- `control_path`
- `mask_path`

`scripts/step3_instantstyle.py` 读取这两个输入后，构造 worker 命令时同时传入 `--rgb-image` 和 `--control-image`。

### 4.3 worker 路线

`scripts/workers/instantstyle_worker.py` 改为使用 `StableDiffusionXLControlNetImg2ImgPipeline`，而不是纯 ControlNet txt2img。

运行时输入分成三层：

- `pil_image` 仍然是 style reference。
- `image=rgb.png` 作为 init image。
- `control_image=control.png` 作为几何约束。

核心参数建议：

- `strength=0.45`
- `controlnet_conditioning_scale=0.7`
- `num_inference_steps=30`

`control.png` 直接送进 pipeline，不再经过 `cv2.Canny`。

### 4.4 背景处理

因为 img2img 会把输入图当作可见底图，`rgb.png` 在进入 pipeline 前应当先做一次透明背景处理：

- 前景按原纹理保留。
- 透明区域使用中性背景填充，避免黑底污染扩散到主体边缘。
- `control.png` 在进入 pipeline 前也要用 alpha 变成纯结构的 RGB 图，背景统一压成黑底或近黑底，避免透明通道在不同库里被不一致解释。

输出后仍按当前逻辑用 `mask.png` 恢复 alpha。

## 5. 数据流

1. `step3_sample_views.py` 用自适应相机导出六视角。
2. `view_sampling.py` 写出 textured `rgb.png` 和 geometry-only `control.png`。
3. `views/manifest.json` 记录 `rgb_path`、`control_path`、`mask_path`。
4. `step3_instantstyle.py` 读取两个图像路径并启动 worker。
5. `instantstyle_worker.py` 用 `rgb.png` 做 img2img 底图，用 `control.png` 锁骨架。
6. 输出的 `stylized.png` 继续按 mask 恢复透明背景。
7. `step4_retexture.py` 保持不变，继续读取 stylized views。

## 6. 文件职责

### 6.1 `lib/view_sampling.py`

- 保持 textured RGB 输出。
- 新增或调整 `control.png` 构造逻辑，使其只依赖几何信息。
- 不再让 `control.png` 参与纹理表达。

### 6.2 `scripts/step3_instantstyle.py`

- 读取 `rgb_path` 和 `control_path`。
- 构造 worker 命令时同时传入 `--rgb-image` 与 `--control-image`。
- 保持外部 CLI 不变。

### 6.3 `scripts/workers/instantstyle_worker.py`

- 将底模切到 `StableDiffusionXLControlNetImg2ImgPipeline`。
- 去掉 `build_canny_control_map()` 这条路径。
- 让 `rgb.png` 成为真正的 init image。

### 6.4 `views/manifest.json`

- 新增 `rgb_path`。
- 保留 `control_path` 和 `mask_path`。
- 可选记录 `strength`、`control_scale`，方便回看 run 参数。

## 7. 测试策略

### 7.1 step3 命令测试

- `build_instantstyle_command()` 必须同时包含 `--rgb-image` 和 `--control-image`。
- `run_step()` 读取 manifest 时必须校验 `rgb_path`、`control_path`、`mask_path` 都存在。

### 7.2 worker 行为测试

- fake pipeline 要能收到 `image=rgb` 和 `control_image=control`。
- fake pipeline 要能收到 `strength`。
- 不能再出现 `cv2.Canny()` 调用。

### 7.3 control 图测试

- `control.png` 不得依赖 `rgb.png` 的颜色纹理。
- `control.png` 背景仍应保持透明或前景掩码一致。
- `control.png` 的高响应区域应集中在主体轮廓和结构转折。

### 7.4 集成回归

在现有 run 上重跑：

1. `step3_sample_views.py`
2. `step3_instantstyle.py`
3. `step4_retexture.py`
4. `step5_build_viewer.py`

关注：

- `views/*/rgb.png` 是否保留完整纹理。
- `views/*/control.png` 是否只表达骨架。
- `stylize/*/stylized.png` 是否比纯 control Canny 方案更完整、更少碎块。

## 8. 风险与缓解

### 8.1 img2img 强度过高

风险：底图纹理被重新绘制掉。

缓解：默认把 `strength` 压在 0.45 左右，并保留可调范围。

### 8.2 control 图太弱

风险：几何约束不足，视角间变形。

缓解：`control.png` 只锁骨架，但它必须由轮廓、depth、normal 共同组成，而不是单纯灰图。

### 8.3 透明背景引入黑边

风险：PIL / diffusers 在读 RGBA 时把透明区处理成黑底。

缓解：进入 pipeline 前先做中性背景填充，输出后再恢复 alpha。

### 8.4 现有 ControlNet 与几何图不完全匹配

风险：当前缓存的是 canny ControlNet，几何图效果可能不是最优。

缓解：先把输入链路改对，避免 Canny 绑死纹理；如果后续仍然不稳，再考虑切换到 depth/normal ControlNet。

## 9. 成功标准

- `step3` 同时读取 `rgb.png` 和 `control.png`。
- `control.png` 不再承担纹理职责，也不再做 Canny。
- `rgb.png` 作为 img2img 底图真正进入 Stable Diffusion。
- 生成结果相较原方案更完整、更少黑块和碎片化。
- 现有相机自适应和 step4 流程不被破坏。
