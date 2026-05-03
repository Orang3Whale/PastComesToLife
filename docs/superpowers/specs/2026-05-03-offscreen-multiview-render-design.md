# 多视角离屏直接渲染设计

## 1. 背景

当前多视角阶段曾经通过自研光栅化逻辑生成 `views/*/rgb.png`：读取 `mesh_raw.glb` 的 `baseColorTexture`，按 UV 对每个屏幕像素做纹理采样，再写出 RGB、depth、mask、normal 和 control。

这个路径的问题是它把 SF3D 输出的稀疏 atlas 直接暴露出来。真实样例中，原始纹理 atlas 大量区域为黑色；viewer 里旋转模型时视觉上还能接受，但离线导出的 `rgb.png` 会出现大面积黑块。原因是 viewer 渲染的是三维 mesh 表面，而旧导出路径本质是 texture-space 硬采样，二者不是同一条渲染路径。

本设计将 `sample_views` 改为直接离屏渲染 mesh，目标是让多视角图像和三维可视化的白模/中性材质外观一致，不再依赖 SF3D 原始纹理质量。

## 2. 目标

- `views/*/rgb.png` 来自离屏直接渲染的 mesh 主体，而不是 UV atlas 采样。
- 每个视角继续使用已有的自适应相机距离，保证主体完整进入画面。
- `rgb.png`、`mask.png`、`depth.npy`、`normal.png` 和 `control.png` 使用同一相机和同一几何结果生成，避免坐标不一致。
- 输出图背景保持透明，主体外区域 alpha 为 0。
- 不新增模型下载，不依赖生成式 inpainting。
- 保持 `step3_sample_views.py` 的外部 CLI 参数兼容。

## 3. 非目标

- 不在本次设计中修复 SF3D 原始 UV atlas 本身。
- 不在本次设计中改变 `InstantStyle` 的模型加载方式或风格化参数。
- 不在本次设计中重写 `step4_retexture` 的多视角回烘算法。
- 不追求和浏览器 `model-viewer` 像素级一致；目标是渲染语义一致，即直接渲染几何主体，而不是采样稀疏纹理。

## 4. 推荐方案

使用 `pyrender.OffscreenRenderer` 作为 `sample_views` 的主渲染后端。

渲染时不使用 SF3D 的 `baseColorTexture` 作为颜色来源，而是将 mesh 转成中性材质：

- 主体使用浅灰/白色 PBR 或 unlit 材质。
- 场景加入稳定的环境光和方向光，避免背光视角全黑。
- 相机使用已有 `CameraView.pose` 和 `fovy_deg`。
- 背景透明，由 depth 是否命中主体派生 mask。

这样 `rgb.png` 成为“几何主体的干净视角渲染”，不再继承 SF3D 纹理黑洞。`control.png` 继续由 depth/normal/mask 派生，作为风格化结构约束。

## 5. 数据流

新的 `sample_views` 数据流：

1. `step3_sample_views.py` 加载 `sf3d/mesh_raw.glb`。
2. `camera_views.build_six_view_spec()` 计算六个 canonical 视角和自适应距离。
3. `view_sampling.render_view_assets()` 调用离屏渲染器。
4. 每个视角输出：
   - `rgb.png`：离屏渲染的中性材质主体图，背景透明。
   - `depth.npy`：离屏 depth buffer，背景为 0。
   - `depth.png`：depth 可视化。
   - `normal.png`：从几何或 depth 派生的 normal/control 参考图，背景透明。
   - `mask.png`：由 depth 命中区域生成。
   - `control.png`：由 normal/depth/mask 组合生成，背景透明。
   - `camera.json`：当前视角相机信息。
5. `views/manifest.json` 增加渲染模式字段，例如 `"render_mode": "mesh_offscreen"`。

## 6. 组件边界

### 6.1 `lib/view_sampling.py`

保留公共入口：

- `render_view_assets(mesh, views, resolution)`
- `write_view_assets(view_root, assets)`

内部职责调整：

- 删除或停用 RGB 的 UV 采样路径。
- 新增 `pyrender` 后端封装，用于渲染 color/depth。
- 将 mask 统一由 depth 命中生成。
- 将 control 继续限定在 mask 内，背景 alpha 为 0。

### 6.2 `lib/camera_views.py`

继续作为相机规格来源，不负责渲染。

保留当前自适应距离逻辑，避免每个视角截断主体。

### 6.3 `scripts/step3_sample_views.py`

继续作为步骤编排入口。

外部参数保持不变：

- `--run-dir`
- `--view-resolution`
- `--camera-distance`
- `--camera-fovy-deg`

输出 manifest 需要记录：

- `render_mode`
- `view_resolution`
- `camera_distance`
- `camera_fovy_deg`
- 每个视角的资产路径

### 6.4 `step4_retexture`

本设计不改变 `step4_retexture` 的输入接口。它继续读取：

- `views/manifest.json`
- `stylize/manifest.json`
- `views/*/depth.npy`
- `stylize/*/stylized.png`

由于 stylized views 将基于更干净的 control 生成，回烘质量预期提升。

## 7. 错误处理

- 如果 `pyrender.OffscreenRenderer` 初始化失败，明确报错说明当前环境缺少可用 EGL/OSMesa/OpenGL 后端。
- 不静默回退到 UV 采样路径，避免重新引入黑块问题。
- 如果某个视角渲染 mask 为空，抛出错误并指明视角名称。
- 如果 depth 全为 0，视为渲染失败。

## 8. 测试策略

### 8.1 单元测试

- 使用带黑色纹理的简单 mesh，验证 `rgb.png` 主体区域不继承原始黑纹理。
- 验证 `mask.png` 与 `rgb.png` alpha 一致。
- 验证 `control.png` 背景 RGB 和 alpha 都为 0。
- 验证 manifest 记录 `render_mode: mesh_offscreen`。

### 8.2 相机测试

保留已有 elongated mesh fit 测试，确保离屏渲染前的相机仍能覆盖完整主体。

### 8.3 真实样例验证

在 `runs/real-chair-starry-multiview-v2` 上重跑：

1. `step3_sample_views.py`
2. `step3_instantstyle.py`
3. `step4_retexture.py`
4. `step5_build_viewer.py`

记录：

- 每个 `views/*/rgb.png` 的主体黑像素比例。
- 每个 `stylize/*/stylized.png` 的 alpha 是否等于对应 mask。
- `retexture/texture_preview.png` 的 UV 区域黑像素比例。

## 9. 风险与缓解

### 9.1 服务器无可用 OpenGL 后端

风险：`pyrender` 离屏渲染依赖 EGL/OSMesa/OpenGL，环境可能不可用。

缓解：

- 实现时先做最小 smoke test。
- 错误信息明确提示可设置 `PYOPENGL_PLATFORM=egl` 或安装 OSMesa。
- 不自动下载额外依赖到系统盘。

### 9.2 RGB 外观与浏览器 viewer 不完全一致

风险：`pyrender` 与 `model-viewer` 的光照、tone mapping、材质实现不同。

缓解：

- 本设计只要求直接渲染语义一致，不要求像素级一致。
- 使用中性材质和稳定光照，降低材质差异影响。

### 9.3 法线图质量不足

风险：如果 normal 继续由 depth gradient 近似生成，边界和细节可能不如真实几何 normal。

缓解：

- 第一版保持当前 depth-derived normal，保证改动集中。
- 后续可增加独立 normal pass，用几何 normal 渲染到颜色 buffer。

## 10. 成功标准

本设计实现后应满足：

- 真实样例的 `views/*/rgb.png` 主体不再出现大面积黑块。
- `views/*/rgb.png` 与 viewer 中看到的白模/中性主体感知一致。
- 背景透明严格由 mask 控制。
- 现有多视角风格化和回烘接口无需重新设计。
- 聚焦测试和真实 run 都能完成。
