# 带纹理离屏多视角渲染设计

## 1. 背景

当前 `sample_views` 已经从旧的 Python UV 采样路径切换到 `pyrender` 离屏渲染，但默认渲染前会调用 `build_neutral_render_mesh()`，把 `mesh.visual` 替换成统一浅灰顶点色。因此 `views/*/rgb.png` 是干净白模，而不是 SF3D mesh 在可视化器中看到的带纹理外观。

这解决了旧路径的大面积黑块问题，但带来新的质量问题：白模视角缺少原始纹理、材质和颜色边界，后续风格化每个视角会更依赖生成模型自行补全，视角间外观更容易不一致。

另一个关键事实是：`step3_instantstyle.py` 当前不读取 `views/*/rgb.png`。它读取 `views/*/control.png`，worker 再对 `control.png` 做 Canny，作为 ControlNet 条件。因此只把 `rgb.png` 改成带纹理还不够；如果 `control.png` 仍然只由 depth/normal 组成，InstantStyle 仍然看不到纹理边缘。

## 2. 目标

- 默认多视角 RGB 改为直接离屏渲染原始 textured mesh，而不是中性白模。
- 背景仍严格透明，主体外 alpha 为 0。
- 主体内部的 atlas 黑洞只做局部修补，不整视角回退白模。
- `control.png` 改为携带修补后的纹理/颜色边缘，让 InstantStyle 的 Canny 条件能看到原始外观结构。
- 保留 depth/normal 的几何约束，避免纯纹理边缘导致结构漂移。
- 输出 manifest 明确记录新的渲染模式和每个视角的修补统计。
- 不新增模型下载，不使用生成式 inpainting。

## 3. 非目标

- 不在本阶段重写 `step4_retexture` 的 UV 回烘算法。
- 不保证和浏览器 viewer 像素级一致；目标是让离屏输入和 viewer 同源，即使用同一个 textured mesh/material。
- 不解决所有合法黑色材质的语义判断问题；黑洞修补采用保守阈值和统计输出，避免静默大范围改色。
- 不改变 `step3_instantstyle.py` 的外部 CLI。

## 4. 推荐方案

使用“textured offscreen render + foreground-only black-hole repair + texture-aware control”的单一路线。

### 4.1 textured offscreen render

`lib/offscreen_renderer.py` 的主路径不再调用 `build_neutral_render_mesh()`。渲染时直接将原始 `trimesh.Trimesh` 传给 `pyrender.Mesh.from_trimesh(mesh, smooth=False)`，保留 `TextureVisuals`、`baseColorTexture`、UV 和材质颜色。

`build_neutral_render_mesh()` 可以保留为测试或诊断 helper，但不能再作为默认 `render_offscreen_view()` 的颜色来源。

### 4.2 黑洞检测

离屏渲染后，继续用 depth buffer 生成 foreground mask：

- `foreground = depth > 0`
- 背景 alpha 始终为 0。
- 黑洞候选只允许出现在 foreground 内。

黑洞候选规则：

- 像素 alpha 在 foreground 内。
- RGB 亮度低于固定低阈值，例如 `luma < 18`。
- 候选像素周围存在非黑 foreground 邻域，说明这是局部洞而不是整块真实黑材质。

每个视角记录：

- `foreground_pixel_count`
- `black_hole_pixel_count_before`
- `black_hole_ratio_before`
- `black_hole_pixel_count_after`
- `black_hole_ratio_after`

如果某个视角 foreground 内黑洞比例异常高，仍然保留 textured render 和局部修补结果，但在 manifest 中记录高比例，不自动整视角白模回退。

### 4.3 局部修补

修补只发生在 foreground 内的黑洞候选区域：

- 背景透明像素不参与修补。
- 有效 foreground 非黑像素作为颜色源。
- 使用最近有效 foreground 像素填充黑洞，优先采用 `scipy.ndimage.distance_transform_edt` 的 nearest valid fill。
- 如果没有有效颜色源，保持原像素并记录修补失败统计。

这不是生成式 inpainting，只是局部颜色传播，目标是消除 atlas 空洞对风格化条件的破坏。

### 4.4 texture-aware control

当前 `control.png` 由 normal/depth 组合生成，然后 worker 对它做 Canny。新设计改为 texture-aware control：

- 主输入：修补后的 textured RGB。
- 辅助输入：depth preview 和 normal RGB。
- 输出仍为 `control.png`，alpha 仍为 foreground mask。

推荐控制图组合：

- RGB 主导，用于保留原始纹理和颜色边界。
- normal/depth 轻量混入，用于保留几何轮廓和大结构。
- 背景 RGB 和 alpha 都为 0。

这样不需要改 `step3_instantstyle.py` 的 CLI 或 worker 输入路径；worker 继续对 `control.png` 做 Canny，但 Canny 的边缘来源将包含纹理、材质和几何。

## 5. 数据流

1. `step3_sample_views.py` 加载 `sf3d/mesh_raw.glb`。
2. `camera_views.build_six_view_spec()` 计算六个自适应视角。
3. `view_sampling.render_view_assets()` 调用 textured offscreen renderer。
4. `offscreen_renderer.render_offscreen_view()` 输出：
   - raw textured color buffer
   - depth buffer
   - alpha mask
   - black-hole repair stats
5. `view_sampling._derive_secondary_maps()` 使用修补后的 textured RGB 派生：
   - `rgb.png`
   - `depth.npy`
   - `depth.png`
   - `normal.png`
   - `mask.png`
   - texture-aware `control.png`
6. `views/manifest.json` 写入：
   - `render_mode: "mesh_textured_offscreen"`
   - 每个视角的资产路径
   - 每个视角的 black-hole 修补统计
7. `step3_instantstyle.py` 不改入口，继续读取 `control.png` 和 `mask.png`。
8. `step4_retexture.py` 不改入口，继续读取 stylized views 和 depth。

## 6. 文件职责

### 6.1 `lib/offscreen_renderer.py`

负责真实 mesh 离屏渲染和黑洞修补。

新增或调整职责：

- 默认渲染保留原始 textured material。
- 提供 foreground-only 黑洞检测。
- 提供 foreground-only 最近邻颜色修补。
- 返回修补统计。
- 保留明确的 EGL/OSMesa/OpenGL 错误信息。

### 6.2 `lib/view_sampling.py`

负责把 renderer 输出转换成现有资产 bundle。

新增或调整职责：

- 使用修补后的 textured RGB 作为 `rgb.png`。
- 用 textured RGB + normal/depth 生成 texture-aware `control.png`。
- 保持 `depth.png`、`normal.png`、`mask.png` 背景透明。

### 6.3 `scripts/step3_sample_views.py`

继续作为步骤编排入口。

新增或调整职责：

- manifest 的 `render_mode` 更新为 `mesh_textured_offscreen`。
- 将每个视角的修补统计写入 manifest。

### 6.4 `step3_instantstyle.py`

不改变外部接口。

它仍读取 `control.png`，但由于 `control.png` 本身变成 texture-aware，InstantStyle 的 Canny 条件会获得原始纹理/材质边缘。

## 7. 测试策略

### 7.1 renderer 单元测试

- 构造带纹理 mesh，验证默认 render path 不调用 `build_neutral_render_mesh()`。
- fake renderer 返回 foreground 内黑块，验证黑洞区域被局部修补。
- 验证透明背景不被修补，alpha 仍为 0。
- 验证修补统计包含 before/after 黑洞比例。
- 验证重复 view name 和 renderer cleanup 行为继续有效。

### 7.2 view sampling 单元测试

- fake offscreen 输出带彩色纹理边缘，验证 `rgb.png` 保留颜色。
- 验证 `control.png` 包含 textured RGB 信息，而不是只包含 normal/depth。
- 验证 `control.png`、`depth.png`、`normal.png` 背景 alpha 为 0。
- 验证 manifest 为 `mesh_textured_offscreen` 并包含修补统计。

### 7.3 instantstyle 回归测试

- 不改 CLI。
- 现有 worker 的 `build_canny_control_map()` 继续只接收 `control.png`。
- 增加测试证明 masked background 不会进入 Canny。

### 7.4 真实样例验证

在 `runs/real-chair-starry-multiview-v2` 上重跑：

1. `step3_sample_views.py`
2. `step3_instantstyle.py`
3. `step4_retexture.py`
4. `step5_build_viewer.py`

记录：

- `views/*/rgb.png` foreground 内 black ratio before/after repair。
- `views/*/control.png` foreground 内 edge density。
- `stylize/*/stylized.png` alpha 是否严格匹配 mask。
- `retexture/texture_preview.png` dark ratio。

## 8. 风险与缓解

### 8.1 合法黑色材质被误判为黑洞

风险：真实黑色纹理也可能被修补。

缓解：

- 黑洞检测只修补 foreground 内的局部低亮度区域。
- 输出每个视角的修补比例。
- 如果修补比例异常高，记录 warning stats，而不是静默白模回退。

### 8.2 texture-aware control 过度约束风格化

风险：Canny 捕获太多纹理细节，可能让风格化变碎。

缓解：

- `control.png` 采用 RGB 主导但混入 normal/depth，第一版避免多路 ControlNet。
- 如果真实 run 显示边缘过密，后续可调低 RGB 权重或先轻度平滑 RGB，再生成 control。

### 8.3 textured render 仍与 viewer 不完全一致

风险：`pyrender` 与浏览器 viewer 的 tone mapping、材质模型不同。

缓解：

- 本设计要求保留 mesh material 和 texture source，不要求像素级一致。
- 验收指标以黑洞比例、透明背景和风格化一致性为主。

## 9. 成功标准

- `views/*/rgb.png` 不再是白模，而是保留原始 mesh 纹理/颜色。
- `views/*/rgb.png` foreground 内黑块比例经修补后显著降低。
- `control.png` 不是纯几何 control，而是包含修补后的纹理/颜色边缘。
- 背景透明不回归。
- `step3_instantstyle.py` 和 `step4_retexture.py` 外部入口保持兼容。
- 聚焦测试通过，并且真实 run 完整跑通。
