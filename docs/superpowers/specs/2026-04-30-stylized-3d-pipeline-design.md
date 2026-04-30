# 单图三维化与风格化流水线设计

## 1. 背景与目标

本项目的短期目标是打通一条可复现的本地流水线：

`输入单张物体图 -> 生成 3D 模型 -> 输入单张风格参考图 -> 生成风格化主视角图 -> 将风格结果回贴到 3D 模型 -> 本地 HTML 可视化查看`

当前工作目录下已有两个上游项目：

- `stable-fast-3d/`：用于单图生成带 UV 和纹理的 3D 网格
- `InstantStyle/`：用于参考图驱动的图像风格化

第一版以路线跑通为最高优先级，强调：

- 可复现
- 可分步调试
- 模块边界清楚
- 结果可视化可直接查看

## 2. 已确认范围

### 2.1 输入范围

第一版仅支持：

- 单个前景主体物体
- 背景可自动抠图
- 输入视角以正面或 3/4 视角为主

暂不以复杂场景、多主体、人物全身、强遮挡作为第一版目标。

### 2.2 贴图目标

第一版采用“主视角优先”策略：

- 重点保证输入图可见面的风格化结果
- 背面和遮挡面允许保留 SF3D 原始纹理或由模型已有结果补全

不在第一版中解决多视角一致重贴图问题。

### 2.3 风格输入

- 风格参考图第一版只支持单张
- 后续允许扩展为多张，但不纳入本次设计实现范围

### 2.4 可视化目标

第一版网页只负责“结果查看 + 对照信息”，同页展示：

- 原始输入图
- 风格参考图
- 风格化结果图
- 最终 3D 模型

不在第一版中实现网页上传并触发后端推理。

### 2.5 Prompt 输入

- 第一版由用户手工提供简短 prompt
- 不做自动 caption 或自动 prompt 生成

### 2.6 验证方式

- 先使用公开样例验证各模块与总链路
- 再切换到用户的真实图片做复验

### 2.7 成功标准

第一版成功标准是“链路完整优先”：

- 从输入图到风格化 3D 模型的流程可以自动跑通
- 允许第一版效果一般
- 允许背面纹理保守处理
- 允许后续通过参数和策略继续优化效果

## 3. 方案对比与结论

评估过的三条路线：

### 方案 A：分环境 + 总控脚本 + 单视角回贴

特点：

- `SF3D` 与 `InstantStyle` 使用独立 Python 环境
- 新增一个轻量 orchestration 层串联各阶段
- 先做主视角风格图回贴，不做多视角完整重建
- 本地网页只做结果展示

优点：

- 依赖冲突风险最低
- 分步调试最直接
- 与“先跑通、可复现”的目标一致
- 后续扩展到 ComfyUI 或 Web 服务时迁移成本低

缺点：

- 第一版不是图形化工作流主导
- 前后两个环境需要单独维护

### 方案 B：ComfyUI 作为主入口

特点：

- 将主要流程包装成 ComfyUI 工作流

优点：

- 后续交互和展示直观

缺点：

- 第一版集成成本高
- 节点间排错成本更高
- 不利于先确认最小技术闭环

### 方案 C：强行统一单环境

特点：

- `SF3D`、`InstantStyle`、编排脚本尽量共用一个环境

优点：

- 表面上入口更简单

缺点：

- 版本冲突风险最高
- 后续维护和复现最差

### 结论

第一版采用方案 A。

理由：

- `SF3D` 与 `InstantStyle/SDXL` 依赖栈差异明显
- 用户要求同时支持“分步命令”和“一条总命令”
- 当前优先级是跑通链路而不是做 UI 封装

## 4. 总体架构

第一版架构分为三层：

### 4.1 上游模型层

- `stable-fast-3d/`
- `InstantStyle/`

原则：

- 不修改上游主逻辑
- 尽量通过脚本调用或子进程调用复用现有能力

### 4.2 编排层

新增自有工程目录 `stylized-3d-pipeline/`，负责：

- 输入输出路径组织
- 分步脚本
- 总控脚本
- 参数记录
- 中间产物管理
- 失败中断与断点续跑

### 4.3 结果查看层

生成静态 HTML 页面用于：

- 结果对照展示
- 加载最终 `GLB`
- 基础交互查看模型

## 5. 数据流设计

主链路按五步执行：

### 5.1 Step 1：输入预处理

输入：

- 普通输入图片

处理：

- 自动抠图
- 主体居中与裁切
- 生成标准 `RGBA PNG`
- 输出 mask 和元信息

输出：

- 供 `SF3D` 使用的标准输入图
- 供风格化步骤参考的内容基准图

### 5.2 Step 2：SF3D 重建

输入：

- 预处理后的 `RGBA PNG`

处理：

- 调用 `stable-fast-3d` 生成带 UV 和初始纹理的网格

输出：

- `mesh_raw.glb`
- SF3D 阶段元信息

说明：

根据仓库代码，`SF3D` 输出的是带纹理的 `GLB`，因此后续“先生成模型再局部重贴图”在技术上成立。

### 5.3 Step 3：InstantStyle 风格化

输入：

- 预处理后的内容图
- 风格参考图
- 用户 prompt

处理：

- 使用 `InstantStyle + 结构约束` 生成风格化主视角图
- 优先保持主体轮廓与主要形状稳定

输出：

- `stylized.png`
- 风格化参数与元信息

### 5.4 Step 4：主视角回贴

输入：

- `mesh_raw.glb`
- `stylized.png`

处理：

- 基于主视角将风格化结果投回前侧可见纹理区域
- 仅覆盖主视角相关 texel
- 背面与遮挡面保留 SF3D 原始纹理

输出：

- `mesh_stylized.glb`
- 纹理预览图
- 回贴元信息

### 5.5 Step 5：HTML 可视化

输入：

- 原始图
- 风格参考图
- 风格化结果图
- 最终 `GLB`

处理：

- 生成静态网页
- 同页展示 2D 对照图与 3D 模型查看器

输出：

- `viewer/index.html`

## 6. 目录结构

建议结构如下：

```text
/root/autodl-tmp/src/
├── InstantStyle/
├── stable-fast-3d/
└── stylized-3d-pipeline/
    ├── README.md
    ├── requirements.txt
    ├── configs/
    │   ├── pipeline.yaml
    │   └── prompts.yaml
    ├── scripts/
    │   ├── step1_preprocess.py
    │   ├── step2_sf3d.py
    │   ├── step3_instantstyle.py
    │   ├── step4_retexture.py
    │   ├── step5_build_viewer.py
    │   └── run_all.py
    ├── lib/
    │   ├── io_paths.py
    │   ├── subprocess_utils.py
    │   ├── image_utils.py
    │   ├── mesh_utils.py
    │   └── viewer_utils.py
    ├── web/
    │   ├── viewer.html
    │   └── assets/
    └── runs/
        └── <run_id>/
            ├── inputs/
            ├── preprocess/
            ├── sf3d/
            ├── stylize/
            ├── retexture/
            └── viewer/
```

原则：

- 上游仓库保持原样
- 自有逻辑集中到单独目录
- 所有运行产物按 `run_id` 隔离

## 7. 环境设计

第一版采用三个环境：

- `/root/autodl-tmp/envs/sf3d`
- `/root/autodl-tmp/envs/instantstyle`
- `/root/autodl-tmp/envs/pipeline`

用途：

- `sf3d`：运行 `stable-fast-3d` 及其依赖
- `instantstyle`：运行 `InstantStyle/SDXL/IP-Adapter` 相关依赖
- `pipeline`：运行编排脚本、文件组织、网页生成、基础检查

选择该方案的原因：

- 避免上游依赖互相污染
- 有利于后续独立升级
- 更容易定位失败来自哪个模块

## 8. 运行目录与产物约定

每次运行都创建独立目录：

`runs/<run_id>/`

建议 `run_id` 支持：

- 自动时间戳
- 用户显式指定 `run-name`

每个阶段固定产物：

### 8.1 inputs/

- `content.png`
- `style.png`
- `prompt.txt`

### 8.2 preprocess/

- `rgba.png`
- `mask.png`
- `meta.json`

### 8.3 sf3d/

- `input.png`
- `mesh_raw.glb`
- `sf3d_meta.json`

### 8.4 stylize/

- `stylized.png`
- `stylize_meta.json`

### 8.5 retexture/

- `mesh_stylized.glb`
- `texture_preview.png`
- `retexture_meta.json`

### 8.6 viewer/

- `index.html`
- `viewer_meta.json`

此外，总控入口记录：

- `run_config.json`

用于保留本次运行的全部显式参数。

## 9. 命令接口设计

### 9.1 分步命令

第一版每一步都能单独运行，便于排错。

示例：

```bash
python scripts/step1_preprocess.py \
  --input /abs/path/content.jpg \
  --run-dir runs/2026-04-30-demo-mug

python scripts/step2_sf3d.py \
  --run-dir runs/2026-04-30-demo-mug \
  --sf3d-python /root/autodl-tmp/envs/sf3d/bin/python

python scripts/step3_instantstyle.py \
  --run-dir runs/2026-04-30-demo-mug \
  --style-image /abs/path/style.jpg \
  --prompt "ceramic mug" \
  --instantstyle-python /root/autodl-tmp/envs/instantstyle/bin/python

python scripts/step4_retexture.py \
  --run-dir runs/2026-04-30-demo-mug

python scripts/step5_build_viewer.py \
  --run-dir runs/2026-04-30-demo-mug
```

### 9.2 总控命令

示例：

```bash
python scripts/run_all.py \
  --input /abs/path/content.jpg \
  --style-image /abs/path/style.jpg \
  --prompt "ceramic mug" \
  --run-name demo-mug
```

总控脚本职责仅限于：

- 创建 `run_dir`
- 保存 `run_config.json`
- 顺序调用五个步骤
- 失败即停并保留中间产物

## 10. 验证节点设计

第一版所有步骤都必须有可检查的通过条件。

### 10.1 preprocess

成功标准：

- `rgba.png` 已生成
- `mask.png` 已生成
- 主体未被明显裁坏

失败策略：

- 不进入 `SF3D`

### 10.2 sf3d

成功标准：

- `mesh_raw.glb` 已生成
- `GLB` 能被 `trimesh` 或 viewer 正常打开

失败策略：

- 保留预处理结果，便于重复调试

### 10.3 instantstyle

成功标准：

- `stylized.png` 已生成
- 主体轮廓大体保留
- 没有明显跑题

失败策略：

- 不覆盖已有结果
- 记录参数与报错

### 10.4 retexture

成功标准：

- `mesh_stylized.glb` 已生成
- 主视角可看出风格变化

允许：

- 背面仍偏原始纹理

### 10.5 viewer

成功标准：

- 页面可同时展示原图、风格图、风格化结果图、最终 `GLB`

## 11. 实用机制

第一版加入两个基础机制：

### 11.1 断点续跑

提供：

- `--resume-from <step_name>`

用于从中间步骤继续。

### 11.2 跳过已完成步骤

提供：

- `--skip-existing`

如果某一步产物已经存在且校验通过，则直接跳过。

作用：

- 调整风格参数时无需重新跑 `SF3D`
- 降低重复推理成本

## 12. 风险与边界

### 12.1 已接受的限制

- 不保证背面风格一致性
- 不保证复杂场景效果
- 不保证第一版观感最优

### 12.2 主要技术风险

- 自动抠图质量可能影响 `SF3D` 重建质量
- `InstantStyle` 结果可能偏离原物体细节，需要后续调参
- 主视角回贴如果实现过于粗糙，可能出现纹理接缝或视角错位

### 12.3 规避策略

- 先用公开样例验证最小闭环
- 每步保存中间结果
- 优先保证“有产物可看”，再逐步优化质量

## 13. 后续扩展方向

不纳入第一版，但后续可扩展：

- 多风格参考图
- 多视角纹理补全
- ComfyUI 工作流封装
- 网页上传与任务触发
- prompt 自动生成
- 人物或复杂主体支持

## 14. 非目标

以下内容明确不属于第一版：

- 复杂场景图的一步到位支持
- 高质量多视角风格一致贴图
- 在线服务化部署
- 统一单环境打包
- 完整 Web 产品化交互

## 15. 当前状态说明

已验证事实：

- 机器具备 `RTX 4080 32GB`
- `InstantStyle/` 与 `stable-fast-3d/` 已存在于工作区
- `SF3D` 的 gated 访问已开通
- 从当前机器直连官方 Hugging Face 存在超时
- `hf-mirror.com` 可用，可作为公开与已授权资源的下载通道

## 16. 实施前提

进入实现阶段前，默认满足以下前提：

- 用户提供 Hugging Face `read` token 以完成依赖拉取
- 各环境允许单独创建并安装依赖
- 第一版先用公开样例，再切真实样例

## 17. Git 状态说明

按照常规流程，spec 应提交到 git。

但当前工作目录 `/root/autodl-tmp/src` 不是一个 git 仓库，因此本次只能写入 spec 文件，无法在该目录完成 git commit。若后续需要版本化该编排层，建议为项目根目录初始化独立仓库，或将编排层移动到现有 git 仓库中管理。
