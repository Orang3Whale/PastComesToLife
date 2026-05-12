from __future__ import annotations

import html
import time
import uuid
from datetime import datetime
from pathlib import Path

import gradio as gr


APP_TITLE = "万物赋新台"
APP_SUBTITLE = "岁月器物的三维留存与跨次元重塑"
PLACEHOLDER_MODEL = Path(__file__).with_name("mesh_stylized4.glb")#当前只是占位


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_task_id() -> str:
    return f"TASK-{datetime.now().strftime('%m%d-%H%M%S')}-{uuid.uuid4().hex[:4].upper()}"


def normalize_history(history: list[dict] | None) -> list[dict]:
    return list(history or [])


def upsert_top(history: list[dict], record: dict) -> list[dict]:
    items = list(history)
    if items and items[0]["task_id"] == record["task_id"]:
        items[0] = record
        return items
    return [record] + items


def preview_text(text: str, limit: int = 28) -> str:
    clean = (text or "").strip()
    if not clean:
        return "未填写"
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + "..."


def html_text(text: str) -> str:
    return html.escape((text or "").strip()).replace("\n", "<br>")


def status_card(task_id: str = "", status: str = "idle", message: str = "等待创建任务") -> str:
    return f"""
    <div class="status-box">
        <div class="status-row">
            <span class="status-badge {status}">{status.upper()}</span>
            <span class="status-text">{message}</span>
        </div>
        <div class="task-id">{task_id or "未生成任务 ID"}</div>
    </div>
    """


def toast_card(message: str = "", detail: str = "", visible: bool = False) -> str:
    if not visible or not message:
        return ""

    detail_html = f"<div class='toast-detail'>{detail}</div>" if detail else ""
    return f"""
    <div class="toast-shell">
        <div class="toast-card">
            <div class="toast-header">
                <span class="toast-badge">处理中</span>
                <span class="toast-dots"><i></i><i></i><i></i></span>
            </div>
            <div class="toast-message">{message}</div>
            {detail_html}
        </div>
    </div>
    """


def progress_card(step: int) -> str:
    steps = ["接收照片", "读取参考图", "提示词注入", "返回模型"]
    items = []
    for index, label in enumerate(steps, start=1):
        state = "done" if index < step else "active" if index == step else "idle"
        items.append(
            f"""
            <div class="progress-item {state}">
                <span>{index}</span>
                <div>{label}</div>
            </div>
            """
        )
    return f"<div class='progress-grid'>{''.join(items)}</div>"


def result_card(record: dict | None) -> str:
    if not record:
        return "<div class='empty-box'>结果将在任务完成后显示。</div>"
    return f"""
    <div class="result-box">
        <dl>
            <dt>参考图</dt><dd>{html_text(record['reference_name'])}</dd>
            <dt>Prompt</dt><dd>{html_text(record['prompt_text'])}</dd>
            <dt>输入</dt><dd>{html_text(record['image_name'])}</dd>
            <dt>输出</dt><dd>{html_text(record['model_path'])}</dd>
            <dt>更新时间</dt><dd>{html_text(record['updated_at'])}</dd>
        </dl>
    </div>
    """


def history_card(history: list[dict] | None) -> str:
    records = normalize_history(history)
    if not records:
        return "<div class='empty-box'>当前没有历史记录。</div>"

    blocks = []
    for record in records[:6]:
        blocks.append(
            f"""
            <div class="history-item">
                <div><strong>{html_text(record['task_id'])}</strong></div>
                <div>{html_text(record['reference_name'])} · {html_text(preview_text(record['prompt_text'], 20))}</div>
                <div>{html_text(record['status'].upper())} · {html_text(record['updated_at'])}</div>
            </div>
            """
        )
    return f"<div class='history-list'>{''.join(blocks)}</div>"


def query_card(task_id: str, history: list[dict] | None) -> str:
    clean_id = (task_id or "").strip()
    if not clean_id:
        return "<div class='empty-box'>输入任务 ID 后可查询。</div>"

    match = next((item for item in normalize_history(history) if item["task_id"] == clean_id), None)
    if not match:
        return f"<div class='empty-box'>未找到任务 {clean_id}。</div>"

    return f"""
    <div class="result-box">
        <dl>
            <dt>任务</dt><dd>{html_text(match['task_id'])}</dd>
            <dt>状态</dt><dd>{html_text(match['status'].upper())}</dd>
            <dt>参考图</dt><dd>{html_text(match['reference_name'])}</dd>
            <dt>Prompt</dt><dd>{html_text(match['prompt_text'])}</dd>
        </dl>
    </div>
    """


def build_record(
    task_id: str,
    image_path: str,
    reference_image_path: str,
    prompt_text: str,
    status: str,
    model_path: str = "",
) -> dict:
    return {
        "task_id": task_id,
        "image_name": Path(image_path).name if image_path else "未上传",
        "reference_name": Path(reference_image_path).name if reference_image_path else "未上传",
        "prompt_text": prompt_text.strip() if prompt_text else "",
        "status": status,
        "model_path": model_path,
        "updated_at": now_text(),
    }


def submit_generation_task(image_path: str, reference_image_path: str, prompt_text: str) -> str:
    if not image_path:
        raise gr.Error("请先上传一张照片。")
    if not reference_image_path:
        raise gr.Error("请先上传一张风格参考图。")
    if not PLACEHOLDER_MODEL.exists():
        raise gr.Error("未找到占位模型 `mesh_raw.glb`。")

    # TODO: 替换为真实后端提交接口。
    _payload = {
        "image_path": image_path,
        "reference_image_path": reference_image_path,
        "prompt_text": prompt_text,
    }
    return build_task_id()


def run_generation(image_path: str, reference_image_path: str, prompt_text: str, history: list[dict] | None):
    history_items = normalize_history(history)
    task_id = submit_generation_task(image_path, reference_image_path, prompt_text)

    queued = build_record(task_id, image_path, reference_image_path, prompt_text, "queued")
    history_items = upsert_top(history_items, queued)
    yield (
        gr.skip(),
        task_id,
        toast_card("任务已接收", "正在准备实景重建与风格融合流程", True),
        status_card(task_id, "queued", "任务已创建"),
        progress_card(1),
        result_card(None),
        history_card(history_items),
        history_items,
    )

    time.sleep(0.7)
    running = build_record(task_id, image_path, reference_image_path, prompt_text, "running")
    history_items = upsert_top(history_items, running)
    yield (
        gr.skip(),
        task_id,
        toast_card("正在读取风格参考图", "提取颜色、材质与构图特征", True),
        status_card(task_id, "running", "正在进行三维重建"),
        progress_card(2),
        result_card(None),
        history_card(history_items),
        history_items,
    )

    time.sleep(0.8)
    stylized = build_record(task_id, image_path, reference_image_path, prompt_text, "running")
    history_items = upsert_top(history_items, stylized)
    yield (
        gr.skip(),
        task_id,
        toast_card("正在注入提示词", "将语义描述写入风格化管线", True),
        status_card(task_id, "running", "正在注入提示词"),
        progress_card(3),
        result_card(None),
        history_card(history_items),
        history_items,
    )

    time.sleep(0.8)
    completed = build_record(task_id, image_path, reference_image_path, prompt_text, "completed", str(PLACEHOLDER_MODEL))
    history_items = upsert_top(history_items, completed)
    yield (
        str(PLACEHOLDER_MODEL),
        task_id,
        toast_card("生成完成", "模型已回传，可旋转查看", True),
        status_card(task_id, "completed", "任务完成"),
        progress_card(4),
        result_card(completed),
        history_card(history_items),
        history_items,
    )

    time.sleep(0.9)
    yield (
        gr.skip(),
        task_id,
        "",
        status_card(task_id, "completed", "任务完成"),
        progress_card(4),
        result_card(completed),
        history_card(history_items),
        history_items,
    )


def reset_demo():
    return (
        None,
        None,
        "",
        None,
        "",
        "",
        status_card(),
        progress_card(1),
        result_card(None),
        query_card("", []),
        history_card([]),
    )

CUSTOM_CSS = """
:root {
    --bg-0: #f6efe1;
    --bg-1: #dfe8dd;
    --bg-2: #cad8d3;
    --ink: #1c2926;
    --soft: #62716c;
    --line: rgba(28, 41, 38, 0.12);
    --card: rgba(255, 252, 246, 0.84);
    --card-strong: rgba(255, 255, 255, 0.94);
    --accent: #1f6a57;
    --accent-2: #2f8f74;
    --warm: #a96d3c;
    --shadow: 0 20px 48px rgba(33, 47, 44, 0.12);
}

html, body {
    min-height: 100%;
}

body {
    color: var(--ink);
    background:
        radial-gradient(circle at 10% 8%, rgba(187, 150, 88, 0.18), transparent 24%),
        radial-gradient(circle at 90% 18%, rgba(47, 143, 116, 0.16), transparent 25%),
        radial-gradient(circle at 75% 92%, rgba(31, 106, 87, 0.10), transparent 26%),
        linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 56%, var(--bg-2) 100%);
    background-attachment: fixed;
}

body::before,
body::after {
    content: "";
    position: fixed;
    inset: -12vmax;
    pointer-events: none;
    z-index: 0;
}

body::before {
    background:
        radial-gradient(circle at 18% 24%, rgba(255, 255, 255, 0.36), transparent 18%),
        radial-gradient(circle at 82% 68%, rgba(255, 255, 255, 0.18), transparent 16%),
        radial-gradient(circle at 50% 50%, rgba(255, 255, 255, 0.08), transparent 42%);
    filter: blur(8px);
    mix-blend-mode: screen;
    animation: floatGlow 18s ease-in-out infinite alternate;
}

body::after {
    background-image:
        linear-gradient(rgba(255, 255, 255, 0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.05) 1px, transparent 1px),
        linear-gradient(135deg, rgba(31, 106, 87, 0.06) 0 1px, transparent 1px 18px);
    background-size: 34px 34px, 34px 34px, 100% 100%;
    opacity: 0.55;
    mask-image: linear-gradient(180deg, rgba(0, 0, 0, 0.26), transparent 78%);
}

@keyframes floatGlow {
    from { transform: translate3d(-1.5%, -1%, 0) scale(1); }
    to { transform: translate3d(1.5%, 1%, 0) scale(1.03); }
}

.gradio-container {
    position: relative;
    z-index: 1;
    max-width: 1320px !important;
    margin: 0 auto !important;
    padding: 18px !important;
}

.workspace-row {
    display: grid !important;
    grid-template-columns: minmax(0, 5fr) minmax(0, 7fr);
    align-items: stretch !important;
    gap: 16px !important;
}

.workspace-row > div {
    min-width: 0 !important;
}

.left-pane,
.right-pane {
    min-width: 0;
}

.reference-prompt-row {
    display: grid !important;
    grid-template-columns: minmax(0, 5fr) minmax(0, 7fr);
    gap: 12px !important;
    align-items: stretch !important;
}

.reference-prompt-row > div {
    min-width: 0 !important;
}

footer {
    display: none !important;
}

.title-wrap {
    position: relative;
    display: flex;
    align-items: end;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 14px;
    padding: 20px 22px;
    border-radius: 28px;
    background:
        linear-gradient(135deg, rgba(255, 250, 242, 0.96), rgba(239, 246, 241, 0.88)),
        radial-gradient(circle at 88% 18%, rgba(47, 143, 116, 0.12), transparent 26%);
    border: 1px solid rgba(255, 255, 255, 0.72);
    box-shadow: var(--shadow);
    overflow: hidden;
    isolation: isolate;
}

.title-wrap::before,
.title-wrap::after {
    content: "";
    position: absolute;
    border-radius: 50%;
    pointer-events: none;
}

.title-wrap::before {
    width: 260px;
    height: 260px;
    right: -92px;
    top: -104px;
    background: radial-gradient(circle, rgba(47, 143, 116, 0.16), transparent 66%);
}

.title-wrap::after {
    width: 180px;
    height: 180px;
    left: -72px;
    bottom: -96px;
    background: radial-gradient(circle, rgba(169, 109, 60, 0.14), transparent 70%);
}

.title-copy {
    position: relative;
    z-index: 1;
}

.eyebrow {
    display: inline-flex;
    align-items: center;
    padding: 6px 10px;
    margin-bottom: 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
    background: rgba(31, 106, 87, 0.08);
}

.title-wrap h1 {
    margin: 0;
    font-size: 34px;
    line-height: 1.08;
    font-family: "STZhongsong", "Songti SC", serif;
    letter-spacing: 0.02em;
}

.title-wrap p {
    margin: 8px 0 0;
    max-width: 760px;
    font-size: 13px;
    line-height: 1.7;
    color: var(--soft);
}

.title-chip {
    position: relative;
    z-index: 1;
    flex-shrink: 0;
    padding: 9px 14px;
    border-radius: 999px;
    background: linear-gradient(135deg, rgba(31, 106, 87, 0.11), rgba(255, 255, 255, 0.55));
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.52);
    color: var(--accent);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.06em;
}

.main-card,
.fold-card {
    position: relative;
    overflow: hidden;
    isolation: isolate;
    padding: 16px;
    border-radius: 26px;
    background: var(--card);
    border: 1px solid rgba(255, 255, 255, 0.6);
    box-shadow: var(--shadow);
    backdrop-filter: blur(10px);
}

.main-card::before,
.fold-card::before {
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    opacity: 0.88;
    z-index: 0;
}

.main-card::after,
.fold-card::after {
    content: "";
    position: absolute;
    right: -48px;
    bottom: -58px;
    width: 180px;
    height: 180px;
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    opacity: 0.8;
    filter: blur(2px);
}

.control-panel::before {
    background:
        radial-gradient(circle at 92% 6%, rgba(169, 109, 60, 0.11), transparent 24%),
        linear-gradient(180deg, rgba(255, 255, 255, 0.18), transparent 28%);
}

.control-panel::after {
    background: radial-gradient(circle, rgba(169, 109, 60, 0.18), transparent 70%);
}

.viewer-card::before {
    background:
        radial-gradient(circle at 80% 10%, rgba(47, 143, 116, 0.18), transparent 28%),
        radial-gradient(circle at 18% 88%, rgba(31, 106, 87, 0.12), transparent 26%),
        linear-gradient(180deg, rgba(17, 30, 28, 0.04), transparent 34%);
}

.viewer-card::after {
    background: radial-gradient(circle, rgba(47, 143, 116, 0.22), transparent 68%);
}

.fold-card {
    background: rgba(255, 253, 248, 0.88);
}

.main-card > *,
.fold-card > * {
    position: relative;
    z-index: 1;
}

.control-panel {
    border-left: 1px solid rgba(31, 106, 87, 0.10);
}

.viewer-card {
    background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.90), rgba(255, 252, 246, 0.82)),
        radial-gradient(circle at 50% 0%, rgba(47, 143, 116, 0.08), transparent 28%);
}

.stage-head {
    display: flex;
    align-items: end;
    justify-content: space-between;
    gap: 14px;
}

.stage-label {
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
}

.stage-title {
    margin-top: 4px;
    font-size: 18px;
    font-weight: 700;
    color: var(--ink);
}

.stage-chip {
    flex-shrink: 0;
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(31, 106, 87, 0.09);
    color: var(--accent);
    font-size: 12px;
    font-weight: 700;
}

#generate-btn {
    background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
    border: none !important;
    color: #fff !important;
    box-shadow: 0 12px 26px rgba(31, 106, 87, 0.20) !important;
}

#generate-btn:hover {
    filter: brightness(1.04);
}

#reset-btn {
    border: 1px solid rgba(34, 105, 87, 0.22) !important;
    color: var(--accent) !important;
}

#output-model {
    border-radius: 22px;
    overflow: hidden;
    background:
        radial-gradient(circle at 50% 0%, rgba(47, 143, 116, 0.16), transparent 26%),
        linear-gradient(180deg, rgba(18, 29, 27, 0.98), rgba(29, 44, 40, 0.92));
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.06),
        0 14px 28px rgba(16, 23, 22, 0.18);
}

#scene-upload {
    border-radius: 20px;
    overflow: hidden;
    background:
        radial-gradient(circle at 86% 12%, rgba(47, 143, 116, 0.10), transparent 24%),
        linear-gradient(180deg, rgba(255, 255, 255, 0.72), rgba(248, 244, 235, 0.82));
    border: 1px solid rgba(31, 106, 87, 0.12);
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.6),
        0 12px 24px rgba(33, 47, 44, 0.06);
}

#reference-image {
    border-radius: 20px;
    overflow: hidden;
    background:
        radial-gradient(circle at 14% 12%, rgba(169, 109, 60, 0.10), transparent 24%),
        linear-gradient(180deg, rgba(255, 255, 255, 0.74), rgba(246, 240, 229, 0.84));
    border: 1px solid rgba(169, 109, 60, 0.14);
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.58),
        0 12px 24px rgba(33, 47, 44, 0.06);
}

#prompt-box {
    border-radius: 20px;
    background:
        radial-gradient(circle at 92% 18%, rgba(47, 143, 116, 0.08), transparent 24%),
        linear-gradient(180deg, rgba(255, 255, 255, 0.82), rgba(248, 243, 234, 0.92));
    border: 1px solid rgba(31, 106, 87, 0.12);
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.64),
        0 12px 24px rgba(33, 47, 44, 0.06);
}

#prompt-box textarea,
#prompt-box input {
    background: transparent !important;
    color: var(--ink) !important;
    border: none !important;
    box-shadow: none !important;
}

#prompt-box textarea {
    min-height: 220px !important;
    height: 220px !important;
    line-height: 1.7 !important;
    resize: none !important;
}

#prompt-box textarea::placeholder,
#prompt-box input::placeholder {
    color: rgba(98, 113, 108, 0.78) !important;
}

#scene-options,
#style-options {
    --choice-border: rgba(31, 106, 87, 0.16);
    --choice-bg: linear-gradient(180deg, rgba(255, 251, 244, 0.96), rgba(245, 239, 228, 0.88));
    --choice-hover-bg: linear-gradient(180deg, rgba(243, 250, 246, 0.98), rgba(234, 243, 238, 0.92));
    --choice-active-bg: linear-gradient(135deg, rgba(31, 106, 87, 0.96), rgba(47, 143, 116, 0.92));
    --choice-shadow: 0 8px 18px rgba(33, 47, 44, 0.08);
}

#scene-options .wrap,
#style-options .wrap {
    gap: 10px;
}

#scene-options label,
#scene-options button,
#style-options label,
#style-options button {
    border-radius: 999px !important;
    border: 1px solid var(--choice-border) !important;
    background: var(--choice-bg) !important;
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.78),
        var(--choice-shadow) !important;
    color: var(--ink) !important;
    transition:
        transform 0.18s ease,
        box-shadow 0.18s ease,
        border-color 0.18s ease,
        background 0.18s ease !important;
}

#scene-options label:hover,
#scene-options button:hover,
#style-options label:hover,
#style-options button:hover {
    transform: translateY(-1px);
    border-color: rgba(31, 106, 87, 0.28) !important;
    background: var(--choice-hover-bg) !important;
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.84),
        0 12px 20px rgba(31, 106, 87, 0.10) !important;
}

#scene-options label:has(input:checked),
#style-options label:has(input:checked),
#scene-options button[aria-pressed="true"],
#style-options button[aria-pressed="true"],
#scene-options button.selected,
#style-options button.selected {
    border-color: rgba(31, 106, 87, 0.08) !important;
    background: var(--choice-active-bg) !important;
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.16),
        0 14px 24px rgba(31, 106, 87, 0.18) !important;
    color: #ffffff !important;
}

#scene-options label:has(input:checked) *,
#style-options label:has(input:checked) *,
#scene-options button[aria-pressed="true"] *,
#style-options button[aria-pressed="true"] *,
#scene-options button.selected *,
#style-options button.selected * {
    color: #ffffff !important;
}

#scene-options input[type="radio"],
#style-options input[type="radio"] {
    accent-color: var(--accent);
}

#scene-options .wrap,
#style-options .wrap {
    gap: 10px !important;
}

#scene-options label,
#style-options label,
#scene-options button,
#style-options button {
    border-radius: 999px !important;
    border: 1px solid rgba(31, 106, 87, 0.14) !important;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(245, 239, 228, 0.96)) !important;
    color: var(--ink) !important;
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.72),
        0 8px 18px rgba(33, 47, 44, 0.06) !important;
    transition:
        transform 0.18s ease,
        border-color 0.18s ease,
        box-shadow 0.18s ease,
        background 0.18s ease !important;
}

#scene-options label *,
#style-options label *,
#scene-options button *,
#style-options button * {
    color: inherit !important;
}

#scene-options label:hover,
#style-options label:hover,
#scene-options button:hover,
#style-options button:hover {
    transform: translateY(-1px);
    border-color: rgba(31, 106, 87, 0.28) !important;
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.78),
        0 12px 22px rgba(33, 47, 44, 0.08) !important;
}

#scene-options label:has(input:checked),
#scene-options button[aria-checked="true"],
#scene-options button[aria-pressed="true"] {
    background: linear-gradient(135deg, rgba(31, 106, 87, 0.98), rgba(47, 143, 116, 0.9)) !important;
    border-color: rgba(31, 106, 87, 0.42) !important;
    color: #ffffff !important;
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.12),
        0 14px 26px rgba(31, 106, 87, 0.18) !important;
}

#style-options label:has(input:checked),
#style-options button[aria-checked="true"],
#style-options button[aria-pressed="true"] {
    background: linear-gradient(135deg, rgba(169, 109, 60, 0.96), rgba(31, 106, 87, 0.9)) !important;
    border-color: rgba(169, 109, 60, 0.42) !important;
    color: #ffffff !important;
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.12),
        0 14px 26px rgba(120, 88, 50, 0.18) !important;
}

#scene-options input[type="radio"],
#style-options input[type="radio"] {
    accent-color: var(--accent);
}

#scene-options label:focus-within,
#style-options label:focus-within,
#scene-options button:focus-visible,
#style-options button:focus-visible {
    outline: none !important;
    box-shadow:
        0 0 0 2px rgba(255, 251, 244, 0.95),
        0 0 0 4px rgba(31, 106, 87, 0.22),
        0 14px 26px rgba(33, 47, 44, 0.1) !important;
}

.toast-shell {
    position: fixed;
    top: 20px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 9999;
    pointer-events: none;
    animation: toastEnter 0.24s ease-out;
}

.toast-card {
    min-width: 280px;
    max-width: 420px;
    padding: 14px 16px;
    border-radius: 18px;
    background:
        linear-gradient(135deg, rgba(255, 251, 244, 0.96), rgba(239, 246, 241, 0.94)),
        radial-gradient(circle at 90% 0%, rgba(47, 143, 116, 0.12), transparent 28%);
    border: 1px solid rgba(255, 255, 255, 0.72);
    box-shadow: 0 18px 42px rgba(24, 36, 33, 0.18);
    backdrop-filter: blur(12px);
}

.toast-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 10px;
}

.toast-badge {
    display: inline-flex;
    align-items: center;
    padding: 5px 10px;
    border-radius: 999px;
    background: rgba(31, 106, 87, 0.10);
    color: var(--accent);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
}

.toast-dots {
    display: inline-flex;
    align-items: center;
    gap: 4px;
}

.toast-dots i {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--accent);
    display: inline-block;
    animation: toastDot 0.9s infinite ease-in-out;
}

.toast-dots i:nth-child(2) {
    animation-delay: 0.12s;
}

.toast-dots i:nth-child(3) {
    animation-delay: 0.24s;
}

.toast-message {
    font-size: 15px;
    font-weight: 700;
    color: var(--ink);
    line-height: 1.5;
}

.toast-detail {
    margin-top: 6px;
    font-size: 12px;
    color: var(--soft);
    line-height: 1.6;
}

@keyframes toastEnter {
    from { opacity: 0; transform: translateX(-50%) translateY(-8px); }
    to { opacity: 1; transform: translateX(-50%) translateY(0); }
}

@keyframes toastDot {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.45; }
    40% { transform: translateY(-3px); opacity: 1; }
}

.status-box,
.result-box,
.empty-box,
.history-item {
    border-radius: 16px;
    border: 1px solid var(--line);
    background: var(--card-strong);
    padding: 12px 14px;
}

.status-row {
    display: flex;
    gap: 10px;
    align-items: center;
    margin-bottom: 8px;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    padding: 5px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
}

.status-badge.idle,
.status-badge.queued {
    background: rgba(169, 109, 60, 0.12);
    color: var(--warm);
}

.status-badge.running {
    background: rgba(31, 106, 87, 0.12);
    color: var(--accent);
}

.status-badge.completed {
    background: rgba(31, 106, 87, 0.14);
    color: var(--accent);
}

.status-text,
.task-id {
    font-size: 13px;
    color: var(--soft);
}

.task-id {
    margin-top: 4px;
    font-family: Consolas, monospace;
}

.progress-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 8px;
}

.progress-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    border-radius: 14px;
    border: 1px solid var(--line);
    background: rgba(255, 255, 255, 0.72);
    font-size: 13px;
}

.progress-item span {
    width: 24px;
    height: 24px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 999px;
    background: rgba(29, 40, 37, 0.08);
    color: var(--soft);
    font-weight: 700;
    flex: 0 0 auto;
}

.progress-item.done span,
.progress-item.active span {
    background: var(--accent);
    color: #fff;
}

.history-list {
    display: grid;
    gap: 10px;
}

.history-item {
    font-size: 13px;
    line-height: 1.6;
}

.empty-box {
    color: var(--soft);
}

.main-card .wrap,
.fold-card .wrap {
    gap: 12px;
}

@media (max-width: 960px) {
    .workspace-row,
    .reference-prompt-row {
        grid-template-columns: 1fr !important;
    }

    .title-wrap {
        flex-direction: column;
        align-items: start;
    }

    .title-wrap h1 {
        font-size: 28px;
    }

    .progress-grid {
        grid-template-columns: 1fr 1fr;
    }
}
"""


with gr.Blocks(title=APP_TITLE) as demo:
    history_state = gr.State([])

    gr.HTML(
        f"""
        <div class="title-wrap">
            <div class="title-copy">
                <div class="eyebrow">记忆可视化</div>
                <h1>{APP_TITLE}</h1>
                <p>{APP_SUBTITLE}</p>
            </div>
            <!--<div class="title-chip">Desktop Preview</div>-->
        </div>
        """
                )

    toast_html = gr.HTML("")

    with gr.Row(equal_height=True, elem_classes=["workspace-row"]):
        with gr.Column(scale=5, min_width=0):
            with gr.Column(elem_classes=["main-card", "control-panel", "left-pane"]):
                input_image = gr.Image(label="上传物件照片", type="filepath", height=260, elem_id="scene-upload")
                with gr.Row(elem_classes=["reference-prompt-row"]):
                    with gr.Column(scale=5, min_width=0):
                        reference_image = gr.Image(label="风格参考图", type="filepath", height=260, elem_id="reference-image")
                    with gr.Column(scale=7, min_width=0):
                        prompt_text = gr.Textbox(
                            label="提示词",
                            placeholder="例如：复古质感、温润材质、保留岁月痕迹、柔和光照、层次丰富",
                            lines=8,
                            elem_id="prompt-box",
                        )
                with gr.Row():
                    generate_btn = gr.Button("开始生成", elem_id="generate-btn", variant="primary")
                    reset_btn = gr.Button("重置", elem_id="reset-btn")

        with gr.Column(scale=7, min_width=0):
            with gr.Column(elem_classes=["main-card", "viewer-card", "right-pane"]):
                gr.HTML(
                    """
                    <div class="stage-head">
                        <div>
                            <div class="stage-label">Object Preview</div>
                            <div class="stage-title">3D展示框</div>
                        </div>
                        <div class="stage-chip">可旋转查看</div>
                    </div>
                    """
                )
                output_model = gr.Model3D(
                    label="",
                    height=560,
                    clear_color=[0.08, 0.12, 0.11, 1.0],
                    camera_position=(270, 90, None),
                    elem_id="output-model",
                    value=None,
                )

    with gr.Accordion("更多", open=False):
        with gr.Column(elem_classes=["fold-card"]):
            current_task_id = gr.Textbox(label="当前任务 ID", interactive=False, placeholder="生成后自动显示")
            status_html = gr.HTML(status_card())
            progress_html = gr.HTML(progress_card(1))
            result_html = gr.HTML(result_card(None))

            query_task_id = gr.Textbox(label="查询任务 ID", placeholder="输入任务 ID 后查询")
            query_btn = gr.Button("查询状态")
            query_result_html = gr.HTML(query_card("", []))

            history_html = gr.HTML(history_card([]))

    generate_btn.click(
        fn=run_generation,
        inputs=[input_image, reference_image, prompt_text, history_state],
        outputs=[
            output_model,
            current_task_id,
            toast_html,
            status_html,
            progress_html,
            result_html,
            history_html,
            history_state,
        ],
        show_progress="hidden",
    )

    query_btn.click(
        fn=query_card,
        inputs=[query_task_id, history_state],
        outputs=[query_result_html],
    )

    reset_btn.click(
        fn=reset_demo,
        outputs=[
            input_image,
            reference_image,
            prompt_text,
            output_model,
            current_task_id,
            toast_html,
            status_html,
            progress_html,
            result_html,
            query_result_html,
            history_html,
        ],
    )

    demo.load(
        fn=lambda: None,
        outputs=[output_model],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        share=False,
        css=CUSTOM_CSS,
        theme=gr.themes.Base(),
    )
