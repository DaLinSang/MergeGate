"""
负反馈通知（飞书）
=================

【业务定位】
    用户点踩时，把"哪条回答、什么原因、完整轨迹链接"推到飞书群。
    目的：让团队能第一时间看到坏案例，而不是等周报统计。

【数据流向】
    上游（谁调它）：
        api/routes/feedback.py → 写完 feedback 后调本函数
    下游（它调谁）：
        settings.feishu_feedback_webhook_url → 飞书机器人 webhook（HTTP POST）

【核心规则】
    ★ 通知失败绝不能影响已保存的反馈。
      所以本函数【永不抛异常】，只返回 (状态, 错误信息)。

【本文件函数地图】
    send_negative_feedback_notification()  ★ 主流程
        ├ _clip()       截断长文本
        └ _trace_url()  生成轨迹链接
"""

import logging
from typing import Any

import httpx

from app.core.config import settings
from app.db.models import AgentRun, ChatFeedback, ChatMessage, Repository, User

logger = logging.getLogger(__name__)


# 点踩原因的中文标签（前端传英文 key，通知里显示中文）
# 和 schemas/feedback.py 里的枚举保持一致
REASON_LABELS = {
    "inaccurate": "内容不准确",
    "not_relevant": "没有解决问题",
    "missing_context": "缺少关键上下文",
    "unreliable_citation": "引用或证据不可靠",
    "tool_error": "工具执行异常",
    "other": "其他",
}


def _clip(value: str | None, limit: int) -> str:
    """压缩空白并截断（用于把长文本塞进通知）。

    【调用链】被 send_negative_feedback_notification() 调用

    注意这里截断后加 "..." 是在 limit 之外加的（会略微超长），
    因为通知场景不需要精确控制长度。
    """
    text = " ".join((value or "").split())
    return text if len(text) <= limit else f"{text[:limit]}..."


def _trace_url(run_id: Any) -> str:
    """生成"执行轨迹"的可点击链接。

    【调用链】被 send_negative_feedback_notification() 调用

    【为什么要可配置 base_url】
        本地开发时前端地址是 localhost，通知里放 localhost 链接没用。
        配置 settings.feedback_trace_base_url 可以指向线上地址。

    【没有配置时返回相对路径】
        飞书会显示成纯文本，至少能看出路径是什么。
    """
    path = f"/api/chat/feedback/trace/{run_id}"
    base_url = (settings.feedback_trace_base_url or "").strip().rstrip("/")
    return f"{base_url}{path}" if base_url else path


async def send_negative_feedback_notification(
    *,
    feedback: ChatFeedback,
    repo: Repository,
    message: ChatMessage,
    run: AgentRun | None,
    reporter: User | None,
) -> tuple[str, str | None]:
    """发送一条负反馈通知到飞书。

    【调用链】_clip() / _trace_url() → httpx.post → 本函数
    【被谁调用】api/routes/feedback.py（点踩且 review 需要人工介入时）

    【数据流向】
        入：feedback（反馈记录）、repo、message（被评价的回答）、
            run（关联的运行记录，可能为 None）、reporter（反馈人）
        出：(status, error)
            status = "sent" / "failed" / "not_configured"
            error  = 失败原因（已截断到 1000 字符）

    【为什么要拼这么多字段】
        收到通知的人第一眼要能判断"要不要现在处理"：
            哪个项目？谁反馈的？什么原因？问了什么？答了什么？
        少了任何一项，都得再去后台查，通知就失去了"即时性"价值。

    【为什么 run 可能为 None】
        老数据（agent_runs 表上线前的回答）没有 run_id。
        这时通知里要明确写"该历史回答未关联 run_id"，
        而不是留空让人以为是 bug。
    """
    webhook_url = (settings.feishu_feedback_webhook_url or "").strip()
    # 没配置 → 直接返回（不是错误，只是没开这个功能）
    if not webhook_url:
        return "not_configured", None

    run_id = feedback.run_id or "未关联"
    lines = [
        "DevFlow AI 收到一条负反馈",
        f"项目：{repo.full_name}",
        f"反馈人：{reporter.name if reporter else '内部用户'}",
        f"trace_id：{run_id}",
        f"message_id：{feedback.assistant_message_id}",
        # 路由优先取 run 的（更准确），退回 message 的，最后 unknown
        f"路由：{run.route if run else message.route or 'unknown'}",
        f"原因：{REASON_LABELS.get(feedback.reason or '', '未填写')}",
    ]
    # 用户写的补充说明（可选）
    if feedback.comment:
        lines.append(f"补充说明：{_clip(feedback.comment, 300)}")

    # 有关联运行记录 → 附上问题和回答（截断，避免超过飞书消息长度限制）
    if run:
        lines.extend(
            [
                f"用户问题：{_clip(run.user_message, 300)}",
                f"Agent 回答：{_clip(run.final_answer, 500)}",
            ]
        )

    lines.append(f"执行轨迹：{_trace_url(run_id)}" if feedback.run_id else "执行轨迹：该历史回答未关联 run_id")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                webhook_url,
                json={"msg_type": "text", "content": {"text": "\n".join(lines)}},
            )
            response.raise_for_status()
            payload = response.json()
            # 飞书 webhook 即使 HTTP 200 也可能业务失败，要看 StatusCode / code
            status_code = payload.get("StatusCode", payload.get("code", 0)) if isinstance(payload, dict) else 0
            if status_code not in {0, "0", None}:
                raise RuntimeError(str(payload.get("StatusMessage") or payload.get("msg") or payload))
    except Exception as exc:  # Notification failure must not lose the persisted feedback.
        # ★ 只记 warning，不往上抛 —— 反馈本身已经存库了，通知失败不该让接口报错
        logger.warning("发送飞书负反馈通知失败：%s", exc)
        return "failed", _clip(str(exc), 1000)
    return "sent", None
