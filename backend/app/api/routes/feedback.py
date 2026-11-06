"""
反馈路由
========

【路由清单】
    GET   /feedback/metrics          反馈指标看板（覆盖率 / 好评率 / 差评原因分布）
    GET   /feedback/trace/{run_id}   按 run_id 拉完整执行轨迹（排查"这条回答为什么差"）
    GET   /feedback                  列反馈
    POST  /feedback                  提交 / 更新反馈（好/差评）
    PATCH /feedback/{feedback_id}    人工处理差评（标记已解决 / 已忽略）

【数据流向】
    用户点👍/👎
      → POST /feedback（本文件）
      → 校验消息归属 → 找 run_id（从 message.meta）
      → chat_feedback 表
      → 如果是差评且首次 → services/feedback_notifications.py 发飞书通知

    GET /trace/{run_id}（排查链路）
      → agent_runs（模型侧记录）
      → chat_messages（用户问 + AI 答）
      → chat_sessions（transcript_json：工具调用、记忆更新）
      → 一次性返回给前端"轨迹"页

【要点】
    1. 反馈是"一条消息一条"，重复提交是【更新】不是新增 —— 所以先按
       assistant_message_id 查已有记录。
    2. 差评通知只在"从非差评变成差评"或"还没通知过"时发，避免用户反复点👎刷通知。
    3. 改回好评时把 review_status 置 resolved、通知状态置 not_applicable ——
       好评不需要人工处理。
    4. 通知发送放在 commit 之后：通知失败不能影响已存的反馈（见 feedback_notifications.py）。
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.models import AgentRun, ChatFeedback, ChatMessage, ChatSession, Conversation, Repository
from app.db.session import get_db
from app.schemas.feedback import ChatFeedbackCreate, ChatFeedbackResponse, ChatFeedbackReview
from app.services.feedback_notifications import send_negative_feedback_notification
from app.services.permissions import ensure_demo_user

router = APIRouter()


def _run_id_from_message(message: ChatMessage) -> uuid.UUID | None:
    """从消息的 meta 里取 run_id。

    run_id 是串起"消息 ↔ 模型运行记录 ↔ 反馈"的关键字段。
    存在 meta（JSON）里而不是独立列，所以这里要容错解析。
    """
    raw_run_id = (message.meta or {}).get("run_id")
    if not raw_run_id:
        return None
    try:
        return uuid.UUID(str(raw_run_id))
    except (TypeError, ValueError):
        return None


def _message_or_404(db: Session, payload: ChatFeedbackCreate) -> ChatMessage:
    """定位被反馈的那条 assistant 消息。

    四个条件全要匹配（id + repo + conversation + role=assistant）：
    只按 id 查的话，构造一个别的仓库的消息 id 就能跨仓库提交反馈 —— 越权。
    """
    message = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.id == payload.assistant_message_id,
            ChatMessage.repo_id == payload.repo_id,
            ChatMessage.conversation_id == payload.conversation_id,
            ChatMessage.role == "assistant",
        )
        .one_or_none()
    )
    if message is None:
        raise HTTPException(status_code=404, detail="未找到对应的 Agent 回答")
    return message


def _serialize_message(message: ChatMessage | None) -> dict[str, Any] | None:
    """消息 → dict。轨迹页要原始消息内容，但不需要 ORM 对象。"""
    if message is None:
        return None
    return {
        "id": str(message.id),
        "role": message.role,
        "content": message.content,
        "route": message.route,
        "tool_calls": message.tool_calls or [],
        "metadata": message.meta or {},
        "created_at": message.created_at.isoformat() if message.created_at else None,
    }


@router.get("/metrics")
async def feedback_metrics(
    repo_id: uuid.UUID = Query(...),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """反馈指标看板。

    两个容易混淆的比率，别用错：
      helpful_rate    = 好评 / 已评价     → 衡量"评价过的里面有多少满意"
      negative_feedback_rate = 差评 / 全部回答 → 衡量"整体输出有多少被吐槽"
    """
    rows = db.query(ChatFeedback).filter(ChatFeedback.repo_id == repo_id).all()
    assistant_messages = db.query(ChatMessage).filter(ChatMessage.repo_id == repo_id, ChatMessage.role == "assistant").count()
    helpful = sum(row.rating == "helpful" for row in rows)
    unhelpful = sum(row.rating == "unhelpful" for row in rows)
    rated = len(rows)

    # 差评原因分布：产品侧据此决定优化方向（检索差 / 答非所问 / 太慢）
    reasons: dict[str, int] = {}
    for row in rows:
        if row.reason:
            reasons[row.reason] = reasons.get(row.reason, 0) + 1

    return {
        "repo_id": str(repo_id),
        "assistant_messages": assistant_messages,
        "rated_messages": rated,
        # 覆盖率低说明"用户懒得点" —— 低于 10% 时上面的好评率参考价值有限
        "feedback_coverage": round(rated / assistant_messages, 4) if assistant_messages else 0.0,
        "helpful": helpful,
        "unhelpful": unhelpful,
        "helpful_rate": round(helpful / rated, 4) if rated else 0.0,
        "unhelpful_rate": round(unhelpful / rated, 4) if rated else 0.0,
        "negative_feedback_rate": round(unhelpful / assistant_messages, 4) if assistant_messages else 0.0,
        # 还没被人工处理掉的差评
        "open_reviews": sum(row.rating == "unhelpful" and row.review_status in {"open", "in_review"} for row in rows),
        "reason_counts": reasons,
    }


@router.get("/trace/{run_id}")
async def feedback_trace(run_id: uuid.UUID, db: Session = Depends(get_db)) -> dict[str, Any]:
    """拉一次运行的完整轨迹：反馈 + 模型运行 + 两条消息 + 会话 transcript。

    设计上要"尽量都能找到"：run 或 feedback 有一个存在就返回，
    所以下面每个来源都做了兜底（找不到就 None，不 404）。
    """
    run = db.get(AgentRun, run_id)
    feedback = db.query(ChatFeedback).filter(ChatFeedback.run_id == run_id).one_or_none()
    if run is None and feedback is None:
        raise HTTPException(status_code=404, detail="未找到对应的执行轨迹")

    # 找 assistant 消息：先按 feedback 记录直接拿 id
    assistant_message = db.get(ChatMessage, feedback.assistant_message_id) if feedback else None
    if assistant_message is None and run is not None:
        # 兜底：没有反馈记录时，在同一个会话里找 meta.run_id 匹配的那条
        candidates = (
            db.query(ChatMessage)
            .filter(ChatMessage.conversation_id == run.conversation_id, ChatMessage.role == "assistant")
            .order_by(ChatMessage.created_at.desc())
            .all()
        )
        assistant_message = next((item for item in candidates if _run_id_from_message(item) == run_id), None)

    # 会话封存记录：里面有工具调用、记忆更新等"过程数据"
    session = (
        db.query(ChatSession).filter(ChatSession.assistant_message_id == assistant_message.id).one_or_none()
        if assistant_message
        else None
    )
    user_message = db.get(ChatMessage, session.user_message_id) if session and session.user_message_id else None
    # repo / conversation 从 feedback 或 run 任一处取（两者都有这两个字段）
    repo_id = feedback.repo_id if feedback else run.repo_id if run else None
    conversation_id = feedback.conversation_id if feedback else run.conversation_id if run else None
    repo = db.get(Repository, repo_id) if repo_id else None
    conversation = db.get(Conversation, conversation_id) if conversation_id else None
    transcript = session.transcript_json if session else {}

    return {
        "trace_id": str(run_id),
        "repository": {"id": str(repo.id), "full_name": repo.full_name} if repo else None,
        "conversation": {"id": str(conversation.id), "title": conversation.title} if conversation else None,
        "feedback": ChatFeedbackResponse.model_validate(feedback).model_dump(mode="json") if feedback else None,
        "run": {
            "id": str(run.id),
            "status": run.status,
            "route": run.route,
            "user_message": run.user_message,
            "final_answer": run.final_answer,
            "tool_calls": run.tool_calls or [],
            "created_at": run.created_at.isoformat() if run.created_at else None,
        }
        if run
        else None,
        "messages": {
            "user": _serialize_message(user_message),
            "assistant": _serialize_message(assistant_message),
        },
        "session": {
            "id": str(session.id),
            "status": session.status,
            "digest": session.digest,
            "agent_events": transcript.get("agent_events", []),
            "tool_use_results": transcript.get("tool_use_results", []),
            "memory_update": transcript.get("memory_update", {}),
            "sealed_at": session.sealed_at.isoformat() if session.sealed_at else None,
        }
        if session
        else None,
    }


@router.get("", response_model=list[ChatFeedbackResponse])
async def list_feedback(
    repo_id: uuid.UUID = Query(...),
    conversation_id: uuid.UUID | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[ChatFeedback]:
    """列反馈，新的在前。"""
    query = db.query(ChatFeedback).filter(ChatFeedback.repo_id == repo_id)
    if conversation_id:
        query = query.filter(ChatFeedback.conversation_id == conversation_id)
    return query.order_by(ChatFeedback.created_at.desc()).limit(limit).all()


@router.post("", response_model=ChatFeedbackResponse)
async def create_feedback(payload: ChatFeedbackCreate, db: Session = Depends(get_db)) -> ChatFeedback:
    """提交反馈（同一条消息重复提交 = 更新）。"""
    # 三重归属校验：repo 存在、conversation 存在、且 conversation 属于该 repo
    repo = db.get(Repository, payload.repo_id)
    conversation = db.get(Conversation, payload.conversation_id)
    if repo is None or conversation is None or conversation.repo_id != repo.id:
        raise HTTPException(status_code=404, detail="未找到对应的项目会话")

    message = _message_or_404(db, payload)
    run_id = _run_id_from_message(message)
    run = db.get(AgentRun, run_id) if run_id else None
    reporter = ensure_demo_user(db)

    # 一条消息只留一条反馈记录，重复提交走更新
    feedback = db.query(ChatFeedback).filter(ChatFeedback.assistant_message_id == message.id).one_or_none()

    # 是否要发通知：必须是差评，且（之前没有 / 之前不是差评 / 还没通知成功过）
    notify = payload.rating == "unhelpful" and (feedback is None or feedback.rating != "unhelpful" or feedback.notified_at is None)

    if feedback is None:
        feedback = ChatFeedback(
            repo_id=repo.id,
            conversation_id=conversation.id,
            assistant_message_id=message.id,
            run_id=run_id,
            user_id=reporter.id,
            rating=payload.rating,
        )
        db.add(feedback)

    previous_rating = feedback.rating
    feedback.rating = payload.rating
    # reason 只在差评时有意义，改回好评要清掉，否则指标会统计到过期的差评原因
    feedback.reason = payload.reason if payload.rating == "unhelpful" else None
    feedback.comment = (payload.comment or "").strip() or None
    feedback.run_id = run_id
    feedback.user_id = reporter.id

    if payload.rating == "unhelpful":
        if previous_rating != "unhelpful":
            # 首次变差评：开一条待处理工单
            feedback.review_status = "open"
            feedback.review_note = None
            feedback.reviewed_at = None
        if notify:
            feedback.notification_status = "pending"
            feedback.notification_error = None
    else:
        # 改回好评：差评工单自动关闭，不再需要通知
        feedback.review_status = "resolved"
        feedback.review_note = None
        feedback.reviewed_at = datetime.now(timezone.utc)
        feedback.notification_status = "not_applicable"
        feedback.notification_error = None

    # 先提交反馈本身，再发通知 —— 通知失败不能回滚掉用户的反馈
    db.commit()
    db.refresh(feedback)

    if notify:
        notification_status, notification_error = await send_negative_feedback_notification(
            feedback=feedback,
            repo=repo,
            message=message,
            run=run,
            reporter=reporter,
        )
        feedback.notification_status = notification_status
        feedback.notification_error = notification_error
        # 只有成功才记 notified_at，失败的话下次还允许重试
        feedback.notified_at = datetime.now(timezone.utc) if notification_status == "sent" else None
        db.commit()
        db.refresh(feedback)
    return feedback


@router.patch("/{feedback_id}", response_model=ChatFeedbackResponse)
async def review_feedback(
    feedback_id: uuid.UUID,
    payload: ChatFeedbackReview,
    db: Session = Depends(get_db),
) -> ChatFeedback:
    """人工处理差评：标记已解决 / 已忽略，并写处理备注。"""
    feedback = db.get(ChatFeedback, feedback_id)
    if feedback is None:
        raise HTTPException(status_code=404, detail="未找到反馈记录")
    feedback.review_status = payload.review_status
    feedback.review_note = (payload.review_note or "").strip() or None
    # 只有终态才记处理时间；in_review 表示还在看，不算处理完
    feedback.reviewed_at = datetime.now(timezone.utc) if payload.review_status in {"resolved", "dismissed"} else None
    db.commit()
    db.refresh(feedback)
    return feedback
