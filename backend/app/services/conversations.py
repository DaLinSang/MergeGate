"""
会话管理（多会话 + 级联删除 + 数据回填）
=====================================

【业务定位】
    一个仓库下可以有多个对话。本文件管四件事：
        1. 保证"至少有一个会话"（没选会话时的默认落点）
        2. 创建 / 列表 / 删除会话
        3. 删除会话时**级联清理**所有关联数据（11 张表）
        4. 回填历史数据（老数据的 conversation_id 是 NULL）

【数据流向】
    上游（谁调它）：
        api/routes/conversations.py   会话列表 / 新建 / 删除
        api/routes/chat.py            每轮对话前 ensure_conversation
        app/main.py                   启动时 backfill_all_default_conversations
    下游（它写谁）：
        conversations 表（主）
        以及删除时涉及的 11 张关联表（见 delete_conversation）

【核心规则】
    1. **必须有默认会话**：用户不选会话时，对话要有地方落。
       所以 ensure_default_conversation 会被到处调用。
    2. **删除是软删除**（status="deleted"），但关联数据是**硬删除**。
       为什么：会话记录本身留着（便于审计"这个会话被删过"），
       但消息/记忆/证据这些"内容"必须真删（否则越积越多，且用户以为删干净了）。
    3. **删除后必须返回一个替代会话**：前端删完要跳转，不能返回空。

【本文件函数地图】
    ensure_default_conversation()  ★ 保证有默认会话（被调用最频繁）
    ensure_conversation()          按 ID 取，取不到就退到默认
    list_conversations()           列表（顺手做回填）
    create_conversation()          新建（自动起名）
    delete_conversation()          ★ 级联删除（11 张表）
    touch_conversation()           更新活跃时间 + 自动起标题
    backfill_all_default_conversations()  全库回填（启动时）
    _backfill_repo_conversation()  单仓库回填
    _default_title()               生成"新会话 N"
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.db.models import (
    AgentRun,
    AgentTaskRun,
    AgentWorkflowRun,
    ChatFeedback,
    ChatMessage,
    ChatSession,
    Conversation,
    ConversationMemory,
    EvidenceItem,
    MemoryCandidate,
    RecallEvent,
    Repository,
)


def _default_title(db: Session, repo_id: uuid.UUID) -> str:
    """生成默认会话标题（"默认会话" 或 "新会话 N"）。

    【调用链】被 create_conversation() 调用

    【为什么用 count 而不是 max(number)】
        conversations 表没有"序号"字段。用现有数量 + 1 作为新会话编号，
        简单够用（重名不影响功能，只是显示）。
    """
    existing = db.query(Conversation).filter(Conversation.repo_id == repo_id).count()
    return "默认会话" if existing == 0 else f"新会话 {existing + 1}"


def ensure_default_conversation(db: Session, repo_id: uuid.UUID) -> Conversation:
    """保证仓库至少有一个活跃会话，返回它（★ 被调用最频繁的函数）。

    【调用链】被 ensure_conversation() / list_conversations() / delete_conversation() /
              _backfill_repo_conversation() 调用
    【被谁调用】api/routes/chat.py（每轮对话前）、api/routes/conversations.py

    【数据流向】
        入：db、repo_id
        出：Conversation（已 flush，有 id）

    【为什么取 created_at 最早的】
        "默认会话"应该是最早创建的那个 —— 它承载了"这个仓库最初的所有对话"。
        取最新的会导致每次重建默认会话都换一个，历史对话就散了。

    【为什么只 flush 不 commit】
        调用方通常在业务事务中间。commit 由外层控制。
    """
    conversation = (
        db.query(Conversation)
        .filter(Conversation.repo_id == repo_id, Conversation.status == "active")
        .order_by(Conversation.created_at.asc())
        .first()
    )
    if conversation:
        return conversation

    conversation = Conversation(repo_id=repo_id, title="默认会话", status="active")
    db.add(conversation)
    db.flush()
    return conversation


def ensure_conversation(db: Session, repo_id: uuid.UUID, conversation_id: uuid.UUID | None = None) -> Conversation:
    """按 ID 取会话；取不到（不存在 / 已删 / 跨仓库）就退到默认会话。

    【调用链】ensure_default_conversation() → 本函数
    【被谁调用】api/routes/chat.py

    【数据流向】
        入：repo_id、conversation_id（可空）
        出：Conversation（一定是有效的）

    【★ 为什么过滤条件里要带 repo_id】
        防止跨仓库访问：A 仓库的会话 ID 不能用来在 B 仓库里对话。
        这是多租户系统最基本的一条隔离规则。

    【为什么"取不到"不报错而是退到默认】
        前端可能缓存了已删除的会话 ID。
        报错会让用户卡住；退到默认会话则能继续用。
    """
    if conversation_id:
        conversation = (
            db.query(Conversation)
            .filter(Conversation.id == conversation_id, Conversation.repo_id == repo_id, Conversation.status == "active")
            .one_or_none()
        )
        if conversation:
            return conversation
    return ensure_default_conversation(db, repo_id)


def list_conversations(db: Session, repo_id: uuid.UUID) -> list[Conversation]:
    """列出一个仓库的所有活跃会话（按最近活跃排序）。

    【调用链】ensure_default_conversation() / _backfill_repo_conversation() → 本函数
    【被谁调用】api/routes/conversations.py（侧边栏会话列表）

    【为什么列表前要先"确保默认会话 + 回填"】
        老数据（会话功能上线前的消息）conversation_id 是 NULL，
        不挂到某个会话下就永远看不到。
        所以每次列表都顺手回填一次（成本很低，幂等）。

    【排序：updated_at 降序，created_at 升序】
        最近活跃的在前；同一时间创建的按创建顺序（保证稳定）。
    """
    ensure_default_conversation(db, repo_id)
    _backfill_repo_conversation(db, repo_id)
    db.flush()
    return (
        db.query(Conversation)
        .filter(Conversation.repo_id == repo_id, Conversation.status == "active")
        .order_by(Conversation.updated_at.desc(), Conversation.created_at.asc())
        .all()
    )


def create_conversation(db: Session, repo_id: uuid.UUID, title: str | None = None) -> Conversation:
    """新建会话（标题为空就自动起名）。

    【调用链】_default_title() → 本函数
    【被谁调用】api/routes/conversations.py、delete_conversation()（删完没替代时）

    【(title or "").strip() or ... 这个写法】
        先转字符串去空格；如果是空串（假值），就走 or 右边自动起名。
        一行代码处理"None / 空串 / 纯空格"三种情况。
    """
    conversation = Conversation(repo_id=repo_id, title=(title or "").strip() or _default_title(db, repo_id))
    db.add(conversation)
    db.flush()
    return conversation


def delete_conversation(db: Session, repo_id: uuid.UUID, conversation_id: uuid.UUID) -> Conversation:
    """删除会话，并级联清理所有关联数据（★ 本文件最复杂的函数）。

    【调用链】ensure_default_conversation() / create_conversation() → 本函数
    【被谁调用】api/routes/conversations.py（DELETE 接口）

    【数据流向】
        入：repo_id、conversation_id
        出：替代会话（删完后前端该跳转到哪个）

    【★ 级联删除的 11 张表，顺序有讲究】
        必须先删"子表"，再删"父表"，否则外键约束会报错：
            ① ChatFeedback            （引用 chat_messages / agent_runs）
            ② MemoryCandidate         （引用 chat_sessions）—— 按 conversation_id
            ③ MemoryCandidate         （按 session_id，覆盖"只关联 session 不关联 conversation"的情况）
            ④ RecallEvent             （引用 conversations）
            ⑤ AgentTaskRun            （引用 agent_workflow_runs）← 子表先删
            ⑥ AgentWorkflowRun        （父表后删）
            ⑦ ChatMessage
            ⑧ ChatSession
            ⑨ AgentRun
            ⑩ ConversationMemory
            ⑪ EvidenceItem
        最后才软删 conversation 本身。

    【★ 为什么要先查 session_ids / workflow_ids 再删】
        MemoryCandidate 和 AgentTaskRun 没有 conversation_id 字段，
        只能通过 session_id / workflow_id 间接定位。
        而这些 ID 在删掉 ChatSession / AgentWorkflowRun 之后就查不到了，
        所以必须【先收集 ID，再删父表】。

    【★ 为什么 conversation 只是软删（status="deleted"）】
        留一条记录便于审计（"这个会话存在过、被删了"）。
        但内容（消息/记忆/证据）是真删 —— 用户点删除就是不想让内容再存在。

    【★ 为什么删完要返回替代会话】
        前端删完必须跳到某个会话。如果返回空，前端会白屏。
        逻辑：取剩下最近活跃的；一个都没有就新建一个"默认会话"。
    """
    conversation = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.repo_id == repo_id, Conversation.status == "active")
        .one_or_none()
    )
    # 会话不存在（可能已被删）→ 直接给默认会话，不报错
    if conversation is None:
        return ensure_default_conversation(db, repo_id)

    # ── 步骤 1：先收集"间接关联"的 ID（删父表后就查不到了）──
    session_ids = [
        item.id
        for item in db.query(ChatSession.id).filter(ChatSession.conversation_id == conversation.id).all()
    ]
    workflow_ids = [
        item.id
        for item in db.query(AgentWorkflowRun.id).filter(AgentWorkflowRun.conversation_id == conversation.id).all()
    ]

    # ── 步骤 2：按"子表 → 父表"顺序删除 ──
    db.query(ChatFeedback).filter(ChatFeedback.conversation_id == conversation.id).delete(synchronize_session=False)
    db.query(MemoryCandidate).filter(MemoryCandidate.conversation_id == conversation.id).delete(synchronize_session=False)
    if session_ids:
        # 兜住"只挂了 session_id、没挂 conversation_id"的候选
        db.query(MemoryCandidate).filter(MemoryCandidate.session_id.in_(session_ids)).delete(synchronize_session=False)
    db.query(RecallEvent).filter(RecallEvent.conversation_id == conversation.id).delete(synchronize_session=False)
    if workflow_ids:
        # AgentTaskRun 是 AgentWorkflowRun 的子表，必须先删
        db.query(AgentTaskRun).filter(AgentTaskRun.workflow_id.in_(workflow_ids)).delete(synchronize_session=False)
    db.query(AgentWorkflowRun).filter(AgentWorkflowRun.conversation_id == conversation.id).delete(synchronize_session=False)
    db.query(ChatMessage).filter(ChatMessage.conversation_id == conversation.id).delete(synchronize_session=False)
    db.query(ChatSession).filter(ChatSession.conversation_id == conversation.id).delete(synchronize_session=False)
    db.query(AgentRun).filter(AgentRun.conversation_id == conversation.id).delete(synchronize_session=False)
    db.query(ConversationMemory).filter(ConversationMemory.conversation_id == conversation.id).delete(synchronize_session=False)
    db.query(EvidenceItem).filter(EvidenceItem.conversation_id == conversation.id).delete(synchronize_session=False)

    # ── 步骤 3：软删会话本身 ──
    conversation.status = "deleted"
    conversation.updated_at = datetime.now(timezone.utc)
    db.flush()

    # ── 步骤 4：找一个替代会话给前端 ──
    replacement = (
        db.query(Conversation)
        .filter(Conversation.repo_id == repo_id, Conversation.status == "active")
        .order_by(Conversation.updated_at.desc(), Conversation.created_at.asc())
        .first()
    )
    return replacement or create_conversation(db, repo_id, "默认会话")


def touch_conversation(
    db: Session,
    conversation: Conversation,
    *,
    user_message: str | None = None,
    increment: int = 0,
) -> None:
    """更新会话的活跃时间 / 消息数，并在首轮自动用问题当标题。

    【调用链】被 api/routes/chat.py 调用（每轮对话后）

    【数据流向】
        入：conversation、user_message（本轮用户问题）、increment（消息数增量，通常 2）
        出：无（原地修改）

    【★ 自动起标题的三个条件，缺一不可】
        ① 有 user_message（不是空）
        ② message_count == 0（这是第一轮）
        ③ 标题还是默认的（"默认会话" / "新会话 N"）
        为什么：用户手动改过标题就不能覆盖；非首轮也不该改。

    【标题截断到 40 字符 + 换行换空格】
        侧边栏宽度有限。换行会让标题变成两行，很难看。

    【为什么用 flush 不用 commit】
        和其他业务写操作在同一事务里，由外层统一提交。
    """
    # 首轮 → 用问题当标题
    if user_message and conversation.message_count == 0 and conversation.title.startswith(("默认会话", "新会话")):
        title = user_message.strip().replace("\n", " ")
        if title:
            conversation.title = title[:40]

    conversation.message_count = (conversation.message_count or 0) + increment
    conversation.updated_at = datetime.now(timezone.utc)
    db.flush()


def backfill_all_default_conversations(db: Session) -> None:
    """给所有仓库回填默认会话（启动时跑一次）。

    【调用链】ensure_default_conversation() / _backfill_repo_conversation() → 本函数
    【被谁调用】app/main.py 的 startup 事件

    【为什么需要回填】
        conversations 表是后加的。老数据的 chat_messages / chat_sessions / agent_runs
        的 conversation_id 是 NULL —— 在会话列表里看不到，等于"历史丢失"。
        启动时统一挂到默认会话下，历史就回来了。

    【为什么这里要 commit】
        它是独立任务（不在请求事务里）。不 commit 就等于没做。
    """
    for repo in db.query(Repository).all():
        ensure_default_conversation(db, repo.id)
        _backfill_repo_conversation(db, repo.id)
    db.commit()


def _backfill_repo_conversation(db: Session, repo_id: uuid.UUID) -> None:
    """把某个仓库里 conversation_id 为 NULL 的历史数据挂到默认会话。

    【调用链】ensure_default_conversation() → 本函数
    【被谁调用】list_conversations() / backfill_all_default_conversations()

    【三张表都要回填】
        ChatMessage   消息
        ChatSession   封存记录
        AgentRun      运行记录
        少了任何一张，前端对应页面就是空的。

    【★ 为什么用 update() 而不是逐条改】
        可能有几千条。update 是一条 SQL 批量改，逐条改要几千次往返。

    【为什么最后要重算 message_count】
        回填后消息数变了。不重算的话侧边栏显示的数字是旧的（甚至 0）。
        用 count() 重算最准确（不依赖增量累加，避免历史累加错误）。
    """
    conversation = ensure_default_conversation(db, repo_id)
    for model in [ChatMessage, ChatSession, AgentRun]:
        db.query(model).filter(model.repo_id == repo_id, model.conversation_id.is_(None)).update(
            {model.conversation_id: conversation.id},
            synchronize_session=False,
        )
    conversation.message_count = db.query(ChatMessage).filter(ChatMessage.conversation_id == conversation.id).count()
