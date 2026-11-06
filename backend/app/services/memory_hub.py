"""
记忆中枢（召回日志 + 记忆候选）
==============================

【业务定位】
    Agent 的记忆有两条线，本文件管的是"可控"的那一条：
        自动写入（thread_memories / conversation_memories）→ 由 chat_memory.py 负责
        人工审核（memory_candidates）                      → 由本文件负责
    外加一个"观测"职责：记录每次召回事件，方便排查"Agent 为什么没找到答案"。

    为什么记忆要人工审核：
        Agent 从对话里提炼的"事实"可能是幻觉。如果直接进 RAG，
        错误知识会被反复检索、反复引用，越用越脏。所以必须有一道人工闸门。

【数据流向】
    上游（谁调它）：
        api/routes/chat.py        → record_recall_event()    每次工具召回后记一笔
        api/routes/knowledge.py   → memory_status() / list_* / approve_* / reject_*
        services/chat_memory.py   → capture_memory_candidates() 每轮结束提炼候选
                                    record_recall_event()       记忆工具召回后记一笔
    下游（它写谁 / 依赖谁）：
        recall_events 表          ← record_recall_event()
        memory_candidates 表      ← capture_memory_candidates()
        documents 表 + Milvus     ← approve_memory_candidate() 经 rag/vector_store.add_document()
        project_index_states 表   ← memory_status() 读
        rag/vector_store.milvus_runtime_status() ← memory_status() 读向量库健康状态

【核心规则】
    1. 没有 repo_id 就不产生候选（记忆必须归属某个仓库，否则无法隔离）。
    2. 候选去重：同 repo + 同 kind + 同内容，且状态是 pending/approved → 不再重复插入。
    3. 只有 status=pending 的候选能被批准；已批准的直接返回，不重复建文档。
    4. 批准后候选的 meta 里写回 approved_document_id —— 保证"这条记忆从哪来"可追溯。

【本文件函数地图】（按业务顺序读）
    ① 观测       record_recall_event() → list_recall_events()
    ② 总览       memory_status()        （前端记忆面板的唯一数据源）
    ③ 提炼入库   capture_memory_candidates() → _candidate_exists() → _candidate_title()
    ④ 人工审核   list_memory_candidates() → approve_memory_candidate() / reject_memory_candidate()
    ⑤ 序列化     _recall_result_payload / _recall_event_payload / _candidate_payload
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import (
    Document,
    EvidenceItem,
    MemoryCandidate,
    ProjectIndexState,
    RecallEvent,
    Repository,
)
from app.services.rag.vector_store import add_document, milvus_runtime_status


# ════════════════════════════════════════════════════════════════
# ① 观测：召回事件
# ════════════════════════════════════════════════════════════════

def record_recall_event(
    db: Session,
    *,
    repo_id: uuid.UUID | None,
    conversation_id: uuid.UUID | None,
    tool_name: str,
    query: str,
    results: list[dict[str, Any]] | None = None,
    citations: list[dict[str, Any]] | None = None,
    scope: str = "project",
    mode: str = "hybrid",
    status: str = "success",
    metadata: dict[str, Any] | None = None,
) -> RecallEvent:
    """记一条"Agent 调了某个记忆/检索工具"的日志。

    【调用链】_recall_result_payload() → 本函数 → db.add/flush
    【被谁调用】
        api/routes/chat.py      → 每轮对话结束后汇总记录
        services/chat_memory.py → 记忆工具（recall）调用后记录

    【数据流向】
        入：repo_id / conversation_id（归属）、tool_name（哪个工具）、query（查了什么）、
            results（召回结果，只留前 12 条）、citations（引用）、
            scope（检索范围）、mode（检索方式）、status（成功/空/报错）
        出：RecallEvent（已 flush，拿到自增主键，未 commit —— 由调用方的事务统一提交）

    【为什么要截断到 12 条】
        日志是"排查线索"不是"完整数据"。存全量会让表迅速膨胀，
        而且真正的证据已经在 documents / evidence_items 里了。

    【为什么只 flush 不 commit】
        本函数通常在业务事务中间调用，commit 由外层控制 ——
        否则日志提交了、业务回滚了，会出现"有日志没结果"的假象。
    """
    # 结果做"瘦身 + 限流"：只保留关键字段，最多 12 条
    safe_results = [_recall_result_payload(item) for item in (results or [])[:12]]

    event = RecallEvent(
        repo_id=repo_id,
        conversation_id=conversation_id,
        tool_name=tool_name,
        query=query.strip(),
        scope=scope,
        mode=mode,
        status=status,
        # 注意：这里存的是【原始】结果条数，不是截断后的 —— 排查时要知道真实召回量
        result_count=len(results or []),
        results_json=safe_results,
        citations_json=(citations or [])[:12],
        meta=metadata or {},
    )
    db.add(event)
    db.flush()
    return event


def list_recall_events(
    db: Session,
    *,
    repo_id: uuid.UUID,
    conversation_id: uuid.UUID | None = None,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """查召回日志（按时间倒序）。

    【调用链】_recall_event_payload() → 本函数
    【被谁调用】api/routes/knowledge.py（前端"记忆面板 → 召回记录"）

    【数据流向】
        入：repo_id（必传，仓库隔离）、conversation_id（可选，只看某会话）、limit
        出：dict 列表（已序列化，可直接返回给前端）

    limit 被夹在 1~100：防止前端传 999999 把数据库拖死。
    """
    query = db.query(RecallEvent).filter(RecallEvent.repo_id == repo_id)
    # 可选：只看某个会话的召回
    if conversation_id:
        query = query.filter(RecallEvent.conversation_id == conversation_id)
    rows = query.order_by(RecallEvent.created_at.desc()).limit(min(max(limit, 1), 100)).all()
    return [_recall_event_payload(row) for row in rows]


# ════════════════════════════════════════════════════════════════
# ② 总览：记忆面板的一次性统计
# ════════════════════════════════════════════════════════════════

def memory_status(db: Session, repo: Repository) -> dict[str, Any]:
    """汇总一个仓库的"记忆与知识"状态（前端记忆面板的唯一数据源）。

    【调用链】追这个函数会进入：
        1. ProjectIndexState 查询            → 项目文档扫描状态
        2. Document.group_by(source_type)    → 知识库各类型条数
        3. EvidenceItem.group_by(source_type)→ 证据库各类型条数
        4. MemoryCandidate.group_by(status)  → 候选审核状态分布
        5. RecallEvent 查询                  → 总召回次数 + 最近一次时间
        6. milvus_runtime_status()           → 向量库健康（rag/vector_store.py）

    【被谁调用】api/routes/knowledge.py（GET /knowledge/memory-status）

    【数据流向】
        入：db、repo（当前仓库对象）
        出：嵌套 dict → 前端一次性渲染，避免前端发 5 个请求

    【为什么合并成一个接口】
        前端面板要同时展示"索引 / 文档 / 证据 / 召回 / 候选 / 向量库"六块。
        分 6 次请求会有 6 次往返，且数据可能不同步。合并后一次查完，快照一致。
    """
    # ── 1. 项目文档扫描状态（一仓库一行）──
    index_state = db.query(ProjectIndexState).filter(ProjectIndexState.repo_id == repo.id).one_or_none()

    # ── 2~4. 三个分组统计（group_by 一次查完，避免 N 次 count）──
    source_counts = dict(
        db.query(Document.source_type, func.count(Document.id))
        .filter(Document.repo_id == repo.id)
        .group_by(Document.source_type)
        .all()
    )
    evidence_counts = dict(
        db.query(EvidenceItem.source_type, func.count(EvidenceItem.id))
        .filter(EvidenceItem.repo_id == repo.id)
        .group_by(EvidenceItem.source_type)
        .all()
    )
    candidate_counts = dict(
        db.query(MemoryCandidate.status, func.count(MemoryCandidate.id))
        .filter(MemoryCandidate.repo_id == repo.id)
        .group_by(MemoryCandidate.status)
        .all()
    )

    # ── 5. 最近一次召回（用于显示"上次活跃时间"）──
    latest_recall = (
        db.query(RecallEvent)
        .filter(RecallEvent.repo_id == repo.id)
        .order_by(RecallEvent.created_at.desc())
        .first()
    )

    return {
        "repo_id": repo.id,
        # 索引状态：没扫过就返回 missing + 0，前端不用判空
        "index": {
            "status": index_state.status if index_state else "missing",
            "docs_indexed": index_state.docs_indexed if index_state else 0,
            "docs_total": index_state.docs_total if index_state else 0,
            "last_scan_at": index_state.last_scan_at.isoformat() if index_state and index_state.last_scan_at else None,
            "error_message": index_state.error_message if index_state else None,
            "summary_json": index_state.summary_json if index_state else None,
        },
        # 知识库文档：总数 + 按来源类型分布
        "documents": {
            "total": sum(int(value) for value in source_counts.values()),
            "by_source_type": source_counts,
        },
        # 证据库：同上
        "evidence": {
            "total": sum(int(value) for value in evidence_counts.values()),
            "by_source_type": evidence_counts,
        },
        # 召回：总次数 + 最近时间
        "recall": {
            "total": db.query(RecallEvent).filter(RecallEvent.repo_id == repo.id).count(),
            "latest_at": latest_recall.created_at.isoformat() if latest_recall and latest_recall.created_at else None,
        },
        # 记忆候选：总数 + 状态分布 + 待审核数（前端红点用 pending）
        "candidates": {
            "total": sum(int(value) for value in candidate_counts.values()),
            "by_status": candidate_counts,
            "pending": int(candidate_counts.get("pending") or 0),
        },
        # 向量库健康：直接透传 milvus 的运行状态
        "vector": milvus_runtime_status(),
    }


# ════════════════════════════════════════════════════════════════
# ③ 提炼入库：把"本轮新记忆"变成待审核候选
# ════════════════════════════════════════════════════════════════

def capture_memory_candidates(
    db: Session,
    *,
    repo_id: uuid.UUID | None,
    conversation_id: uuid.UUID | None,
    session_id: uuid.UUID | None,
    memory_update: dict[str, Any],
) -> list[MemoryCandidate]:
    """把一轮对话提炼出的记忆，转成"待审核候选"入库。

    【调用链】追这个函数会进入：
        1. _candidate_exists()  → 本文件，去重（同 repo+kind+content 且未拒绝过就跳过）
        2. _candidate_title()   → 本文件，生成"决策：xxx"这样的标题
        3. db.add / db.flush

    【被谁调用】services/chat_memory.py::SessionSealer（每轮对话结束后）

    【数据流向】
        入：repo_id（必填）、conversation_id / session_id（溯源用）、
            memory_update（structured_memory_from_model 或 fallback_memory_update 的产物）
        出：新建的 MemoryCandidate 列表（status 全部是 pending）

    【关键规则】
        1. repo_id 为 None 直接返回空 —— 记忆必须归属仓库，否则无法做隔离。
        2. 五种 kind 分别处理：decision / fact / task / preference / repo_context
        3. summary 不产生候选 —— 摘要太泛，不适合作为"可引用的知识"
        4. 去重条件包含 status in (pending, approved)：
           已经拒绝过的（rejected）允许再次提出（可能上次是误拒）。
    """
    # 没有仓库归属，不产生任何候选
    if repo_id is None:
        return []

    # kind → 该 kind 对应的内容列表
    specs = [
        ("decision", memory_update.get("decisions") or []),
        ("fact", memory_update.get("facts") or []),
        ("task", memory_update.get("tasks") or []),
        ("preference", memory_update.get("user_preferences") or []),
        ("repo_context", memory_update.get("repo_context") or []),
    ]

    created: list[MemoryCandidate] = []

    for kind, values in specs:
        for value in values:
            content = str(value or "").strip()
            # 空内容 / 已存在 → 跳过
            if not content or _candidate_exists(db, repo_id, kind, content):
                continue

            candidate = MemoryCandidate(
                repo_id=repo_id,
                conversation_id=conversation_id,
                session_id=session_id,
                kind=kind,
                title=_candidate_title(kind, content),
                content=content,
                status="pending",          # 一律从 pending 开始，必须人工审核
                source="session_sealer",   # 标记来源：会话封存器自动提炼
                meta={
                    "memory_update_key": kind,                                    # 来自哪个字段
                    "captured_at": datetime.now(timezone.utc).isoformat(),        # 抓取时间（UTC）
                },
            )
            db.add(candidate)
            created.append(candidate)

    # 有新增才 flush（避免空事务）
    if created:
        db.flush()
    return created


def list_memory_candidates(
    db: Session,
    *,
    repo_id: uuid.UUID,
    status: str | None = "pending",
    limit: int = 50,
) -> list[dict[str, Any]]:
    """列出记忆候选（默认只看待审核的）。

    【调用链】_candidate_payload() → 本函数
    【被谁调用】api/routes/knowledge.py（前端候选列表）

    status 的两个特殊值：
        None  → 不加状态过滤（等价于全部）
        "all" → 同上（前端用 "all" 更直观，所以两种都支持）
    """
    query = db.query(MemoryCandidate).filter(MemoryCandidate.repo_id == repo_id)
    if status and status != "all":
        query = query.filter(MemoryCandidate.status == status)
    rows = query.order_by(MemoryCandidate.created_at.desc()).limit(min(max(limit, 1), 100)).all()
    return [_candidate_payload(row) for row in rows]


# ════════════════════════════════════════════════════════════════
# ④ 人工审核：批准（晋升为知识）/ 拒绝
# ════════════════════════════════════════════════════════════════

def approve_memory_candidate(db: Session, repo: Repository, candidate_id: uuid.UUID) -> tuple[MemoryCandidate, Document | None]:
    """批准一条记忆候选 → 晋升为 Document 进 RAG。

    【调用链】追这个函数会进入：
        1. _candidate_or_none()                     → 本文件，取候选（带仓库隔离）
        2. rag/vector_store.add_document()          → 写 documents 表 + Milvus 向量
           └ 内部还会走 embeddings.embed_text() 和 milvus 插入
        3. db.flush()

    【被谁调用】api/routes/knowledge.py（POST /knowledge/memory-candidates/{id}/approve）

    【数据流向】
        入：db、repo（当前仓库）、candidate_id
        出：(candidate, doc)
            candidate.status 变 approved，meta 里写回 approved_document_id
            doc = 新建的 Document（source_type="memory_note"），同时已写入 Milvus

    【幂等设计】
        已批准的直接返回 (candidate, None) —— 不重复建文档。
        为什么要幂等：前端可能重复点击，或者网络重试导致请求发两次。

    【为什么 add_document 传 commit=False】
        这里要和"改候选状态"放在同一个事务里。
        如果文档写成功、候选状态没改，会出现"知识已进库但候选还是 pending"的脏状态。
    """
    candidate = _candidate_or_none(db, repo.id, candidate_id)
    if candidate is None:
        raise ValueError("candidate_not_found")

    # 已批准 → 幂等返回，不再建文档
    if candidate.status == "approved":
        return candidate, None

    # ── 晋升为知识文档（source_type=memory_note，区别于上传文件/Issue）──
    doc = add_document(
        db,
        repo.id,
        "memory_note",
        uuid.uuid4(),                 # 新生成的 source_id（候选本身没有对应业务对象）
        candidate.title,
        candidate.content,
        {
            "display_title": candidate.title,
            "candidate_id": str(candidate.id),
            "candidate_kind": candidate.kind,
            "source": "approved_memory_candidate",
            # 把候选原有 meta 一并带过去（保留 captured_at 等溯源信息）
            **(candidate.meta or {}),
        },
        commit=False,
    )

    # ── 更新候选状态（与上面同一个事务）──
    candidate.status = "approved"
    candidate.reviewed_at = datetime.now(timezone.utc)
    # 写回文档 ID：以后能从这条候选反查它变成了哪条知识
    candidate.meta = {**(candidate.meta or {}), "approved_document_id": str(doc.id)}
    db.flush()
    return candidate, doc


def reject_memory_candidate(db: Session, repo: Repository, candidate_id: uuid.UUID) -> MemoryCandidate:
    """拒绝一条记忆候选（不建文档，只改状态）。

    【调用链】_candidate_or_none() → 本函数
    【被谁调用】api/routes/knowledge.py

    【数据流向】
        入：db、repo、candidate_id
        出：candidate（status=rejected，reviewed_at 已填）

    为什么保留而不是删除：
        拒绝记录本身有价值 —— 下次提炼出同样内容时，_candidate_exists 会放行（因为只挡 pending/approved），
        但如果要分析"哪类记忆总被拒"，数据还在。
    """
    candidate = _candidate_or_none(db, repo.id, candidate_id)
    if candidate is None:
        raise ValueError("candidate_not_found")

    candidate.status = "rejected"
    candidate.reviewed_at = datetime.now(timezone.utc)
    db.flush()
    return candidate


# ════════════════════════════════════════════════════════════════
# ⑤ 内部工具：查询 / 去重 / 标题 / 序列化
# ════════════════════════════════════════════════════════════════

def _candidate_or_none(db: Session, repo_id: uuid.UUID, candidate_id: uuid.UUID) -> MemoryCandidate | None:
    """按 (repo_id, candidate_id) 取候选 —— 必须带 repo_id 做隔离。

    【调用链】被 approve_memory_candidate() / reject_memory_candidate() 调用

    为什么要带 repo_id：
        如果只按 candidate_id 查，拿到别的仓库的候选就能跨仓库批准。
        这是多租户系统最常见的越权漏洞。
    """
    return (
        db.query(MemoryCandidate)
        .filter(MemoryCandidate.repo_id == repo_id, MemoryCandidate.id == candidate_id)
        .one_or_none()
    )


def _candidate_exists(db: Session, repo_id: uuid.UUID, kind: str, content: str) -> bool:
    """判断同样的候选是否已经存在（去重）。

    【调用链】被 capture_memory_candidates() 调用

    去重键：repo_id + kind + content（完全相等，不做模糊匹配）
    状态范围：pending / approved —— 这两种状态代表"还在流程里"。

    为什么 rejected 不参与去重：
        上次被拒可能是误判。允许再次提出，让人再判断一次。
    """
    return (
        db.query(MemoryCandidate.id)
        .filter(
            MemoryCandidate.repo_id == repo_id,
            MemoryCandidate.kind == kind,
            MemoryCandidate.content == content,
            MemoryCandidate.status.in_(["pending", "approved"]),
        )
        .first()
        is not None
    )


def _candidate_title(kind: str, content: str) -> str:
    """生成候选标题，形如 "决策：统一用 PostgreSQL 存元数据"。

    【调用链】_clip() → 本函数
    【被谁调用】capture_memory_candidates()

    为什么要加中文前缀：
        前端列表里一眼能区分类型；且这个标题会作为 Document 的 title 参与检索，
        带类型词能提升"决策类问题"的召回率。
    """
    labels = {
        "decision": "决策",
        "fact": "事实",
        "task": "任务",
        "preference": "偏好",
        "repo_context": "项目上下文",
    }
    return f"{labels.get(kind, kind)}：{_clip(content, 80)}"


def _recall_result_payload(item: dict[str, Any]) -> dict[str, Any]:
    """把一条召回结果瘦身成"日志版"（只留关键字段 + 截断片段）。

    【调用链】_clip() → 本函数
    【被谁调用】record_recall_event()

    与原始结果的区别：去掉大字段（完整正文），snippet 截到 260 字符。
    日志要的是"能不能看懂召回对不对"，不是"完整原文"。
    """
    return {
        "title": str(item.get("title") or ""),
        "source_type": str(item.get("source_type") or ""),
        "source_id": str(item.get("source_id") or ""),
        "score": item.get("score"),                      # 保留原始分数（不做格式化，方便对比）
        "snippet": _clip(str(item.get("snippet") or ""), 260),
        "metadata": item.get("metadata") or {},
        "retrieval": item.get("retrieval") or {},        # 检索明细（各阶段分数）
    }


def _recall_event_payload(row: RecallEvent) -> dict[str, Any]:
    """把 RecallEvent ORM 对象转成 API 返回结构。

    【调用链】被 list_recall_events() 调用

    注意 metadata 字段：ORM 属性叫 meta（因为 metadata 是 SQLAlchemy 保留字），
    但 API 对外的字段名是 metadata —— 前端不用感知这个技术细节。
    """
    return {
        "id": row.id,
        "repo_id": row.repo_id,
        "conversation_id": row.conversation_id,
        "tool_name": row.tool_name,
        "query": row.query,
        "scope": row.scope,
        "mode": row.mode,
        "status": row.status,
        "result_count": row.result_count,
        "results": row.results_json or [],
        "citations": row.citations_json or [],
        "metadata": row.meta or {},
        "created_at": row.created_at,
    }


def _candidate_payload(row: MemoryCandidate) -> dict[str, Any]:
    """把 MemoryCandidate ORM 对象转成 API 返回结构。

    【调用链】被 list_memory_candidates() 调用
    """
    return {
        "id": row.id,
        "repo_id": row.repo_id,
        "conversation_id": row.conversation_id,
        "session_id": row.session_id,
        "kind": row.kind,
        "title": row.title,
        "content": row.content,
        "status": row.status,
        "source": row.source,
        "metadata": row.meta or {},
        "created_at": row.created_at,
        "reviewed_at": row.reviewed_at,
    }


def _clip(text: str, limit: int) -> str:
    """按【字符数】截断并加省略号（不是按 token）。

    【调用链】被 _candidate_title() / _recall_result_payload() 调用

    为什么按字符而不是 token：
        这里只用于展示和日志，不需要精确的 token 控制，
        字符数更直观（80 字符 ≈ 一行标题）。
    注意 limit-1：给省略号留 1 个字符位置。
    """
    cleaned = " ".join((text or "").split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "..."
