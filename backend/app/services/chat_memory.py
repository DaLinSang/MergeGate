"""
对话记忆中枢
============

【业务定位】
    整个系统里"记忆"这条线的总控。它管四件事：
        1. 消息存取     —— 对话原文（chat_messages）
        2. 证据库       —— 可关键词检索的轻量摘要（evidence_items）
        3. 上下文组装   —— 把"仓库 + 会话 + 记忆 + 证据 + 最近对话"拼成 prompt
        4. 渐进压缩     —— 快爆窗口时，把旧对话压成摘要（插入压缩边界）
    另外还有"会话封存"（一轮结束后提炼记忆）和"记忆工具"（给 Agent 调用的 MCP 工具）。

    一句话：Agent 能"记住"上下文，全靠这个文件。

【数据流向】
    上游（谁调它）：
        api/routes/chat.py                 → MessageStore.append / ContextAssembler.assemble
                                             / ProgressiveContextManager.ensure_headroom
                                             / SessionSealer.seal
        services/agents/chat_agent.py      → MemoryTools.*（Agent 的工具）
        services/memory_hub.py             → 被本文件调用（候选记忆、召回日志）
    下游（它写谁 / 依赖谁）：
        chat_messages 表        ← MessageStore.append / _insert_boundary
        evidence_items 表       ← EvidenceStore.rebuild_repo
        chat_sessions 表        ← SessionSealer.seal
        conversation_memories / thread_memories 表 ← SessionSealer._update_thread_memory
        services/rag/retrieval  ← EvidenceStore.search（向量召回）
        services/context_compression ← 所有 token 预算与压缩算法

【核心规则】
    1. **两种读法，绝不能混**：
        recent()   = 模型看到的"活动投影"（压缩边界之后 + 保留的消息）
        timeline() = 用户看到的"完整流水"（永不丢，含边界之前的原文）
        —— 压缩只改变模型视野，不删用户历史。这是本文件最重要的设计。
    2. **压缩只插边界，不删消息**：插入一条 role=system 的"边界消息"，
       模型从边界往后读；原文一条不动，随时可以回查。
    3. **任何一步失败都不能影响已经完成的对话**：
       记忆提炼失败 → 用规则兜底；持久化失败 → 只记 warning，session 照样封存。
    4. **三层压缩，从便宜到贵**：
       microcompact（清旧工具结果）→ compress_messages（丢旧消息）→ 插入压缩边界（摘要）

【本文件类地图】（7 个类，按"一轮对话的生命周期"排列）

    ┌─ 第 1 层：存取 ───────────────────────────────────
    │  MessageStore        消息读写（两种读法：recent / timeline）
    │      ├ append()              写一条
    │      ├ recent()              ★ 活动投影（模型看的）
    │      ├ timeline()            ★ 完整流水（用户看的）
    │      ├ last_compact_boundary()  找最近的压缩边界
    │      └ active_for_context() / all_for_context()
    │
    │  EvidenceStore       证据库（Issue/PR/CI/记忆的轻量摘要）
    │      ├ rebuild_repo()        重建整个仓库的证据
    │      └ search()              先走向量，再走关键词
    │
    ├─ 第 2 层：组装 ───────────────────────────────────
    │  AssembledContext    组装结果（dataclass）
    │  ContextAssembler    ★ 把各种来源拼成 prompt
    │      └ assemble()            预算分配 + 三层压缩 + 覆盖率统计
    │
    ├─ 第 3 层：压缩 ───────────────────────────────────
    │  ProgressiveContextManager  ★ 快爆窗口时插压缩边界
    │      ├ ensure_headroom()     判断要不要压（含熔断器）
    │      ├ _compact()            真正的压缩流程（多档重试）
    │      ├ _summary_for()        生成摘要（三种来源）
    │      ├ _insert_boundary()    插入边界消息
    │      └ _rehydration_state()  抢救"正在编辑的文件/激活的技能"
    │
    ├─ 第 4 层：沉淀 ───────────────────────────────────
    │  SessionSealer       ★ 一轮结束后提炼记忆
    │      ├ seal()                封存 + 提炼 + 写候选
    │      └ _update_thread_memory()  合并进长期记忆
    │
    └─ 第 5 层：工具 ───────────────────────────────────
       MemoryTools         给 Agent 调的三个 MCP 工具
           ├ search_evidence()       语义+关键词混合检索
           ├ get_thread_context()    精确回查原始消息
           └ read_session_events()   回查已封存的会话
"""

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import (
    AgentRun,
    ChatMessage,
    ChatSession,
    ChatSession as ChatSessionModel,
    Conversation,
    ConversationMemory,
    Document,
    EvidenceItem,
    Issue,
    PullRequest,
    Repository,
    ThreadMemory,
    WorkflowRun,
)
from app.services.context_compression import (
    COMPACT_BOUNDARY_SUBTYPE,
    COMPACT_SUMMARY_SYSTEM_PROMPT,
    adjust_keep_start_to_preserve_api_invariants,
    build_compaction_summary_payload,
    calculate_context_pressure,
    clip_to_token_budget,
    compress_messages,
    default_context_budget,
    dumps_compact,
    ensure_compaction_summary_preserves_key_facts,
    estimate_tokens,
    extract_relevant_snippet,
    fallback_memory_update,
    fallback_compaction_summary,
    format_compaction_summary,
    is_compact_boundary_message,
    messages_after_compact_boundary,
    memory_document_text,
    merge_memory,
    microcompact_messages,
    render_structured_memory,
    structured_memory_from_model,
)
from app.services.llm.client import LLMClient
from app.services.memory_hub import capture_memory_candidates, record_recall_event
from app.services.rag.indexing import CI_LOG_CHUNK_CHARS, sanitize_ci_log
from app.services.rag.retrieval import search_similar_documents
from app.services.rag.source_policy import RAG_SOURCE_TYPES
from app.services.rag.vector_store import MilvusUnavailableError


# ════════════════════════════════════════════════════════════════
# 模块级工具
# ════════════════════════════════════════════════════════════════

def _repo_filter(model: Any, repo_id: uuid.UUID | None) -> Any:
    """生成"按仓库过滤"的 SQL 条件。

    【调用链】被 MessageStore / EvidenceStore / ContextAssembler / SessionSealer 的所有查询调用

    【为什么 repo_id 为 None 时要写成 IS NULL】
        系统支持"不选仓库的通用对话"（repo_id 为 NULL）。
        如果直接写 `repo_id == None`，SQL 会变成 `repo_id = NULL`，
        结果永远是空（SQL 的三值逻辑：NULL = NULL 是 unknown，不是 true）。
        必须用 IS NULL 才能匹配到那些行。
    """
    return model.repo_id.is_(None) if repo_id is None else model.repo_id == repo_id


def _conversation_filter(model: Any, conversation_id: uuid.UUID | None) -> Any:
    """生成"按会话过滤"的 SQL 条件（同上，None 要走 IS NULL）。

    【调用链】被 MessageStore / MemoryTools 的查询调用
    """
    return model.conversation_id.is_(None) if conversation_id is None else model.conversation_id == conversation_id


def _tokens(text: str) -> list[str]:
    """分词（英文词 + 中文整段）。

    【调用链】被 EvidenceStore.search() / MemoryTools.get_thread_context() 调用

    【注意和 rag/retrieval._tokens 的区别】
        这里【不】拆中文单字/双字，而是把连续中文当成一个 token。
        因为这里用于"子串包含"判断（`term in content`），
        整段中文反而能匹配到完整短语，误报更少。
    """
    return re.findall(r"[a-zA-Z0-9_\-\u4e00-\u9fff]+", text.lower())


def _clip(text: str, limit: int) -> str:
    """按字符数截断并加省略号。

    【调用链】被 EvidenceStore / MemoryTools 调用

    limit-3 是给 "..." 留位置。
    """
    text = (text or "").strip()
    return text if len(text) <= limit else text[: max(0, limit - 3)].rstrip() + "..."


# ════════════════════════════════════════════════════════════════
# 第 1 层：消息存取
# ════════════════════════════════════════════════════════════════

class MessageStore:
    """消息读写。

    【核心设计：两种读法】
        recent()   → 模型看的"活动投影"（压缩边界之后 + 边界里保留的消息）
        timeline() → 用户看的"完整流水"（一条不删）

        为什么要分两种：
            上下文压缩只应该缩小"模型看到的内容"，不应该删掉用户的历史记录。
            否则用户会发现"我之前说的话不见了"，而模型只是"看不到"而已。
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def append(
        self,
        *,
        repo_id: uuid.UUID | None,
        conversation_id: uuid.UUID | None,
        role: str,
        content: str,
        route: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """写一条消息。

        【调用链】被 api/routes/chat.py 调用（用户消息、助手回答各一次）
        【下游】db.add + flush（不 commit，由外层事务统一提交）

        【为什么只 flush 不 commit】
            一轮对话要写"用户消息 + 助手回答 + agent_run"多条记录。
            如果每条都 commit，中途失败会留下半截数据（有提问没回答）。
            统一由外层 commit，才能保证"要么全成功要么全失败"。
        """
        message = ChatMessage(
            repo_id=repo_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            route=route,
            tool_calls=tool_calls or [],
            meta=meta or {},
        )
        self.db.add(message)
        self.db.flush()
        return message

    def recent(self, repo_id: uuid.UUID | None, conversation_id: uuid.UUID | None, limit: int = 20) -> list[ChatMessage]:
        """返回【活动投影】—— 模型当前应该看到的消息，不是完整流水。

        【调用链】追这个函数会进入：
            1. last_compact_boundary()   → 找最近的压缩边界
            2. db.query(ChatMessage)     → 按 ID 取"保留的消息"
            3. db.query(ChatMessage)     → 取边界之后的所有消息
        【被谁调用】ContextAssembler.assemble() / active_for_context()

        【数据流向】
            入：repo_id、conversation_id、limit
            出：消息列表（已按时间排序，最多 limit 条）

        Return the active context projection, not the complete transcript.

        New compact boundaries keep references to the original recent messages
        instead of copying them.  Older databases can still contain
        ``preserved_after_compact`` copies, so this method understands both
        layouts during the migration period.

        【两种历史布局，都要兼容】
            新布局：边界消息的 meta.preserved_message_ids 存"要保留的消息 ID"（只存引用，不复制）
            旧布局：边界之后插入 meta.subtype="preserved_after_compact" 的【副本】消息
            兼容方式：先按 ID 取保留的原消息，再取边界之后的消息，
                     但跳过那些"已经被 preserved_ids 覆盖的副本"。
        """
        boundary = self.last_compact_boundary(repo_id, conversation_id)
        # 从没压缩过 → 直接走完整时间线
        if boundary is None:
            return self.timeline(repo_id, conversation_id, limit=limit, include_system=True)

        # ── 步骤 1：取边界里记录的"要保留的消息 ID" ──
        boundary_meta = boundary.meta or {}
        preserved_ids = [
            value
            for value in boundary_meta.get("preserved_message_ids") or []
            if value
        ]

        preserved_rows: list[ChatMessage] = []
        if preserved_ids:
            # 解析 ID（可能有脏数据，逐个 try）
            parsed_ids: list[uuid.UUID] = []
            for value in preserved_ids:
                try:
                    parsed_ids.append(uuid.UUID(str(value)))
                except ValueError:
                    continue
            if parsed_ids:
                # 一次查完，用 dict 建立 id → 行 的映射
                by_id = {
                    row.id: row
                    for row in self.db.query(ChatMessage)
                    .filter(
                        _repo_filter(ChatMessage, repo_id),
                        _conversation_filter(ChatMessage, conversation_id),
                        ChatMessage.id.in_(parsed_ids),
                    )
                    .all()
                }
                # 按 preserved_ids 的原始顺序还原（保持对话顺序）
                preserved_rows = [by_id[item] for item in parsed_ids if item in by_id]

        # ── 步骤 2：取边界之后的所有消息 ──
        rows_after_boundary = (
            self.db.query(ChatMessage)
            .filter(
                _repo_filter(ChatMessage, repo_id),
                _conversation_filter(ChatMessage, conversation_id),
                ChatMessage.created_at >= boundary.created_at,
            )
            .order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc())
            .all()
        )

        # ── 步骤 3：拼接 [边界] + [保留的旧消息] + [边界之后的新消息] ──
        projected = [boundary, *preserved_rows]
        projected.extend(
            row
            for row in rows_after_boundary
            # 排除边界自己（已经在 projected 里）
            if row.id != boundary.id
            # 排除已经通过 preserved_ids 取到的（避免重复）
            and str(row.id) not in preserved_ids
            # 排除旧布局的副本（已经有原消息了，副本是多余的）
            and not (
                preserved_ids
                and (row.meta or {}).get("subtype") == "preserved_after_compact"
            )
        )
        # 只留最后 limit 条（最近的最重要）
        return projected[-max(1, limit) :]

    def timeline(
        self,
        repo_id: uuid.UUID | None,
        conversation_id: uuid.UUID | None,
        *,
        limit: int | None = 500,
        include_system: bool = False,
        before_message_id: uuid.UUID | None = None,
        after_message_id: uuid.UUID | None = None,
    ) -> list[ChatMessage]:
        """返回【完整流水】—— 用户看到的历史，压缩不影响它。

        【调用链】被 recent()（无边界时）/ MemoryTools.get_thread_context() 调用

        Return the durable transcript while hiding compatibility copies.

        Compact boundaries change the model's active projection; they do not
        erase the transcript.  This method is the authoritative read path for
        UI history and exact-memory tools.

        【数据流向】
            入：repo_id、conversation_id、limit、include_system、
                before_message_id / after_message_id（游标分页）
            出：消息列表（时间升序）

        【为什么需要 before/after 游标】
            历史可能有几千条。前端要"往上翻"（before）或"从某条往后读"（after）。
            用 (created_at, id) 组合做游标，而不是 offset ——
            offset 分页在"边翻边新增消息"时会错位、重复。
        """
        query = self.db.query(ChatMessage).filter(
            _repo_filter(ChatMessage, repo_id),
            _conversation_filter(ChatMessage, conversation_id),
        )
        # 默认不返回 system 消息（压缩边界、失败记录这些不该出现在用户历史里）
        if not include_system:
            query = query.filter(ChatMessage.role != "system")

        # ── 游标：只看这条消息【之前】的 ──
        if before_message_id:
            before = (
                self.db.query(ChatMessage)
                .filter(
                    _repo_filter(ChatMessage, repo_id),
                    _conversation_filter(ChatMessage, conversation_id),
                    ChatMessage.id == before_message_id,
                )
                .one_or_none()
            )
            if before and before.created_at:
                # 用 (created_at, id) 复合比较：时间相同时靠 id 保证严格顺序
                query = query.filter(
                    or_(
                        ChatMessage.created_at < before.created_at,
                        and_(
                            ChatMessage.created_at == before.created_at,
                            ChatMessage.id < before.id,
                        ),
                    )
                )

        # ── 游标：只看这条消息【之后】的 ──
        if after_message_id:
            after = (
                self.db.query(ChatMessage)
                .filter(
                    _repo_filter(ChatMessage, repo_id),
                    _conversation_filter(ChatMessage, conversation_id),
                    ChatMessage.id == after_message_id,
                )
                .one_or_none()
            )
            if after and after.created_at:
                query = query.filter(
                    or_(
                        ChatMessage.created_at > after.created_at,
                        and_(
                            ChatMessage.created_at == after.created_at,
                            ChatMessage.id > after.id,
                        ),
                    )
                )

        rows = query.order_by(ChatMessage.created_at.asc(), ChatMessage.id.asc()).all()

        # 过滤掉旧布局的兼容副本（用户不该看到重复消息）
        durable = [
            row
            for row in rows
            if (row.meta or {}).get("subtype") != "preserved_after_compact"
        ]

        if limit is None:
            return durable
        # 用 after 游标（往后翻）→ 取开头 limit 条
        if after_message_id and not before_message_id:
            return durable[: max(1, limit)]
        # 默认（往上翻 / 取最近）→ 取最后 limit 条
        return durable[-max(1, limit) :]

    def active_for_context(
        self,
        repo_id: uuid.UUID | None,
        conversation_id: uuid.UUID | None,
        limit: int = 1000,
    ) -> list[ChatMessage]:
        """给上下文组装用的活动消息（recent 的别名，limit 更大）。

        【调用链】recent() → 本函数
        【被谁调用】ContextAssembler / ProgressiveContextManager._raw_segment_tokens()

        为什么单独开一个方法而不是直接调 recent：
            语义不同 —— 这里要"尽可能多的活动消息"（1000 条）用于算总 token，
            而 recent 默认只要 20 条用于拼 prompt。
        """
        return self.recent(repo_id, conversation_id, limit=limit)

    def all_for_context(self, repo_id: uuid.UUID | None, conversation_id: uuid.UUID | None, limit: int = 200) -> list[ChatMessage]:
        """取最近 limit 条消息（含 system），按时间升序返回。

        【调用链】被测试和部分诊断接口调用

        【和 active_for_context 的区别】
            这个不处理压缩边界，直接按时间倒序取 N 条再翻正。
            用于"我就想要最近 N 条原始记录"的场景。
        """
        rows = (
            self.db.query(ChatMessage)
            .filter(_repo_filter(ChatMessage, repo_id), _conversation_filter(ChatMessage, conversation_id))
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
            .all()
        )
        # 倒序取完要翻回来，保证时间升序
        return list(reversed(rows))

    def last_compact_boundary(self, repo_id: uuid.UUID | None, conversation_id: uuid.UUID | None) -> ChatMessage | None:
        """找最近的压缩边界消息。

        【调用链】is_compact_boundary_message() → 本函数
        【被谁调用】recent() / ContextAssembler.assemble() / _consecutive_failures()

        【为什么先取 50 条再逐条判断，而不是直接在 SQL 里过滤 subtype】
            meta 是 JSON 列，跨数据库的 JSON 查询写法不统一。
            取最近 50 条 system 消息在应用层筛，简单可靠
            （system 消息本来就很少，50 条足够覆盖）。
        """
        rows = (
            self.db.query(ChatMessage)
            .filter(_repo_filter(ChatMessage, repo_id), _conversation_filter(ChatMessage, conversation_id), ChatMessage.role == "system")
            .order_by(ChatMessage.created_at.desc())
            .limit(50)
            .all()
        )
        for row in rows:
            # 复用 context_compression 里的判定函数（保证两边标准一致）
            if is_compact_boundary_message({"role": row.role, "meta": row.meta or {}}):
                return row
        return None


# ════════════════════════════════════════════════════════════════
# 第 1 层：证据库
# ════════════════════════════════════════════════════════════════

class EvidenceStore:
    """证据库 —— Issue / PR / CI / 记忆的"轻量摘要"，用于快速取证。

    【和 RAG Document 的区别】
        Document     = 完整正文，进 Milvus 做语义检索（重，准）
        EvidenceItem = 摘要文本，用于关键词快速匹配（轻，快）

        为什么要两套：
            向量检索有网络/模型延迟，短查询（如"#42 是什么"）用关键词更快。
            Agent 需要"先快速看一眼有没有"，再决定要不要走完整 RAG。
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def rebuild_repo(self, repo: Repository | None, conversation: Conversation | None = None) -> None:
        """重建一个仓库（或会话）的证据库。

        【调用链】追这个函数会进入：
            1. 删除旧证据（会话级：只删该会话 + 全局的；仓库级：删该仓库全部）
            2. _add() × N  → 逐类写入（仓库/Issue/PR/CI/文档/记忆/会话/运行记录）
        【被谁调用】api/routes/chat.py（每轮对话前刷新）、启动时的数据回填

        【数据流向】
            入：repo（可空）、conversation（可空）
            出：evidence_items 表被重建（flush，不 commit）

        【为什么会话级只删"该会话 + conversation_id 为 NULL"】
            全局证据（repo 级的 Issue/PR）在多个会话间共享，不能因为
            某一个会话刷新就删掉。所以会话级重建只清"属于这个会话的"和"全局的"，
            然后重新生成 —— 相当于"刷新当前会话可见的证据"。

        【为什么要重建而不是增量更新】
            证据是"派生数据"，重建成本低（一次查询 + 几百次 insert），
            增量更新要处理"删掉的 Issue 怎么办"，复杂且容易出错。
        """
        repo_id = repo.id if repo else None
        conversation_id = conversation.id if conversation else None

        # ── 步骤 1：删除旧证据 ──
        if conversation_id:
            # 会话级：删该会话的 + 全局的（不动其他会话）
            self.db.query(EvidenceItem).filter(
                _repo_filter(EvidenceItem, repo_id),
                or_(EvidenceItem.conversation_id == conversation_id, EvidenceItem.conversation_id.is_(None)),
            ).delete(synchronize_session=False)
        else:
            # 仓库级：全删
            self.db.query(EvidenceItem).filter(_repo_filter(EvidenceItem, repo_id)).delete(synchronize_session=False)

        # ── 步骤 2：重建（只处理仓库级数据，会话级数据下面单独处理）──
        if repo:
            # ① 仓库本身
            self._add(
                repo_id,
                None,
                "repository",
                str(repo.id),
                repo.full_name,
                "\n".join(filter(None, [repo.description or "", f"默认分支：{repo.default_branch or '未知'}"])),
                {"anchor": f"repo:{repo.full_name}"},
            )

            # ② 所有 Issue
            for issue in self.db.query(Issue).filter(Issue.repo_id == repo.id).all():
                self._add(
                    repo.id,
                    None,
                    "issue",
                    str(issue.id),
                    f"Issue #{issue.number}: {issue.title}",
                    "\n".join(filter(None, [issue.title, issue.body or "", f"state: {issue.state}", f"labels: {', '.join(issue.labels or [])}"])),
                    {"number": issue.number, "state": issue.state},
                )

            # ③ 所有 PR
            for pr in self.db.query(PullRequest).filter(PullRequest.repo_id == repo.id).all():
                self._add(
                    repo.id,
                    None,
                    "pull_request",
                    str(pr.id),
                    f"PR #{pr.number}: {pr.title}",
                    "\n".join(filter(None, [pr.title, pr.body or "", f"state: {pr.state}", f"base: {pr.base_branch}", f"head: {pr.head_branch}"])),
                    {"number": pr.number, "state": pr.state},
                )

            # ④ 失败的 CI（成功的没价值）
            for run in self.db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo.id).all():
                if run.conclusion != "failure" or not (run.logs_text or "").strip():
                    continue
                # ★ 脱敏必须在截断之前（否则半截 token 匹配不到规则，会漏进库）
                sanitized_logs = sanitize_ci_log(run.logs_text or "")
                safe_logs = sanitized_logs[:CI_LOG_CHUNK_CHARS]
                self._add(
                    repo.id,
                    None,
                    "workflow_run",
                    str(run.id),
                    f"CI: {run.name}",
                    "\n".join(filter(None, [run.name, run.status, run.conclusion or "", safe_logs])),
                    {
                        "status": run.status,
                        "conclusion": run.conclusion,
                        "logs_truncated": len(sanitized_logs) > CI_LOG_CHUNK_CHARS,
                        "secrets_redacted": True,
                    },
                )

            # ⑤ 所有进 RAG 的文档（直接用它们的正文当证据）
            for doc in (
                self.db.query(Document)
                .filter(Document.repo_id == repo.id, Document.source_type.in_(RAG_SOURCE_TYPES))
                .all()
            ):
                self._add(repo.id, None, doc.source_type, str(doc.source_id), doc.title, doc.content, doc.meta or {})

        # ── 步骤 3：记忆（会话级用 ConversationMemory，仓库级用 ThreadMemory）──
        memories = (
            self.db.query(ConversationMemory).filter(ConversationMemory.conversation_id == conversation_id).all()
            if conversation_id
            else self.db.query(ThreadMemory).filter(_repo_filter(ThreadMemory, repo_id)).all()
        )
        for memory in memories:
            self._add(
                repo_id,
                conversation_id,
                "memory",
                str(memory.id),
                "Thread memory",
                memory_document_text(memory),
                {"sessions_incorporated": memory.sessions_incorporated},
            )

        # ── 步骤 4：最近 30 个会话摘要 + 最近 30 次运行记录 ──
        session_query = self.db.query(ChatSession).filter(_repo_filter(ChatSession, repo_id))
        run_query = self.db.query(AgentRun).filter(_repo_filter(AgentRun, repo_id))
        if conversation_id:
            session_query = session_query.filter(ChatSession.conversation_id == conversation_id)
            run_query = run_query.filter(AgentRun.conversation_id == conversation_id)

        for session in session_query.order_by(ChatSession.created_at.desc()).limit(30):
            self._add(repo_id, conversation_id, "session", str(session.id), f"Session digest: {session.route}", session.digest, {"route": session.route})
        for run in run_query.order_by(AgentRun.created_at.desc()).limit(30):
            self._add(repo_id, conversation_id, "agent_run", str(run.id), f"Agent run: {run.route}", f"{run.user_message}\n{run.final_answer}", {"route": run.route})

        self.db.flush()

    def search(self, repo_id: uuid.UUID | None, conversation_id: uuid.UUID | None, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """检索证据：先走向量（准），不够再走关键词（快）。

        【调用链】追这个函数会进入：
            1. search_similar_documents()   → 向量+BM25 混合检索（rag/retrieval）
            2. extract_relevant_snippet()   → 关键词命中后抽相关片段
        【被谁调用】ContextAssembler.assemble() / MemoryTools.search_evidence()

        【数据流向】
            入：repo_id、conversation_id、query、limit
            出：证据列表（统一结构：title/source_type/source_id/snippet/score/metadata/retrieval）

        【两阶段策略】
            阶段 1：走向量检索，够 limit 条就直接返回（质量最好）
            阶段 2：向量不够（或 Milvus 挂了）→ 用关键词在 evidence_items 里捞
                    取最近 500 条候选，算词频得分

        【为什么关键词阶段要限制 500 条】
            evidence_items 可能有几千条。全表扫 + Python 算分太慢。
            按 created_at 倒序取 500 条（最近的更可能相关），是个务实的折中。
        """
        query_terms = _tokens(query)
        if not query_terms:
            return []

        results: list[dict[str, Any]] = []
        # 已收录的 (source_type, source_id)，用于两阶段之间去重
        seen: set[tuple[str, str]] = set()

        # ── 阶段 1：向量检索（需要 repo_id）──
        if repo_id:
            try:
                rag_results = search_similar_documents(
                    self.db,
                    repo_id,
                    query,
                    source_type=None,
                    limit=max(limit, 8),        # 多取几条，留去重余量
                )
            except MilvusUnavailableError:
                # Milvus 挂了不影响取证 —— 静默降级到关键词
                rag_results = []

            for item in rag_results:
                key = (str(item.get("source_type") or ""), str(item.get("source_id") or ""))
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    {
                        "title": item.get("title") or "",
                        "source_type": item.get("source_type") or "",
                        "source_id": item.get("source_id") or "",
                        "snippet": item.get("snippet") or "",
                        "score": item.get("score") or 0,
                        "metadata": item.get("metadata") or {},
                        "retrieval": item.get("retrieval") or {},
                    }
                )
                # 向量结果够了 → 直接返回，不用跑阶段 2
                if len(results) >= limit:
                    return results

        # ── 阶段 2：关键词兜底 ──
        candidates = (
            self.db.query(EvidenceItem)
            .filter(
                _repo_filter(EvidenceItem, repo_id),
                # 会话级 + 全局的都算
                or_(EvidenceItem.conversation_id == conversation_id, EvidenceItem.conversation_id.is_(None)),
            )
            .order_by(EvidenceItem.created_at.desc())
            .limit(500)
            .all()
        )

        ranked: list[tuple[int, EvidenceItem]] = []
        for item in candidates:
            key = (item.source_type, item.source_id)
            if key in seen:
                continue
            haystack = f"{item.title}\n{item.content}".lower()
            # 得分 = 所有查询词出现次数之和（朴素的词频打分）
            score = sum(haystack.count(term) for term in query_terms)
            if score:
                ranked.append((score, item))

        # 分数降序，同分按时间新的在前
        ranked.sort(key=lambda pair: (pair[0], pair[1].created_at), reverse=True)

        for score, item in ranked:
            key = (item.source_type, item.source_id)
            if key in seen:
                continue
            seen.add(key)
            results.append(
                {
                    "title": item.title,
                    "source_type": item.source_type,
                    "source_id": item.source_id,
                    # 关键词命中后，用 query 抽取最相关的片段（而不是整段塞进去）
                    "snippet": extract_relevant_snippet(query, item.content, max_tokens=90),
                    "score": score,
                    "metadata": item.meta or {},
                    "retrieval": {"sources": ["evidence_item_keyword"]},
                }
            )
            if len(results) >= limit:
                break
        return results

    def _add(self, repo_id: uuid.UUID | None, conversation_id: uuid.UUID | None, source_type: str, source_id: str, title: str, content: str, meta: dict[str, Any]) -> None:
        """写一条证据（空内容直接跳过）。

        【调用链】被 rebuild_repo() 调用（8 处）

        【为什么空内容要跳过】
            空证据会被关键词检索命中（"这个词出现 0 次"不会命中，
            但 title 可能命中），结果给出一个没有正文的条目 —— 纯噪声。
        """
        if not content.strip():
            return
        self.db.add(EvidenceItem(repo_id=repo_id, conversation_id=conversation_id, source_type=source_type, source_id=source_id, title=title, content=content, meta=meta))


# ════════════════════════════════════════════════════════════════
# 第 2 层：上下文组装
# ════════════════════════════════════════════════════════════════

@dataclass
class AssembledContext:
    """上下文组装的产物。

    【业务含义】这就是"最终要拼进 prompt 的东西"的快照。
    【谁产出】ContextAssembler.assemble()
    【谁消费】services/agents/chat_agent.py（拼成 SystemMessage + HumanMessage）
             前端"上下文用量"面板（读 compression_stats）
    """

    system_context: str                                    # 系统上下文（仓库/会话/记忆/证据，已按预算裁剪）
    history_messages: list[dict[str, str]]                 # 历史对话（已压缩，role + content）
    evidence: list[dict[str, Any]]                         # 相关证据（用于展示引用）
    compression_stats: dict[str, Any] = field(default_factory=dict)  # 压缩统计（预算/覆盖率/压力）


class ContextAssembler:
    """上下文组装器 —— 把 5 个来源拼成一个 prompt，且保证不超预算。

    【5 个来源，各有独立预算】
        system      当前仓库 + 当前会话（稳定，几乎不变）
        memory      长期记忆 + 上次会话摘要 + 压缩边界
        evidence    相关证据（RAG + 关键词）
        recent      最近对话（会被压缩）
        tool        工具观察结果（预留给 agent 的 tool 消息）
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def assemble(self, repo: Repository | None, conversation: Conversation | None, current_message: str, limit: int = 16) -> AssembledContext:
        """【核心】组装上下文。

        【调用链】追这个函数会进入：
            1. default_context_budget()            → 分预算（context_compression）
            2. MessageStore.last_compact_boundary() → 找压缩边界
            3. EvidenceStore.search()               → 取证（向量 + 关键词）
            4. MessageStore.recent()                → 取活动消息
            5. calculate_context_pressure()         → 算压力
            6. microcompact_messages()              → 第一层压缩（清旧工具结果）
            7. compress_messages()                  → 第二层压缩（丢旧消息）
            8. build_context()（内部闭包）           → 拼系统上下文
        【被谁调用】
            api/routes/chat.py                          ← 每轮对话
            ProgressiveContextManager.ensure_headroom()  ← 判断要不要压缩

        【数据流向】
            入：repo、conversation、current_message、limit（历史条数）
            出：AssembledContext（system_context + history + evidence + stats）

        【三阶段预算控制（重要，容易看漏）】
            ① 先按 raw_history（原始条数）算压力 → 需要就先 microcompact
            ② 再 compress_messages 到 recent 预算
            ③ 拼完 system_context 后【再算一次】总压力
               —— 如果还是超，且前面没做过 microcompact，就补做一次
               这是"第二轮兜底"，因为 system_context 本身也可能很大（记忆 + 证据）

        【为什么要判断 current_message_in_history】
            调用方可能已经把当前问题写进库了（append 之后再 assemble）。
            这时历史里最后一条就是当前问题，不能再算一次 pending_user_tokens，
            否则会重复计算、虚高压力。
        """
        # ── 步骤 0：分预算 ──
        budget = default_context_budget()
        repo_id = repo.id if repo else None
        conversation_id = conversation.id if conversation else None
        message_store = MessageStore(self.db)
        boundary = message_store.last_compact_boundary(repo_id, conversation_id)

        # ── 步骤 1：取记忆（会话级优先，否则仓库级）──
        memory = (
            self.db.query(ConversationMemory).filter(ConversationMemory.conversation_id == conversation_id).first()
            if conversation_id
            else self.db.query(ThreadMemory).filter(_repo_filter(ThreadMemory, repo_id)).first()
        )

        # ── 步骤 2：取上一次已封存的会话摘要（跨轮次的连续性）──
        latest_session = (
            self.db.query(ChatSession)
            .filter(
                _repo_filter(ChatSession, repo_id),
                _conversation_filter(ChatSession, conversation_id),
                # 只认封存成功的（sealing 是中间态，内容还不完整）
                ChatSession.status.in_(["sealed", "sealed_with_warnings"]),
            )
            .order_by(ChatSession.created_at.desc())
            .first()
        )

        # ── 步骤 3：取证（8 条，给 evidence 预算留余量）──
        evidence = EvidenceStore(self.db).search(repo_id, conversation_id, current_message, limit=8)

        # ── 步骤 4：取活动消息（多取一些，压缩才有选择空间）──
        history_fetch_limit = max(limit, 30)
        recent = message_store.recent(repo_id, conversation_id, limit=history_fetch_limit)

        # 转成 dict（下游压缩函数只认 dict），只留 user/assistant
        raw_history = [
            {
                "role": item.role,
                "content": item.content,
                "route": item.route,
                "tool_calls": item.tool_calls or [],
                "meta": item.meta or {},
            }
            for item in recent
            if item.role in {"user", "assistant"}
        ]

        micro_stats: dict[str, Any] | None = None

        # ── 步骤 5：算原始 token（含 tool_calls，因为它们也占预算）──
        raw_history_tokens = sum(
            estimate_tokens(str(item.get("role") or ""))
            + estimate_tokens(str(item.get("content") or ""))
            + (
                estimate_tokens(dumps_compact(item.get("tool_calls")))
                if item.get("tool_calls")
                else 0
            )
            for item in raw_history
        )

        # 当前问题是否已经在历史里（避免重复计算）
        current_message_in_history = bool(
            raw_history
            and raw_history[-1].get("role") == "user"
            and " ".join(str(raw_history[-1].get("content") or "").split())
            == " ".join(str(current_message or "").split())
        )
        pending_user_tokens = 0 if current_message_in_history else estimate_tokens(current_message)

        raw_history_pressure = calculate_context_pressure(raw_history_tokens + pending_user_tokens)

        # ── 步骤 6：第一层压缩（只在压力大时做，因为要重写内容）──
        history_source = raw_history
        if raw_history_pressure.should_warn:
            history_source, micro_stats = microcompact_messages(raw_history, current_message)

        # ── 步骤 7：第二层压缩（按 recent 预算丢旧消息）──
        history, history_stats = compress_messages(history_source, budget.recent_tokens)

        def build_context(next_history: list[dict[str, str]], next_evidence: list[dict[str, Any]]) -> str:
            """内部闭包：把各来源拼成 system_context。

            【为什么要做成闭包】
                后面"第二轮兜底"需要重新拼一次（history 和 evidence 变了）。
                做成闭包可以直接复用，不用重复写一遍拼接逻辑。
            """
            system_sections: list[str] = []

            # ① 当前仓库 / 当前会话（最稳定，放最前面 —— 也利于 prompt cache 命中）
            if repo:
                system_sections.append(f"[Current Repository]\n{repo.full_name}\n[/Current Repository]")
            if conversation:
                system_sections.append(f"[Current Conversation]\n{conversation.title}\n[/Current Conversation]")

            sections: list[str] = []
            stable_system = clip_to_token_budget("\n\n".join(system_sections), budget.system_tokens)
            if stable_system:
                sections.append(stable_system)

            # ② 记忆区（压缩边界 + 恢复状态 + 长期记忆 + 上次摘要）
            memory_sections: list[str] = []

            if boundary and boundary.content:
                transcript_ref = (boundary.meta or {}).get("transcript_ref") or {}
                rehydration = (boundary.meta or {}).get("rehydration") or {}
                # 告诉模型"原文还能查回来"，避免它以为信息永久丢失
                recovery_hint = (
                    "\nExact transcript recovery: call devflow_get_thread_context with this conversation"
                    if transcript_ref
                    else ""
                )
                # 恢复工作状态（正在编辑的文件、激活的技能）—— 比摘要更具体
                if rehydration:
                    memory_sections.append(
                        "[Rehydrated Working State]\n"
                        + dumps_compact(rehydration)
                        + "\n[/Rehydrated Working State]"
                    )
                memory_sections.append(
                    f"[Compaction Boundary]\n{boundary.content}{recovery_hint}\n[/Compaction Boundary]"
                )

            if memory and memory_document_text(memory).strip():
                # 记忆预算只用一半：留空间给后面的会话摘要
                memory_text = render_structured_memory(memory, max_tokens=max(256, budget.memory_tokens // 2))
                memory_sections.append(f"[Thread Memory: {memory.sessions_incorporated} sessions]\n{memory_text}\n[/Thread Memory]")

            if latest_session and latest_session.digest:
                memory_sections.append(f"[Previous Session Digest]\n{latest_session.digest}\n[/Previous Session Digest]")

            memory_context = clip_to_token_budget("\n\n".join(memory_sections), budget.memory_tokens)
            if memory_context:
                sections.append(memory_context)

            # ③ 证据区（每条平均分配预算）
            if next_evidence:
                evidence_budget = max(120, budget.evidence_tokens // max(len(next_evidence), 1))
                lines = ["[Related Evidence]"]
                for item in next_evidence:
                    snippet = clip_to_token_budget(item["snippet"], evidence_budget)
                    lines.append(f"- [{item['source_type']}] {item['title']}: {snippet}")
                lines.append("[/Related Evidence]")
                sections.append(clip_to_token_budget("\n".join(lines), budget.evidence_tokens))

            return "\n\n".join(sections)

        # ── 步骤 8：拼第一次，然后算总压力 ──
        system_context = build_context(history, evidence)

        history_prompt_tokens = sum(
            estimate_tokens(item.get("role")) + estimate_tokens(item.get("content"))
            for item in history
        )
        initial_prompt_tokens = estimate_tokens(system_context) + history_prompt_tokens + pending_user_tokens
        initial_pressure = calculate_context_pressure(initial_prompt_tokens)

        # ── 步骤 9：第二轮兜底（system_context 本身太大时补做 microcompact）──
        if initial_pressure.should_warn and micro_stats is None:
            micro_raw_history, micro_stats = microcompact_messages(raw_history, current_message)
            history, history_stats = compress_messages(micro_raw_history, budget.recent_tokens)
            system_context = build_context(history, evidence)

        # ── 步骤 10：最终统计 ──
        history_prompt_tokens = sum(
            estimate_tokens(item.get("role")) + estimate_tokens(item.get("content"))
            for item in history
        )
        assembled_prompt_tokens = estimate_tokens(system_context) + history_prompt_tokens + pending_user_tokens
        final_pressure = calculate_context_pressure(assembled_prompt_tokens)
        microcompacted_messages = int((micro_stats or {}).get("compacted_messages") or 0)

        # 历史覆盖率：告诉用户/调试者"模型到底看到了多少历史"
        history_coverage = {
            "source_messages": len(raw_history),                                   # 原始条数
            "prompt_messages": len(history),                                       # 实际进 prompt 的条数
            "omitted_messages": int(history_stats.get("omitted_messages") or 0),   # 被整条丢弃
            "truncated_messages": int(history_stats.get("truncated_messages") or 0),  # 被截断内容
            "exact_suffix_messages": int(history_stats.get("exact_suffix_messages") or 0),  # 尾部原文保留
            "source_limit_reached": len(recent) >= history_fetch_limit,            # 是否因为 limit 截断
            "has_compaction_boundary": boundary is not None,                       # 是否有压缩边界
            "microcompacted_messages": microcompacted_messages,                    # 被清空结果的条数
        }
        # "完整且精确" = 五个降级因素全都没有
        history_coverage["complete_and_exact"] = not any(
            [
                history_coverage["omitted_messages"],
                history_coverage["truncated_messages"],
                history_coverage["source_limit_reached"],
                history_coverage["has_compaction_boundary"],
                history_coverage["microcompacted_messages"],
            ]
        )

        return AssembledContext(
            system_context=system_context,
            history_messages=history,
            evidence=evidence,
            compression_stats={
                "budget": budget.__dict__,                  # 五份预算
                "history": history_stats,                   # 历史压缩统计
                "history_coverage": history_coverage,       # 覆盖率
                "microcompact": micro_stats,                # 轻量压缩统计（没做则为 None）
                "pressure": final_pressure.as_dict(),       # 最终压力
                "initial_pressure": initial_pressure.as_dict(),
                "raw_history_pressure": raw_history_pressure.as_dict(),
                "compact_boundary": {
                    "id": str(boundary.id) if boundary else None,
                    "created_at": boundary.created_at.isoformat() if boundary and boundary.created_at else None,
                    "mode": (boundary.meta or {}).get("mode") if boundary else None,
                },
                "system_context_estimated_tokens": estimate_tokens(system_context),
                "history_estimated_tokens": history_prompt_tokens,
                "pending_user_estimated_tokens": pending_user_tokens,
                "assembled_prompt_estimated_tokens": assembled_prompt_tokens,
                # 拆解：prompt 的 token 都花在哪了（调试"为什么这么快就满了"最有用）
                "prompt_breakdown": {
                    "system_memory_evidence": estimate_tokens(system_context),
                    "recent_history": history_prompt_tokens,
                    "pending_user": pending_user_tokens,
                    "tool_observation_budget": budget.tool_observation_tokens,
                    "reserved_response": budget.reserved_response_tokens,
                },
                "evidence_items": len(evidence),
                "mode": "progressive_budgeted_context",
            },
        )


# ════════════════════════════════════════════════════════════════
# 第 3 层：渐进压缩（快爆窗口时插压缩边界）
# ════════════════════════════════════════════════════════════════

class ProgressiveContextManager:
    """渐进式上下文压缩 —— 上下文快满时，把旧对话压成摘要并插一条"边界"。

    【为什么叫"渐进"】
        不是"满了才一次性处理"，而是分档：warning 只提示，
        auto_compact / aggressive / manual_path 逐档加大压缩力度。
        每档都会重新评估"压完够不够"，不够就试下一档。

    【三道安全阀，防止压缩把系统拖垮】
        ① 自动压缩总开关（settings.context_auto_compact_enabled）
        ② microcompact 够用就不做完整压缩（便宜的手段优先）
        ③ 熔断器：连续失败 N 次后停止尝试（settings.context_max_compaction_failures）
    """

    def __init__(self, db: Session, llm: LLMClient | None = None) -> None:
        self.db = db
        self.llm = llm or LLMClient()

    async def ensure_headroom(self, repo: Repository | None, conversation: Conversation | None, current_message: str) -> dict[str, Any]:
        """确保上下文有足够余量（不够就压缩）。

        【调用链】追这个函数会进入：
            1. ContextAssembler.assemble()      → 先组装一次，看实际压力
            2. calculate_context_pressure()     → 算原始/组装后两个压力
            3. _consecutive_failures()          → 熔断器检查
            4. _compact()                       → 真正压缩
            5. _record_failure()                → 失败记录
        【被谁调用】api/routes/chat.py（每轮对话开始前）

        【数据流向】
            入：repo、conversation、current_message
            出：dict，含 attempted / compacted / reason / pressure 等
                没压缩时也有 reason（便于排查"为什么没压"）

        【为什么同时算"原始压力"和"组装后压力"】
            raw_segment_tokens：压缩边界之后的原始消息总量（真实的"未压缩体积"）
            assembled_tokens：组装后实际进 prompt 的量（已经过 microcompact + compress）
            取两者最大值做判定 —— 因为组装后的数字可能因为裁剪而偏小，
            但原始体积仍然很大，下一轮照样会爆。

        【三个提前返回的分支（都很重要）】
            ① prompt 已经不紧张，但 raw 很紧张，且 microcompact 起了作用
               → 说明轻量压缩够用，不必做完整压缩（省钱省时间）
            ② 压力等级是 normal/warning → 不用压
            ③ 熔断器打开（连续失败太多次）→ 不再尝试，避免每次都白花时间
        """
        # 没有仓库或会话 → 无从压起
        if repo is None or conversation is None:
            return {"attempted": False, "reason": "no_repo_or_conversation"}

        # ── 步骤 1：先组装一次，拿到真实压力 ──
        assembled = ContextAssembler(self.db).assemble(repo, conversation, current_message)
        assembled_pressure = assembled.compression_stats.get("pressure") or {}
        micro_stats = assembled.compression_stats.get("microcompact") or {}

        raw_segment_tokens = self._raw_segment_tokens(repo.id, conversation.id) + estimate_tokens(current_message)
        assembled_tokens = int(
            assembled.compression_stats.get("assembled_prompt_estimated_tokens")
            or assembled.compression_stats.get("system_context_estimated_tokens")
            or 0
        )

        raw_pressure_state = calculate_context_pressure(raw_segment_tokens)
        assembled_pressure_state = calculate_context_pressure(assembled_tokens)
        # 取两者最大 —— 保守判定
        pressure_state = calculate_context_pressure(max(raw_segment_tokens, assembled_tokens))

        pressure = {
            **pressure_state.as_dict(),
            "raw_segment_estimated_tokens": raw_segment_tokens,
            "assembled_context_estimated_tokens": assembled_tokens,
            "raw_stage": raw_pressure_state.stage,
            "assembled_stage": assembled_pressure.get("stage") or assembled_pressure_state.stage,
            "microcompact_estimated_tokens_saved": micro_stats.get("estimated_tokens_saved", 0),
            "microcompact_compacted_messages": micro_stats.get("compacted_messages", 0),
        }

        raw_stage = raw_pressure_state.stage
        prompt_stage = str(pressure.get("assembled_stage") or assembled_pressure_state.stage)
        stage = str(pressure.get("stage") or "normal")

        # ── 安全阀 ①：自动压缩总开关 ──
        if not settings.context_auto_compact_enabled:
            return {"attempted": False, "reason": "auto_compact_disabled", "pressure": pressure}

        # ── 安全阀 ②：轻量压缩已经够用 ──
        # 条件：prompt 侧已经不紧张（normal/warning），但原始侧很紧张，
        #       且 microcompact 确实清掉了一些内容 → 说明"清工具结果"就够了
        if (
            prompt_stage in {"normal", "warning"}
            and raw_stage in {"auto_compact", "aggressive", "manual_path"}
            and int(micro_stats.get("compacted_messages") or 0) > 0
        ):
            return {
                "attempted": False,
                "reason": "microcompact_sufficient",
                "pressure": pressure,
                "microcompact": micro_stats,
            }

        # ── 压力不大 → 不压 ──
        if stage in {"normal", "warning"}:
            return {"attempted": False, "reason": stage, "pressure": pressure}

        # ── 安全阀 ③：熔断器 ──
        failures = self._consecutive_failures(repo.id, conversation.id)
        if failures >= settings.context_max_compaction_failures:
            return {
                "attempted": False,
                "reason": "circuit_breaker_open",
                "consecutive_failures": failures,
                "pressure": pressure,
            }

        # ── 真正压缩 ──
        try:
            result = await self._compact(repo, conversation, current_message, pressure, stage)
            return {"attempted": True, **result, "pressure": pressure, "consecutive_failures": 0}
        except Exception as exc:  # pragma: no cover - defensive circuit breaker path
            # 压缩失败 → 回滚（可能写到一半），记一条失败，返回失败信息
            # 注意：这里【不抛异常】—— 压缩失败不该让整轮对话挂掉
            self.db.rollback()
            failure = self._record_failure(repo.id, conversation.id, current_message, stage, exc)
            self.db.flush()
            return {
                "attempted": True,
                "compacted": False,
                "reason": "compaction_failed",
                "failure_id": str(failure.id),
                "error": str(exc),
                "pressure": pressure,
                "consecutive_failures": failures + 1,
            }

    async def _compact(
        self,
        repo: Repository,
        conversation: Conversation,
        current_message: str,
        pressure: dict[str, Any],
        stage: str,
    ) -> dict[str, Any]:
        """真正的压缩流程：定档 → 逐档尝试 → 选出够用的一档 → 插边界。

        【调用链】追这个函数会进入：
            1. MessageStore.active_for_context()      → 取活动消息
            2. messages_after_compact_boundary()      → 只要边界之后的（不重复压）
            3. _compaction_plan_for_stage()           → 定档参数
            4. _rehydration_state()                   → 抢救工作状态
            5. _messages_to_keep()                    → 算保留哪些
            6. _summary_for()                         → 生成摘要（LLM / 结构化记忆 / 兜底）
            7. _post_compact_tokens()                 → 算压完多大
            8. _insert_boundary()                     → 插边界消息
        【被谁调用】ensure_headroom()

        【数据流向】
            入：repo、conversation、current_message、pressure（压力详情）、stage（当前档位）
            出：dict，含 compacted / mode / messages_summarized / compaction_attempts 等

        【"逐档尝试"是什么意思】
            当前是 auto_compact → 依次尝试 [auto_compact, aggressive, manual_path]
            每一档都真的生成一次摘要、算一次压后 token；
            一旦压完降到 normal/warning 就停（不再试更狠的档）。
            这样既保证效果，又不会无谓地压得太狠。

        【为什么要 messages_after_compact_boundary(include_boundary=False)】
            已经压过的部分不该再压一次（会重复总结，浪费 token 且信息衰减）。
            只压"上次边界之后"的新内容。
        """
        store = MessageStore(self.db)
        all_rows = store.active_for_context(repo.id, conversation.id, limit=500)
        all_messages = [self._row_to_message(row) for row in all_rows]

        # 只取最后一次压缩边界之后的（不重复压历史）
        segment_messages = messages_after_compact_boundary(all_messages, include_boundary=False)
        message_ids = {str(item.get("id")) for item in segment_messages if item.get("id")}
        segment_rows = [row for row in all_rows if str(row.id) in message_ids]

        # 只有这几种角色能进摘要（system 消息如边界/失败记录不算）
        candidate_rows = [
            row
            for row in segment_rows
            if row.role in {"user", "assistant", "tool", "tool_use", "tool_result"}
        ]

        # ── 定档：当前档 + 更狠的档 ──
        plan_stages = [stage]
        if stage == "auto_compact":
            plan_stages.extend(["aggressive", "manual_path"])
        elif stage == "aggressive":
            plan_stages.append("manual_path")

        plans = [self._compaction_plan_for_stage(item) for item in plan_stages]

        # 消息太少，压了也没意义（保留数 + 1 都不到）
        if len(candidate_rows) <= min(int(item["keep_recent"]) for item in plans) + 1:
            return {"compacted": False, "reason": "not_enough_messages"}

        attempts: list[dict[str, Any]] = []
        selected: tuple[
            dict[str, Any],
            list[ChatMessage],
            list[ChatMessage],
            str,
            str,
            dict[str, Any],
            int,
        ] | None = None

        # 先抢救"正在编辑的文件 / 激活的技能"（摘要里可能丢，这里单独保下来）
        rehydration = self._rehydration_state(candidate_rows)

        # ── 逐档尝试 ──
        for index, plan in enumerate(plans, start=1):
            keep_rows = self._messages_to_keep(candidate_rows, keep_count=int(plan["keep_recent"]))
            keep_ids = {row.id for row in keep_rows}
            summarize_rows = [row for row in candidate_rows if row.id not in keep_ids]
            if not summarize_rows:
                continue

            summary, mode, preservation_report = await self._summary_for(
                repo,
                conversation,
                summarize_rows,
                current_message,
                max_summary_tokens=int(plan["summary_tokens"]),
            )

            # 档位前缀（区分是 auto 压的还是 aggressive 压的）
            mode_prefix = str(plan.get("mode_prefix") or "")
            if mode_prefix:
                mode = f"{mode_prefix}_{mode}"

            post_compact_tokens = self._post_compact_tokens(
                summary,
                keep_rows,
                current_message=current_message,
                rehydration=rehydration,
            )
            post_pressure = calculate_context_pressure(post_compact_tokens)

            attempts.append(
                {
                    "attempt": index,
                    "plan": plan,
                    "post_compact_estimated_tokens": post_compact_tokens,
                    "post_compact_stage": post_pressure.stage,
                }
            )

            selected = (
                plan,
                keep_rows,
                summarize_rows,
                summary,
                mode,
                preservation_report,
                post_compact_tokens,
            )

            # 压到正常范围了 → 不用试更狠的档
            if post_pressure.stage in {"normal", "warning"}:
                break

        if selected is None:
            return {"compacted": False, "reason": "nothing_to_summarize"}

        plan, keep_rows, summarize_rows, summary, mode, preservation_report, post_compact_tokens = selected

        # ── 插入边界消息 ──
        boundary = self._insert_boundary(
            repo,
            conversation,
            summary,
            mode,
            pressure,
            summarize_rows,
            keep_rows,
            preservation_report,
            rehydration,
        )

        return {
            "compacted": True,
            "mode": mode,
            "stage": stage,
            "boundary_id": str(boundary.id),
            "messages_summarized": len(summarize_rows),
            "messages_preserved": len(keep_rows),
            "post_compact_estimated_tokens": post_compact_tokens,
            "post_compact_pressure": calculate_context_pressure(post_compact_tokens).as_dict(),
            "compaction_plan": plan,
            "compaction_attempts": attempts,      # 每档的尝试结果（调试用）
            "rehydration": rehydration,
        }

    def _compaction_plan_for_stage(self, stage: str) -> dict[str, Any]:
        """按档位给出压缩参数（保留多少条、摘要给多少 token）。

        【调用链】被 _compact() 调用

        【三档的力度】
            manual_path  最狠：保留 1/3，摘要 50%   （用户手动触发或已到硬上限）
            aggressive   中等：保留 1/2，摘要 67%
            默认(auto)   温和：用配置值

        【为什么摘要 token 也要减】
            压得越狠，摘要本身也得越短，否则"摘要 + 保留消息"还是会超。
        """
        default_keep = max(2, settings.context_compact_keep_recent_messages)
        default_summary_tokens = max(256, settings.context_compact_summary_tokens)

        if stage == "manual_path":
            return {
                "keep_recent": max(2, min(default_keep, max(2, default_keep // 3))),
                "summary_tokens": min(default_summary_tokens, max(256, int(default_summary_tokens * 0.5))),
                "mode_prefix": "manual_path",
            }
        if stage == "aggressive":
            return {
                "keep_recent": max(2, min(default_keep, max(3, default_keep // 2))),
                "summary_tokens": min(default_summary_tokens, max(256, int(default_summary_tokens * 0.67))),
                "mode_prefix": "aggressive",
            }
        return {"keep_recent": default_keep, "summary_tokens": default_summary_tokens, "mode_prefix": ""}

    def _messages_to_keep(self, rows: list[ChatMessage], keep_count: int | None = None) -> list[ChatMessage]:
        """算出"压缩后要原样保留"的消息。

        【调用链】adjust_keep_start_to_preserve_api_invariants() → 本函数
        【被谁调用】_compact()

        【数据流向】
            入：rows（候选消息，时间升序）、keep_count（保留几条）
            出：要保留的消息列表

        【两道修正】
            ① adjust_keep_start_to_preserve_api_invariants()
               保证 tool_use / tool_result 成对（否则 API 会 400）
            ② 如果保留区第一条是 assistant，而它前面是 user
               → 把那条 user 也带上（否则模型看到"回答"却没有"提问"）
        """
        keep = max(2, keep_count if keep_count is not None else settings.context_compact_keep_recent_messages)
        start_index = max(0, len(rows) - keep)

        # 修正 ①：工具调用成对性
        start_index = adjust_keep_start_to_preserve_api_invariants(
            [self._row_to_message(row) for row in rows],
            start_index,
        )
        keep_rows = rows[start_index:]

        # 修正 ②：别让保留区以 assistant 开头
        if (
            keep_rows
            and keep_rows[0].role == "assistant"
            and start_index > 0
            and rows[start_index - 1].role == "user"
        ):
            keep_rows = [rows[start_index - 1], *keep_rows]
        return keep_rows

    async def _summary_for(
        self,
        repo: Repository,
        conversation: Conversation,
        rows: list[ChatMessage],
        current_message: str,
        max_summary_tokens: int,
    ) -> tuple[str, str, dict[str, Any]]:
        """生成压缩摘要 —— 三种来源，按"质量从高到低"尝试。

        【调用链】追这个函数会进入：
            1. memory_document_text() / render_structured_memory()  → 用结构化记忆当摘要
            2. build_compaction_summary_payload()                   → 拼给 LLM 的输入
            3. llm.chat_text(COMPACT_SUMMARY_SYSTEM_PROMPT, ...)    → 调模型写摘要
            4. fallback_compaction_summary()                        → 纯规则兜底
            5. ensure_compaction_summary_preserves_key_facts()      → 补漏关键信息
        【被谁调用】_compact()

        【数据流向】
            入：repo、conversation、rows（要压缩的消息）、current_message、max_summary_tokens
            出：(摘要文本, 模式名, 保真报告)

        【三种模式，为什么是这个顺序】
            session_memory     已有结构化记忆（facts/decisions/tasks）→ 直接渲染，最准且免费
            full_compaction    LLM 写摘要（质量好但要钱要时间）
            fallback_extractive 纯规则抽取（质量差但永不失败）

        【为什么每种都要过 ensure_compaction_summary_preserves_key_facts】
            模型写摘要常漏文件路径、命令、错误码。
            这一步校验并把漏掉的片段追加到末尾 —— 规则兜底比模型兜底可靠。
        """
        messages = [self._row_to_message(row) for row in rows]

        # ── 模式 1：已有结构化记忆 → 直接用它当摘要 ──
        memory = self.db.query(ConversationMemory).filter(ConversationMemory.conversation_id == conversation.id).first()
        if memory and memory_document_text(memory).strip():
            summary = render_structured_memory(memory, max_tokens=max_summary_tokens)
            continuation = (
                "This conversation was compacted using structured session memory. "
                "Continue from the preserved recent messages and the current user request.\n\n"
                f"{summary}"
            )
            summary, report = ensure_compaction_summary_preserves_key_facts(messages, continuation, max_summary_tokens)
            return summary, "session_memory", report

        # ── 模式 2：LLM 写摘要 ──
        payload = build_compaction_summary_payload(messages)
        llm_summary = await self.llm.chat_text(
            COMPACT_SUMMARY_SYSTEM_PROMPT,
            f"Current user request that triggered pressure:\n{current_message}\n\nConversation segment to compact:\n{payload}",
        )
        summary = format_compaction_summary(llm_summary)

        # ── 模式 3：LLM 返回空 → 规则兜底 ──
        if not summary:
            summary = fallback_compaction_summary(messages, max_tokens=max_summary_tokens)
            summary, report = ensure_compaction_summary_preserves_key_facts(messages, summary, max_summary_tokens)
            return summary, "fallback_extractive", report

        summary, report = ensure_compaction_summary_preserves_key_facts(messages, summary, max_summary_tokens)
        return summary, "full_compaction", report

    def _insert_boundary(
        self,
        repo: Repository,
        conversation: Conversation,
        summary: str,
        mode: str,
        pressure: dict[str, Any],
        summarized_rows: list[ChatMessage],
        keep_rows: list[ChatMessage],
        preservation_report: dict[str, Any],
        rehydration: dict[str, Any],
    ) -> ChatMessage:
        """插入一条"压缩边界"消息 —— 压缩的物理落点。

        【调用链】被 _compact() 调用
        【下游】写 chat_messages 表（role=system, meta.subtype=compact_boundary）

        【数据流向】
            入：summary（摘要正文）、mode（哪种压缩）、pressure（当时压力）、
                summarized_rows（被总结的）、keep_rows（保留的）、
                preservation_report（保真报告）、rehydration（工作状态）
            出：ChatMessage（边界消息）

        【这条消息为什么这么设计】
            它不是给人看的，是给【程序】读的：
                role=system + meta.subtype=compact_boundary  → recent() 靠它划分视野
                meta.preserved_message_ids                    → 保留哪些原消息
                meta.transcript_ref                           → 原文还能用哪个工具查回来
                meta.rehydration                              → 正在编辑的文件 / 激活的技能
                meta.preservation                             → 保真报告（漏了什么）

        【时间戳为什么要微调】
            created_at 默认用 now()，但如果库里已有消息的时间 >= now（时钟偏差），
            边界会排在那些消息后面，recent() 的 `created_at >= boundary.created_at`
            就会漏掉它们。所以这里保证边界时间严格大于所有被压缩消息。
        """
        now = datetime.now(timezone.utc)
        latest_source_time = max(
            (row.created_at for row in [*summarized_rows, *keep_rows] if row.created_at),
            default=None,
        )
        # 保证边界时间严格大于已有消息（+1 微秒）
        if latest_source_time:
            # 有 tzinfo 和没 tzinfo 不能直接比，要对齐
            comparable_now = now if latest_source_time.tzinfo else now.replace(tzinfo=None)
            if latest_source_time >= comparable_now:
                now = latest_source_time + timedelta(microseconds=1)

        boundary = ChatMessage(
            repo_id=repo.id,
            conversation_id=conversation.id,
            role="system",
            content=summary,
            route="context_compaction",
            tool_calls=[],
            meta={
                "subtype": COMPACT_BOUNDARY_SUBTYPE,
                "trigger": "auto",
                "mode": mode,
                "pressure": pressure,
                # 溯源：这次压缩覆盖了哪一段
                "pre_compact_message_id": str(summarized_rows[-1].id) if summarized_rows else None,
                "summarized_from_message_id": str(summarized_rows[0].id) if summarized_rows else None,
                "summarized_to_message_id": str(summarized_rows[-1].id) if summarized_rows else None,
                # 保留区（recent() 靠这个还原）
                "preserved_start_message_id": str(keep_rows[0].id) if keep_rows else None,
                "preserved_message_ids": [str(row.id) for row in keep_rows],
                # 告诉模型"原文还能查"（避免它以为信息永久丢失）
                "transcript_ref": {
                    "repo_id": str(repo.id),
                    "conversation_id": str(conversation.id),
                    "tool": "devflow_get_thread_context",
                },
                "messages_summarized": len(summarized_rows),
                "messages_preserved": len(keep_rows),
                "summary_estimated_tokens": estimate_tokens(summary),
                "preservation": preservation_report,
                "rehydration": rehydration,
            },
            created_at=now,
        )
        self.db.add(boundary)
        self.db.flush()
        return boundary

    def _post_compact_tokens(
        self,
        summary: str,
        keep_rows: list[ChatMessage],
        *,
        current_message: str = "",
        rehydration: dict[str, Any] | None = None,
    ) -> int:
        """算压缩后的大小（= 摘要 + 保留消息 + 当前问题 + 恢复状态）。

        【调用链】被 _compact() 调用（每档试完都算一次，判断够不够）

        【为什么要算全这四项】
            只算摘要会低估 —— 保留的消息、当前问题、rehydration 都占预算。
            低估会导致"以为压够了，实际还是超"，然后下一轮又要压。
        """
        return (
            estimate_tokens(summary)
            + sum(estimate_tokens(row.content) for row in keep_rows)
            + estimate_tokens(current_message)
            + estimate_tokens(dumps_compact(rehydration or {}))
        )

    def _rehydration_state(self, rows: list[ChatMessage]) -> dict[str, Any]:
        """从历史里抢救"工作状态"——正在编辑的文件、激活的技能、用过的工具。

        【调用链】被 _compact() 调用

        【数据流向】
            入：rows（要压缩的消息）
            出：{working_files, active_skills, recent_tools, recovery}

        【为什么需要这个】
            摘要（尤其是 LLM 写的）经常丢掉"我正在改哪个文件"这种具体信息。
            压缩后 Agent 会忘了自己在干什么，从头开始找 —— 体验极差。
            所以这里从 tool_calls 和 agent_events 里把这三类信息单独捞出来，
            作为结构化数据挂在边界的 meta 里，注入 prompt 时用
            [Rehydrated Working State] 段落展示。

        【为什么各字段的 limit 不同】
            working_files 5：文件是最关键的，但太多也没用（人一次只改几个）
            active_skills 6：技能激活的上下文很重要
            recent_tools 8：工具名只是"最近在干什么"的线索

        【remember() 里的 del target[0:...]】
            超过上限时从【头部】删 —— 保留最新的（append 在尾部）。
        """
        working_files: list[str] = []
        active_skills: list[str] = []
        recent_tools: list[str] = []

        def remember(target: list[str], value: Any, limit: int) -> None:
            """去重加入，超限时丢最旧的。"""
            clean = str(value or "").strip()
            if not clean or clean in target:
                return
            target.append(clean)
            if len(target) > limit:
                del target[0 : len(target) - limit]

        for row in rows:
            calls = [item for item in (row.tool_calls or []) if isinstance(item, dict)]
            events = [item for item in (row.meta or {}).get("agent_events") or [] if isinstance(item, dict)]

            # 来源 1：tool_calls（结构化字段）
            for call in calls:
                remember(recent_tools, call.get("tool_name"), 8)
                arguments = call.get("arguments") or {}
                if isinstance(arguments, dict):
                    # 兼容 path / file_path 两种参数名
                    remember(working_files, arguments.get("path") or arguments.get("file_path"), 5)

            # 来源 2：agent_events（Agent 自己上报的事件流）
            for event in events:
                event_type = str(event.get("type") or "")
                if event_type in {"tool_use", "tool_result"}:
                    remember(recent_tools, event.get("tool_name"), 8)
                arguments = event.get("arguments") or {}
                if isinstance(arguments, dict):
                    remember(working_files, arguments.get("path") or arguments.get("file_path"), 5)
                if event_type == "skill_activated":
                    remember(active_skills, event.get("skill_name") or event.get("skill_title"), 6)

        return {
            "working_files": working_files,
            "active_skills": active_skills,
            "recent_tools": recent_tools,
            # 告诉模型"要更多细节就用这两个工具"（和边界的 recovery_hint 呼应）
            "recovery": {
                "exact_transcript_tool": "devflow_get_thread_context",
                "sealed_session_tool": "devflow_read_session_events",
            },
        }

    def _raw_segment_tokens(self, repo_id: uuid.UUID, conversation_id: uuid.UUID) -> int:
        """算"当前压缩段"的原始 token 总量（未压缩体积）。

        【调用链】messages_after_compact_boundary() → 本函数
        【被谁调用】ensure_headroom()

        【为什么要单独算这个】
            组装后的 assembled_tokens 是"已经裁剪过"的数字，可能看起来很小。
            但原始体积仍然很大，下一轮又会爆。
            这个数字用于判断"轻量压缩是否已经够用"。
        """
        rows = MessageStore(self.db).active_for_context(repo_id, conversation_id, limit=1000)
        # 只看边界之后的（已压缩的部分不该再算）
        messages = messages_after_compact_boundary([self._row_to_message(row) for row in rows], include_boundary=False)
        total = 0
        for message in messages:
            total += estimate_tokens(str(message.get("role") or ""))
            total += estimate_tokens(str(message.get("content") or ""))
            # tool_calls 也占预算（序列化后可能很大）
            if message.get("tool_calls"):
                total += estimate_tokens(dumps_compact(message.get("tool_calls")))
        return total

    def _row_to_message(self, row: ChatMessage) -> dict[str, Any]:
        """ORM 对象 → dict（压缩函数只认 dict）。

        【调用链】被 _compact() / _messages_to_keep() / _summary_for() / _raw_segment_tokens() 调用

        注意把 id 转成 str：下游要做集合运算和 JSON 序列化，UUID 对象不方便。
        """
        return {
            "id": str(row.id),
            "role": row.role,
            "content": row.content,
            "route": row.route,
            "tool_calls": row.tool_calls or [],
            "meta": row.meta or {},
        }

    def _consecutive_failures(self, repo_id: uuid.UUID, conversation_id: uuid.UUID) -> int:
        """数"上次成功压缩之后连续失败了几次"（熔断器用）。

        【调用链】被 ensure_headroom() 调用

        【数据流向】
            入：repo_id、conversation_id
            出：连续失败次数

        【为什么要"从上次边界之后"开始数】
            成功压缩会插一条 boundary。失败记录都在它之后。
            一旦成功，之前的失败就翻篇了 —— 所以只看边界之后的。

        【为什么遇到非失败记录就 break】
            记录是按时间倒序取的。只要碰到一条不是 compact_failure 的，
            就说明"连续"中断了，不用再往前数。
        """
        boundary = MessageStore(self.db).last_compact_boundary(repo_id, conversation_id)
        query = self.db.query(ChatMessage).filter(
            ChatMessage.repo_id == repo_id,
            ChatMessage.conversation_id == conversation_id,
            ChatMessage.role == "system",
            ChatMessage.route == "context_compaction",
        )
        if boundary and boundary.created_at:
            query = query.filter(ChatMessage.created_at > boundary.created_at)

        count = 0
        for row in query.order_by(ChatMessage.created_at.desc()).limit(settings.context_max_compaction_failures).all():
            if (row.meta or {}).get("subtype") == "compact_failure":
                count += 1
            else:
                break
        return count

    def _record_failure(self, repo_id: uuid.UUID, conversation_id: uuid.UUID, current_message: str, stage: str, exc: Exception) -> ChatMessage:
        """记一条压缩失败（熔断器的计数依据）。

        【调用链】被 ensure_headroom() 调用（压缩抛异常时）

        【为什么要落库而不是只打日志】
            熔断器要跨请求生效 —— 日志是进程内的，落库才能被下一次请求读到。
            另外失败原因对排查很有价值（是 LLM 超时？还是数据问题？）。
        """
        failure = ChatMessage(
            repo_id=repo_id,
            conversation_id=conversation_id,
            role="system",
            content=clip_to_token_budget(f"Context compaction failed during {stage}: {exc}", 240),
            route="context_compaction",
            tool_calls=[],
            meta={
                "subtype": "compact_failure",
                "stage": stage,
                # 当前问题截断存（用于排查"哪类问题容易触发失败"）
                "current_message": clip_to_token_budget(current_message, 120),
            },
            created_at=datetime.now(timezone.utc),
        )
        self.db.add(failure)
        return failure


# ════════════════════════════════════════════════════════════════
# 第 4 层：会话封存（一轮结束后提炼记忆）
# ════════════════════════════════════════════════════════════════

class SessionSealer:
    """会话封存器 —— 一轮对话结束后，把"这轮发生了什么"沉淀成记忆。

    【为什么叫"封存"】
        这一轮结束后就固定下来，不再改动（status=sealed）。
        它同时承担三件事：
            1. 存档：完整记录（消息 + 工具轨迹 + 事件流 + 模型用量）
            2. 提炼：从这轮对话里抽出 facts / decisions / tasks / preferences
            3. 候选：把提炼结果写成"待审核候选"（人工批准后才进 RAG）

    【设计原则：任何一步失败都不能影响已完成的对话】
        记忆提炼失败 → 用规则兜底
        持久化失败 → 只记 warning，session 照样封存
        因为用户已经拿到回答了，不能因为"记忆没存上"就报错。
    """

    def __init__(self, db: Session, llm: LLMClient | None = None) -> None:
        self.db = db
        self.llm = llm or LLMClient()

    async def seal(
        self,
        *,
        repo: Repository | None,
        conversation: Conversation | None,
        user_message: ChatMessage,
        assistant_message: ChatMessage,
        route: str,
        tool_calls: list[dict[str, Any]],
        agent_events: list[dict[str, Any]] | None = None,
        model_usage: list[dict[str, Any]] | None = None,
        context_diagnostics: dict[str, Any] | None = None,
    ) -> ChatSessionModel:
        """封存一轮对话。

        【调用链】追这个函数会进入：
            1. fallback_memory_update()          → 先用规则算一份兜底记忆
            2. ChatSession(...) + flush          → 先落"原始记录"（phase=raw_persisted）
            3. _build_memory_update()            → 调 LLM 提炼记忆
            4. _update_thread_memory()           → 合并进长期记忆
            5. capture_memory_candidates()       → 写待审核候选（memory_hub）
            6. 更新 session 状态为 sealed
        【被谁调用】api/routes/chat.py（一轮回答结束后）

        【数据流向】
            入：repo、conversation、user_message、assistant_message、route、
                tool_calls、agent_events、model_usage、context_diagnostics
            出：ChatSession（status = sealed / sealed_with_warnings）

        【★ 关键顺序：先落原始记录，再做提炼】
            第 2 步先写一条 status="sealing" 的 session（含完整原始记录），
            然后再去调 LLM 提炼。
            这样即使 LLM 超时/崩溃，原始记录也已经存下来了 ——
            用户可以回查"这轮到底发生了什么"，只是没有结构化记忆而已。

        【为什么要 begin_nested()（SAVEPOINT）】
            记忆合并 + 候选写入是一组操作。如果失败，只回滚这一组，
            不能把前面已经落库的 session 一起回滚掉。
        """
        agent_events = agent_events or []
        model_usage = model_usage or []
        context_diagnostics = context_diagnostics or {}

        # ── 步骤 1：先用规则算一份兜底记忆（后面无论成败都用得上）──
        fallback_update = fallback_memory_update(
            user_message.content,
            assistant_message.content,
            route,
            tool_calls,
        )
        fallback_digest = str(fallback_update.get("summary") or "")

        # ── 步骤 2：先落原始记录（phase=raw_persisted）──
        session = ChatSession(
            repo_id=repo.id if repo else None,
            conversation_id=conversation.id if conversation else None,
            user_message_id=user_message.id,
            assistant_message_id=assistant_message.id,
            route=route,
            transcript_json={
                "messages": [
                    {"role": "user", "content": user_message.content, "created_at": self._iso(user_message.created_at)},
                    {"role": "assistant", "content": assistant_message.content, "created_at": self._iso(assistant_message.created_at)},
                ],
                "tool_calls": tool_calls,
                "agent_events": agent_events,
                # 单独抽一份工具结果（便于按类型筛选）
                "tool_use_results": [
                    event for event in agent_events if event.get("type") in {"tool_use", "tool_result"}
                ],
                "model_usage": model_usage,
                "context_diagnostics": context_diagnostics,
                "memory_update": fallback_update,
                "sealing": {"phase": "raw_persisted", "errors": []},
            },
            digest=fallback_digest,
            status="sealing",       # 中间态：原始记录已存，提炼还没做
            sealed_at=None,
        )
        self.db.add(session)
        self.db.flush()

        errors: list[str] = []
        memory_update = fallback_update
        digest = fallback_digest

        # ── 步骤 3：调 LLM 提炼记忆（失败就用兜底）──
        try:
            memory_update = await self._build_memory_update(
                user_message.content,
                assistant_message.content,
                route,
                tool_calls,
                fallback_digest,
                repo_id=repo.id if repo else None,
                conversation_id=conversation.id if conversation else None,
            )
            # _digest 是内部字段，pop 掉不写进记忆
            digest = str(memory_update.pop("_digest", "") or memory_update.get("summary") or fallback_digest)
        except Exception as exc:  # model/network failures must not invalidate the completed chat turn
            errors.append(f"memory_extraction_failed: {exc}")
            memory_update = fallback_update
            digest = fallback_digest

        # ── 步骤 4+5：合并进长期记忆 + 写候选（同一 SAVEPOINT）──
        try:
            with self.db.begin_nested():
                self._update_thread_memory(
                    repo.id if repo else None,
                    conversation.id if conversation else None,
                    digest,
                    memory_update,
                )
                capture_memory_candidates(
                    self.db,
                    repo_id=repo.id if repo else None,
                    conversation_id=conversation.id if conversation else None,
                    session_id=session.id,
                    memory_update=memory_update,
                )
                self.db.flush()
        except Exception as exc:  # keep raw ChatSession even when enrichment fails
            errors.append(f"memory_persistence_failed: {exc}")

        # ── 步骤 6：收尾（写回最终记忆 + 状态）──
        transcript = dict(session.transcript_json or {})
        # 去掉 _ 开头的内部字段（_digest / _update_mode）再存档
        transcript["memory_update"] = {
            key: value for key, value in memory_update.items() if not key.startswith("_")
        }
        transcript["sealing"] = {
            "phase": "completed",
            "errors": errors,
            # 证据库刷新延到下一次请求（避免这里多花几百毫秒）
            "evidence_refresh": "deferred_to_next_request",
        }
        session.transcript_json = transcript
        session.digest = digest
        # 有错误也照样 sealed，只是多一个 warning 标记
        session.status = "sealed_with_warnings" if errors else "sealed"
        session.sealed_at = datetime.now(timezone.utc)
        self.db.flush()
        return session

    async def _build_memory_update(
        self,
        user_message: str,
        assistant_message: str,
        route: str,
        tool_calls: list[dict[str, Any]],
        digest: str,
        *,
        repo_id: uuid.UUID | None,
        conversation_id: uuid.UUID | None,
    ) -> dict[str, Any]:
        """调 LLM 提炼这轮对话的记忆。

        【调用链】fallback_memory_update() / _memory_state() / structured_memory_from_model() → 本函数
        【被谁调用】seal()

        【数据流向】
            入：本轮消息、route、tool_calls、digest（兜底摘要）、repo_id、conversation_id
            出：记忆 dict（含内部字段 _update_mode / _digest）

        【为什么让模型返回"完整快照"而不是"增量"】
            让模型做 merge 容易出错（漏字段、重复条目）。
            改成"给它现有记忆 + 本轮记录，让它返回更新后的完整快照"，
            模型只需做一次改写，结果更可控。
            （对应 _update_mode="snapshot"，后续 _update_thread_memory 会走 snapshot 分支）

        【为什么要区分会话级 / 仓库级】
            有会话 → 更新 ConversationMemory（只在本会话有效）
            没会话 → 更新 ThreadMemory（整个仓库的长期记忆）
        """
        fallback = fallback_memory_update(user_message, assistant_message, route, tool_calls)

        # 取当前记忆（会话级优先）
        if repo_id and conversation_id:
            current_memory: ConversationMemory | ThreadMemory | None = (
                self.db.query(ConversationMemory)
                .filter(ConversationMemory.conversation_id == conversation_id)
                .first()
            )
        else:
            current_memory = (
                self.db.query(ThreadMemory)
                .filter(_repo_filter(ThreadMemory, repo_id))
                .first()
            )
        existing_memory = self._memory_state(current_memory) if current_memory else {}

        prompt = (
            "请增量更新软件工程 Agent 的数据库会话记忆。"
            "输入包含已有记忆和本轮原始记录；返回更新后的完整快照，而不是只返回本轮增量。"
            "只返回 JSON，包含 digest、summary、facts、decisions、open_questions、tasks、"
            "user_preferences、repo_context、citations。summary 必须描述当前工作状态和下一步；"
            "保留重要文件、函数、命令、错误修复和用户纠正，不要编造。"
        )
        parsed = await self.llm.chat_json(
            prompt,
            {
                "route": route,
                "digest": digest,
                "existing_memory": existing_memory,
                "user_message": user_message,
                "assistant_message": assistant_message,
                "tool_calls": tool_calls,
            },
        )

        update = structured_memory_from_model(parsed) if parsed else fallback
        # 模型没给 summary → 用兜底的（summary 是记忆的核心字段，不能空）
        if not update.get("summary"):
            update["summary"] = fallback["summary"]
        if parsed:
            # 标记"这是完整快照"，让 _update_thread_memory 走覆盖而不是合并
            update["_update_mode"] = "snapshot"
            update["_digest"] = str(parsed.get("digest") or parsed.get("summary") or digest)
        return update

    def _memory_state(self, memory: ConversationMemory | ThreadMemory) -> dict[str, Any]:
        """把 ORM 记忆对象转成 dict（给 LLM 当输入）。

        【调用链】被 _build_memory_update() / _update_thread_memory() 调用

        【为什么用 getattr】
            ConversationMemory 和 ThreadMemory 字段几乎一样，但不完全一致
            （ThreadMemory 有 repo_context 等）。用 getattr 带默认值最省事。
        """
        return {
            "summary": memory.summary or "",
            "facts": getattr(memory, "facts", None) or [],
            "decisions": memory.decisions or [],
            "open_questions": memory.open_questions or [],
            "tasks": getattr(memory, "tasks", None) or [],
            "user_preferences": getattr(memory, "user_preferences", None) or [],
            "repo_context": getattr(memory, "repo_context", None) or [],
            "citations": getattr(memory, "citations", None) or [],
        }

    def _new_memory(self, repo_id: uuid.UUID | None, conversation_id: uuid.UUID | None) -> ConversationMemory | ThreadMemory:
        """创建一条空记忆（有会话 → 会话级；否则 → 仓库级）。

        【调用链】被 _update_thread_memory() 调用

        所有列表字段初始化为空列表（不是 None）——
        下游到处做 `or []` 和迭代，None 会多一层判断。
        """
        if repo_id and conversation_id:
            memory = ConversationMemory(
                repo_id=repo_id,
                conversation_id=conversation_id,
                summary="",
                facts=[],
                decisions=[],
                open_questions=[],
                tasks=[],
                user_preferences=[],
                repo_context=[],
                citations=[],
                sessions_incorporated=0,
            )
        else:
            memory = ThreadMemory(
                repo_id=repo_id,
                summary="",
                facts=[],
                decisions=[],
                open_questions=[],
                tasks=[],
                user_preferences=[],
                repo_context=[],
                citations=[],
                sessions_incorporated=0,
            )
        self.db.add(memory)
        self.db.flush()
        return memory

    def _update_thread_memory(
        self,
        repo_id: uuid.UUID | None,
        conversation_id: uuid.UUID | None,
        digest: str,
        memory_update: dict[str, Any],
    ) -> ConversationMemory | ThreadMemory | None:
        """把本轮记忆写进长期记忆（两种模式：覆盖 / 合并）。

        【调用链】structured_memory_from_model() / merge_memory() → 本函数
        【被谁调用】seal()（在 SAVEPOINT 里）

        【数据流向】
            入：repo_id、conversation_id、digest、memory_update（可能含 _update_mode）
            出：更新后的记忆对象

        【两种模式的差别】
            snapshot（模型给了完整快照）→ 直接覆盖，summary 截断到 1100 token
            merge（兜底路径）→ 用 merge_memory 合并旧的 + 新的（去重保序）

        为什么 summary 固定 1100：记忆每次都要注入 prompt，
        不设上限会让它慢慢膨胀到吃掉整个预算。
        """
        # 取现有记忆
        if repo_id and conversation_id:
            memory: ConversationMemory | ThreadMemory | None = self.db.query(ConversationMemory).filter(ConversationMemory.conversation_id == conversation_id).first()
        else:
            memory = self.db.query(ThreadMemory).filter(_repo_filter(ThreadMemory, repo_id)).first()

        if not memory:
            memory = self._new_memory(repo_id, conversation_id)

        if memory_update.get("_update_mode") == "snapshot":
            # 覆盖模式：模型已经给了完整快照
            merged = {
                **structured_memory_from_model(memory_update),
                "summary": clip_to_token_budget(str(memory_update.get("summary") or digest), 1100),
            }
        else:
            # 合并模式：兜底路径，旧 + 新 去重保序
            merged = merge_memory(self._memory_state(memory), memory_update, digest, max_tokens=1100)

        # 逐字段写回（不用 setattr 循环，是为了让 IDE 能跳转、类型检查能生效）
        memory.summary = merged["summary"]
        memory.facts = merged["facts"]
        memory.decisions = merged["decisions"]
        memory.open_questions = merged["open_questions"]
        memory.tasks = merged["tasks"]
        memory.user_preferences = merged["user_preferences"]
        memory.repo_context = merged["repo_context"]
        memory.citations = merged["citations"]
        # 累计已合并的会话数（记忆面板显示"基于 N 轮对话"）
        memory.sessions_incorporated = (memory.sessions_incorporated or 0) + 1
        return memory

    def _iso(self, value: datetime | None) -> str | None:
        """datetime → ISO 字符串（存档用，JSON 不能直接存 datetime）。"""
        return value.isoformat() if value else None


# ════════════════════════════════════════════════════════════════
# 第 5 层：记忆工具（给 Agent 调的 MCP 工具）
# ════════════════════════════════════════════════════════════════

class MemoryTools:
    """三个记忆工具，供 Agent 在对话中主动调用。

    【为什么要给 Agent 工具，而不是全塞进 prompt】
        prompt 预算有限。与其把全部历史塞进去，不如给 Agent "查"的能力：
            需要证据   → search_evidence
            需要原文   → get_thread_context
            需要旧会话 → read_session_events

    【三个工具的分工】
        search_evidence      "有没有相关资料"（模糊，走向量+关键词）
        get_thread_context   "我之前具体说了什么"（精确，走完整流水）
        read_session_events  "上一轮是怎么做的"（历史封存包）
    """

    def __init__(self, db: Session, repo: Repository | None, conversation: Conversation | None = None) -> None:
        self.db = db
        self.repo = repo
        self.repo_id = repo.id if repo else None
        self.conversation = conversation
        self.conversation_id = conversation.id if conversation else None

    async def search_evidence(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """工具：检索证据（语义 + 关键词）。

        【调用链】EvidenceStore.search() / record_recall_event() → 本函数
        【被谁调用】services/agents/chat_agent.py 的工具分发

        【数据流向】
            入：input_data（query / message / limit）
            出：统一工具返回结构（route / tool_calls / answer / citations / results）

        【为什么参数兼容 query 和 message】
            不同模型调工具时参数名可能不一致（有的传 query，有的传 message）。
            两个都接，减少"参数不匹配"导致的调用失败。

        【为什么要记 record_recall_event】
            Agent 说"没找到"时，用户第一个问题就是"它到底查了什么"。
            召回日志能直接回答这个 —— 而且能看出"查到了 0 条"还是"压根没查"。
        """
        query = str(input_data.get("query") or input_data.get("message") or "")
        results = EvidenceStore(self.db).search(self.repo_id, self.conversation_id, query, limit=int(input_data.get("limit") or 5))

        answer = "No evidence matched this query." if not results else "\n".join(
            f"- [{item['source_type']}] {item['title']}: {item['snippet']}" for item in results
        )
        citations = [{"type": item["source_type"], "id": item["source_id"], "title": item["title"]} for item in results]

        record_recall_event(
            self.db,
            repo_id=self.repo_id,
            conversation_id=self.conversation_id,
            tool_name="devflow_search_evidence",
            query=query,
            results=results,
            citations=citations,
            scope="conversation" if self.conversation_id else "project",
            mode="hybrid",
            metadata={"limit": int(input_data.get("limit") or 5)},
        )
        return {
            "route": "memory_search",
            "tool_calls": [{"tool_name": "search_evidence", "arguments": {"query": query}, "status": "success"}],
            "answer": answer,
            "citations": citations,
            "results": results,
        }

    async def get_thread_context(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """工具：精确回查原始对话（完整流水，不受压缩影响）。

        【调用链】MessageStore.timeline() / record_recall_event() → 本函数
        【被谁调用】services/agents/chat_agent.py

        【数据流向】
            入：input_data（keyword / limit / message_id / before_message_id /
                          after_message_id / offset / max_chars）
            出：工具返回结构（answer 里带 [message_id=... chars=a:b/c] 这类定位头）

        【为什么要有 offset / max_chars 分页】
            一条消息可能几万字。一次全返回会撑爆工具结果预算。
            所以按字符切片，并在末尾提示"继续调用 message_id=X, offset=Y"，
            让 Agent 自己决定要不要继续读 —— 这是"按需读取"的核心设计。

        【为什么 message_id 模式要排除 preserved_after_compact】
            那是旧布局的副本，返回它会让 Agent 看到重复内容。
        """
        keyword = str(input_data.get("keyword") or input_data.get("query") or "").strip().lower()
        limit = min(max(int(input_data.get("limit") or 30), 1), 100)
        message_id = self._uuid_or_none(input_data.get("message_id"))
        before_message_id = self._uuid_or_none(input_data.get("before_message_id"))
        after_message_id = self._uuid_or_none(input_data.get("after_message_id"))
        offset = max(int(input_data.get("offset") or 0), 0)
        max_chars = min(max(int(input_data.get("max_chars") or 6000), 200), 16000)

        store = MessageStore(self.db)

        # ── 取消息：按 ID 精确取，或按游标/关键词取 ──
        if message_id:
            message = (
                self.db.query(ChatMessage)
                .filter(
                    _repo_filter(ChatMessage, self.repo_id),
                    _conversation_filter(ChatMessage, self.conversation_id),
                    ChatMessage.id == message_id,
                    # 只取对话类消息（system 消息不给 Agent 看）
                    ChatMessage.role.in_(["user", "assistant", "tool", "tool_result"]),
                )
                .one_or_none()
            )
            messages = [message] if message and (message.meta or {}).get("subtype") != "preserved_after_compact" else []
        else:
            messages = store.timeline(
                self.repo_id,
                self.conversation_id,
                limit=None,                 # 先全取，下面再筛
                before_message_id=before_message_id,
                after_message_id=after_message_id,
            )
            # 关键词过滤（子串匹配）
            if keyword:
                terms = _tokens(keyword)
                messages = [msg for msg in messages if any(term in msg.content.lower() for term in terms)]
            # 只留最后 limit 条
            messages = messages[-limit:]

        # ── 拼输出：每条带定位头，超长自动分页 ──
        lines: list[str] = []
        results: list[dict[str, Any]] = []
        remaining_chars = max_chars

        for message in messages:
            content = message.content or ""
            # message_id 模式支持 offset（继续读上一条没读完的部分）
            chunk_offset = offset if message_id else 0
            chunk = content[chunk_offset : chunk_offset + remaining_chars]
            if not chunk and content:
                continue

            chunk_end = chunk_offset + len(chunk)
            truncated = chunk_end < len(content)
            created_at = message.created_at.isoformat() if message.created_at else ""

            # 定位头：告诉模型"这是哪条消息、读了哪一段"
            lines.append(
                f"[message_id={message.id} role={message.role} created_at={created_at} chars={chunk_offset}:{chunk_end}/{len(content)}]"
            )
            lines.append(chunk)
            # 没读完 → 告诉模型怎么继续（这是"按需读取"的关键提示）
            if truncated:
                lines.append(
                    f"[内容未读完；继续调用 message_id={message.id}, offset={chunk_end}]"
                )

            results.append(
                {
                    "title": f"{message.role} message",
                    "source_type": "chat_message",
                    "source_id": str(message.id),
                    "snippet": chunk,
                    "metadata": {
                        "role": message.role,
                        "created_at": created_at,
                        "offset": chunk_offset,
                        "next_offset": chunk_end if truncated else None,
                        "total_chars": len(content),
                        "truncated": truncated,
                    },
                }
            )
            remaining_chars -= len(chunk)
            if remaining_chars <= 0:
                break

        citations = [
            {"type": "chat_message", "id": item["source_id"], "title": item["title"]}
            for item in results
        ]

        record_recall_event(
            self.db,
            repo_id=self.repo_id,
            conversation_id=self.conversation_id,
            tool_name="devflow_get_thread_context",
            query=keyword or str(message_id or ""),
            results=results,
            citations=citations,
            scope="conversation",
            mode="exact_transcript",
            metadata={
                "limit": limit,
                "message_id": str(message_id) if message_id else None,
                "before_message_id": str(before_message_id) if before_message_id else None,
                "after_message_id": str(after_message_id) if after_message_id else None,
                "offset": offset,
                "max_chars": max_chars,
            },
        )
        return {
            "route": "thread_context",
            "tool_calls": [{"tool_name": "get_thread_context", "arguments": {"keyword": keyword, "message_id": str(message_id or "")}, "status": "success"}],
            "answer": "\n".join(lines) if lines else "当前会话还没有匹配的消息。",
            "citations": citations,
            "results": results,
        }

    async def read_session_events(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """工具：读取已封存的会话记录（含工具轨迹）。

        【调用链】dumps_compact() / record_recall_event() → 本函数
        【被谁调用】services/agents/chat_agent.py

        【数据流向】
            入：input_data（session_id / include_transcript / offset / max_chars / limit）
            出：工具返回结构（answer 里带 Session 头 + 摘要 + 可选完整 transcript）

        【什么时候用】
            "上一轮我们是怎么修这个 bug 的" —— 需要看历史工具调用轨迹。
            get_thread_context 只能看消息文本，看不到 tool_calls 的细节；
            这个工具能看到完整 transcript（含每步工具参数）。

        【include_transcript 的取舍】
            True  → 包含完整 transcript（信息全，但可能几万字，要分页）
            False → 只看 digest（摘要，短）
        """
        session_id = input_data.get("session_id")
        include_transcript = bool(input_data.get("include_transcript", True))
        offset = max(int(input_data.get("offset") or 0), 0)
        max_chars = min(max(int(input_data.get("max_chars") or 8000), 400), 20000)

        query = self.db.query(ChatSession).filter(
            _repo_filter(ChatSession, self.repo_id),
            _conversation_filter(ChatSession, self.conversation_id),
        ).order_by(ChatSession.created_at.desc())

        # ── 取会话：指定 ID 就取一条，否则取最近 N 条 ──
        if session_id:
            parsed_session_id = self._uuid_or_none(session_id)
            if parsed_session_id is None:
                # 非法 ID → 返回空（不报错，让 Agent 自己决定下一步）
                sessions = []
            else:
                sessions = query.filter(ChatSession.id == parsed_session_id).limit(1).all()
        else:
            sessions = query.limit(min(max(int(input_data.get("limit") or 5), 1), 20)).all()

        lines: list[str] = []
        results: list[dict[str, Any]] = []
        remaining_chars = max_chars

        for session in sessions:
            header = f"Session {session.id} route={session.route} sealed_at={session.sealed_at}"
            payload = session.digest
            if include_transcript:
                # transcript 是结构化 JSON，用 dumps_compact 压成紧凑字符串
                payload += "\ntranscript=" + dumps_compact(session.transcript_json or {})

            chunk_offset = offset if session_id else 0
            chunk = payload[chunk_offset : chunk_offset + remaining_chars]
            chunk_end = chunk_offset + len(chunk)
            truncated = chunk_end < len(payload)

            lines.append(header)
            lines.append(chunk)
            if truncated:
                lines.append(
                    f"[会话记录未读完；继续调用 session_id={session.id}, offset={chunk_end}。]"
                )

            results.append(
                {
                    "title": f"Session {session.route}",
                    "source_type": "session",
                    "source_id": str(session.id),
                    "snippet": chunk,
                    "metadata": {
                        "sealed_at": session.sealed_at.isoformat() if session.sealed_at else None,
                        "include_transcript": include_transcript,
                        "offset": chunk_offset,
                        "next_offset": chunk_end if truncated else None,
                        "total_chars": len(payload),
                        "truncated": truncated,
                    },
                }
            )
            remaining_chars -= len(chunk)
            if remaining_chars <= 0:
                break

        citations = [
            {"type": "session", "id": item["source_id"], "title": item["title"]}
            for item in results
        ]

        record_recall_event(
            self.db,
            repo_id=self.repo_id,
            conversation_id=self.conversation_id,
            tool_name="devflow_read_session_events",
            query=str(session_id or "latest sessions"),
            results=results,
            citations=citations,
            scope="conversation",
            mode="sealed_transcript",
            metadata={"include_transcript": include_transcript, "offset": offset, "max_chars": max_chars},
        )
        return {
            "route": "session_events",
            "tool_calls": [{"tool_name": "read_session_events", "arguments": {"session_id": str(session_id or "")}, "status": "success"}],
            "answer": "\n\n".join(lines) if lines else "No sealed sessions found yet.",
            "citations": citations,
            "results": results,
        }

    @staticmethod
    def _uuid_or_none(value: Any) -> uuid.UUID | None:
        """宽容地把输入转成 UUID，失败返回 None。

        【调用链】被 get_thread_context() / read_session_events() 调用

        【为什么要宽容】
            参数来自 LLM 生成的工具调用，可能是空串、"null"、乱码。
            直接抛异常会让整个工具调用失败；返回 None 则退化成"不按 ID 过滤"，
            Agent 还能拿到结果。
        """
        if not value:
            return None
        try:
            return uuid.UUID(str(value))
        except (TypeError, ValueError):
            return None
