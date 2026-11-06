"""
Issue 路由
==========

【路由清单】
    GET  /repos/{repo_id}/issues              列某仓库的 Issue
    GET  /issues/{issue_id}                   查单个 Issue
    POST /issues/{issue_id}/analyze           Issue 分析（一次性返回结构化结论）
    POST /issues/{issue_id}/analyze/stream    Issue 分析（SSE 流式：先出过程，再出结论）
    GET  /issues/{issue_id}/similar           找相似 Issue（查重用）

【数据流向】
    analyze（数据流 4 的"判断"环节）
      → _build_issue_analysis_context()   五路取证
          ① 相似历史 Issue  → rag/retrieval.search_similar_documents(source_type="issue")
          ② 代码线索        → code_search.search_repository_code()（不进 RAG，实时搜）
          ③ 项目文档/记忆   → rag/retrieval(source_types=knowledge_file/memory_note/project_doc)
          ④ 会话上下文      → chat_memory.ContextAssembler.assemble()
          ⑤ 团队画像        → documents(source_type="team_member")
      → services/agents/issue_agent.py::IssueAgent.run()
      → analysis_results 表（input_snapshot 记下"当时用了多少条证据"）

【要点】
    1. 五路取证里，代码走【实时检索】而不是 RAG —— 源码有精确结构，语义检索不如 grep 准。
    2. stream 版本会先流式输出"公开分析过程"（public reasoning），再给最终 JSON。
       注意 prompt 里明确要求"不输出隐藏思维链"——只输出可审计的摘要。
    3. 流式版失败时 db.rollback() 再发 error 事件：SSE 已经返回 200 了，
       没法改状态码，只能靠事件体告诉前端失败。
    4. similar 接口不做 dedup 过滤（会返回自己）—— 调用方自己判断。
"""

import asyncio
import json
from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.models import AnalysisResult, Conversation, Document, Issue, Repository
from app.db.session import get_db
from app.schemas.issues import IssueAnalysisResponse, IssueResponse
from app.services.agents.issue_agent import IssueAgent
from app.services.chat_memory import ContextAssembler
from app.services.code_search import search_repository_code
from app.services.rag.retrieval import search_similar_documents

router = APIRouter()

# 项目类文档的三种来源（不含 issue / pull_request / code_file）
PROJECT_DOC_TYPES = {"knowledge_file", "memory_note", "project_doc"}

# 流式"公开分析过程"的提示词。
# 【关键】这里刻意要求"不输出隐藏链路思维"—— 对外只暴露可审计的摘要，
# 不是模型的私有推理链。
PUBLIC_REASONING_PROMPT = """
你是 DevFlow AI 的 Issue Agent。请用中文流式输出一段“公开分析过程”，帮助用户理解你如何分析当前 Issue。
要求：
- 只输出面向用户的可审计分析摘要，不输出隐藏链路思维或私有推理链。
- 简洁说明你检查了哪些证据、如何判断结论/优先级/复杂度/负责人。
- 不要编造输入中没有的文件、人员、Issue 或 PR。
- 不要输出最终 JSON。
"""


def _encode_sse(event: str, data: dict) -> str:
    """拼一条 SSE 消息。格式必须是 `event: xxx\\ndata: {...}\\n\\n`（末尾空行是分隔符）。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _document_evidence(items: list[dict], default_source_type: str) -> list[dict]:
    """把检索结果规整成统一的证据结构。

    snippet 或 title 为空的直接丢掉 —— 没有正文的证据没法给用户核对。
    """
    evidence = []
    for item in items:
        snippet = str(item.get("snippet") or "").strip()
        title = str(item.get("title") or "").strip()
        if not snippet or not title:
            continue
        evidence.append(
            {
                "source_type": item.get("source_type") or default_source_type,
                "title": title,
                "snippet": snippet,
                "source_id": str(item.get("source_id") or item.get("id") or ""),
                "score": item.get("score"),
                "metadata": item.get("metadata") or {},
            }
        )
    return evidence


def _fallback_project_documents(db: Session, repo_id: UUID) -> list[dict]:
    """向量检索没结果时的兜底：直接取最近 4 篇项目文档。

    为什么需要：新仓库的文档可能还没索引完，纯靠向量检索会一条都取不到，
    导致 Agent 在"完全没背景"的情况下瞎猜。
    """
    rows = (
        db.query(Document)
        .filter(Document.repo_id == repo_id, Document.source_type.in_(PROJECT_DOC_TYPES))
        .order_by(Document.created_at.desc())
        .limit(4)
        .all()
    )
    return [
        {
            "source_type": doc.source_type,
            "title": doc.title,
            "snippet": doc.content[:260],
            "source_id": str(doc.source_id),
            "score": None,
            "metadata": doc.meta or {},
        }
        for doc in rows
    ]


def _conversation_evidence(db: Session, issue: Issue, conversation_id: UUID | None, query: str) -> list[dict]:
    """从当前会话里捞相关证据（用户可能刚才已经聊过这个 Issue）。"""
    if not conversation_id:
        return []
    conversation = db.get(Conversation, conversation_id)
    # 会话必须属于同一个仓库，否则会把别的项目的信息串进来
    if conversation is None or conversation.repo_id != issue.repo_id:
        return []
    repo = db.get(Repository, issue.repo_id)
    assembled = ContextAssembler(db).assemble(repo, conversation, query, limit=8)
    evidence = _document_evidence(assembled.evidence, "conversation")
    # 会话的摘要（system_context）本身也是一条证据 —— 它包含"之前聊了什么"
    if assembled.system_context.strip():
        evidence.append(
            {
                "source_type": "conversation",
                "title": f"会话上下文：{conversation.title}",
                "snippet": assembled.system_context[:320],
                "source_id": str(conversation.id),
                "score": None,
            }
        )
    return evidence[:5]


def _issue_payload(issue: Issue) -> dict:
    """Issue ORM → 传给 Agent 的纯 dict（只挑 Agent 需要的字段）。"""
    return {
        "id": str(issue.id),
        "title": issue.title,
        "body": issue.body,
        "labels": issue.labels,
        "number": issue.number,
        "state": issue.state,
        "assignees": issue.assignees,
        "author": issue.author,
    }


def _build_issue_analysis_context(db: Session, issue: Issue, conversation_id: UUID | None = None) -> tuple[str, dict, dict]:
    """五路取证，返回 (query, context, counts)。

    counts 会存进 analysis_results.input_snapshot ——
    事后排查"为什么这次分析不准"时，能看出是哪一路证据取空了。
    """
    query = f"{issue.title}\n{issue.body or ''}"
    repo = db.get(Repository, issue.repo_id)

    # ① 相似历史 Issue（查重）
    similar_docs = search_similar_documents(db, str(issue.repo_id), query, "issue", 4)
    duplicates = [
        {
            "issue_id": item["source_id"],
            "title": item["title"],
            # 分数夹到 [0,1]：融合分偶尔会因浮点误差略微超出
            "score": max(0.0, min(1.0, float(item["score"]))),
        }
        for item in similar_docs
        # 排除自己 —— 否则每个 Issue 都会"重复自己"
        if item["source_id"] != str(issue.id)
    ]

    # ② 代码线索：走实时检索（不进 RAG）
    code_docs = _document_evidence(
        search_repository_code(repo, query, limit=4) if repo else [],
        "code",
    )

    # ③ 项目文档 / 记忆笔记
    project_docs = _document_evidence(
        search_similar_documents(
            db,
            str(issue.repo_id),
            query,
            limit=4,
            source_types=sorted(PROJECT_DOC_TYPES),
        ),
        "document",
    )
    if not project_docs:
        project_docs = _fallback_project_documents(db, issue.repo_id)

    # ④ 会话上下文
    chat_evidence = _conversation_evidence(db, issue, conversation_id, query)

    # ⑤ 团队画像（负责人推荐的依据）
    team_members = [
        {
            # 名字优先取 meta，兜底从标题里剥掉前缀
            "name": (doc.meta or {}).get("name") or doc.title.replace("团队成员：", "").replace("Team member: ", ""),
            "role": (doc.meta or {}).get("role") or "",
            "strengths": (doc.meta or {}).get("strengths") or "",
            "techStack": (doc.meta or {}).get("tech_stack") or "",
        }
        for doc in db.query(Document).filter(Document.repo_id == issue.repo_id, Document.source_type == "team_member").all()
    ]

    context = {
        "repo_id": str(issue.repo_id),
        "duplicate_candidates": duplicates,
        "code_references": code_docs,
        "project_documents": project_docs,
        "conversation_evidence": chat_evidence,
        "team_members": team_members,
    }
    counts = {
        "duplicates": len(duplicates),
        "code": len(code_docs),
        "documents": len(project_docs),
        "conversation": len(chat_evidence),
        "team_members": len(team_members),
    }
    return query, context, counts


def _persist_issue_analysis(
    db: Session,
    issue: Issue,
    result: dict,
    model_name: str,
    conversation_id: UUID | None,
    evidence_counts: dict,
) -> AnalysisResult:
    """落库分析结果。

    input_snapshot 里存的是"输入侧快照"（用了哪些证据、多少条），
    和 result_json（输出侧）分开 —— 复盘时能区分"输入不够"和"模型判断错"。
    """
    analysis = AnalysisResult(
        target_type="issue",
        target_id=issue.id,
        analysis_type="issue_analysis",
        input_snapshot={
            "issue_id": str(issue.id),
            "conversation_id": str(conversation_id) if conversation_id else None,
            "evidence_counts": evidence_counts,
        },
        result_json=result,
        model_name=model_name,
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis


def _trace_item(kind: str, title: str, content: str, status: str = "done") -> dict:
    """构造一条前端"分析轨迹"条目（前端按 kind 显示不同图标）。"""
    return {"kind": kind, "title": title, "content": content, "status": status}


def _public_reasoning_fallback(issue: Issue, context: dict) -> list[str]:
    """LLM 不可用时的兜底分析过程（纯模板，说实话：只报数量，不下结论）。"""
    duplicates = context.get("duplicate_candidates") or []
    code_refs = context.get("code_references") or []
    docs = context.get("project_documents") or []
    members = context.get("team_members") or []
    return [
        f"- 我先读取 Issue #{issue.number} 的标题、正文、标签和当前状态，确认它描述的是「{issue.title}」。\n",
        f"- 然后检索相似历史 Issue，找到 {len(duplicates)} 个候选；如果相似度很高，会优先建议合并到已有 Issue。\n",
        f"- 接着查看代码和项目文档证据：代码线索 {len(code_refs)} 条，文档/记忆线索 {len(docs)} 条，用于判断影响范围和复杂度。\n",
        f"- 最后结合团队画像，当前可参考成员 {len(members)} 人；负责人只会从已配置成员或明确方向中选择。\n",
        "- 这些公开分析步骤会汇总成结构化结论：结论、优先级、复杂度、负责人、证据、checklist 和可选草稿。\n",
    ]


async def _stream_public_reasoning(agent: IssueAgent, issue: Issue, payload: dict, context: dict) -> AsyncIterator[str]:
    """流式产出"公开分析过程"。

    只传 counts 和每类前 3 条预览 —— 目的是让模型知道"有哪些证据"，
    而不是把全部证据塞进去（那样输出会变成逐条复述）。
    """
    streamed = False
    user_payload = {
        "issue": payload,
        "evidence_counts": {
            "duplicates": len(context.get("duplicate_candidates") or []),
            "code_references": len(context.get("code_references") or []),
            "project_documents": len(context.get("project_documents") or []),
            "conversation_evidence": len(context.get("conversation_evidence") or []),
            "team_members": len(context.get("team_members") or []),
        },
        "context_preview": {
            "duplicates": context.get("duplicate_candidates", [])[:3],
            "code_references": context.get("code_references", [])[:3],
            "project_documents": context.get("project_documents", [])[:3],
            "team_members": context.get("team_members", [])[:3],
        },
    }
    async for chunk in agent.llm.chat_text_stream(
        PUBLIC_REASONING_PROMPT,
        [{"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}],
    ):
        streamed = True
        yield chunk
    if streamed:
        return
    # 模型一个字都没吐（没配 key / 调用失败）→ 用模板兜底，保证前端不空屏
    for line in _public_reasoning_fallback(issue, context):
        await asyncio.sleep(0.06)  # 轻微延迟，模拟打字机效果
        yield line


async def _issue_analysis_sse(issue_id: UUID, conversation_id: UUID | None, db: Session) -> AsyncIterator[str]:
    """SSE 生成器：trace（步骤）→ thinking_delta（流式过程）→ final（结论）。

    事件顺序就是前端的展示顺序，不能乱：
      trace       步骤卡片（"正在检索代码…"）
      thinking_delta  流式文字增量
      final       最终结构化结果
      error       失败（此时 HTTP 已经是 200，只能靠事件告知）
    """
    issue = db.get(Issue, issue_id)
    if issue is None:
        yield _encode_sse("error", {"message": "未找到 Issue"})
        return
    agent = IssueAgent()
    payload = _issue_payload(issue)
    try:
        yield _encode_sse("trace", _trace_item("step", "读取 Issue", f"读取 Issue #{issue.number} 的标题、正文、标签、状态和已分配人员。", "running"))
        # 五路取证（同步操作，但很快）
        query, context, counts = _build_issue_analysis_context(db, issue, conversation_id)
        yield _encode_sse("trace", _trace_item("tool", "检索相似历史", f"找到 {counts['duplicates']} 个相似 Issue 候选。"))
        yield _encode_sse("trace", _trace_item("tool", "检索代码与文档", f"找到 {counts['code']} 条代码线索、{counts['documents']} 条项目文档/记忆线索。"))
        yield _encode_sse("trace", _trace_item("tool", "读取团队画像", f"读取 {counts['team_members']} 位团队成员，用于建议负责人。"))
        if counts["conversation"]:
            # 只有真的有会话证据才显示这一步，避免前端出现"0 条"的空卡片
            yield _encode_sse("trace", _trace_item("tool", "读取会话上下文", f"找到 {counts['conversation']} 条与当前会话相关的证据。"))
        yield _encode_sse("trace", _trace_item("thinking", "LLM 公开分析过程", "开始综合 Issue、证据和团队画像。", "running"))
        async for chunk in _stream_public_reasoning(agent, issue, payload, context):
            yield _encode_sse("thinking_delta", {"content": chunk})
        # 过程播完再跑正式分析（拿结构化 JSON）
        result = await agent.run(payload, context)
        analysis = _persist_issue_analysis(db, issue, result, agent.llm.model, conversation_id, counts)
        yield _encode_sse("trace", _trace_item("result", "生成结构化结论", f"结论：{result.get('conclusion')}；优先级：{result.get('priority')}；复杂度：{result.get('complexity')}；负责人：{result.get('suggested_owner')}。"))
        yield _encode_sse("final", {"analysis_id": str(analysis.id), "result": result})
    except Exception as exc:
        # 响应头已经发出去了，改不了状态码，只能回滚事务 + 发 error 事件
        db.rollback()
        yield _encode_sse("error", {"message": f"分析失败：{exc}"})


@router.get("/repos/{repo_id}/issues", response_model=list[IssueResponse])
async def list_issues(repo_id: UUID, db: Session = Depends(get_db)) -> list[Issue]:
    """列 Issue，编号大的在前（新的在前）。"""
    return db.query(Issue).filter(Issue.repo_id == repo_id).order_by(Issue.number.desc()).all()


@router.get("/issues/{issue_id}", response_model=IssueResponse)
async def get_issue(issue_id: UUID, db: Session = Depends(get_db)) -> Issue:
    """查单个 Issue 详情。"""
    issue = db.get(Issue, issue_id)
    if issue is None:
        raise HTTPException(status_code=404, detail="未找到 Issue")
    return issue


@router.post("/issues/{issue_id}/analyze", response_model=IssueAnalysisResponse)
async def analyze_issue(
    issue_id: UUID,
    conversation_id: UUID | None = Query(default=None),
    db: Session = Depends(get_db),
) -> IssueAnalysisResponse:
    """一次性分析（不流式）。前端"重新分析"按钮用它。"""
    issue = db.get(Issue, issue_id)
    if issue is None:
        raise HTTPException(status_code=404, detail="未找到 Issue")
    agent = IssueAgent()
    _query, context, counts = _build_issue_analysis_context(db, issue, conversation_id)
    result = await agent.run(_issue_payload(issue), context)
    analysis = _persist_issue_analysis(db, issue, result, agent.llm.model, conversation_id, counts)
    return IssueAnalysisResponse(analysis_id=analysis.id, result=result)


@router.post("/issues/{issue_id}/analyze/stream")
async def analyze_issue_stream(
    issue_id: UUID,
    conversation_id: UUID | None = Query(default=None),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """流式分析。返回 text/event-stream，事件类型见 _issue_analysis_sse。"""
    return StreamingResponse(_issue_analysis_sse(issue_id, conversation_id, db), media_type="text/event-stream")


@router.get("/issues/{issue_id}/similar")
async def similar_issues(issue_id: UUID, db: Session = Depends(get_db)) -> list[dict]:
    """找相似 Issue（查重用）。注意结果里会包含它自己。"""
    issue = db.get(Issue, issue_id)
    if issue is None:
        raise HTTPException(status_code=404, detail="未找到 Issue")
    return search_similar_documents(db, str(issue.repo_id), f"{issue.title}\n{issue.body or ''}", "issue", 5)
