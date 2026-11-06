"""
CI 根因分析路由
===============

【路由清单】
    GET  /repos/{repo_id}/workflow-runs               列 workflow 运行记录
    GET  /workflow-runs/{run_id}                      查单次运行
    GET  /workflow-runs/{run_id}/failed-jobs          只看失败 job 和失败 step
    POST /workflow-runs/{run_id}/analyze              CI 根因分析（一次性）
    POST /workflow-runs/{run_id}/analyze/stream       CI 根因分析（SSE 六步流式）

【数据流向】
    analyze（数据流 4 的 CI 分支）
      → _prepare_ci_pr_snapshot()   先猜"这次 CI 是哪个 PR 触发的"，再检出那个分支
      → _build_ci_analysis_context()  七路取证
          ① 失败 job / step（从 run.jobs JSON 里筛）
          ② 首个错误块（_first_error 从日志里找关键词，前后各取几行）
          ③ 日志里的文件路径（_related_files_from_logs 正则）
          ④ 代码（主分支实时检索）
          ⑤ workflow 配置（限定 .github/workflows 目录）
          ⑥ 项目文档 / 记忆
          ⑦ 最近 5 个 PR + 会话上下文
      → _augment_ci_context_with_snapshot()  用 PR 快照补代码 + 图谱影响面
      → services/agents/ci_debug_agent.py::CIDebugAgent.run()
      → analysis_results 表

【要点】
    1. CI 日志必须先脱敏再送语义检索（sanitize_ci_log）：
       日志里常有 token / 私钥 / 环境变量，直接进向量库等于泄漏。
    2. _first_error 的窗口是"前 2 行 + 后 5 行"：
       前面少后面多，因为真正的堆栈信息在报错行之后。
    3. _related_pr_for_run 先按 head_branch 在日志里匹配，匹配不上才退回"最近更新的 PR"。
       直接取最近 PR 可能是错的（CI 不一定由它触发），所以先试着精确匹配。
    4. 快照可能拿不到（没有关联 PR / 仓库没 clone），此时退回主分支代码继续分析。
"""

import asyncio
import json
import re
from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import or_
from sqlalchemy.orm import Session, selectinload

from app.db.models import AnalysisResult, CodeRelation, CodeSymbol, Conversation, Document, PullRequest, Repository, WorkflowRun
from app.db.session import get_db
from app.schemas.ci import CIAnalysisResponse, WorkflowRunResponse
from app.services.agents.ci_debug_agent import CIDebugAgent
from app.services.chat_memory import ContextAssembler
from app.services.code_search import search_repository_code
from app.services.rag.indexing import sanitize_ci_log
from app.services.rag.retrieval import search_similar_documents
from app.services.worktree_manager import prepare_pr_worktree

router = APIRouter()

PROJECT_DOC_TYPES = {"knowledge_file", "memory_note", "project_doc"}

# 流式"公开排查过程"的提示词。同样要求不暴露私有推理链。
PUBLIC_REASONING_PROMPT = """
你是 DevFlow AI 的 CI 根因分析 Agent。请用中文流式输出一段“公开排查过程”，帮助用户理解你如何按计划定位失败 CI。
要求：
- 只输出面向用户的可审计分析摘要，不输出隐藏链路思维或私有推理链。
- 按计划说明你检查了失败 job、失败 step、首个错误块、失败类型、相关代码/配置、最近 PR，并如何判断是否阻塞合并。
- 不要编造输入中没有的 job、step、文件、PR 或错误日志。
- 不要输出最终 JSON。
"""


def _encode_sse(event: str, data: dict) -> str:
    """拼 SSE 消息。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _trace_item(kind: str, title: str, content: str, status: str = "done") -> dict:
    """构造前端步骤卡片。"""
    return {"kind": kind, "title": title, "content": content, "status": status}


def _document_evidence(items: list[dict], default_source_type: str) -> list[dict]:
    """检索结果 → 统一证据结构。缺正文或标题的丢掉。"""
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


def _dedupe_evidence(items: list[dict], limit: int) -> list[dict]:
    """按 (类型, id, 标题) 去重后截断（快照代码和主分支代码会重复）。"""
    output = []
    seen = set()
    for item in items:
        key = (item.get("source_type"), item.get("source_id"), item.get("title"))
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
        if len(output) >= limit:
            break
    return output


def _fallback_project_documents(db: Session, repo_id: UUID) -> list[dict]:
    """向量检索空结果时的兜底：最近 4 篇项目文档。"""
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


def _conversation_evidence(db: Session, run: WorkflowRun, conversation_id: UUID | None, query: str) -> list[dict]:
    """当前会话里和这次 CI 相关的证据。"""
    if not conversation_id:
        return []
    conversation = db.get(Conversation, conversation_id)
    if conversation is None or conversation.repo_id != run.repo_id:
        return []
    repo = db.get(Repository, run.repo_id)
    assembled = ContextAssembler(db).assemble(repo, conversation, query, limit=8)
    evidence = _document_evidence(assembled.evidence, "conversation")
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


def _ci_plan() -> list[str]:
    """固定六步排查计划。"""
    return [
        "找出失败 job 和失败 step。",
        "提取首个关键错误块。",
        "判断失败类型：依赖、测试、构建、权限、环境变量或未知。",
        "搜索相关代码、测试文件和配置文件。",
        "关联最近 PR 或变更模块。",
        "输出根因、修复步骤和阻塞判断。",
    ]


def _workflow_payload(run: WorkflowRun) -> dict:
    """WorkflowRun ORM → Agent 需要的 dict。"""
    return {
        "id": str(run.id),
        "name": run.name,
        "status": run.status,
        "conclusion": run.conclusion,
        "jobs": run.jobs or [],
        "logs_text": run.logs_text,
        "html_url": run.html_url,
    }


def _failed_jobs(run: WorkflowRun) -> list[dict]:
    """筛出失败 job，并在每个 job 里挂上它的失败 step。

    run.jobs 是 JSON 列（存 GitHub 返回的 jobs 数组），
    这里做一次"扁平化 + 过滤"，让 Agent 不用自己解析嵌套结构。
    """
    return [
        {
            **job,
            "failed_steps": [step for step in job.get("steps", []) if step.get("conclusion") == "failure"],
        }
        for job in (run.jobs or [])
        if job.get("conclusion") == "failure"
    ]


def _first_error(logs: str | None) -> str | None:
    """从日志里提取"首个错误块"。

    窗口是前 2 行 + 后 5 行：前面少后面多 ——
    因为真正的堆栈 / 原因通常在报错行之后。
    """
    if not logs or not logs.strip():
        return None
    lines = [line.strip() for line in logs.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        lowered = line.lower()
        # 关键词覆盖：Python / Node / 通用
        if any(token in lowered for token in ["error", "failed", "failure", "exception", "traceback", "assertionerror", "npm err", "fatal"]):
            start = max(0, index - 2)
            end = min(len(lines), index + 5)
            # 900 字符上限：再长就超预算了
            return "\n".join(lines[start:end])[:900]
    # 一个关键词都没命中 → 返回日志开头 6 行（可能是自定义错误格式）
    return "\n".join(lines[:6])[:900]


def _related_files_from_logs(logs: str | None) -> list[str]:
    """从日志里正则捞出文件路径（堆栈里的文件）。

    扩展名白名单限制在常见代码/配置，避免把 URL、时间戳误当路径。
    """
    if not logs:
        return []
    candidates = re.findall(r"[\w./\\-]+\.(?:py|ts|tsx|js|jsx|json|ya?ml|toml|ini|cfg|md)", logs)
    normalized = []
    for item in candidates:
        # 剥掉堆栈里常见的包裹符号：`file.py:12` → file.py
        cleaned = item.strip("`'\"()[]{}:,;").replace("\\", "/")
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized[:8]


def _related_pr_for_run(db: Session, run: WorkflowRun) -> PullRequest | None:
    """猜这次 CI 是哪个 PR 触发的。

    策略：先拿最近 8 个 PR，逐个看 head_branch 有没有出现在日志里
    （GitHub Actions 日志里常打 refs/pull/123/merge 或分支名）。
    都匹配不上就退回"最近更新的那个" —— 不保证对，但比没有强。
    """
    candidates = (
        db.query(PullRequest)
        .options(selectinload(PullRequest.files))
        .filter(PullRequest.repo_id == run.repo_id)
        .order_by(PullRequest.updated_at.desc(), PullRequest.created_at.desc())
        .limit(8)
        .all()
    )
    if not candidates:
        return None
    logs = f"{run.name}\n{run.logs_text or ''}".lower()
    for pr in candidates:
        if pr.head_branch and pr.head_branch.lower() in logs:
            return pr
    return candidates[0]


def _snapshot_summary(snapshot: dict) -> dict:
    """快照结果精简（比 PR 版多带一个 pull_request 字段）。"""
    return {
        "status": snapshot.get("status"),
        "message": snapshot.get("message"),
        "name": snapshot.get("snapshot_name") or "PR 分支快照",
        "branch": snapshot.get("branch"),
        "commit_sha": snapshot.get("commit_sha"),
        "pr_id": snapshot.get("pr_id"),
        "pr_number": snapshot.get("pr_number"),
        "indexed_symbols": snapshot.get("indexed_symbols"),
        "pull_request": snapshot.get("pull_request"),
    }


def _prepare_ci_pr_snapshot(db: Session, run: WorkflowRun, enabled: bool) -> dict:
    """准备 CI 关联 PR 的分支快照。任何一步失败都返回带 status 的 dict。"""
    if not enabled:
        return {"status": "skipped", "message": "未请求 PR 分支快照。"}
    repo = db.get(Repository, run.repo_id)
    if repo is None:
        return {"status": "missing_repo", "message": "未找到仓库。"}
    pr = _related_pr_for_run(db, run)
    if pr is None:
        # 没有关联 PR 很常见（比如 push 到 main 触发的 CI）—— 不算错误
        return {"status": "missing_pr", "message": "未找到可关联的 PR。"}
    snapshot = prepare_pr_worktree(db, repo, pr)
    # 把 PR 基础信息也带上，前端能显示"关联到 PR #123"
    snapshot["pull_request"] = {
        "id": str(pr.id),
        "number": pr.number,
        "title": pr.title,
        "head_branch": pr.head_branch,
        "state": pr.state,
    }
    return snapshot


def _ci_code_graph_impact(db: Session, pr_id: UUID, related_files: list[str]) -> dict:
    """CI 版的图谱影响面：按"日志里出现的文件"筛符号和关系。"""
    paths = list(dict.fromkeys(related_files))
    symbol_query = db.query(CodeSymbol).filter(CodeSymbol.pr_id == pr_id)
    if paths:
        symbol_query = symbol_query.filter(CodeSymbol.path.in_(paths))
    symbols = symbol_query.order_by(CodeSymbol.path.asc(), CodeSymbol.start_line.asc()).limit(30).all()

    symbol_names = [symbol.name for symbol in symbols]
    relation_query = db.query(CodeRelation).filter(CodeRelation.pr_id == pr_id)
    relation_filters = []
    if paths:
        relation_filters.append(CodeRelation.path.in_(paths))
    if symbol_names:
        relation_filters.extend([CodeRelation.source_name.in_(symbol_names), CodeRelation.target_name.in_(symbol_names)])
    if relation_filters:
        relation_query = relation_query.filter(or_(*relation_filters))
    relations = relation_query.limit(60).all()

    return {
        "related_files": paths[:30],
        "symbols": [
            {
                "name": symbol.name,
                "kind": symbol.kind,
                "path": symbol.path,
                "start_line": symbol.start_line,
                "branch": symbol.branch,
                "commit_sha": symbol.commit_sha,
            }
            for symbol in symbols
        ],
        "relations": [
            {
                "source": relation.source_name,
                "target": relation.target_name,
                "type": relation.relation_type,
                "path": relation.path,
                "line": (relation.meta or {}).get("line"),
            }
            for relation in relations
        ],
    }


def _augment_ci_context_with_snapshot(db: Session, run: WorkflowRun, query: str, context: dict, counts: dict, snapshot: dict) -> None:
    """用快照补代码证据 + 图谱影响面（原地修改）。"""
    context["pr_snapshot"] = _snapshot_summary(snapshot)
    counts["snapshot_code"] = 0
    counts["impacted_symbols"] = 0
    counts["impacted_relations"] = 0
    if snapshot.get("status") != "ready" or not snapshot.get("pr_id"):
        # 没快照也要写这个字段（空的），前端才不会拿到 undefined
        context["code_graph_impact"] = {"related_files": context.get("related_files_from_logs") or [], "symbols": [], "relations": []}
        return

    repo = db.get(Repository, run.repo_id)
    snapshot_docs = _document_evidence(
        search_repository_code(
            repo,
            query,
            limit=8,
            checkout_path=snapshot.get("snapshot_path"),
            metadata={
                "checkout_kind": "pr_snapshot",
                "pr_id": str(snapshot["pr_id"]),
                "pr_number": snapshot.get("pr_number"),
                "branch": snapshot.get("branch"),
                "commit_sha": snapshot.get("commit_sha"),
            },
        )
        if repo and snapshot.get("snapshot_path")
        else [],
        "code",
    )
    # 快照代码优先，再拼主分支的，去重取 10 条
    context["code_references"] = _dedupe_evidence([*snapshot_docs, *(context.get("code_references") or [])], 10)
    pr_uuid = UUID(str(snapshot["pr_id"]))
    impact = _ci_code_graph_impact(db, pr_uuid, context.get("related_files_from_logs") or [])
    context["code_graph_impact"] = impact
    counts["snapshot_code"] = len(snapshot_docs)
    counts["impacted_symbols"] = len(impact["symbols"])
    counts["impacted_relations"] = len(impact["relations"])
    counts["code"] = len(context["code_references"])


def _build_ci_analysis_context(db: Session, run: WorkflowRun, conversation_id: UUID | None = None) -> tuple[str, dict, dict]:
    """七路取证，返回 (code_query, context, counts)。

    【关键】semantic_query 走 sanitize_ci_log 脱敏后才送检索 ——
    CI 日志里常有 token / 私钥，直接进向量库等于永久泄漏。
    code_query 不脱敏：它是拿去 grep 代码的，不走向量库。
    """
    repo = db.get(Repository, run.repo_id)
    first_error = _first_error(run.logs_text)
    failed_jobs = _failed_jobs(run)
    # 把所有 job 的失败 step 拍平成一个列表
    failed_steps = [
        step
        for job in failed_jobs
        for step in job.get("failed_steps", [])
        if isinstance(step, dict)
    ]
    related_files = _related_files_from_logs(run.logs_text)
    code_query = "\n".join(
        filter(None, [run.name, run.conclusion or "", first_error or "", run.logs_text[:5000] if run.logs_text else ""])
    )
    # ★ 脱敏后再做语义检索
    semantic_query = sanitize_ci_log(
        "\n".join(filter(None, [run.name, first_error or "", run.logs_text[:3000] if run.logs_text else ""]))
    )

    # ① 代码（主分支）
    code_docs = _document_evidence(
        search_repository_code(repo, code_query, limit=6) if repo else [],
        "code",
    )
    # ② 项目文档 / 记忆
    project_docs = _document_evidence(
        search_similar_documents(
            db,
            str(run.repo_id),
            semantic_query,
            limit=4,
            source_types=sorted(PROJECT_DOC_TYPES),
        ),
        "document",
    )
    if not project_docs:
        project_docs = _fallback_project_documents(db, run.repo_id)
    # ③ workflow 配置（限定 .github/workflows：CI 失败常常是 yml 配错）
    workflow_docs = _document_evidence(
        search_repository_code(repo, run.name, limit=4, relative_path=".github/workflows") if repo else [],
        "workflow",
    )
    # ④ 最近 5 个 PR（用来关联"这次改动是不是它引起的"）
    recent_prs = [
        {
            "number": pr.number,
            "title": pr.title,
            "state": pr.state,
            "author": pr.author,
            "updated_at": pr.updated_at.isoformat() if pr.updated_at else None,
        }
        for pr in (
            db.query(PullRequest)
            .filter(PullRequest.repo_id == run.repo_id)
            .order_by(PullRequest.updated_at.desc())
            .limit(5)
            .all()
        )
    ]
    # ⑤ 会话上下文
    chat_evidence = _conversation_evidence(db, run, conversation_id, semantic_query)

    context = {
        "repo_id": str(run.repo_id),
        "plan": _ci_plan(),
        "failed_jobs": failed_jobs,
        "failed_steps": failed_steps,
        "first_error": first_error,
        "related_files_from_logs": related_files,
        "code_references": code_docs,
        "workflow_references": workflow_docs,
        "project_documents": project_docs,
        "recent_prs": recent_prs,
        "conversation_evidence": chat_evidence,
    }
    counts = {
        "failed_jobs": len(failed_jobs),
        "failed_steps": len(failed_steps),
        "code": len(code_docs),
        "workflow": len(workflow_docs),
        "documents": len(project_docs),
        "recent_prs": len(recent_prs),
        "conversation": len(chat_evidence),
        "related_files": len(related_files),
    }
    return code_query, context, counts


def _persist_ci_analysis(
    db: Session,
    run: WorkflowRun,
    result: dict,
    model_name: str,
    conversation_id: UUID | None,
    evidence_counts: dict,
) -> AnalysisResult:
    """落库 CI 分析结果。target_type 用 "workflow_run"。"""
    analysis = AnalysisResult(
        target_type="workflow_run",
        target_id=run.id,
        analysis_type="ci_debug",
        input_snapshot={
            "run_id": str(run.id),
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


def _public_reasoning_fallback(run: WorkflowRun, context: dict, counts: dict) -> list[str]:
    """LLM 不可用时的兜底过程（纯模板）。"""
    first_error = context.get("first_error") or "日志中没有明确错误块"
    return [
        f"- 我先制定 CI 排查计划，再读取 workflow「{run.name}」的状态、结论、job 和日志。\n",
        f"- 当前找到失败 job {counts['failed_jobs']} 个、失败 step {counts['failed_steps']} 个，首个关键错误块是：{str(first_error)[:180]}。\n",
        f"- 我从日志中提取到 {counts['related_files']} 个相关文件，并检索代码/配置线索 {counts['code'] + counts['workflow']} 条，其中 PR 快照代码 {counts.get('snapshot_code', 0)} 条。\n",
        f"- 同时参考最近 PR {counts['recent_prs']} 条、项目文档/记忆 {counts['documents']} 条、会话证据 {counts['conversation']} 条。\n",
        "- 最后输出失败类型、根因、修复步骤，以及该 CI 是否阻塞合并。\n",
    ]


async def _stream_public_reasoning(agent: CIDebugAgent, run: WorkflowRun, payload: dict, context: dict, counts: dict) -> AsyncIterator[str]:
    """流式输出公开排查过程。只给预览，不给全量日志。"""
    streamed = False
    user_payload = {
        "workflow_run": payload,
        "plan": context.get("plan") or [],
        "evidence_counts": counts,
        "context_preview": {
            "failed_jobs": context.get("failed_jobs", [])[:3],
            "first_error": context.get("first_error"),
            "related_files_from_logs": context.get("related_files_from_logs", [])[:5],
            "recent_prs": context.get("recent_prs", [])[:3],
            "pr_snapshot": context.get("pr_snapshot"),
            "code_graph_impact": context.get("code_graph_impact"),
            "code_references": context.get("code_references", [])[:3],
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
    for line in _public_reasoning_fallback(run, context, counts):
        await asyncio.sleep(0.06)
        yield line


async def _ci_analysis_sse(run_id: UUID, conversation_id: UUID | None, db: Session) -> AsyncIterator[str]:
    """SSE 六步 CI 排查。"""
    run = db.get(WorkflowRun, run_id)
    if run is None:
        yield _encode_sse("error", {"message": "未找到 workflow 运行"})
        return
    agent = CIDebugAgent()
    payload = _workflow_payload(run)
    plan = _ci_plan()
    try:
        yield _encode_sse("trace", _trace_item("step", "制定排查计划", "\n".join(f"{index + 1}. {item}" for index, item in enumerate(plan)), "done"))
        yield _encode_sse("trace", _trace_item("step", "执行 1/6：定位失败 job/step", f"读取 workflow「{run.name}」的 jobs、steps、状态和结论。", "running"))

        # 先猜关联 PR 并起快照（拿不到也继续）
        snapshot = _prepare_ci_pr_snapshot(db, run, enabled=True)
        yield _encode_sse(
            "trace",
            _trace_item(
                "tool",
                "关联 PR 分支快照",
                str(snapshot.get("message") or f"状态：{snapshot.get('status')}；代码图符号：{snapshot.get('indexed_symbols') or 0}。"),
            ),
        )

        _query, context, counts = _build_ci_analysis_context(db, run, conversation_id)
        _augment_ci_context_with_snapshot(db, run, _query, context, counts, snapshot)
        context["plan"] = plan

        failed_job_names = [str(job.get("name")) for job in context.get("failed_jobs", []) if job.get("name")]
        yield _encode_sse("trace", _trace_item("tool", "执行 1/6：定位失败 job/step", f"失败 job：{', '.join(failed_job_names[:3]) or '未提供'}；失败 step 数：{counts['failed_steps']}。"))
        yield _encode_sse("trace", _trace_item("tool", "执行 2/6：提取首个错误块", str(context.get("first_error") or "日志中没有提取到明确 error 块。")))
        yield _encode_sse("trace", _trace_item("tool", "执行 3/6：判断失败类型", "根据日志关键词、失败命令和 job/step 名称进行分类。"))
        yield _encode_sse("trace", _trace_item("tool", "执行 4/6：搜索代码和配置", f"找到 {counts['code']} 条代码线索，其中 PR 快照代码 {counts.get('snapshot_code', 0)} 条；workflow 配置线索 {counts['workflow']} 条、日志文件路径 {counts['related_files']} 个。"))
        yield _encode_sse("trace", _trace_item("tool", "执行 5/6：关联最近 PR", f"读取最近 PR {counts['recent_prs']} 条，会话证据 {counts['conversation']} 条。"))
        yield _encode_sse("trace", _trace_item("thinking", "LLM 公开排查过程", "开始综合失败日志、代码配置、最近 PR 和项目文档。", "running"))

        async for chunk in _stream_public_reasoning(agent, run, payload, context, counts):
            yield _encode_sse("thinking_delta", {"content": chunk})

        result = await agent.run(payload, context)
        analysis = _persist_ci_analysis(db, run, result, agent.llm.model, conversation_id, counts)
        yield _encode_sse("trace", _trace_item("result", "执行 6/6：输出根因和阻塞判断", f"失败类型：{result.get('failure_type')}；阻塞合并：{'是' if result.get('is_merge_blocking') else '否'}。"))
        yield _encode_sse("final", {"analysis_id": str(analysis.id), "result": result})
    except Exception as exc:
        db.rollback()
        yield _encode_sse("error", {"message": f"分析失败：{exc}"})


@router.get("/repos/{repo_id}/workflow-runs", response_model=list[WorkflowRunResponse])
async def list_workflow_runs(repo_id: UUID, db: Session = Depends(get_db)) -> list[WorkflowRun]:
    """列 workflow 运行记录，新的在前。"""
    return db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo_id).order_by(WorkflowRun.created_at.desc()).all()


@router.get("/workflow-runs/{run_id}", response_model=WorkflowRunResponse)
async def get_workflow_run(run_id: UUID, db: Session = Depends(get_db)) -> WorkflowRun:
    """查单次运行详情（含 jobs 和完整日志）。"""
    run = db.get(WorkflowRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="未找到 workflow 运行")
    return run


@router.get("/workflow-runs/{run_id}/failed-jobs")
async def get_failed_jobs(run_id: UUID, db: Session = Depends(get_db)) -> dict:
    """只看失败 job / step（前端"失败明细"折叠面板用）。

    注意这里又写了一遍筛选逻辑，和 _failed_jobs() 重复 —— 历史遗留，行为一致。
    """
    run = db.get(WorkflowRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="未找到 workflow 运行")
    failed_jobs = [
        {
            **job,
            "failed_steps": [step for step in job.get("steps", []) if step.get("conclusion") == "failure"],
        }
        for job in (run.jobs or [])
        if job.get("conclusion") == "failure"
    ]
    return {"run_id": str(run.id), "workflow": run.name, "failed_jobs": failed_jobs}


@router.post("/workflow-runs/{run_id}/analyze", response_model=CIAnalysisResponse)
async def analyze_workflow_run(
    run_id: UUID,
    conversation_id: UUID | None = Query(default=None),
    use_pr_snapshot: bool = Query(default=True),
    db: Session = Depends(get_db),
) -> CIAnalysisResponse:
    """一次性 CI 根因分析。"""
    run = db.get(WorkflowRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="未找到 workflow 运行")
    agent = CIDebugAgent()
    snapshot = _prepare_ci_pr_snapshot(db, run, enabled=use_pr_snapshot)
    _query, context, counts = _build_ci_analysis_context(db, run, conversation_id)
    _augment_ci_context_with_snapshot(db, run, _query, context, counts, snapshot)
    result = await agent.run(_workflow_payload(run), context)
    analysis = _persist_ci_analysis(db, run, result, agent.llm.model, conversation_id, counts)
    return CIAnalysisResponse(analysis_id=analysis.id, result=result)


@router.post("/workflow-runs/{run_id}/analyze/stream")
async def analyze_workflow_run_stream(
    run_id: UUID,
    conversation_id: UUID | None = Query(default=None),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """流式 CI 分析（SSE）。"""
    return StreamingResponse(_ci_analysis_sse(run_id, conversation_id, db), media_type="text/event-stream")
