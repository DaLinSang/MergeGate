"""
Pull Request 路由
=================

【路由清单】
    GET  /repos/{repo_id}/pull-requests                 列 PR（带 files + review_comments）
    GET  /pull-requests/{pr_id}                         查单个 PR
    POST /pull-requests/{pr_id}/analyze                 PR 审查（一次性返回）
    POST /pull-requests/{pr_id}/analyze/stream          PR 审查（SSE 六步流式）
    POST /pull-requests/{pr_id}/snapshot                准备 PR 分支快照（worktree）
    POST /pull-requests/{pr_id}/worktree                同上（历史别名）
    POST /repos/{repo_id}/pr-snapshots/cleanup          清理过期快照
    POST /pull-requests/{pr_id}/review-checklist        只取 checklist（内部仍跑完整分析）

【数据流向】
    analyze（数据流 4 的 PR 分支）
      → _prepare_pr_snapshot_context()   起 git worktree，检出 PR 分支
      → _build_pr_analysis_context()     七路取证
          ① 代码（主分支实时检索）
          ② 项目文档 / 记忆
          ③ 历史 Issue
          ④ 会话上下文
          ⑤ CI 运行（最近 8 条，取失败的前 4 条 + 日志片段）
          ⑥ review comments
          ⑦ 团队画像
      → _augment_pr_context_with_snapshot()  用 PR 快照再补一轮代码 + 图谱影响面
      → services/agents/pr_review_agent.py::PRReviewAgent.run()
      → analysis_results 表

【要点】
    1. 为什么要 PR 快照：主分支代码和 PR 分支不一样。不检出 PR 分支，
       审查时看到的代码可能是"这个 PR 想改但还没改"的样子 —— 结论必然错。
    2. 快照失败不阻塞审查：status != "ready" 时退回主分支代码，并在 trace 里说明。
       宁可"看得不全"也不要"直接报错"。
    3. selectinload(files, review_comments)：这俩是懒加载关系，
       不预加载会触发 N+1 查询（每个 PR 多查两次）。
    4. sensitive_files 是纯关键词判断（auth / security / migration / payment / ci…），
       用来提醒模型"这些文件要重点看"，不是结论。
"""

import asyncio
import json
from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import or_
from sqlalchemy.orm import Session, selectinload

from app.db.models import AnalysisResult, CodeRelation, CodeSymbol, Conversation, Document, PullRequest, Repository, WorkflowRun
from app.db.session import get_db
from app.schemas.pull_requests import PRAnalysisResponse, PullRequestResponse
from app.services.agents.pr_review_agent import PRReviewAgent
from app.services.chat_memory import ContextAssembler
from app.services.code_search import search_repository_code
from app.services.rag.retrieval import search_similar_documents
from app.services.worktree_manager import cleanup_pr_snapshots, prepare_pr_worktree

router = APIRouter()

PROJECT_DOC_TYPES = {"knowledge_file", "memory_note", "project_doc"}

# 流式"公开执行过程"的提示词。四个建议值写死，防止模型自创第五种。
PUBLIC_REASONING_PROMPT = """
你是 DevFlow AI 的 PR 合并就绪度 Agent。请用中文流式输出一段“公开执行过程”，帮助用户理解你如何按计划审查当前 PR。
要求：
- 只输出面向用户的可审计分析摘要，不输出隐藏链路思维或私有推理链。
- 按计划说明你检查了 PR 目标、diff 风险、相关代码/文档、CI、Review 评论，并如何形成合入建议。
- 最终建议只能围绕：建议合入 / 修改后合入 / 暂缓 / 拒绝。
- 不要编造输入中没有的文件、CI、Issue、人员或评论。
- 不要输出最终 JSON。
"""


def _encode_sse(event: str, data: dict) -> str:
    """拼 SSE 消息。末尾必须有两个换行。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _trace_item(kind: str, title: str, content: str, status: str = "done") -> dict:
    """构造一条前端展示的步骤卡片。"""
    return {"kind": kind, "title": title, "content": content, "status": status}


def _document_evidence(items: list[dict], default_source_type: str) -> list[dict]:
    """检索结果 → 统一证据结构。没有正文或标题的丢掉（无法核对）。"""
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
    """按 (类型, id, 标题) 去重后截断。

    为什么需要：PR 快照代码和主分支代码可能检索到同一批文件，
    不去重的话证据列表里全是重复项，浪费上下文预算。
    """
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
    """向量检索没结果时，直接取最近 4 篇项目文档兜底。"""
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


def _conversation_evidence(db: Session, pr: PullRequest, conversation_id: UUID | None, query: str) -> list[dict]:
    """当前会话里和这个 PR 相关的证据。"""
    if not conversation_id:
        return []
    conversation = db.get(Conversation, conversation_id)
    # 会话必须属于同一仓库，防跨项目串数据
    if conversation is None or conversation.repo_id != pr.repo_id:
        return []
    repo = db.get(Repository, pr.repo_id)
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


def _pr_payload(pr: PullRequest) -> dict:
    """PR ORM → Agent 需要的 dict（含 files 和 comments 的完整内容）。"""
    return {
        "id": str(pr.id),
        "number": pr.number,
        "title": pr.title,
        "body": pr.body,
        "state": pr.state,
        "author": pr.author,
        "base_branch": pr.base_branch,
        "head_branch": pr.head_branch,
        "merged_at": pr.merged_at.isoformat() if pr.merged_at else None,
        "files": [
            {
                "filename": file.filename,
                "status": file.status,
                "additions": file.additions,
                "deletions": file.deletions,
                "patch": file.patch,
            }
            for file in pr.files
        ],
        "comments": [
            {"body": comment.body, "path": comment.path, "line": comment.line, "author": comment.author}
            for comment in pr.review_comments
        ],
    }


def _pr_review_plan() -> list[str]:
    """固定的六步审查计划。流式过程会逐步展示，让用户知道 AI 在干什么。"""
    return [
        "理解 PR 标题、描述、变更文件和关联上下文。",
        "按模块拆分 diff 风险。",
        "读取相关源代码、测试文件和项目文档。",
        "检查 CI 状态和失败信号。",
        "检索团队约定、历史对话和 review comments。",
        "形成合入建议和 review comments。",
    ]


def _build_pr_analysis_context(db: Session, pr: PullRequest, conversation_id: UUID | None = None) -> tuple[str, dict, dict]:
    """七路取证，返回 (code_query, context, counts)。

    注意 code_query 和 semantic_query 是两个不同的检索词：
      code_query    = 标题 + 正文 + 文件名 + diff 前 5000 字（给代码检索用）
      semantic_query = 标题 + 正文 + review 评论（给语义检索用）
    分开的原因：diff 里全是代码符号，拿去向量检索会污染语义。
    """
    repo = db.get(Repository, pr.repo_id)
    file_names = [file.filename for file in pr.files]
    # 只取前 12 个文件的 diff：大 PR 全塞进去会爆上下文
    diff_text = "\n".join(f"{file.filename}\n{file.patch or ''}" for file in pr.files[:12])
    code_query = "\n".join(filter(None, [pr.title, pr.body or "", " ".join(file_names), diff_text[:5000]]))
    review_text = "\n".join(comment.body or "" for comment in pr.review_comments)
    semantic_query = "\n".join(filter(None, [pr.title, pr.body or "", review_text[:3000]]))

    # ① 代码（主分支实时检索）
    code_docs = _document_evidence(
        search_repository_code(repo, code_query, limit=6) if repo else [],
        "code",
    )
    # ② 项目文档 / 记忆
    project_docs = _document_evidence(
        search_similar_documents(
            db,
            str(pr.repo_id),
            semantic_query,
            limit=4,
            source_types=sorted(PROJECT_DOC_TYPES),
        ),
        "document",
    )
    if not project_docs:
        project_docs = _fallback_project_documents(db, pr.repo_id)
    # ③ 历史 Issue（PR 常常在修某个 Issue）
    related_issues = _document_evidence(
        search_similar_documents(db, str(pr.repo_id), semantic_query, "issue", 4),
        "issue",
    )
    # ④ 会话上下文
    chat_evidence = _conversation_evidence(db, pr, conversation_id, semantic_query)

    # ⑤ CI：只看最近 8 条，失败的前 4 条带日志片段
    recent_runs = (
        db.query(WorkflowRun)
        .filter(WorkflowRun.repo_id == pr.repo_id)
        .order_by(WorkflowRun.created_at.desc())
        .limit(8)
        .all()
    )
    failed_runs = [run for run in recent_runs if run.conclusion == "failure"]

    # ⑦ 团队画像（评审人推荐用）
    team_members = [
        {
            "name": (doc.meta or {}).get("name") or doc.title.replace("团队成员：", "").replace("Team member: ", ""),
            "role": (doc.meta or {}).get("role") or "",
            "strengths": (doc.meta or {}).get("strengths") or "",
            "techStack": (doc.meta or {}).get("tech_stack") or "",
        }
        for doc in db.query(Document).filter(Document.repo_id == pr.repo_id, Document.source_type == "team_member").all()
    ]

    # diff 统计 + 敏感文件识别（纯关键词，只作提示不作结论）
    total_additions = sum(file.additions or 0 for file in pr.files)
    total_deletions = sum(file.deletions or 0 for file in pr.files)
    sensitive_files = [
        name
        for name in file_names
        if any(token in name.lower() for token in ["auth", "security", "migration", "schema", "payment", "permission", "config", "ci", "workflow"])
    ]

    context = {
        "repo_id": str(pr.repo_id),
        "plan": _pr_review_plan(),
        "diff_stats": {
            "files": len(pr.files),
            "additions": total_additions,
            "deletions": total_deletions,
            "sensitive_files": sensitive_files,
        },
        "code_references": code_docs,
        "project_documents": project_docs,
        "related_issues": related_issues,
        "conversation_evidence": chat_evidence,
        "review_comments": _pr_payload(pr)["comments"],
        "ci_summary": {
            "recent_runs": len(recent_runs),
            "failed_runs": len(failed_runs),
            "failed": [
                # 日志只截 260 字：CI 日志动辄几十万行，全塞进去毫无意义
                {"name": run.name, "conclusion": run.conclusion, "logs_text": (run.logs_text or "")[:260]}
                for run in failed_runs[:4]
            ],
        },
        "team_members": team_members,
    }
    counts = {
        "files": len(pr.files),
        "comments": len(pr.review_comments),
        "code": len(code_docs),
        "documents": len(project_docs),
        "issues": len(related_issues),
        "conversation": len(chat_evidence),
        "failed_ci": len(failed_runs),
        "team_members": len(team_members),
    }
    return code_query, context, counts


def _snapshot_summary(snapshot: dict) -> dict:
    """快照准备结果 → 精简摘要（完整信息太啰嗦，前端只需要这几个字段）。"""
    return {
        "status": snapshot.get("status"),
        "message": snapshot.get("message"),
        "name": snapshot.get("snapshot_name") or "PR 分支快照",
        "branch": snapshot.get("branch"),
        "commit_sha": snapshot.get("commit_sha"),
        "pr_id": snapshot.get("pr_id"),
        "pr_number": snapshot.get("pr_number"),
        "indexed_symbols": snapshot.get("indexed_symbols"),
    }


def _prepare_pr_snapshot_context(db: Session, pr: PullRequest, enabled: bool) -> dict:
    """准备 PR 分支快照。失败也返回 dict（带 status），不抛异常 —— 审查要继续。"""
    if not enabled:
        return {"status": "skipped", "message": "未请求 PR 分支快照。"}
    repo = db.get(Repository, pr.repo_id)
    if repo is None:
        return {"status": "missing_repo", "message": "未找到仓库。"}
    return prepare_pr_worktree(db, repo, pr)


def _pr_code_graph_impact(db: Session, pr: PullRequest) -> dict:
    """算这个 PR 的代码图谱影响面：改了哪些文件、涉及哪些符号、牵连哪些关系。"""
    # dict.fromkeys 去重且保序（一个 PR 可能有多个文件同名？理论上不会，但保险）
    changed_paths = list(dict.fromkeys(file.filename for file in pr.files if file.filename))
    symbol_query = db.query(CodeSymbol).filter(CodeSymbol.repo_id == pr.repo_id, CodeSymbol.pr_id == pr.id)
    if changed_paths:
        symbol_query = symbol_query.filter(CodeSymbol.path.in_(changed_paths))
    symbols = symbol_query.order_by(CodeSymbol.path.asc(), CodeSymbol.start_line.asc()).limit(30).all()

    symbol_names = [symbol.name for symbol in symbols]
    relation_query = db.query(CodeRelation).filter(CodeRelation.repo_id == pr.repo_id, CodeRelation.pr_id == pr.id)
    # 三种命中方式任一：改动的文件里的边、源符号命中、目标符号命中
    relation_filters = []
    if changed_paths:
        relation_filters.append(CodeRelation.path.in_(changed_paths))
    if symbol_names:
        relation_filters.extend([CodeRelation.source_name.in_(symbol_names), CodeRelation.target_name.in_(symbol_names)])
    if relation_filters:
        relation_query = relation_query.filter(or_(*relation_filters))
    relations = relation_query.limit(60).all()

    return {
        "changed_paths": changed_paths[:30],
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


def _augment_pr_context_with_snapshot(db: Session, pr: PullRequest, query: str, context: dict, counts: dict, snapshot: dict) -> None:
    """用 PR 快照补充上下文（原地修改 context 和 counts）。

    快照没准备好时也要写 code_graph_impact（空的），
    否则前端拿不到这个字段会报 undefined。
    """
    context["pr_snapshot"] = _snapshot_summary(snapshot)
    counts["snapshot_code"] = 0
    counts["impacted_symbols"] = 0
    counts["impacted_relations"] = 0
    if snapshot.get("status") != "ready":
        context["code_graph_impact"] = {"changed_paths": [file.filename for file in pr.files], "symbols": [], "relations": []}
        return

    repo = db.get(Repository, pr.repo_id)
    # 在快照目录里检索代码（读到的是 PR 分支的真实代码）
    snapshot_docs = _document_evidence(
        search_repository_code(
            repo,
            query,
            limit=8,
            checkout_path=snapshot.get("snapshot_path"),
            metadata={
                # checkout_kind 标记"这段代码来自 PR 快照"——前端会显示不同图标
                "checkout_kind": "pr_snapshot",
                "pr_id": str(pr.id),
                "pr_number": pr.number,
                "branch": snapshot.get("branch"),
                "commit_sha": snapshot.get("commit_sha"),
            },
        )
        if repo and snapshot.get("snapshot_path")
        else [],
        "code",
    )
    # 快照代码排在前面（更相关），再拼主分支的，去重后取 10 条
    context["code_references"] = _dedupe_evidence([*snapshot_docs, *(context.get("code_references") or [])], 10)
    impact = _pr_code_graph_impact(db, pr)
    context["code_graph_impact"] = impact
    counts["snapshot_code"] = len(snapshot_docs)
    counts["impacted_symbols"] = len(impact["symbols"])
    counts["impacted_relations"] = len(impact["relations"])
    counts["code"] = len(context["code_references"])


def _persist_pr_analysis(
    db: Session,
    pr: PullRequest,
    result: dict,
    model_name: str,
    conversation_id: UUID | None,
    evidence_counts: dict,
) -> AnalysisResult:
    """落库 PR 审查结果。input_snapshot 记证据数量，便于复盘。"""
    analysis = AnalysisResult(
        target_type="pull_request",
        target_id=pr.id,
        analysis_type="pr_review",
        input_snapshot={
            "pr_id": str(pr.id),
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


def _public_reasoning_fallback(pr: PullRequest, context: dict, counts: dict) -> list[str]:
    """LLM 不可用时的兜底过程（纯模板）。"""
    stats = context.get("diff_stats") or {}
    return [
        f"- 我先制定 Review 计划，再读取 PR #{pr.number} 的标题、描述、分支和 {counts['files']} 个变更文件。\n",
        f"- Diff 规模为 +{stats.get('additions', 0)} / -{stats.get('deletions', 0)}，其中敏感文件 {len(stats.get('sensitive_files') or [])} 个。\n",
        f"- 我检索了相关代码 {counts['code']} 条，其中 PR 快照代码 {counts.get('snapshot_code', 0)} 条；项目文档/记忆 {counts['documents']} 条、历史 Issue {counts['issues']} 条和会话证据 {counts['conversation']} 条。\n",
        f"- CI 侧读取到 {counts['failed_ci']} 个失败信号，并结合 {counts['comments']} 条 review comment 判断是否阻塞。\n",
        "- 最后将风险点、阻塞项、测试建议汇总为合入建议：建议合入、修改后合入、暂缓或拒绝。\n",
    ]


async def _stream_public_reasoning(agent: PRReviewAgent, pr: PullRequest, payload: dict, context: dict, counts: dict) -> AsyncIterator[str]:
    """流式输出公开执行过程。只给预览（每类前 3 条），不给全量。"""
    streamed = False
    user_payload = {
        "pull_request": payload,
        "plan": context.get("plan") or [],
        "evidence_counts": counts,
        "context_preview": {
            "diff_stats": context.get("diff_stats"),
            "pr_snapshot": context.get("pr_snapshot"),
            "code_graph_impact": context.get("code_graph_impact"),
            "code_references": context.get("code_references", [])[:3],
            "project_documents": context.get("project_documents", [])[:3],
            "ci_summary": context.get("ci_summary"),
            "review_comments": context.get("review_comments", [])[:3],
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
    for line in _public_reasoning_fallback(pr, context, counts):
        await asyncio.sleep(0.06)
        yield line


async def _pr_analysis_sse(pr_id: UUID, conversation_id: UUID | None, db: Session) -> AsyncIterator[str]:
    """SSE 六步审查。事件顺序 = 前端进度展示顺序。"""
    # 预加载 files / review_comments，避免 N+1
    pr = db.query(PullRequest).options(selectinload(PullRequest.files), selectinload(PullRequest.review_comments)).filter(PullRequest.id == pr_id).one_or_none()
    if pr is None:
        yield _encode_sse("error", {"message": "未找到 PR"})
        return
    agent = PRReviewAgent()
    payload = _pr_payload(pr)
    plan = _pr_review_plan()
    try:
        yield _encode_sse("trace", _trace_item("step", "制定 Review 计划", "\n".join(f"{index + 1}. {item}" for index, item in enumerate(plan)), "done"))
        yield _encode_sse("trace", _trace_item("step", "执行 1/6：理解 PR", f"读取 PR #{pr.number} 的标题、描述、分支、作者和变更文件列表。", "running"))

        # 第一步就先起快照 —— 后面所有代码检索都依赖它
        snapshot = _prepare_pr_snapshot_context(db, pr, enabled=True)
        yield _encode_sse(
            "trace",
            _trace_item(
                "tool",
                "准备 PR 分支快照",
                str(snapshot.get("message") or f"状态：{snapshot.get('status')}；代码图符号：{snapshot.get('indexed_symbols') or 0}。"),
            ),
        )

        _query, context, counts = _build_pr_analysis_context(db, pr, conversation_id)
        _augment_pr_context_with_snapshot(db, pr, _query, context, counts, snapshot)
        context["plan"] = plan

        yield _encode_sse("trace", _trace_item("tool", "执行 2/6：拆分 diff 风险", f"扫描 {counts['files']} 个文件，识别敏感文件 {len((context.get('diff_stats') or {}).get('sensitive_files') or [])} 个。"))
        yield _encode_sse("trace", _trace_item("tool", "执行 3/6：读取代码和测试", f"找到 {counts['code']} 条代码线索，其中 PR 快照代码 {counts.get('snapshot_code', 0)} 条；项目文档/记忆线索 {counts['documents']} 条。"))
        yield _encode_sse("trace", _trace_item("tool", "执行 4/6：检查 CI", f"读取最近 CI 信号，失败 run 数量：{counts['failed_ci']}。"))
        yield _encode_sse("trace", _trace_item("tool", "执行 5/6：检索上下文", f"找到 {counts['issues']} 条历史 Issue、{counts['conversation']} 条会话证据、{counts['comments']} 条 review comment。"))
        yield _encode_sse("trace", _trace_item("thinking", "LLM 公开执行过程", "开始综合 PR diff、代码、CI、文档和 review comment。", "running"))

        async for chunk in _stream_public_reasoning(agent, pr, payload, context, counts):
            yield _encode_sse("thinking_delta", {"content": chunk})

        result = await agent.run(payload, context)
        analysis = _persist_pr_analysis(db, pr, result, agent.llm.model, conversation_id, counts)
        yield _encode_sse("trace", _trace_item("result", "执行 6/6：形成合入建议", f"结论：{result.get('merge_recommendation')}；风险点：{len(result.get('risk_points') or [])}；阻塞项：{len(result.get('blocking_issues') or [])}。"))
        yield _encode_sse("final", {"analysis_id": str(analysis.id), "result": result})
    except Exception as exc:
        # 响应头已发出，只能靠事件告知失败
        db.rollback()
        yield _encode_sse("error", {"message": f"分析失败：{exc}"})


@router.get("/repos/{repo_id}/pull-requests", response_model=list[PullRequestResponse])
async def list_pull_requests(repo_id: UUID, db: Session = Depends(get_db)) -> list[PullRequest]:
    """列 PR（带 files + review_comments），编号大的在前。"""
    return (
        db.query(PullRequest)
        .options(selectinload(PullRequest.files), selectinload(PullRequest.review_comments))
        .filter(PullRequest.repo_id == repo_id)
        .order_by(PullRequest.number.desc())
        .all()
    )


@router.get("/pull-requests/{pr_id}", response_model=PullRequestResponse)
async def get_pull_request(pr_id: UUID, db: Session = Depends(get_db)) -> PullRequest:
    """查单个 PR 详情。"""
    pr = db.query(PullRequest).options(selectinload(PullRequest.files), selectinload(PullRequest.review_comments)).filter(PullRequest.id == pr_id).one_or_none()
    if pr is None:
        raise HTTPException(status_code=404, detail="未找到 PR")
    return pr


@router.post("/pull-requests/{pr_id}/analyze", response_model=PRAnalysisResponse)
async def analyze_pull_request(
    pr_id: UUID,
    conversation_id: UUID | None = Query(default=None),
    use_pr_snapshot: bool = Query(default=True),
    use_worktree: bool | None = Query(default=None),
    db: Session = Depends(get_db),
) -> PRAnalysisResponse:
    """一次性 PR 审查。

    use_worktree 是历史参数名，use_pr_snapshot 是新名字 —— 两个都支持，
    传了 use_worktree 就以它为准（向后兼容老前端）。
    """
    pr = db.query(PullRequest).options(selectinload(PullRequest.files), selectinload(PullRequest.review_comments)).filter(PullRequest.id == pr_id).one_or_none()
    if pr is None:
        raise HTTPException(status_code=404, detail="未找到 PR")
    agent = PRReviewAgent()
    snapshot_enabled = use_pr_snapshot if use_worktree is None else use_worktree
    snapshot = _prepare_pr_snapshot_context(db, pr, snapshot_enabled)
    _query, context, counts = _build_pr_analysis_context(db, pr, conversation_id)
    _augment_pr_context_with_snapshot(db, pr, _query, context, counts, snapshot)
    result = await agent.run(_pr_payload(pr), context)
    analysis = _persist_pr_analysis(db, pr, result, agent.llm.model, conversation_id, counts)
    return PRAnalysisResponse(analysis_id=analysis.id, result=result)


@router.post("/pull-requests/{pr_id}/analyze/stream")
async def analyze_pull_request_stream(
    pr_id: UUID,
    conversation_id: UUID | None = Query(default=None),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """流式 PR 审查（SSE）。"""
    return StreamingResponse(_pr_analysis_sse(pr_id, conversation_id, db), media_type="text/event-stream")


@router.post("/pull-requests/{pr_id}/snapshot")
async def prepare_pull_request_snapshot(pr_id: UUID, db: Session = Depends(get_db)) -> dict:
    """单独准备 PR 快照（前端可以提前点"准备代码"）。"""
    pr = db.query(PullRequest).options(selectinload(PullRequest.files)).filter(PullRequest.id == pr_id).one_or_none()
    if pr is None:
        raise HTTPException(status_code=404, detail="未找到 PR")
    return _prepare_pr_snapshot_context(db, pr, enabled=True)


@router.post("/pull-requests/{pr_id}/worktree")
async def prepare_pull_request_worktree(pr_id: UUID, db: Session = Depends(get_db)) -> dict:
    """同上，历史路径名（老前端在用，保留兼容）。"""
    pr = db.query(PullRequest).options(selectinload(PullRequest.files)).filter(PullRequest.id == pr_id).one_or_none()
    if pr is None:
        raise HTTPException(status_code=404, detail="未找到 PR")
    return _prepare_pr_snapshot_context(db, pr, enabled=True)


@router.post("/repos/{repo_id}/pr-snapshots/cleanup")
async def cleanup_pull_request_snapshots(
    repo_id: UUID,
    max_age_hours: int = Query(default=168, ge=1, le=24 * 90),
    db: Session = Depends(get_db),
) -> dict:
    """清理过期 PR 快照。默认 168 小时（7 天），最长 90 天。

    worktree 很占磁盘（每个 PR 一份完整代码），不清会越积越多。
    """
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    return cleanup_pr_snapshots(db, repo, max_age_hours=max_age_hours)


@router.post("/pull-requests/{pr_id}/review-checklist")
async def review_checklist(pr_id: UUID, db: Session = Depends(get_db)) -> dict:
    """只取 checklist。注意：内部仍然跑完整审查（没有"只算 checklist"的轻量路径）。"""
    response = await analyze_pull_request(pr_id, db=db)
    return {"pr_id": str(pr_id), "checklist": response.result.review_checklist}
