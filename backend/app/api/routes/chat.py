"""
对话路由 —— 【整个系统的总入口】
==============================

【路由清单】（7 个）
    GET  /chat/workflow-runs                列 Agent 工作流运行记录（含子任务）
    GET  /chat/history                      拉对话历史（timeline 完整流水）
    POST /chat/plan                         只生成执行计划（不执行，返回 WorkflowSpec 给用户确认）
    POST /chat/plan/execute/stream          执行已确认的计划（SSE）
    POST /chat                              ChatAgent 对话（一次性返回）
    POST /chat/stream                       ChatAgent 对话（SSE 流式）
    POST /chat/recommendations              推荐下一步该问什么

【数据流向】（数据流 4 的完整链路，这是全项目最长的一条）

    POST /chat/stream
      ┌─ ① 准备阶段（_conversation_sse 开头，四步固定顺序）
      │   ensure_conversation()              没有会话就建一个
      │   ProgressiveContextManager.ensure_headroom()   上下文快满了先压缩
      │   MessageStore.append(role="user")   把用户消息落库
      │   EvidenceStore.rebuild_repo()       刷新证据库（否则检索不到最新数据）
      │   ContextAssembler.assemble()        组装 system_context + history + evidence
      │
      ├─ ② 注入工具（_build_conversation_tools）
      │   14 个工具：workspace.* / repo_health / rag.search / issue / pr / ci /
      │              report / safety / workflow / devflow_*（MCP 记忆工具）
      │
      ├─ ③ 跑 Agent（ChatAgent().run_stream）
      │   内部：model_decision ⇄ tool_action 循环（最多 12 步）
      │   工具实际执行 → 回到本文件的 _build_conversation_tools 里的闭包
      │
      └─ ④ 收尾（收到 final 事件时）
          MessageStore.append(role="assistant")   存回答
          AgentRun 落一条运行记录
          touch_conversation()                    更新会话计数
          SessionSealer.seal()                    封存会话 → 提炼记忆候选
          db.commit()

【要点】
    1. 【四步准备顺序不能乱】必须先 ensure_headroom 再 append 用户消息 ——
       反了的话，刚插入的消息可能立刻被压缩掉。
    2. 【EvidenceStore 每次都要 rebuild】用户在别的页面刚改的知识，
       这一步不刷就检索不到，模型会答"没有相关信息"。
    3. 【两条路径共享同一套准备逻辑】_conversation_response / _conversation_sse /
       _conversation_plan_execute_sse 三处几乎逐行重复。这是历史遗留，
       改一处要记得改另外两处。
    4. 【流式里的落库时机】只有收到 final 事件才落库 —— 中途失败会走 except 分支，
       落一条 route="error" 的错误消息。两种情况都会 commit，不会丢消息。
    5. 【工作区工具的安全边界】workspace.* 系列有四道校验：
       路径必须在仓库内（relative_to）、跳过 .git/node_modules、
       拒绝 .env 等敏感文件、拒绝二进制和超大文件。
"""

import uuid
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session, selectinload

from app.core.config import settings
from app.db.models import ActionDraft, AgentRun, AgentTaskRun, AgentWorkflowRun, ChatMessage, Conversation, Issue, PullRequest, Repository, WorkflowRun
from app.db.session import get_db
from app.schemas.chat import ChatHistoryMessage, ChatPlanExecuteRequest, ChatRequest, ChatResponse, PromptRecommendationRequest, PromptRecommendationResponse
from app.services.agents.ci_debug_agent import CIDebugAgent
from app.services.agents.chat_agent import ChatAgent, encode_sse
from app.services.agents.issue_agent import IssueAgent
from app.services.agents.planner_agent import PlannerAgent
from app.services.agents.pr_review_agent import PRReviewAgent
from app.services.agents.recommendation_agent import RecommendationAgent
from app.services.agents.report_agent import ReportAgent
from app.services.agents.safety_agent import SafetyAgent
from app.services.agents.workflow_orchestrator import WorkflowOrchestrator
from app.services.chat_memory import ContextAssembler, EvidenceStore, MessageStore, ProgressiveContextManager, SessionSealer
from app.services.code_search import query_terms, search_checkout_code
from app.services.conversations import ensure_conversation, touch_conversation
from app.services.memory_hub import record_recall_event
from app.services.mcp_client import StdioMcpClient
from app.services.rag.retrieval import search_similar_documents
from app.services.rag.vector_store import add_document
from app.schemas.workflow import WorkflowSpec, WorkflowTask, WorkflowTaskResult
from app.services.skills import SkillRegistry, get_skill_registry
from app.services.permissions import ensure_demo_user, write_audit_log

router = APIRouter()

# 项目根目录（用于 repo 为空时的工作区）
PROJECT_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = Path(__file__).resolve().parents[3]

# 工作区遍历时跳过的目录：这些要么体积巨大，要么是构建产物/版本库内部数据
WORKSPACE_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".next",
    ".nuxt",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "coverage",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

# 【安全】这些文件名一律不读 —— 里面是密钥
WORKSPACE_SECRET_FILENAMES = {".env", ".env.local", ".env.production", ".npmrc", ".pypirc"}
# 无扩展名但属于文本的常见文件
WORKSPACE_TEXT_FILENAMES = {"Dockerfile", "Makefile", "README", "LICENSE", "package.json", "requirements.txt", "pyproject.toml"}
# 允许读取的文本扩展名白名单
WORKSPACE_TEXT_EXTENSIONS = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".json",
    ".md",
    ".mdx",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".sql",
    ".sh",
    ".ps1",
    ".bat",
    ".cmd",
    ".html",
    ".css",
    ".scss",
}


@router.get("/workflow-runs")
async def list_workflow_runs(
    repo_id: uuid.UUID | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """列工作流运行记录（含每个工作流下的子任务）。

    两次查询拼装（不是 N+1）：先查 workflow，再一次性查所有子任务，然后内存分组。
    """
    query = db.query(AgentWorkflowRun)
    if repo_id is not None:
        query = query.filter(AgentWorkflowRun.repo_id == repo_id)
    rows = query.order_by(AgentWorkflowRun.created_at.desc()).limit(limit).all()

    workflow_ids = [row.id for row in rows]
    tasks = (
        db.query(AgentTaskRun)
        .filter(AgentTaskRun.workflow_id.in_(workflow_ids))
        .order_by(AgentTaskRun.started_at.asc(), AgentTaskRun.task_id.asc())
        .all()
        if workflow_ids
        else []
    )
    tasks_by_workflow: dict[uuid.UUID, list[AgentTaskRun]] = {}
    for task in tasks:
        tasks_by_workflow.setdefault(task.workflow_id, []).append(task)

    return [
        {
            "id": str(row.id),
            "repo_id": str(row.repo_id) if row.repo_id else None,
            "conversation_id": str(row.conversation_id) if row.conversation_id else None,
            "goal": row.goal,
            "status": row.status,
            "spec": row.spec_json or {},
            "observation": row.observation_json or {},
            "final_answer": row.final_answer,
            "metrics": row.metrics or {},
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "completed_at": row.completed_at.isoformat() if row.completed_at else None,
            "tasks": [
                {
                    "id": str(task.id),
                    "task_id": task.task_id,
                    "agent_name": task.agent_name,
                    "task_type": task.task_type,
                    "status": task.status,
                    "summary": str((task.result_json or {}).get("summary") or ""),
                    "error": task.error,
                    "started_at": task.started_at.isoformat() if task.started_at else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                }
                for task in tasks_by_workflow.get(row.id, [])
            ],
        }
        for row in rows
    ]


@router.get("/history", response_model=list[ChatHistoryMessage])
async def chat_history(
    repo_id: uuid.UUID = Query(...),
    conversation_id: uuid.UUID | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    before_message_id: uuid.UUID | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    """拉对话历史。

    用的是 MessageStore.timeline()（用户视角的完整流水），
    不是 recent()（模型视角的压缩投影）—— 用户要看到自己说过的每一句。
    before_message_id 支持向上翻页（游标分页，比 offset 稳定）。
    """
    _repo_or_404(db, repo_id)
    conversation = ensure_conversation(db, repo_id, conversation_id)
    conversation_id = conversation.id
    # ensure_conversation 可能创建了默认会话，要 commit
    db.commit()
    rows = [
        item
        for item in MessageStore(db).timeline(
            repo_id,
            conversation_id,
            limit=limit,
            before_message_id=before_message_id,
        )
        # 只返回用户和助手消息，system 消息（压缩边界等）不展示给用户
        if item.role in {"user", "assistant"}
    ]
    return [
        {
            "id": str(item.id),
            "conversation_id": str(item.conversation_id) if item.conversation_id else None,
            "role": item.role,
            "content": item.content,
            "route": item.route,
            "tool_calls": item.tool_calls or [],
            # 以下四个都从 meta 取：诊断信息，前端"详情"面板展示
            "agent_events": (item.meta or {}).get("agent_events", []),
            "compression_stats": (item.meta or {}).get("compression_stats", {}),
            "progressive_compaction": (item.meta or {}).get("progressive_compaction", {}),
            "model_usage": (item.meta or {}).get("model_usage", []),
            "context_diagnostics": (item.meta or {}).get("context_diagnostics", {}),
            "created_at": item.created_at.isoformat() if item.created_at else None,
        }
        for item in rows
    ]


def _repo_or_404(db: Session, repo_id: uuid.UUID) -> Repository:
    """取仓库，不存在就 404。"""
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到指定仓库。")
    return repo


def _client_context(value: dict[str, Any] | None) -> dict[str, Any]:
    """过滤前端传来的 context。

    【安全】把"服务端拥有"的键全部剔除 —— 否则前端可以伪造
    evidence（假证据）、tools（换掉工具实现）、repo_id（越权访问别的仓库）。
    """
    server_owned = {
        "repo_id",
        "conversation_id",
        "tools",
        "server_context",
        "server_messages",
        "evidence",
        "compression_stats",
        "progressive_compaction",
        "registered_skills",
        "skill_description_by_tool",
    }
    return {key: item for key, item in (value or {}).items() if key not in server_owned}


def _item_time(item: Any) -> datetime:
    """排序键：优先 updated_at，其次 created_at。"""
    return item.updated_at or item.created_at or datetime.min


def _join_items(items: list[str], empty_text: str) -> str:
    """列表拼成 "; " 分隔的字符串，空列表返回占位文案。"""
    return "; ".join(items) if items else empty_text


def _source_type_from_message(message: str) -> str | None:
    """从用户问题里猜要检索哪类来源（给 rag.search 当默认过滤条件）。

    纯关键词匹配，顺序从具体到宽泛。
    """
    lowered = message.lower()
    if "issue" in lowered:
        return "issue"
    if "pr" in lowered or "pull request" in lowered or "diff" in lowered:
        return "pull_request"
    if "ci" in lowered or "workflow" in lowered:
        return "workflow_run"
    if any(token in lowered for token in ["knowledge", "知识库", "上传资料", "上传文档"]):
        return "knowledge_file"
    return None


def _risky_action(message: str) -> str | None:
    """判断这句话是不是"要写 GitHub"。

    返回 None 表示只读（安全）；非 None 表示需要走草稿确认。
    【安全】宁可多判（误判成写操作只是多一个确认步骤），不能漏判。
    """
    lowered = message.lower()
    if any(token in lowered for token in ["comment", "reply", "post"]):
        return "comment"
    if any(token in lowered for token in ["create issue", "new issue"]):
        return "create_issue"
    if any(token in lowered for token in ["send", "send_report"]):
        return "send_report"
    if any(token in lowered for token in ["close issue", "label", "write"]):
        return "write_back"
    return None


def _date_window_from_context(context: dict[str, Any]) -> tuple[datetime, datetime]:
    """从 context 算时间窗口。

    优先级：显式 start_date/end_date > time_range（如 "7d"）> 默认 7 天。
    """
    end_at = datetime.now(timezone.utc)
    time_range = str(context.get("time_range") or "7d").lower()
    # "7d" → 7；格式不对就用 7
    days = int(time_range[:-1]) if time_range.endswith("d") and time_range[:-1].isdigit() else 7
    # 减 (days-1) 而不是 days：让"最近 7 天"包含今天
    start_at = end_at - timedelta(days=max(1, days) - 1)
    if context.get("start_date"):
        start_at = datetime.fromisoformat(str(context["start_date"])).replace(tzinfo=timezone.utc)
    if context.get("end_date"):
        # 结束日期补齐到当天 23:59:59，否则会漏掉最后一天
        end_at = datetime.fromisoformat(str(context["end_date"])).replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
    return start_at, end_at


def _within(value: datetime | None, start_at: datetime, end_at: datetime) -> bool:
    """判断时间是否落在窗口内（自动补时区）。"""
    if not value:
        return False
    current = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return start_at <= current <= end_at


def _repo_payload(repo_id: uuid.UUID, db: Session) -> tuple[list[Issue], list[PullRequest], list[WorkflowRun]]:
    """一次取齐三类事实数据。

    PR 用 selectinload 预加载 files / review_comments，避免后面 N+1。
    """
    issues = db.query(Issue).filter(Issue.repo_id == repo_id).all()
    prs = (
        db.query(PullRequest)
        .options(selectinload(PullRequest.files), selectinload(PullRequest.review_comments))
        .filter(PullRequest.repo_id == repo_id)
        .all()
    )
    runs = db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo_id).all()
    return issues, prs, runs


def _recommendation_payload(
    repo: Repository | None,
    issues: list[Issue],
    prs: list[PullRequest],
    runs: list[WorkflowRun],
    exclude: list[str],
    limit: int,
) -> dict[str, Any]:
    """给推荐 Agent 的输入：仓库 + 三类数据 + 已排除项 + 数量上限。"""
    return {
        "repo": {"id": str(repo.id), "full_name": repo.full_name} if repo else None,
        "issues": [
            {
                "id": str(item.id),
                "number": item.number,
                "title": item.title,
                "state": item.state,
                "labels": item.labels or [],
                "assignees": item.assignees or [],
                "created_at": item.created_at,
                "updated_at": item.updated_at,
                "closed_at": item.closed_at,
            }
            for item in issues
        ],
        "pull_requests": [
            {
                "id": str(item.id),
                "number": item.number,
                "title": item.title,
                "state": item.state,
                "author": item.author,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
                "merged_at": item.merged_at,
            }
            for item in prs
        ],
        "workflow_runs": [
            {
                "id": str(item.id),
                "name": item.name,
                "status": item.status,
                "conclusion": item.conclusion,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
            }
            for item in runs
        ],
        "exclude": exclude,
        "limit": limit,
    }


def _repo_health_answer(repo: Repository, issues: list[Issue], prs: list[PullRequest], runs: list[WorkflowRun]) -> str:
    """仓库健康度摘要（纯模板拼字符串，不调 LLM）。"""
    open_issues = [item for item in issues if item.state == "open"]
    open_prs = [item for item in prs if item.state == "open"]
    merged_prs = [item for item in prs if item.merged_at]
    failed_runs = [item for item in runs if item.conclusion == "failure"]
    # 只取最近 3 条，避免摘要过长
    latest_issues = sorted(open_issues, key=_item_time, reverse=True)[:3]
    latest_prs = sorted(prs, key=_item_time, reverse=True)[:3]
    issue_titles = _join_items([f"#{item.number} {item.title}" for item in latest_issues], "没有打开的 Issue")
    pr_titles = _join_items([f"#{item.number} {item.title}" for item in latest_prs], "没有 PR 记录")
    ci_text = f"已同步 {len(failed_runs)} 次失败 CI 运行" if failed_runs else "还没有同步到失败 CI 运行"
    return (
        f"当前仓库：{repo.full_name}\n\n"
        f"- Issues：{len(open_issues)} 个打开。近期关注：{issue_titles}。\n"
        f"- PRs：{len(open_prs)} 个打开，{len(merged_prs)} 个已合并。近期 PR：{pr_titles}。\n"
        f"- CI：{ci_text}。\n\n"
        "建议下一步：先看最新打开的 Issue，或检查最新 PR 风险；如果同步后出现失败 CI，可以让我继续排障。"
    )


def _semantic_search(repo: Repository, message: str, context: dict[str, Any], db: Session) -> tuple[str, list[dict[str, str]], list[dict[str, Any]]]:
    """检索 + 拼成带编号的文本，返回 (answer, citations, raw_results)。"""
    filters = {
        "label": context.get("label"),
        "path": context.get("path"),
        "module": context.get("module"),
        "state": context.get("state"),
        "start_date": context.get("start_date"),
        "end_date": context.get("end_date"),
    }
    results = search_similar_documents(
        db,
        repo.id,
        message,
        # source_type 优先用前端指定的，没指定就从问题里猜
        source_type=context.get("source_type") or _source_type_from_message(message),
        limit=int(context.get("limit", 5)),
        # 只传有值的过滤条件（None 会被当成"要求该字段为 null"）
        metadata_filters={key: value for key, value in filters.items() if value},
    )
    if not results:
        return "没有找到足够相似的历史记录。可以尝试放宽标签、路径或时间过滤条件。", [], []
    lines = ["找到以下相关历史记录："]
    citations: list[dict[str, str]] = []
    for index, item in enumerate(results, start=1):
        meta = item.get("metadata") or {}
        number = meta.get("number")
        prefix = f"#{number} " if number else ""
        lines.append(f"- [{index}] {item['source_type']} {prefix}{item['title']}：相似度 {item['score']}")
        snippet = str(item.get("snippet") or "").strip()
        if snippet:
            lines.append(f"  证据：{snippet}")
        citations.append({"type": item["source_type"], "id": item["source_id"], "title": item["title"]})
    return "\n".join(lines), citations, results


def _checkout_root() -> Path:
    """代码 clone 的根目录。相对路径时以 backend 为基准。"""
    root = Path(settings.repo_checkout_dir)
    if not root.is_absolute():
        root = BACKEND_ROOT / root
    return root.resolve()


def _repo_checkout_path(repo: Repository) -> Path:
    """算出这个仓库的本地代码目录。三级回退：

      ① repo.local_path（用户连的本地仓库）
      ② {checkout_root}/{repo.id}-{owner__repo}
      ③ 同名但 UUID 不同的旧目录（仓库删了重建后 UUID 会变）
    """
    local_path = getattr(repo, "local_path", None)
    if local_path:
        path = Path(os.path.expandvars(os.path.expanduser(str(local_path))))
        if not path.is_absolute():
            path = BACKEND_ROOT / path
        resolved = path.resolve()
        if resolved.exists():
            return resolved

    safe_name = repo.full_name.replace("/", "__").replace("\\", "__")
    checkout_root = _checkout_root()
    expected = (checkout_root / f"{repo.id}-{safe_name}").resolve()
    if expected.exists():
        return expected

    # 仓库记录可能被删除后重建，而本地 clone 仍保留旧 UUID。
    # 复用同名 checkout，避免工作区工具静默失去源码访问能力。
    candidates = [
        path
        for path in checkout_root.glob(f"*-{safe_name}")
        if path.is_dir() and path.resolve() != expected
    ]
    if candidates:
        # 多个候选时取最新修改的那个（最可能是当前在用的）
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return candidates[0].resolve()
    return expected


def _workspace_root(repo: Repository | None) -> Path:
    """工作区根目录。没有仓库时用项目根（可以读 DevFlow 自己的代码）。"""
    if repo is None:
        return PROJECT_ROOT
    checkout_path = _repo_checkout_path(repo)
    if not checkout_path.exists():
        # 报可操作的错误：告诉用户"去同步仓库"，而不是只说"路径不存在"
        raise ValueError(
            f"{repo.full_name} 的仓库代码还没有克隆到本地。"
            "读取源码前，请先同步仓库或重新连接仓库。"
        )
    return checkout_path


def _workspace_relative(path: Path, workspace_root: Path) -> str:
    """转成相对路径（统一用 / 分隔，Windows 也能用）。"""
    return path.relative_to(workspace_root).as_posix()


def _workspace_path(workspace_root: Path, raw_path: str | None = None) -> Path:
    """把用户给的相对路径解析成绝对路径，并做四道安全校验。

    【安全】这是防止 Agent 读到 /etc/passwd 或 .env 的关键函数。
    """
    normalized_path = str(raw_path or ".").strip()
    if normalized_path in {"", ".", "./", "/", "\\"}:
        normalized_path = "."
    elif not re.match(r"^[a-zA-Z]:", normalized_path):
        # 去掉开头的 / 或 \，让 "/src" 变成 "src"（当作仓库内的绝对路径处理）
        normalized_path = normalized_path.lstrip("/\\") or "."
    candidate = (workspace_root / normalized_path).resolve()

    # 校验 1：必须还在工作区内（防 ../ 逃逸）
    try:
        candidate.relative_to(workspace_root)
    except ValueError as exc:
        raise ValueError("路径位于项目工作区之外。") from exc
    # 校验 2：不能落在 .git / node_modules 这类跳过目录里
    if any(part in WORKSPACE_SKIP_DIRS for part in candidate.relative_to(workspace_root).parts):
        raise ValueError("路径位于被跳过的工作区目录中。")
    # 校验 3：不能是敏感配置文件
    if candidate.name in WORKSPACE_SECRET_FILENAMES:
        raise ValueError("不会向对话 Agent 暴露敏感配置文件。")
    return candidate


def _is_workspace_text(path: Path) -> bool:
    """判断这个文件是不是"可读的文本"（白名单 + 黑名单双判）。"""
    # 黑名单优先：敏感文件、日志、数据库文件一律不读
    if path.name in WORKSPACE_SECRET_FILENAMES or path.suffix.lower() in {".log", ".db", ".sqlite", ".sqlite3"}:
        return False
    return path.name in WORKSPACE_TEXT_FILENAMES or path.suffix.lower() in WORKSPACE_TEXT_EXTENSIONS or path.name.lower().endswith("dockerfile")


def _iter_workspace_files(root: Path, workspace_root: Path, max_files: int = 1000):
    """遍历工作区文件（生成器）。

    max_files 上限 1000：防止在大仓库里遍历几十万文件卡死。
    """
    if root.is_file():
        yield root
        return
    count = 0
    for path in root.rglob("*"):
        if count >= max_files:
            break
        if not path.is_file():
            continue
        relative_parts = path.relative_to(workspace_root).parts
        # 跳过目录里的文件直接忽略（不 descend，因为 rglob 已经展开）
        if any(part in WORKSPACE_SKIP_DIRS for part in relative_parts):
            continue
        if not _is_workspace_text(path):
            continue
        count += 1
        yield path


def _read_workspace_text(path: Path, max_bytes: int = 250_000) -> str:
    """读文本文件，带大小 / 二进制 / 编码三重防护。"""
    if not path.exists() or not path.is_file():
        raise ValueError("未找到工作区文件。")
    if not _is_workspace_text(path):
        raise ValueError("只能读取非敏感的文本工作区文件。")
    data = path.read_bytes()
    # 250KB 上限 + 前 4KB 含 \x00 视为二进制
    if len(data) > max_bytes or b"\x00" in data[:4096]:
        raise ValueError("文件过大，或看起来是二进制文件。")
    for encoding in ["utf-8-sig", "utf-8", "gb18030", "latin-1"]:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("无法将工作区文件解码为文本。")


def _draft_for_action(action: str, message: str, repo: Repository, issues: list[Issue], prs: list[PullRequest], runs: list[WorkflowRun]) -> tuple[str, list[dict[str, str]]]:
    """按动作类型生成草稿文本（模板，不调 LLM）。

    注意：草稿只是"建议内容"，真正写 GitHub 要等人工确认。
    """
    citations: list[dict[str, str]] = [{"type": "repository", "id": str(repo.id), "title": repo.full_name}]
    latest_issue = sorted(issues, key=_item_time, reverse=True)[0] if issues else None
    latest_pr = sorted(prs, key=_item_time, reverse=True)[0] if prs else None
    failed_runs = [run for run in runs if run.conclusion == "failure"]

    if action == "create_issue":
        return f"## Issue 草稿\n标题：{message.strip()[:120]}\n\n### 背景\n请补充影响范围、复现步骤和预期行为。\n\n### 验收标准\n- 预期行为清晰\n- 包含验证步骤\n", citations
    # 评论优先挂到 PR（有 PR 时），否则挂 Issue
    if action == "comment" and latest_pr:
        citations.append({"type": "pull_request", "id": str(latest_pr.id), "title": latest_pr.title})
        return f"## PR 评论草稿\n针对 PR #{latest_pr.number}：{latest_pr.title}\n\n合并前请确认测试覆盖、未解决的 Review 评论，以及是否存在高风险文件。\n", citations
    if action == "comment" and latest_issue:
        citations.append({"type": "issue", "id": str(latest_issue.id), "title": latest_issue.title})
        return f"## Issue 评论草稿\n针对 Issue #{latest_issue.number}：{latest_issue.title}\n\n请补充复现步骤、影响范围、预期行为，以及当前是否有临时绕过方案。\n", citations
    if action == "send_report":
        draft = (
            "## 报告发送草稿\n"
            f"仓库：{repo.full_name}\n\n"
            f"- 打开的 Issue：{len([issue for issue in issues if issue.state == 'open'])}\n"
            f"- 打开的 PR：{len([pr for pr in prs if pr.state == 'open'])}\n"
            f"- 失败 CI 运行：{len(failed_runs)}\n\n"
            "发送前请确认收件人、日期范围和敏感日志内容。\n"
        )
        return draft, citations
    return f"## 写操作草稿\n这可能会写入 GitHub 或外部渠道。MVP 模式只生成动作草稿。\n\n用户请求：{message.strip()}\n", citations


def _safe_confidence(value: Any, default: float) -> float:
    """置信度夹到 [0,1]，解析失败用默认值（模型偶尔返回字符串）。"""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _skipped_workflow_result(task: WorkflowTask, summary: str) -> WorkflowTaskResult:
    """构造一个"跳过"的任务结果（没有目标数据时用）。"""
    return WorkflowTaskResult(
        task_id=task.id,
        agent_name=task.agent_name,
        task_type=task.task_type,
        status="skipped",
        summary=summary,
        confidence=0.0,
    )


def _agent_context(context: dict[str, Any]) -> dict[str, Any]:
    """去掉 tools / _skill_registry 再传给子 Agent。

    为什么：tools 是给 ChatAgent 的工具表，_skill_registry 是内部对象，
    传给子 Agent 会污染上下文（而且不可 JSON 序列化）。
    """
    return {key: value for key, value in context.items() if key not in {"tools", "_skill_registry"}}


class WorkflowTargetResolutionError(ValueError):
    """Raised when an explicitly requested workflow target cannot be resolved safely.

    【为什么单独定义这个异常】用户明确说了"分析 Issue #42"，但本地没同步到 #42 时，
    绝不能悄悄退回"最新 Issue" —— 那会让用户以为分析的就是 #42。
    所以要抛这个错，让上层返回 422 并说明原因。
    """


# 从用户话里提取"目标编号"的正则。每种目标类型两个模式（英文 / 中文）。
# (?<![A-Za-z0-9_]) 是词边界保护：防止 "specific 42" 里的 "ci" 被匹配。
_WORKFLOW_TARGET_PATTERNS: dict[str, tuple[str, ...]] = {
    "issue": (
        r"(?<![A-Za-z0-9_])issue(?![A-Za-z0-9_])\s*(?:#|＃|no\.?|number|id)?\s*(\d+)",
        r"(?:工单|议题)\s*(?:#|＃|编号)?\s*(\d+)",
    ),
    "pull_request": (
        r"(?<![A-Za-z0-9_])(?:pr|pull[\s_-]*request)(?![A-Za-z0-9_])\s*(?:#|＃|no\.?|number|id)?\s*(\d+)",
        r"合并请求\s*(?:#|＃|编号)?\s*(\d+)",
    ),
    "workflow_run": (
        r"(?<![A-Za-z0-9_])(?:ci|workflow[\s_-]*run)(?![A-Za-z0-9_])\s*(?:#|＃|no\.?|number|id)?\s*(\d+)",
        r"(?<![A-Za-z0-9_])run(?![A-Za-z0-9_])\s*(?:#|＃|id)\s*(\d+)",
        r"(?:CI\s*运行|工作流运行|流水线)\s*(?:#|＃|编号|ID)?\s*(\d+)",
    ),
}

# 报错文案里用的中文标签
_WORKFLOW_TARGET_LABELS = {
    "issue": "Issue",
    "pull_request": "PR",
    "workflow_run": "CI/Workflow Run",
}


def _requested_workflow_target_number(message: str, target_type: str) -> int | None:
    """从消息里提取目标编号。

    提取到多个就报错 —— 一次工作流只能为一个对象建立 claim，
    否则任务不知道该分析哪个。
    """
    numbers = {
        int(match.group(1))
        for pattern in _WORKFLOW_TARGET_PATTERNS[target_type]
        for match in re.finditer(pattern, message, flags=re.IGNORECASE)
    }
    if len(numbers) > 1:
        references = "、".join(f"#{number}" for number in sorted(numbers))
        raise WorkflowTargetResolutionError(
            f"一次工作流只能为一个 {_WORKFLOW_TARGET_LABELS[target_type]} 建立明确 claim；检测到多个目标：{references}。"
        )
    return next(iter(numbers), None)


def _target_by_number(
    items: list[Any],
    number: int | None,
    *,
    target_type: str,
    number_attribute: str,
) -> Any | None:
    """按编号找目标。找不到就报错（不回退到最新对象）。"""
    if number is None:
        return None
    target = next((item for item in items if getattr(item, number_attribute, None) == number), None)
    if target is None:
        raise WorkflowTargetResolutionError(
            f"当前仓库没有同步 {_WORKFLOW_TARGET_LABELS[target_type]} #{number}，已停止生成计划；不会回退到最新对象。"
        )
    return target


def _build_engineering_workflow_runtime(
    repo: Repository,
    db: Session,
    message: str,
    context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """构建工作流运行时：返回 (workflow_context, task_runners)。

    【核心职责】
      ① 解析用户点名的目标（Issue #42 / PR #7 / CI run 123）
      ② 组装 workflow_context（Planner 用它决定要哪些任务）
      ③ 提供 6 个 task runner（Orchestrator 按计划调度它们）

    task_runners 的 6 个键必须和 Planner 输出的 task_type 对上。
    """
    issues, prs, runs = _repo_payload(repo.id, db)
    failed_runs = [run for run in runs if run.conclusion == "failure"]

    # ── ① 解析显式目标 ──
    requested_issue_number = _requested_workflow_target_number(message, "issue")
    requested_pr_number = _requested_workflow_target_number(message, "pull_request")
    requested_run_number = _requested_workflow_target_number(message, "workflow_run")
    # 用户点名了但本地没同步 → 这里会抛 WorkflowTargetResolutionError
    selected_issue = _target_by_number(
        issues,
        requested_issue_number,
        target_type="issue",
        number_attribute="number",
    )
    selected_pr = _target_by_number(
        prs,
        requested_pr_number,
        target_type="pull_request",
        number_attribute="number",
    )
    selected_run = _target_by_number(
        runs,
        requested_run_number,
        target_type="workflow_run",
        # 注意这里用 github_run_id 而不是 number（WorkflowRun 没有 number）
        number_attribute="github_run_id",
    )
    # 用户没点名 → 用"最新的"作为默认目标
    if selected_issue is None:
        selected_issue = sorted(issues, key=_item_time, reverse=True)[0] if issues else None
    if selected_pr is None:
        selected_pr = sorted(prs, key=_item_time, reverse=True)[0] if prs else None
    # CI 默认取"最新的失败 run"，而不是最新 run —— 用户问 CI 多半是想排障
    if selected_run is None:
        selected_run = sorted(failed_runs, key=_item_time, reverse=True)[0] if failed_runs else None

    def claimed_target(task: WorkflowTask, target_type: str, items: list[Any], fallback: Any | None) -> Any | None:
        """按任务的 claim 字段确定它该操作哪个对象。

        claim 是 Planner 写进任务里的"目标声明"，包含 target_type + target_id。
        【关键】claim.target_id 存在但对象不存在 → 报错，不回退。
        否则会出现"任务说要分析 PR #7，实际分析了最新的 PR #9"。
        """
        if task.claim.target_type != target_type:
            raise WorkflowTargetResolutionError(
                f"任务 {task.id} 的 claim 类型为 {task.claim.target_type!r}，预期为 {target_type!r}。"
            )
        if not task.claim.target_id:
            # 没声明具体 id → 用默认目标
            return fallback
        try:
            target_id = uuid.UUID(str(task.claim.target_id))
        except (TypeError, ValueError) as exc:
            raise WorkflowTargetResolutionError(f"任务 {task.id} 的 claim.target_id 不是有效 UUID。") from exc
        target = next((item for item in items if item.id == target_id), None)
        if target is None:
            raise WorkflowTargetResolutionError(
                f"任务 {task.id} 的 claim 目标不存在或不属于仓库 {repo.full_name}；不会回退到最新对象。"
            )
        return target

    safe_context = _agent_context(context)
    skill_registry = context.get("_skill_registry")
    # 技能激活结果缓存：同一 entrypoint 在一次工作流里只解析一次
    agent_skill_cache: dict[str, dict[str, Any]] = {}

    def agent_skill_context(entrypoint: str) -> dict[str, Any]:
        """按 entrypoint 激活技能，返回要注入 Agent 的 runtime 上下文。

        缓存的原因：一次工作流可能跑多个同类任务，重复激活会重复解析 SKILL.md。
        """
        if entrypoint in agent_skill_cache:
            return agent_skill_cache[entrypoint]
        if not isinstance(skill_registry, SkillRegistry):
            agent_skill_cache[entrypoint] = {}
            return {}
        activation = skill_registry.activate_entrypoint(entrypoint)
        if activation is None:
            agent_skill_cache[entrypoint] = {}
            return {}
        payload = activation.model_dump(mode="json")
        runtime = {
            "active_skills": [payload],
            "skill_instructions": activation.prompt_block(),
            "skill_runtime": {
                **activation.trace_metadata(),
                "entrypoint": activation.entrypoint,
                "output_contract": activation.output_contract,
                "safety_level": activation.safety_level,
                "instructions_loaded": True,
            },
        }
        agent_skill_cache[entrypoint] = runtime
        return runtime

    def with_skill_runtime(result: dict[str, Any], entrypoint: str) -> dict[str, Any]:
        """给任务输出打上"用了哪个技能"的标记（前端显示技能徽章）。"""
        runtime = agent_skill_context(entrypoint).get("skill_runtime")
        return {**result, "_skill_runtime": {**runtime, "validation": "entrypoint_completed"}} if runtime else result

    # ── ② 组装 workflow_context（Planner 的输入）──
    # has_* / *_explicit 这两个字段决定 Planner 选哪些任务：
    #   has_issue=True 才可能排 issue_analysis
    #   issue_explicit=True 表示"用户点名了"，Planner 要严格按这个目标
    workflow_context = {
        **safe_context,
        "repo_id": str(repo.id),
        "repo_full_name": repo.full_name,
        "has_repo": True,
        "has_issue": selected_issue is not None,
        "issue_id": str(selected_issue.id) if selected_issue else None,
        "issue_number": selected_issue.number if selected_issue else None,
        "issue_explicit": requested_issue_number is not None,
        "has_pr": selected_pr is not None,
        "pr_id": str(selected_pr.id) if selected_pr else None,
        "pr_number": selected_pr.number if selected_pr else None,
        "pr_explicit": requested_pr_number is not None,
        "has_ci_run": selected_run is not None,
        "has_failed_ci": selected_run is not None and selected_run.conclusion == "failure",
        "workflow_run_id": str(selected_run.id) if selected_run else None,
        "workflow_run_number": selected_run.github_run_id if selected_run else None,
        "workflow_run_explicit": requested_run_number is not None,
        "requires_safety": bool(_risky_action(message)),
        "team_members": safe_context.get("team_members") or [],
    }

    # ── ③ 六个 task runner ──
    async def run_repo_health(task: WorkflowTask, spec: WorkflowSpec, workflow_context: dict[str, Any]) -> WorkflowTaskResult:
        """仓库健康度（纯模板，不调 LLM）。"""
        answer = _repo_health_answer(repo, issues, prs, runs)
        open_issue_count = len([item for item in issues if item.state == "open"])
        open_pr_count = len([item for item in prs if item.state == "open"])
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary=f"{repo.full_name}: {open_issue_count} open issues, {open_pr_count} open PRs, {len(failed_runs)} failed CI runs.",
            output={
                "answer": answer,
                "open_issues": open_issue_count,
                "open_prs": open_pr_count,
                "failed_ci": len(failed_runs),
            },
            evidence_count=len(issues) + len(prs) + len(runs),
            confidence=0.82,
            citations=[{"type": "repository", "id": str(repo.id), "title": repo.full_name}],
        )

    async def run_issue_task(task: WorkflowTask, spec: WorkflowSpec, workflow_context: dict[str, Any]) -> WorkflowTaskResult:
        """Issue 分析。没数据就 skipped（不是 error）。"""
        issue = claimed_target(task, "issue", issues, selected_issue)
        if issue is None:
            return _skipped_workflow_result(task, "No synced issue is available.")
        result = await IssueAgent().run(
            {
                "id": str(issue.id),
                "title": issue.title,
                "body": issue.body,
                "labels": issue.labels,
                "number": issue.number,
                "state": issue.state,
                "assignees": issue.assignees,
                "author": issue.author,
            },
            {**safe_context, **agent_skill_context("issue_agent.analyze_issue")},
        )
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary=f"Issue #{issue.number}: {result.get('conclusion')} / {result.get('priority')} / owner {result.get('suggested_owner')}.",
            output=with_skill_runtime(result, "issue_agent.analyze_issue"),
            evidence_count=len(result.get("evidence") or []),
            confidence=_safe_confidence(result.get("confidence"), 0.62),
            citations=[{"type": "issue", "id": str(issue.id), "title": issue.title}],
        )

    async def run_pr_task(task: WorkflowTask, spec: WorkflowSpec, workflow_context: dict[str, Any]) -> WorkflowTaskResult:
        """PR 审查。会顺手把 CI 任务的 claim 目标也解析出来，塞进 ci_summary。"""
        pull_request = claimed_target(task, "pull_request", prs, selected_pr)
        if pull_request is None:
            return _skipped_workflow_result(task, "No synced pull request is available.")
        # 找到同一次工作流里的 ci_debug 任务，把它的目标也带上（让 PR 审查知道 CI 状态）
        ci_task = next((item for item in spec.tasks if item.task_type == "ci_debug"), None)
        claimed_ci_run = claimed_target(ci_task, "workflow_run", runs, selected_run) if ci_task else selected_run
        pr_context = {
            **safe_context,
            **agent_skill_context("pr_review_agent.analyze_pr"),
            "repo_id": str(repo.id),
            "ci_summary": {
                "failed_runs": len(failed_runs),
                "latest_failed_run": claimed_ci_run.name if claimed_ci_run and claimed_ci_run.conclusion == "failure" else None,
            },
        }
        result = await PRReviewAgent().run(
            {
                "number": pull_request.number,
                "title": pull_request.title,
                "body": pull_request.body,
                "state": pull_request.state,
                "author": pull_request.author,
                "merged_at": pull_request.merged_at,
                "files": [
                    {
                        "filename": file.filename,
                        "status": file.status,
                        "additions": file.additions,
                        "deletions": file.deletions,
                        "patch": file.patch,
                    }
                    for file in pull_request.files
                ],
                "comments": [
                    {
                        "body": comment.body,
                        "path": comment.path,
                        "line": comment.line,
                        "author": comment.author,
                    }
                    for comment in pull_request.review_comments
                ],
            },
            pr_context,
        )
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary=f"PR #{pull_request.number}: {result.get('merge_recommendation')} with {len(result.get('risk_points') or [])} risk points.",
            output=with_skill_runtime(result, "pr_review_agent.analyze_pr"),
            evidence_count=len(pull_request.files) + len(pull_request.review_comments),
            confidence=_safe_confidence(result.get("confidence"), 0.62),
            citations=[{"type": "pull_request", "id": str(pull_request.id), "title": pull_request.title}],
        )

    async def run_ci_task(task: WorkflowTask, spec: WorkflowSpec, workflow_context: dict[str, Any]) -> WorkflowTaskResult:
        """CI 排障。"""
        workflow_run = claimed_target(task, "workflow_run", runs, selected_run)
        if workflow_run is None:
            return _skipped_workflow_result(task, "No synced CI run is available.")
        result = await CIDebugAgent().run(
            {
                "name": workflow_run.name,
                "status": workflow_run.status,
                "conclusion": workflow_run.conclusion,
                "jobs": workflow_run.jobs or [],
                "logs_text": workflow_run.logs_text,
            },
            {**safe_context, **agent_skill_context("ci_debug_agent.analyze_workflow_run"), "repo_id": str(repo.id)},
        )
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary=f"CI {workflow_run.name}: {result.get('failure_type')}, merge blocking={result.get('is_merge_blocking')}.",
            output=with_skill_runtime(result, "ci_debug_agent.analyze_workflow_run"),
            evidence_count=len(workflow_run.jobs or []) + (1 if workflow_run.logs_text else 0),
            confidence=_safe_confidence(result.get("confidence"), 0.58),
            citations=[{"type": "workflow_run", "id": str(workflow_run.id), "title": workflow_run.name}],
        )

    async def run_rag_task(task: WorkflowTask, spec: WorkflowSpec, workflow_context: dict[str, Any]) -> WorkflowTaskResult:
        """RAG 检索。会记一条 recall_event（可观测）。"""
        query = message or spec.goal
        answer, citations, results = _semantic_search(repo, query, safe_context, db)
        record_recall_event(
            db,
            repo_id=repo.id,
            conversation_id=uuid.UUID(str(context["conversation_id"])) if context.get("conversation_id") else None,
            tool_name="workflow.rag_search",
            query=query,
            results=results,
            citations=citations,
            scope="project",
            mode="hybrid",
            metadata={"route": "workflow_rag_task", "task_id": task.id},
        )
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary=f"Retrieved {len(citations)} related historical records.",
            output={"answer": answer, "citations": citations},
            evidence_count=len(citations),
            # 有命中 0.68，没命中 0.45（低于观察者的低置信阈值，会触发重规划）
            confidence=0.68 if citations else 0.45,
            citations=citations,
        )

    async def run_safety_task(task: WorkflowTask, spec: WorkflowSpec, workflow_context: dict[str, Any]) -> WorkflowTaskResult:
        """安全审查：判定动作风险等级，生成草稿。"""
        action = _risky_action(message) or "write_back"
        draft, citations = _draft_for_action(action, message or spec.goal, repo, issues, prs, runs)
        result = await SafetyAgent().run(
            {"action": action, "draft": draft},
            {**safe_context, **agent_skill_context("safety_agent.classify_action_risk")},
        )
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="success",
            summary=f"Safety risk={result.get('risk_level')}, confirmation required={result.get('requires_confirmation')}.",
            output=with_skill_runtime(result, "safety_agent.classify_action_risk"),
            evidence_count=1,
            confidence=0.9,
            citations=citations,
        )

    # 这 6 个键必须覆盖 Planner 可能输出的所有 task_type
    return workflow_context, {
        "repo_health": run_repo_health,
        "issue_analysis": run_issue_task,
        "pr_review": run_pr_task,
        "ci_debug": run_ci_task,
        "rag_search": run_rag_task,
        "safety_review": run_safety_task,
    }


def _persist_engineering_workflow(
    db: Session,
    repo: Repository | None,
    conversation: Conversation | None,
    report: dict[str, Any],
) -> uuid.UUID:
    """把工作流执行结果落库：1 条 AgentWorkflowRun + N 条 AgentTaskRun。"""
    spec = report.get("spec") or {}
    # task_id → spec 里的任务定义（要拿 claim / objective / dependencies）
    task_by_id = {
        str(task.get("id")): task
        for task in spec.get("tasks", [])
        if isinstance(task, dict) and task.get("id")
    }
    workflow = AgentWorkflowRun(
        repo_id=repo.id if repo else None,
        conversation_id=conversation.id if conversation else None,
        goal=str(spec.get("goal") or ""),
        spec_json=spec,
        observation_json=report.get("observation") or {},
        final_answer=str(report.get("final_answer") or ""),
        metrics=report.get("metrics") or {},
        # 只要有任务出错，整个工作流就标 error（部分成功也算 error，前端会显示详情）
        status="error" if int((report.get("metrics") or {}).get("error_count") or 0) else "success",
        completed_at=datetime.now(timezone.utc),
    )
    db.add(workflow)
    # flush 拿 workflow.id（下面子任务要当外键用），还不 commit
    db.flush()
    for result in report.get("task_results") or []:
        if not isinstance(result, dict):
            continue
        task_id = str(result.get("task_id") or "")
        task = task_by_id.get(task_id, {})
        db.add(
            AgentTaskRun(
                workflow_id=workflow.id,
                task_id=task_id,
                agent_name=str(result.get("agent_name") or ""),
                task_type=str(result.get("task_type") or ""),
                claim_json=task.get("claim") if isinstance(task.get("claim"), dict) else {},
                input_json={"objective": task.get("objective"), "dependencies": task.get("dependencies", [])},
                result_json=result,
                status=str(result.get("status") or "unknown"),
                error=result.get("error"),
            )
        )
    return workflow.id


def _build_conversation_tools(repo: Repository | None, conversation: Conversation | None, db: Session) -> dict[str, Any]:
    """构造 ChatAgent 可用的全部工具（14 个）。

    每个工具都是 async 函数，签名统一为 (input_data, context) → dict，
    返回结构固定：{route, tool_calls, answer, citations}。
    ChatAgent 靠这个统一结构渲染"工具轨迹"。

    【安全】所有工具都在这个闭包里，repo / conversation / db 是闭包捕获的 ——
    模型无法通过参数越权访问别的仓库（repo_id 由服务端填，不取模型给的）。
    """
    mcp_client = StdioMcpClient()

    def agent_context(context: dict[str, Any]) -> dict[str, Any]:
        """剥掉 tools / _skill_registry（同上）。"""
        return {key: value for key, value in context.items() if key not in {"tools", "_skill_registry"}}

    # ── MCP 记忆工具（走 stdio 子进程，调外部 memory server）──
    async def call_memory_mcp(tool_name: str, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """通用 MCP 调用。repo_id / conversation_id 强制由服务端注入。"""
        arguments = dict(input_data)
        if repo:
            arguments["repo_id"] = str(repo.id)
        if conversation:
            arguments["conversation_id"] = str(conversation.id)
        if tool_name == "devflow_search_evidence":
            # 模型有时只传 message 不传 query，这里补一下
            arguments.setdefault("query", input_data.get("message") or "")
        result = await mcp_client.call_tool(tool_name, arguments)
        # MCP 返回 content 数组，只取 type=text 的片段
        text_parts = [
            str(item.get("text") or "")
            for item in result.get("content", [])
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return {
            "route": tool_name.replace("devflow_", ""),
            "tool_calls": [{"tool_name": tool_name, "arguments": arguments, "status": "success"}],
            "answer": "\n".join(text_parts).strip(),
            "citations": [],
        }

    async def search_evidence_mcp(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return await call_memory_mcp("devflow_search_evidence", input_data, context)

    async def get_thread_context_mcp(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return await call_memory_mcp("devflow_get_thread_context", input_data, context)

    async def read_session_events_mcp(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        return await call_memory_mcp("devflow_read_session_events", input_data, context)

    # ── 工作区工具（读本地代码，不进 RAG）──
    async def list_workspace_files(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """列文件。limit 夹在 1~200，pattern 是子串匹配。"""
        workspace_root = _workspace_root(repo)
        root = _workspace_path(workspace_root, str(input_data.get("path") or "."))
        limit = min(max(int(input_data.get("limit") or 80), 1), 200)
        raw_pattern = str(input_data.get("pattern") or "").strip().lower()
        rows: list[str] = []
        for path in _iter_workspace_files(root, workspace_root):
            relative = _workspace_relative(path, workspace_root)
            if raw_pattern and raw_pattern not in relative.lower():
                continue
            rows.append(relative)
            if len(rows) >= limit:
                break
        answer = "\n".join(rows) if rows else "没有找到匹配的工作区文件。"
        return {
            "route": "workspace_list_files",
            "tool_calls": [{"tool_name": "workspace.list_files", "arguments": {"path": str(input_data.get("path") or "."), "pattern": raw_pattern, "limit": limit}, "status": "success"}],
            "answer": answer,
            "citations": [{"type": "workspace", "id": str(workspace_root), "title": repo.full_name if repo else "项目工作区"}],
        }

    async def read_workspace_file(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """读文件片段。默认 160 行、最多 400 行，返回带行号的文本。"""
        raw_path = str(input_data.get("path") or "").strip()
        if not raw_path:
            raise ValueError("path 为必填项")
        workspace_root = _workspace_root(repo)
        path = _workspace_path(workspace_root, raw_path)
        text = _read_workspace_text(path)
        lines = text.splitlines()
        start_line = max(int(input_data.get("start_line") or 1), 1)
        line_count = min(max(int(input_data.get("line_count") or 160), 1), 400)
        end_line = min(start_line + line_count - 1, len(lines))
        # 带行号：模型引用代码时能说"第 42 行"
        numbered = [f"{line_number}: {lines[line_number - 1]}" for line_number in range(start_line, end_line + 1)]
        relative = _workspace_relative(path, workspace_root)
        return {
            "route": "workspace_read_file",
            "tool_calls": [{"tool_name": "workspace.read_file", "arguments": {"path": relative, "start_line": start_line, "line_count": line_count}, "status": "success"}],
            "answer": f"{relative}:{start_line}-{end_line}\n" + "\n".join(numbered),
            "citations": [{"type": "workspace_file", "id": relative, "title": relative}],
        }

    async def search_workspace_code(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """在工作区里搜代码（ripgrep）。"""
        query = str(input_data.get("query") or input_data.get("message") or "").strip()
        # query_terms 为空说明全是停用词/符号，搜也没意义
        if not query_terms(query):
            raise ValueError("query 为必填项")
        workspace_root = _workspace_root(repo)
        relative_path = str(input_data.get("path") or ".")
        # 先校验路径合法（不合法会抛错），再搜
        _workspace_path(workspace_root, relative_path)
        limit = min(max(int(input_data.get("limit") or 12), 1), 50)
        matches = search_checkout_code(workspace_root, query, limit=limit, relative_path=relative_path)
        rows = [item["snippet"] for item in matches]
        citations = [
            {"type": "workspace_file", "id": item["source_id"], "title": item["title"]}
            for item in matches
        ]
        return {
            "route": "workspace_search_code",
            "tool_calls": [{"tool_name": "workspace.search_code", "arguments": {"query": query, "path": relative_path, "limit": limit}, "status": "success"}],
            "answer": "\n".join(rows) if rows else "没有找到匹配的工作区代码。",
            "citations": citations,
        }

    # ── 业务工具（每个对应一个专用 Agent 或 service）──
    async def repo_health(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """仓库健康度。没有仓库时返回提示，不报错。"""
        if repo is None:
            return {"route": "unknown", "tool_calls": [], "answer": "当前还没有可用仓库，请先连接仓库。", "citations": []}
        issues, prs, runs = _repo_payload(repo.id, db)
        return {
            "route": "repo_health",
            "tool_calls": [{"tool_name": "repo_health.generate_summary", "arguments": {"repo_id": str(repo.id)}, "status": "success"}],
            "answer": _repo_health_answer(repo, issues, prs, runs),
            "citations": [{"type": "repository", "id": str(repo.id), "title": repo.full_name}],
        }

    async def semantic_search(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """RAG 检索。

        除了 answer，还返回 retrieved_contexts / retrieval_results ——
        这两个字段会被 RAGAS 评测用来算 context_precision / recall。
        """
        if repo is None:
            return await repo_health(input_data, context)
        original_message = str(input_data.get("message") or "").strip()
        # 模型可能改写 query（query 重写），所以和原问题分开记
        query = str(input_data.get("query") or original_message).strip()
        if not query:
            raise ValueError("query 为必填项")
        answer, citations, results = _semantic_search(repo, query, context, db)
        record_recall_event(
            db,
            repo_id=repo.id,
            conversation_id=conversation.id if conversation else None,
            tool_name="rag.search_similar_documents",
            query=query,
            results=results,
            citations=citations,
            scope="conversation" if conversation else "project",
            mode="hybrid",
            metadata={
                "source_type": context.get("source_type"),
                "limit": context.get("limit", 5),
                "original_message": original_message,
                "query_rewritten": bool(original_message and query != original_message),
            },
        )
        return {
            "route": "semantic_search",
            "tool_calls": [
                {
                    "tool_name": "rag.search_similar_documents",
                    "arguments": {"repo_id": str(repo.id), "query": query},
                    "status": "success",
                }
            ],
            "answer": answer,
            "citations": citations,
            "retrieved_contexts": [
                str(item.get("snippet") or "")
                for item in results
                if str(item.get("snippet") or "").strip()
            ],
            "retrieval_results": [
                {
                    "id": str(item.get("id") or ""),
                    "source_id": str(item.get("source_id") or ""),
                    "source_type": str(item.get("source_type") or ""),
                    "title": str(item.get("title") or ""),
                    "score": item.get("score"),
                    "snippet": str(item.get("snippet") or ""),
                    "retrieval": item.get("retrieval") or {},
                }
                for item in results
            ],
        }

    async def analyze_issue(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Issue 分析。注意这里取的是"最新创建的 Issue"，不看用户指定（工具版简化）。"""
        if repo is None:
            return await repo_health(input_data, context)
        issue = db.query(Issue).filter(Issue.repo_id == repo.id).order_by(Issue.created_at.desc()).first()
        if not issue:
            return {"route": "issue_analysis", "tool_calls": [], "answer": f"仓库 {repo.full_name} 还没有同步到 Issue。", "citations": []}
        result = await IssueAgent().run(
            {
                "id": str(issue.id),
                "title": issue.title,
                "body": issue.body,
                "labels": issue.labels,
                "number": issue.number,
                "state": issue.state,
                "assignees": issue.assignees,
                "author": issue.author,
            },
            agent_context(context),
        )
        return {
            "route": "issue_analysis",
            "tool_calls": [{"tool_name": "issue_agent.analyze_issue", "arguments": {"issue_id": str(issue.id)}, "status": "success"}],
            "answer": (
                f"Issue #{issue.number} 的结论：{result['conclusion']}。\n\n"
                f"- 分类：{result['category']}\n"
                f"- 优先级：{result['priority']}\n"
                f"- 复杂度：{result['complexity']}\n"
                f"- 建议负责人：{result['suggested_owner']}，{result['owner_reason']}\n\n"
                f"理由：{result['conclusion_reason']}\n\n"
                f"下一步：{_join_items(result.get('checklist') or result.get('action_items') or [], '补充复现步骤、影响范围和验收标准。')}"
            ),
            "citations": [{"type": "issue", "id": str(issue.id), "title": issue.title}],
        }

    async def analyze_pr(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """PR 审查（同样取最新创建的 PR）。"""
        if repo is None:
            return await repo_health(input_data, context)
        pr = (
            db.query(PullRequest)
            .options(selectinload(PullRequest.files), selectinload(PullRequest.review_comments))
            .filter(PullRequest.repo_id == repo.id)
            .order_by(PullRequest.created_at.desc())
            .first()
        )
        if not pr:
            return {"route": "pr_review", "tool_calls": [], "answer": f"仓库 {repo.full_name} 还没有同步到 PR。", "citations": []}
        result = await PRReviewAgent().run(
            {
                "number": pr.number,
                "title": pr.title,
                "body": pr.body,
                "files": [{"filename": file.filename, "status": file.status, "patch": file.patch} for file in pr.files],
                "comments": [{"body": comment.body, "path": comment.path, "line": comment.line, "author": comment.author} for comment in pr.review_comments],
            },
            agent_context(context) | {"repo_id": str(repo.id)},
        )
        return {
            "route": "pr_review",
            "tool_calls": [{"tool_name": "pr_review_agent.analyze_pr", "arguments": {"pr_id": str(pr.id)}, "status": "success"}],
            "answer": f"PR #{pr.number} 摘要：{result['summary']}\n\n主要风险：{_join_items(result['risk_points'], '没有明显风险。')}\n测试建议：{_join_items(result['test_suggestions'], '运行相关测试和核心流程检查。')}",
            "citations": [{"type": "pull_request", "id": str(pr.id), "title": pr.title}],
        }

    async def analyze_ci(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """CI 排障。只找"失败的" run；没有失败就退回健康度摘要。"""
        if repo is None:
            return await repo_health(input_data, context)
        run = db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo.id, WorkflowRun.conclusion == "failure").order_by(WorkflowRun.created_at.desc()).first()
        if not run:
            # 没有失败 CI 不是错误 —— 转成健康度摘要并说明
            issues, prs, runs = _repo_payload(repo.id, db)
            return {
                "route": "repo_health",
                "tool_calls": [{"tool_name": "repo_health.generate_summary", "arguments": {"repo_id": str(repo.id), "reason": "no_failed_ci"}, "status": "success"}],
                "answer": _repo_health_answer(repo, issues, prs, runs) + "\n\n说明：这个仓库目前还没有同步到失败 CI 运行。",
                "citations": [{"type": "repository", "id": str(repo.id), "title": repo.full_name}],
            }
        result = await CIDebugAgent().run({"name": run.name, "status": run.status, "conclusion": run.conclusion, "jobs": run.jobs or [], "logs_text": run.logs_text}, agent_context(context) | {"repo_id": str(repo.id)})
        return {
            "route": "ci_debug",
            "tool_calls": [{"tool_name": "ci_debug_agent.analyze_workflow_run", "arguments": {"run_id": str(run.id)}, "status": "success"}],
            "answer": f"最新失败 CI：{run.name}。失败类型：{result['failure_type']}。\n\n{result['failure_summary']}\n\n排查步骤：{_join_items(result['debug_steps'], '没有明确可用的排查步骤。')}",
            "citations": [{"type": "workflow_run", "id": str(run.id), "title": run.name}],
        }

    async def generate_report(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """生成周报，并顺手写进 RAG（之后能被检索到）。

        commit=False：和整个对话一起提交，避免出现"报告入库了但对话失败了"。
        """
        if repo is None:
            return await repo_health(input_data, context)
        issues, prs, runs = _repo_payload(repo.id, db)
        start_at, end_at = _date_window_from_context(context)
        # 区间过滤：任一动作落在窗口内就算
        scoped_issues = [item for item in issues if any(_within(value, start_at, end_at) for value in [item.created_at, item.updated_at, item.closed_at])]
        scoped_prs = [item for item in prs if any(_within(value, start_at, end_at) for value in [item.created_at, item.updated_at, item.merged_at])]
        scoped_runs = [item for item in runs if any(_within(value, start_at, end_at) for value in [item.created_at, item.updated_at])]
        result = await ReportAgent().run(
            {
                "issues": [{"number": item.number, "title": item.title, "state": item.state, "closed_at": item.closed_at} for item in scoped_issues],
                "pull_requests": [{"number": item.number, "title": item.title, "state": item.state, "merged_at": item.merged_at} for item in scoped_prs],
                "workflow_runs": [{"name": item.name, "conclusion": item.conclusion} for item in scoped_runs],
                "metrics": {
                    "issues": len(scoped_issues),
                    "pull_requests": len(scoped_prs),
                    "merged_prs": len([item for item in scoped_prs if item.merged_at]),
                    "open_prs": len([item for item in scoped_prs if item.state == "open"]),
                    "failed_ci": len([item for item in scoped_runs if item.conclusion == "failure"]),
                },
            },
            {"repo_id": str(repo.id), "repo_name": repo.full_name, "start_date": start_at.date().isoformat(), "end_date": end_at.date().isoformat()},
        )
        add_document(
            db,
            repo.id,
            "weekly_report",
            None,
            f"研发周报 {start_at.date().isoformat()} - {end_at.date().isoformat()}",
            result["report_markdown"],
            {
                "display_title": f"研发周报 {start_at.date().isoformat()} - {end_at.date().isoformat()}",
                "start_date": start_at.date().isoformat(),
                "end_date": end_at.date().isoformat(),
                "repo_name": repo.full_name,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                # source=chat：标记这份周报是"对话里生成的"，区别于报告页生成的
                "source": "chat",
            },
            commit=False,
        )
        return {
            "route": "weekly_report",
            "tool_calls": [{"tool_name": "report_agent.generate_weekly_report", "arguments": {"repo_id": str(repo.id)}, "status": "success"}],
            "answer": result["report_markdown"],
            "citations": [{"type": "repository", "id": str(repo.id), "title": repo.full_name}],
        }

    async def check_safety(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """★ 安全闸门：判定风险 + 生成 ActionDraft（不执行）。

        这是"AI 想写 GitHub"在对话里的唯一出口 —— 只落草稿，等人工确认。
        """
        if repo is None:
            return await repo_health(input_data, context)
        issues, prs, runs = _repo_payload(repo.id, db)
        action = _risky_action(input_data["message"]) or "write_back"
        draft, citations = _draft_for_action(action, input_data["message"], repo, issues, prs, runs)
        safety = await SafetyAgent().run({"action": action, "draft": draft}, agent_context(context))

        # 决定草稿挂在谁身上：PR 优先，其次 Issue
        latest_issue = sorted(issues, key=_item_time, reverse=True)[0] if issues else None
        latest_pr = sorted(prs, key=_item_time, reverse=True)[0] if prs else None
        target_type = "repository"
        target_id = None
        metadata: dict[str, Any] = {"source": "chat", "message": input_data["message"]}
        if action == "comment" and latest_pr:
            target_type = "pull_request"
            target_id = latest_pr.id
            metadata["pr_number"] = latest_pr.number
        elif action in {"comment", "create_issue", "close_issue"} and latest_issue:
            target_type = "issue"
            target_id = latest_issue.id
            metadata["issue_number"] = latest_issue.number

        # 动作名 → 草稿类型（GitHub 侧的执行分支看 draft_type）
        draft_type = {
            "comment": "pr_comment" if target_type == "pull_request" else "issue_comment",
            "create_issue": "create_issue",
            "send_report": "send_report",
            "write_back": "comment",
        }.get(action, action)
        user = ensure_demo_user(db)
        action_draft = ActionDraft(
            repo_id=repo.id,
            draft_type=draft_type,
            target_type=target_type,
            target_id=target_id,
            title=f"{draft_type} 草稿",
            # 用 SafetyAgent 改写过的草稿（可能加了风险提示），没有就用模板原文
            content=str(safety.get("draft") or draft),
            risk_level=str(safety.get("risk_level") or "medium"),
            created_by=user.id,
            meta=metadata,
        )
        db.add(action_draft)
        # flush 拿 draft.id（审计日志里要记）
        db.flush()
        write_audit_log(
            db,
            user=user,
            repo_id=repo.id,
            action="action_draft.create_from_chat",
            target_type=target_type,
            target_id=str(target_id) if target_id else None,
            request_json={"action": action, "draft_type": draft_type},
            result_json={"draft_id": str(action_draft.id)},
        )
        return {
            "route": "safety_draft",
            "tool_calls": [{"tool_name": "safety_agent.classify_action_risk", "arguments": {"action": action, "draft_id": str(action_draft.id)}, "status": "success"}],
            "answer": f"这个请求可能写入 GitHub 或发送到外部渠道，已生成待确认安全草稿。\n\n草稿 ID：{action_draft.id}\n\n{str(safety.get('draft') or draft)}",
            "citations": citations,
        }

    async def run_engineering_workflow(input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """★ 多 Agent 协作入口：Planner → Orchestrator（拓扑执行）→ Observer。

        这是"多 Agent"能力在对话里的触发点。模型判断需要多步协作时会调它。
        """
        if repo is None:
            return await repo_health(input_data, context)

        message = str(input_data.get("message") or input_data.get("query") or "").strip()
        workflow_context, task_runners = _build_engineering_workflow_runtime(repo, db, message, context)
        # .run() 内部会先让 Planner 出 spec，再按依赖拓扑执行
        report = await WorkflowOrchestrator(task_runners).run({"message": message or "Run an engineering workflow."}, workflow_context)
        workflow_run_id = _persist_engineering_workflow(db, repo, conversation, report)
        report["metrics"]["workflow_run_id"] = str(workflow_run_id)
        return _engineering_workflow_response(repo, workflow_run_id, report)

    # 工具名 → 实现。这些名字必须和 ChatAgent 的 available_tools 完全一致
    return {
        "workspace.list_files": list_workspace_files,
        "workspace.read_file": read_workspace_file,
        "workspace.search_code": search_workspace_code,
        "repo_health.generate_summary": repo_health,
        "rag.search_similar_documents": semantic_search,
        "issue_agent.analyze_issue": analyze_issue,
        "pr_review_agent.analyze_pr": analyze_pr,
        "ci_debug_agent.analyze_workflow_run": analyze_ci,
        "report_agent.generate_weekly_report": generate_report,
        "safety_agent.classify_action_risk": check_safety,
        "workflow.run_engineering_review": run_engineering_workflow,
        "devflow_search_evidence": search_evidence_mcp,
        "devflow_get_thread_context": get_thread_context_mcp,
        "devflow_read_session_events": read_session_events_mcp,
    }


def _skill_runtime_context() -> dict[str, Any]:
    """注入技能相关的上下文（每次请求都读一遍注册表，支持热加载 SKILL.md）。

    四个键各自的用途：
      _skill_registry             给 _build_engineering_workflow_runtime 用（激活技能）
      registered_skills           给 ChatAgent 拼 System Prompt（告诉模型有哪些技能）
      skill_trace_by_tool         给前端显示"这次用了什么技能"
      skill_description_by_tool   追加到工具描述后面（引导模型优先用技能）
    """
    registry = get_skill_registry()
    return {
        "_skill_registry": registry,
        "registered_skills": registry.prompt_context(),
        "skill_trace_by_tool": registry.trace_metadata_by_tool(),
        "skill_description_by_tool": registry.tool_description_suffixes(),
    }


def _engineering_workflow_response(repo: Repository, workflow_run_id: uuid.UUID, report: dict[str, Any]) -> dict[str, Any]:
    """把工作流报告包装成标准工具响应（route/tool_calls/answer/citations）。"""
    return {
        "route": "engineering_workflow",
        "tool_calls": [
            {
                "tool_name": "workflow.run_engineering_review",
                "arguments": {"repo_id": str(repo.id), "workflow_run_id": str(workflow_run_id)},
                "status": "success",
            },
            # 把子任务的工具调用也拼进来，前端能看到完整调用链
            *report.get("tool_calls", []),
        ],
        "answer": report.get("final_answer") or "",
        "citations": report.get("citations") or [],
        "agent_events": report.get("agent_events") or [],
    }


def _workflow_plan_answer(spec: WorkflowSpec) -> str:
    """把 WorkflowSpec 渲染成给人看的计划文本（"确认后我就按这个执行"）。"""
    lines = [
        "已创建执行计划，确认后我会按这个 WorkflowSpec 执行。",
        "",
        f"目标：{spec.goal}",
        "",
        "任务：",
    ]
    for index, task in enumerate(spec.tasks, start=1):
        deps = f"；依赖：{', '.join(task.dependencies)}" if task.dependencies else ""
        critical = "；关键任务" if task.critical else ""
        lines.append(f"{index}. {task.agent_name} / {task.task_type}: {task.objective}{deps}{critical}")
    if spec.acceptance_criteria:
        lines.extend(["", "验收标准："])
        lines.extend(f"- {item}" for item in spec.acceptance_criteria)
    return "\n".join(lines)


async def _conversation_plan(payload: ChatRequest, db: Session) -> dict[str, Any]:
    """只出计划，不执行（用户先看要做什么，再决定跑不跑）。"""
    repo = _repo_or_404(db, payload.repo_id)
    conversation = ensure_conversation(db, repo.id, payload.conversation_id)
    db.commit()
    context = {
        **_client_context(payload.context),
        **_skill_runtime_context(),
        "repo_id": str(repo.id),
        "conversation_id": str(conversation.id),
    }
    # 只拿 workflow_context（Planner 的输入），task_runners 这里用不上
    workflow_context, _task_runners = _build_engineering_workflow_runtime(repo, db, payload.message, context)
    spec_data = await PlannerAgent().run({"message": payload.message}, workflow_context)
    # 用 Pydantic 校验 Planner 的输出（模型可能少字段 / 多字段）
    spec = WorkflowSpec.model_validate(spec_data)
    spec_json = spec.model_dump(mode="json")
    return {
        "route": "engineering_plan",
        "conversation_id": str(conversation.id),
        "answer": _workflow_plan_answer(spec),
        "plan": spec_json,
        "tool_calls": [
            {
                "tool_name": "planner_agent.create_workflow_spec",
                "arguments": {"repo_id": str(repo.id), "workflow_id": spec.workflow_id},
                "status": "success",
            }
        ],
        # agent_events 里放完整的任务定义，前端画流程图用
        "agent_events": [
            {
                "type": "workflow_spec",
                "workflow_id": spec.workflow_id,
                "goal": spec.goal,
                "tasks": [
                    {
                        "id": task.id,
                        "agent_name": task.agent_name,
                        "task_type": task.task_type,
                        "dependencies": task.dependencies,
                        "claim": task.claim.model_dump(mode="json"),
                    }
                    for task in spec.tasks
                ],
            }
        ],
    }


def _workflow_execution_trace_events(report: dict[str, Any], workflow_tool_id: str) -> list[dict[str, Any]]:
    """把工作流执行过程转成前端认识的 tool_result 事件序列。

    三种内部事件各转一条：
      workflow_task_result      → 单个任务完成
      workflow_replan           → Planner 补了一次任务
      workflow_observer_result  → 观察者给出质量评估
    """
    events = [
        {
            "event": "tool_result",
            "data": {
                "tool_use_id": workflow_tool_id,
                "tool_name": "workflow.run_engineering_review",
                "status": "success",
                "route": "engineering_workflow",
                "content": {
                    "answer_preview": "Confirmed plan executed.",
                    "citations_count": len(report.get("citations") or []),
                },
            },
        }
    ]
    for item in report.get("agent_events") or []:
        if not isinstance(item, dict):
            continue
        event_type = item.get("type")
        if event_type == "workflow_task_result":
            events.append(
                {
                    "event": "tool_result",
                    "data": {
                        "tool_use_id": f"{item.get('workflow_id')}_{item.get('task_id')}",
                        "tool_name": f"workflow.{item.get('task_type')}",
                        "status": item.get("status") or "success",
                        "route": "engineering_workflow",
                        "content": {
                            "answer_preview": item.get("summary") or "",
                            "citations_count": item.get("evidence_count") or 0,
                        },
                    },
                }
            )
        elif event_type == "workflow_replan":
            events.append(
                {
                    "event": "tool_result",
                    "data": {
                        "tool_use_id": f"{item.get('workflow_id')}_replan_{item.get('iteration')}",
                        "tool_name": "planner_agent.replan_workflow_spec",
                        "status": "success",
                        "route": "engineering_workflow",
                        "content": {
                            "answer_preview": (
                                f"Replan #{item.get('iteration')}: {item.get('reason')}; "
                                f"added tasks={', '.join(str(task) for task in item.get('added_tasks') or []) or 'none'}"
                            ),
                            "citations_count": 0,
                        },
                    },
                }
            )
        elif event_type == "workflow_observer_result":
            events.append(
                {
                    "event": "tool_result",
                    "data": {
                        "tool_use_id": f"{item.get('workflow_id')}_observer",
                        "tool_name": "workflow.observer",
                        "status": "success",
                        "route": "engineering_workflow",
                        "content": {
                            "answer_preview": f"Observer confidence={item.get('overall_confidence')}, findings={len(item.get('findings') or [])}",
                            "citations_count": 0,
                        },
                    },
                }
            )
    return events


async def _conversation_plan_execute_sse(payload: ChatPlanExecuteRequest, db: Session):
    """执行用户已确认的计划（SSE）。

    和 _conversation_sse 的区别：
      它不调 ChatAgent（不重新判断），直接拿用户确认的 spec 交给 Orchestrator 跑。
    """
    repo = _repo_or_404(db, payload.repo_id)

    # ── 准备阶段（和对话路径完全一致的四步）──
    conversation = ensure_conversation(db, repo.id, payload.conversation_id)
    if conversation:
        db.commit()
    compaction_stats = await ProgressiveContextManager(db).ensure_headroom(repo, conversation, payload.message)
    if compaction_stats.get("attempted"):
        db.commit()
    message_store = MessageStore(db)
    user_message = message_store.append(repo_id=repo.id, conversation_id=conversation.id, role="user", content=payload.message)
    EvidenceStore(db).rebuild_repo(repo, conversation)
    db.commit()
    assembled = ContextAssembler(db).assemble(repo, conversation, payload.message)
    context = {
        **_client_context(payload.context),
        **_skill_runtime_context(),
        "repo_id": str(repo.id),
        "conversation_id": str(conversation.id),
        "server_context": assembled.system_context,
        "server_messages": assembled.history_messages,
        "evidence": assembled.evidence,
        "compression_stats": assembled.compression_stats,
        "progressive_compaction": compaction_stats,
    }
    # 工具调用 id 用 spec 的 workflow_id（前端据此把事件挂到对应卡片）
    workflow_tool_id = f"confirmed_{payload.workflow_spec.get('workflow_id') or uuid.uuid4().hex[:8]}"
    try:
        spec = WorkflowSpec.model_validate(payload.workflow_spec)
        # 先发一个 tool_use 事件：前端立刻显示"正在执行计划"
        yield encode_sse(
            {
                "event": "tool_use",
                "data": {
                    "id": workflow_tool_id,
                    "tool_name": "workflow.run_engineering_review",
                    "arguments": {"repo_id": str(repo.id), "workflow_id": spec.workflow_id, "confirmed_plan": True},
                },
            }
        )
        workflow_context, task_runners = _build_engineering_workflow_runtime(repo, db, payload.message, context)
        # run_spec：直接用给定 spec 跑，不让 Planner 重新规划
        report = await WorkflowOrchestrator(task_runners).run_spec(spec, workflow_context)
        workflow_run_id = _persist_engineering_workflow(db, repo, conversation, report)
        report["metrics"]["workflow_run_id"] = str(workflow_run_id)

        # 把每个任务的执行结果逐个推给前端（能看到"第 2 个任务完成了"）
        for event in _workflow_execution_trace_events(report, workflow_tool_id):
            yield encode_sse(event)

        result = _engineering_workflow_response(repo, workflow_run_id, report)
        run_id = uuid.uuid4()
        result["run_id"] = str(run_id)
        result["compression_stats"] = assembled.compression_stats
        result["progressive_compaction"] = compaction_stats
        # 按 48 字符切片做打字机效果（不是真流式，答案已经完整拿到了）
        for index in range(0, len(result["answer"]), 48):
            yield encode_sse({"event": "delta", "data": {"content": result["answer"][index : index + 48]}})

        # ── 收尾（和对话路径一致）──
        assistant_message = message_store.append(
            repo_id=repo.id,
            conversation_id=conversation.id,
            role="assistant",
            content=result["answer"],
            route=result["route"],
            tool_calls=result["tool_calls"],
            meta={
                "run_id": str(run_id),
                "agent_events": result.get("agent_events", []),
                "model_usage": result.get("model_usage", []),
                "context_diagnostics": result.get("context_diagnostics", {}),
                "compression_stats": assembled.compression_stats,
                "progressive_compaction": compaction_stats,
            },
        )
        result["message_id"] = str(assistant_message.id)
        db.add(AgentRun(id=run_id, repo_id=repo.id, conversation_id=conversation.id, user_message=payload.message, route=result["route"], tool_calls=result["tool_calls"], final_answer=result["answer"], status="success"))
        # increment=2：一问一答算 2 条
        touch_conversation(db, conversation, user_message=payload.message, increment=2)
        # 封存会话 → 提炼记忆候选（这是"沉淀"那一步）
        await SessionSealer(db).seal(
            repo=repo,
            conversation=conversation,
            user_message=user_message,
            assistant_message=assistant_message,
            route=result["route"],
            tool_calls=result["tool_calls"],
            agent_events=result.get("agent_events", []),
            model_usage=result.get("model_usage", []),
            context_diagnostics=result.get("context_diagnostics", {}),
        )
        db.commit()
        yield encode_sse({"event": "final", "data": result})
    except Exception as exc:
        # 失败也要落库：用户得在历史里看到"这次失败了"，而不是消息凭空消失
        db.rollback()
        run_id = uuid.uuid4()
        answer = f"执行计划失败：{exc}"
        assistant_message = message_store.append(
            repo_id=repo.id,
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
            route="error",
            tool_calls=[],
            meta={"run_id": str(run_id), "error": str(exc)},
        )
        db.add(AgentRun(id=run_id, repo_id=repo.id, conversation_id=conversation.id, user_message=payload.message, route="error", tool_calls=[], final_answer=answer, status="error"))
        touch_conversation(db, conversation, user_message=payload.message, increment=2)
        await SessionSealer(db).seal(
            repo=repo,
            conversation=conversation,
            user_message=user_message,
            assistant_message=assistant_message,
            route="error",
            tool_calls=[],
            agent_events=[],
        )
        db.commit()
        yield encode_sse(
            {
                "event": "final",
                "data": {
                    "run_id": str(run_id),
                    "message_id": str(assistant_message.id),
                    "route": "error",
                    "tool_calls": [],
                    "answer": answer,
                    "citations": [],
                    "agent_events": [],
                },
            }
        )


async def _conversation_response(payload: ChatRequest, db: Session) -> dict[str, Any]:
    """对话（一次性返回）。核心路径，SSE 版和它逻辑一致。"""
    repo = _repo_or_404(db, payload.repo_id)
    conversation = ensure_conversation(db, repo.id, payload.conversation_id)
    db.commit()

    # ── 步骤 1：上下文快满了先压缩（必须在 append 用户消息之前）──
    compaction_stats = await ProgressiveContextManager(db).ensure_headroom(repo, conversation, payload.message)
    if compaction_stats.get("attempted"):
        db.commit()

    # ── 步骤 2：用户消息落库 ──
    message_store = MessageStore(db)
    user_message = message_store.append(repo_id=repo.id, conversation_id=conversation.id, role="user", content=payload.message)

    # ── 步骤 3：刷新证据库（否则检索不到别处刚改的知识）──
    EvidenceStore(db).rebuild_repo(repo, conversation)
    db.commit()

    # ── 步骤 4：组装上下文 ──
    assembled = ContextAssembler(db).assemble(repo, conversation, payload.message)
    context = {
        **_client_context(payload.context),
        **_skill_runtime_context(),
        "repo_id": str(repo.id),
        "conversation_id": str(conversation.id),
        "server_context": assembled.system_context,
        "server_messages": assembled.history_messages,
        "evidence": assembled.evidence,
        "compression_stats": assembled.compression_stats,
        "progressive_compaction": compaction_stats,
    }
    # 工具表最后注入（其他键都是数据，只有 tools 是可执行函数）
    context["tools"] = _build_conversation_tools(repo, conversation, db)

    # ── 步骤 5：跑 ChatAgent ──
    result = await ChatAgent().run({"message": payload.message}, context)
    run_id = uuid.uuid4()
    result["run_id"] = str(run_id)
    result["compression_stats"] = assembled.compression_stats
    result["progressive_compaction"] = compaction_stats

    # ── 步骤 6：落库 + 封存 ──
    assistant_message: ChatMessage | None = None
    assistant_message = message_store.append(
        repo_id=repo.id,
        conversation_id=conversation.id,
        role="assistant",
        content=result["answer"],
        route=result["route"],
        tool_calls=result["tool_calls"],
        meta={
            "run_id": str(run_id),
            "agent_events": result.get("agent_events", []),
            "model_usage": result.get("model_usage", []),
            "context_diagnostics": result.get("context_diagnostics", {}),
            "compression_stats": assembled.compression_stats,
            "progressive_compaction": compaction_stats,
        },
    )
    result["message_id"] = str(assistant_message.id)
    db.add(AgentRun(id=run_id, repo_id=repo.id, conversation_id=conversation.id, user_message=payload.message, route=result["route"], tool_calls=result["tool_calls"], final_answer=result["answer"], status="success"))
    touch_conversation(db, conversation, user_message=payload.message, increment=2)
    await SessionSealer(db).seal(
        repo=repo,
        conversation=conversation,
        user_message=user_message,
        assistant_message=assistant_message,
        route=result["route"],
        tool_calls=result["tool_calls"],
        agent_events=result.get("agent_events", []),
        model_usage=result.get("model_usage", []),
        context_diagnostics=result.get("context_diagnostics", {}),
    )
    db.commit()
    return result


async def _conversation_sse(payload: ChatRequest, db: Session):
    """对话（SSE 流式）。准备阶段和 _conversation_response 完全一致。"""
    repo = _repo_or_404(db, payload.repo_id)
    conversation = ensure_conversation(db, repo.id, payload.conversation_id)
    db.commit()
    compaction_stats = await ProgressiveContextManager(db).ensure_headroom(repo, conversation, payload.message)
    if compaction_stats.get("attempted"):
        db.commit()
    message_store = MessageStore(db)
    user_message = message_store.append(repo_id=repo.id, conversation_id=conversation.id, role="user", content=payload.message)
    EvidenceStore(db).rebuild_repo(repo, conversation)
    db.commit()
    assembled = ContextAssembler(db).assemble(repo, conversation, payload.message)
    context = {
        **_client_context(payload.context),
        **_skill_runtime_context(),
        "repo_id": str(repo.id),
        "conversation_id": str(conversation.id),
        "server_context": assembled.system_context,
        "server_messages": assembled.history_messages,
        "evidence": assembled.evidence,
        "compression_stats": assembled.compression_stats,
        "progressive_compaction": compaction_stats,
    }
    context["tools"] = _build_conversation_tools(repo, conversation, db)
    try:
        async for event in ChatAgent().run_stream({"message": payload.message}, context):
            # 收到 final 事件时才落库 —— 中间事件（tool_use / delta）直接透传
            if event["event"] == "final":
                run_id = uuid.uuid4()
                event["data"]["run_id"] = str(run_id)
                event["data"]["compression_stats"] = assembled.compression_stats
                event["data"]["progressive_compaction"] = compaction_stats
                assistant_message = message_store.append(
                    repo_id=repo.id,
                    conversation_id=conversation.id,
                    role="assistant",
                    content=event["data"]["answer"],
                    route=event["data"]["route"],
                    tool_calls=event["data"]["tool_calls"],
                    meta={
                        "run_id": str(run_id),
                        "agent_events": event["data"].get("agent_events", []),
                        "model_usage": event["data"].get("model_usage", []),
                        "context_diagnostics": event["data"].get("context_diagnostics", {}),
                        "compression_stats": assembled.compression_stats,
                        "progressive_compaction": compaction_stats,
                    },
                )
                event["data"]["message_id"] = str(assistant_message.id)
                db.add(AgentRun(id=run_id, repo_id=repo.id, conversation_id=conversation.id, user_message=payload.message, route=event["data"]["route"], tool_calls=event["data"]["tool_calls"], final_answer=event["data"]["answer"], status="success"))
                touch_conversation(db, conversation, user_message=payload.message, increment=2)
                await SessionSealer(db).seal(
                    repo=repo,
                    conversation=conversation,
                    user_message=user_message,
                    assistant_message=assistant_message,
                    route=event["data"]["route"],
                    tool_calls=event["data"]["tool_calls"],
                    agent_events=event["data"].get("agent_events", []),
                    model_usage=event["data"].get("model_usage", []),
                    context_diagnostics=event["data"].get("context_diagnostics", {}),
                )
                db.commit()
            yield encode_sse(event)
    except Exception as exc:
        # 响应头已发出，改不了状态码 → 发一个 route="error" 的 final 事件
        db.rollback()
        run_id = uuid.uuid4()
        answer = f"请求失败：{exc}"
        failed_event = {
            "event": "final",
            "data": {
                "run_id": str(run_id),
                "route": "error",
                "tool_calls": [],
                "answer": answer,
                "citations": [],
            },
        }
        assistant_message = message_store.append(
            repo_id=repo.id,
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
            route="error",
            tool_calls=[],
            meta={"run_id": str(run_id), "error": str(exc)},
        )
        failed_event["data"]["message_id"] = str(assistant_message.id)
        db.add(AgentRun(id=run_id, repo_id=repo.id, conversation_id=conversation.id, user_message=payload.message, route="error", tool_calls=[], final_answer=answer, status="error"))
        touch_conversation(db, conversation, user_message=payload.message, increment=2)
        await SessionSealer(db).seal(
            repo=repo,
            conversation=conversation,
            user_message=user_message,
            assistant_message=assistant_message,
            route="error",
            tool_calls=[],
            agent_events=[],
        )
        db.commit()
        yield encode_sse(failed_event)


@router.post("/plan")
async def chat_plan(payload: ChatRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    """只生成计划。

    目标解析失败返回 422（不是 400）：语义上"请求能理解，但无法安全处理"。
    """
    try:
        return await _conversation_plan(payload, db)
    except WorkflowTargetResolutionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/plan/execute/stream")
async def chat_plan_execute_stream(payload: ChatPlanExecuteRequest, db: Session = Depends(get_db)) -> StreamingResponse:
    """执行用户确认过的计划（SSE）。"""
    return StreamingResponse(_conversation_plan_execute_sse(payload, db), media_type="text/event-stream")


@router.post("", response_model=ChatResponse)
async def chat(payload: ChatRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    """对话主入口（一次性返回）。"""
    return await _conversation_response(payload, db)


@router.post("/stream")
async def chat_stream(payload: ChatRequest, db: Session = Depends(get_db)) -> StreamingResponse:
    """对话主入口（SSE 流式，前端默认用这个）。"""
    return StreamingResponse(_conversation_sse(payload, db), media_type="text/event-stream")


@router.post("/recommendations", response_model=PromptRecommendationResponse)
async def prompt_recommendations(payload: PromptRecommendationRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
    """推荐下一步该问什么（输入框上方的建议气泡）。"""
    repo = _repo_or_404(db, payload.repo_id)
    issues, prs, runs = _repo_payload(repo.id, db)
    result = await RecommendationAgent().run(
        _recommendation_payload(repo, issues, prs, runs, payload.exclude, payload.limit),
        {"repo_id": str(repo.id)},
    )
    return {
        "agent": result["agent"],
        "suggestions": result["suggestions"],
    }
