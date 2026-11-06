"""
仓库路由
========

【路由清单】
    POST   /repos/connect              连接仓库（三种方式：本地路径 / 远程 clone / demo）
    GET    /repos                      列已连接仓库
    POST   /repos/{repo_id}/sync       手动同步（拉 Issue/PR/CI）
    POST   /repos/{repo_id}/select     标记"当前选中"（靠刷新 updated_at 实现）
    DELETE /repos/{repo_id}            删除仓库（级联清 20+ 张表 + Milvus + 磁盘文件）
    GET    /repos/{repo_id}/sync-status 同步状态

【数据流向】
    connect（系统的最上游，数据流 1 的起点）
      前端"连接仓库"
        → 三种模式判定
            ① local_path  → _inspect_local_repository()  读 .git 拿 remote/origin
            ② clone_parent_dir → _clone_target_from_parent()  定下载目录
            ③ owner/repo  → 直接用
        → GitHubClient.get_repo() 验证仓库存在（无 token 时可降级 demo）
        → repositories 表（token 加密后存 github_token_encrypted）
        → sync_repository()  同步 Issue/PR/CI（同步执行）
        → 后台 sync_repository_code_analysis_by_id()  克隆 + 建代码图谱

【要点】
    1. 连接是 upsert：按 full_name 查，存在就更新（重新连接会覆盖 token / 配置）。
    2. token 一定加密存（encrypt_token），从不落明文、也从不回传给前端。
    3. 无 token 且 GitHub 验证失败时，如果全局开了 demo_mode，降级成"本地缓存模式" ——
       只建仓库记录，不报错。这样演示环境不配 token 也能跑通。
    4. 删除接口是全项目最重的操作：手写级联删 20+ 张表。
       顺序原则：先删子表（AnalysisResult / PRFile / AgentTaskRun），再删父表，
       最后删 Repository —— 否则外键约束会拦。
    5. 删库先 commit 再删 Milvus 和磁盘文件：向量库和文件系统不参与事务，
       失败也不该回滚数据库（否则会出现"库里有记录但向量没了"更糟的状态）。
"""

import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.db.models import ActionDraft, AgentRun, AgentTaskRun, AgentWorkflowRun, AnalysisResult, AuditLog, ChatFeedback, ChatMessage, ChatSession, CodeRelation, CodeSymbol, Conversation, ConversationMemory, Document, EvidenceItem, Issue, KnowledgeBaseConfig, KnowledgeGraphEdge, KnowledgeSourceDocument, MemoryCandidate, PRFile, PRReviewComment, ProjectIndexState, PullRequest, RecallEvent, Repository, RetrievalTestRun, ThreadMemory, WorkflowRun, Workspace
from app.db.session import get_db
from app.schemas.repos import RepoConnectRequest, RepoConnectResponse, RepoResponse, RepoSyncRequest, RepoSyncResponse
from app.core.config import settings
from app.core.security import encrypt_token
from app.services.github.client import GitHubClient
from app.services.code_analysis import sync_repository_code_analysis_by_id
from app.services.rag.vector_store import delete_milvus_documents
from app.services.repos_sync import sync_repository

router = APIRouter()

# 目前只实现了这两种 provider。gitlab / gitee 只是占位，连了会报错
SUPPORTED_CONNECT_PROVIDERS = {"github", "github_compatible"}
# 用来判断"是不是官方 GitHub"：不是的话自动降级成 github_compatible
GITHUB_HOSTS = {"github.com", "www.github.com"}


@dataclass(frozen=True)
class GitRemoteInfo:
    """从 git remote 解析出来的仓库信息。"""
    owner: str
    repo: str
    clone_url: str
    host: str | None = None
    default_branch: str | None = None


def _clean_text(value: str | None) -> str | None:
    """去空白；空串当 None 处理（前端传空字符串很常见）。"""
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _resolve_user_path(raw_path: str, field_name: str) -> Path:
    """把用户输入的路径变成绝对路径。

    三步：去掉引号（用户常复制带引号的路径）→ 展开 ~ 和 $VAR → 相对路径拼 cwd。
    """
    cleaned = raw_path.strip().strip('"').strip("'")
    if not cleaned:
        raise HTTPException(status_code=400, detail=f"{field_name} cannot be empty.")
    expanded = os.path.expandvars(os.path.expanduser(cleaned))
    path = Path(expanded)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _git_output(args: list[str], cwd: Path) -> str:
    """跑一条 git 命令并返回 stdout。所有失败都转成 400（用户输入问题）。"""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,  # 30 秒超时：防止 git 卡在网络请求上（比如私有仓库等密码）
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail="未找到 git 可执行文件，无法检查本地仓库。") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise HTTPException(status_code=400, detail=f"Failed to inspect local repository: {detail}") from exc
    return result.stdout.strip()


def _parse_git_remote_url(remote_url: str) -> GitRemoteInfo:
    """解析 git remote URL，支持三种写法：
        https://github.com/owner/repo.git
        git@github.com:owner/repo.git        （SCP 风格）
        ssh://git@host/owner/repo.git
    """
    value = remote_url.strip()
    host: str | None = None
    path = ""
    # SCP 风格：user@host:path（注意里面没有 "://"）
    scp_like = re.match(r"^(?:[^@]+@)?(?P<host>[^:]+):(?P<path>.+)$", value)

    if scp_like and "://" not in value:
        host = scp_like.group("host")
        path = scp_like.group("path")
    else:
        parsed = urlparse(value)
        host = parsed.hostname or parsed.netloc or None
        path = parsed.path

    parts = [item.strip() for item in path.replace("\\", "/").strip("/").split("/") if item.strip()]
    if len(parts) < 2:
        raise HTTPException(status_code=400, detail=f"Cannot parse owner/repository from git remote URL: {remote_url}")

    # 倒数第二段是 owner，最后一段是 repo（去掉 .git 后缀）
    owner = parts[-2]
    repo = re.sub(r"\.git$", "", parts[-1], flags=re.IGNORECASE)
    if not owner or not repo:
        raise HTTPException(status_code=400, detail=f"Cannot parse owner/repository from git remote URL: {remote_url}")
    return GitRemoteInfo(owner=owner, repo=repo, clone_url=value, host=host.lower() if host else None)


def _inspect_local_repository(raw_path: str) -> tuple[Path, GitRemoteInfo]:
    """检查本地路径是不是 git 仓库，并取出 remote origin 信息。"""
    path = _resolve_user_path(raw_path, "local_path")
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Local repository path does not exist or is not a directory: {path}")

    # 注意：要传子目录也能过，git 会向上找 .git
    inside_work_tree = _git_output(["rev-parse", "--is-inside-work-tree"], path).lower()
    if inside_work_tree != "true":
        raise HTTPException(status_code=400, detail=f"Local path is not inside a git repository: {path}")

    # show-toplevel 拿到仓库根目录（用户可能传的是子目录）
    repo_root = Path(_git_output(["rev-parse", "--show-toplevel"], path)).resolve()
    remote_url = _git_output(["remote", "get-url", "origin"], repo_root)
    remote_info = _parse_git_remote_url(remote_url)
    # 当前分支当作默认分支（本地仓库拿不到 GitHub 的 default_branch）
    current_branch = _git_output(["branch", "--show-current"], repo_root) or None
    return repo_root, GitRemoteInfo(
        owner=remote_info.owner,
        repo=remote_info.repo,
        clone_url=remote_info.clone_url,
        host=remote_info.host,
        default_branch=current_branch,
    )


def _infer_provider(requested_provider: str | None, remote_host: str | None) -> str:
    """推断 provider：用户说 github 但 host 不是 github.com → 当作 github_compatible。"""
    provider = requested_provider or "github"
    if remote_host and remote_host not in GITHUB_HOSTS and provider == "github":
        return "github_compatible"
    return provider


def _effective_api_base_url(provider: str, raw_base_url: str | None, remote_host: str | None) -> str | None:
    """确定 API base URL。

    github → 固定官方地址
    github_compatible → 用户填的 / 从 host 推导（GitHub Enterprise 是 https://host/api/v3）
    """
    if provider == "github":
        return "https://api.github.com"
    base_url = _clean_text(raw_base_url)
    if base_url:
        return base_url.rstrip("/")
    if remote_host and remote_host not in GITHUB_HOSTS:
        return f"https://{remote_host}/api/v3"
    return None


def _clone_target_from_parent(raw_parent: str, repo_name: str) -> Path:
    """算出 clone 的目标目录，并做三道安全检查。"""
    parent = _resolve_user_path(raw_parent, "clone_parent_dir")
    if parent.exists() and not parent.is_dir():
        raise HTTPException(status_code=400, detail=f"Download parent path is not a directory: {parent}")

    target = (parent / repo_name).resolve()
    if target.exists() and not target.is_dir():
        raise HTTPException(status_code=400, detail=f"Download target exists and is not a directory: {target}")
    # 目标存在、不是 git 仓库、而且里面有文件 → 拒绝（否则会覆盖用户的东西）
    if target.exists() and not (target / ".git").exists() and any(target.iterdir()):
        raise HTTPException(
            status_code=400,
            detail=f"Download target alreadyconnect exists and is not an empty git repository: {target}",
        )
    return target


@router.post("/connect", response_model=RepoConnectResponse)
async def connect_repo(payload: RepoConnectRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)) -> RepoConnectResponse:
    """连接仓库。这是整个系统的入口 —— 后面所有数据都从这一步开始流。"""
    local_path = _clean_text(payload.local_path)
    clone_parent_dir = _clean_text(payload.clone_parent_dir)
    owner = _clean_text(payload.owner)
    repo_name = _clean_text(payload.repo)
    requested_provider = payload.provider or "github"
    token = payload.token or payload.github_token
    checkout_path: Path | None = None
    checkout_mode = "managed"  # managed=我们管代码副本；local=用用户本地那份
    remote_info: GitRemoteInfo | None = None

    # ── 模式判定：本地路径优先，其次是 owner/repo ──
    if local_path:
        checkout_path, remote_info = _inspect_local_repository(local_path)
        # 本地模式以 remote 为准，忽略用户填的 owner/repo（避免不一致）
        owner = remote_info.owner
        repo_name = remote_info.repo
        checkout_mode = "local"
    elif not owner or not repo_name:
        raise HTTPException(status_code=400, detail="Provide either a local repository path or owner/repository.")

    provider = _infer_provider(requested_provider, remote_info.host if remote_info else None)
    base_url = _effective_api_base_url(provider, payload.api_base_url, remote_info.host if remote_info else None)
    if provider not in SUPPORTED_CONNECT_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"{provider} real connection is not implemented yet. Use demo mode or github_compatible with a GitHub-compatible REST API.",
        )

    # 远程模式 + 指定了下载目录 → 算出 clone 目标
    if not local_path and clone_parent_dir:
        checkout_path = _clone_target_from_parent(clone_parent_dir, repo_name or "repository")

    full_name = f"{owner}/{repo_name}"
    effective_demo_mode = payload.demo_mode
    fallback_message: str | None = None

    # demo 模式的兜底数据（不请求 GitHub 也能建出记录）
    repo_data = {
        "id": None,
        "description": "Demo repository for DevFlow AI",
        "default_branch": remote_info.default_branch if remote_info and remote_info.default_branch else "main",
        "clone_url": remote_info.clone_url if remote_info else f"https://github.com/{owner}/{repo_name}.git",
    }
    if not payload.demo_mode:
        try:
            # 真连一次 GitHub 验证仓库存在 + 拿元数据
            repo_data = await GitHubClient(token, base_url).get_repo(owner or "", repo_name or "")
        except Exception as exc:
            # 有 token 还失败 → 说明 token 错了或没权限，必须报错
            # 没 token 且全局开了 demo_mode → 降级成"本地缓存模式"
            if token or not settings.demo_mode:
                raise HTTPException(status_code=400, detail=f"Repository connection failed: {exc}") from exc
            effective_demo_mode = True
            fallback_message = (
                f"GitHub verification failed without a token: {exc}. "
                "Added the repository in local cache mode; add a token or retry later for full GitHub sync."
            )

    # ── upsert：按 full_name 查，有就更新 ──
    repo = db.query(Repository).filter(Repository.full_name == full_name).one_or_none()
    if repo is None:
        repo = Repository(owner=owner or "", name=repo_name or "", full_name=full_name)
        db.add(repo)
    repo.owner = owner or repo.owner
    repo.name = repo_name or repo.name
    repo.provider = provider
    repo.api_base_url = base_url
    if token:
        # 【安全】token 只以密文形式存库，且从不回传给前端
        repo.github_token_encrypted = encrypt_token(token, settings.token_encryption_key)
    repo.description = repo_data.get("description")
    repo.default_branch = repo_data.get("default_branch")
    repo.clone_url = repo_data.get("clone_url") or (remote_info.clone_url if remote_info else f"https://github.com/{owner}/{repo_name}.git")
    repo.local_path = str(checkout_path) if checkout_path else None
    repo.checkout_mode = checkout_mode
    repo.github_id = repo_data.get("id")
    db.commit()
    db.refresh(repo)

    # ── 首次同步 ──
    sync_result: RepoSyncResponse | None = None
    if not effective_demo_mode:
        # 同步执行（用户在等，且初次数据量有限），limit 由前端控制
        sync_result = await sync_repository(
            db,
            repo,
            RepoSyncRequest(
                sync_issues=payload.sync_issues,
                sync_pull_requests=payload.sync_pull_requests,
                sync_workflow_runs=payload.sync_workflow_runs,
                limit=payload.initial_sync_limit,
            ),
        )
        # 代码克隆 + 图谱分析很慢，扔后台
        background_tasks.add_task(sync_repository_code_analysis_by_id, repo.id)
    elif fallback_message:
        # 降级模式也试着建代码图谱（本地路径可能已经有代码）
        background_tasks.add_task(sync_repository_code_analysis_by_id, repo.id)

    return RepoConnectResponse(
        repo_id=repo.id,
        full_name=repo.full_name,
        provider=repo.provider,
        api_base_url=repo.api_base_url,
        clone_url=repo.clone_url,
        local_path=repo.local_path,
        checkout_mode=repo.checkout_mode,
        default_branch=repo.default_branch,
        connected=True,
        demo_mode=effective_demo_mode,
        message=(
            fallback_message
            or (
                "Demo mode only created a local repository record."
                if effective_demo_mode
                else "Repository connection verified. Initial Issue/PR/CI sync completed; code checkout and graph analysis started."
            )
        ),
        # 三种同步状态：成功 / 降级兜底 / 跳过
        sync_status=sync_result.status if sync_result else ("fallback" if fallback_message else "skipped"),
        synced=sync_result.synced if sync_result else {"issues": 0, "pull_requests": 0, "workflow_runs": 0},
    )


@router.get("", response_model=list[RepoResponse])
async def list_repos(db: Session = Depends(get_db)) -> list[Repository]:
    """列已连接仓库（排除 knowledge_base 虚拟仓库）。"""
    return (
        db.query(Repository)
        .filter(Repository.provider != "knowledge_base")
        .order_by(Repository.updated_at.desc())
        .all()
    )


@router.post("/{repo_id}/sync", response_model=RepoSyncResponse)
async def sync_repo(
    repo_id: UUID,
    payload: RepoSyncRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> RepoSyncResponse:
    """手动刷新：同步 Issue/PR/CI，同时后台重建代码图谱。"""
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    # 和连接仓库、定时任务、webhook 走的是同一个 sync_repository —— 规则只维护一份
    result = await sync_repository(db, repo, payload)
    background_tasks.add_task(sync_repository_code_analysis_by_id, repo.id)
    return result


@router.post("/{repo_id}/select")
async def select_repo(repo_id: UUID, db: Session = Depends(get_db)) -> dict:
    """标记"当前选中仓库"。

    实现很取巧：没有 selected 字段，直接刷 updated_at ——
    前端列表按 updated_at 倒序，所以被选的会跳到第一个。
    """
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    repo.updated_at = datetime.now(timezone.utc)
    db.commit()
    return {"repo_id": str(repo.id), "full_name": repo.full_name, "selected": True}


@router.delete("/{repo_id}")
async def delete_repo(repo_id: UUID, db: Session = Depends(get_db)) -> dict:
    """★ 删除仓库 —— 全项目最重的操作，级联清 20+ 张表。

    顺序原则：先删"引用别人的子表"，再删"被引用的父表"，最后删 Repository。
    另外先把要删的 id 收集起来 —— 父表删完就查不到子表 id 了。
    """
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")

    # ── 第一步：收集所有下游 id（必须在删任何东西之前做）──
    issue_ids = [item.id for item in db.query(Issue.id).filter(Issue.repo_id == repo.id).all()]
    pr_ids = [item.id for item in db.query(PullRequest.id).filter(PullRequest.repo_id == repo.id).all()]
    run_ids = [item.id for item in db.query(WorkflowRun.id).filter(WorkflowRun.repo_id == repo.id).all()]
    conversation_ids = [item.id for item in db.query(Conversation.id).filter(Conversation.repo_id == repo.id).all()]
    session_ids = [
        item.id
        for item in db.query(ChatSession.id)
        .filter(
            or_(
                ChatSession.repo_id == repo.id,
                # 条件式 in_：列表为空时传 False，避免 "IN ()" 语法错误
                ChatSession.conversation_id.in_(conversation_ids) if conversation_ids else False,
            )
        )
        .all()
    ]
    workflow_ids = [
        item.id
        for item in db.query(AgentWorkflowRun.id)
        .filter(
            or_(
                AgentWorkflowRun.repo_id == repo.id,
                AgentWorkflowRun.conversation_id.in_(conversation_ids) if conversation_ids else False,
            )
        )
        .all()
    ]

    # ── 第二步：删分析结果（按 target_type + target_id 关联，没有 repo_id 字段）──
    for target_type, ids in [
        ("issue", issue_ids),
        ("pull_request", pr_ids),
        ("workflow_run", run_ids),
    ]:
        if ids:
            db.query(AnalysisResult).filter(
                AnalysisResult.target_type == target_type,
                AnalysisResult.target_id.in_(ids),
            ).delete(synchronize_session=False)

    # 先记下磁盘文件路径，等数据库提交后再删文件
    knowledge_sources = (
        db.query(KnowledgeSourceDocument)
        .filter(KnowledgeSourceDocument.repo_id == repo.id)
        .all()
    )
    stored_paths = [Path(source.stored_path) for source in knowledge_sources]

    # ── 第三步：删 PR 子资源（PRFile / PRReviewComment 依赖 pr_ids）──
    if pr_ids:
        db.query(PRFile).filter(PRFile.pr_id.in_(pr_ids)).delete(synchronize_session=False)
        db.query(PRReviewComment).filter(PRReviewComment.pr_id.in_(pr_ids)).delete(synchronize_session=False)

    # ── 第四步：删所有带 repo_id 的直属表 ──
    db.query(Document).filter(Document.repo_id == repo.id).delete(synchronize_session=False)
    db.query(RetrievalTestRun).filter(RetrievalTestRun.repo_id == repo.id).delete(synchronize_session=False)
    db.query(KnowledgeSourceDocument).filter(KnowledgeSourceDocument.repo_id == repo.id).delete(synchronize_session=False)
    db.query(KnowledgeBaseConfig).filter(KnowledgeBaseConfig.repo_id == repo.id).delete(synchronize_session=False)
    db.query(KnowledgeGraphEdge).filter(KnowledgeGraphEdge.repo_id == repo.id).delete(synchronize_session=False)
    db.query(ProjectIndexState).filter(ProjectIndexState.repo_id == repo.id).delete(synchronize_session=False)
    db.query(CodeRelation).filter(CodeRelation.repo_id == repo.id).delete(synchronize_session=False)
    db.query(CodeSymbol).filter(CodeSymbol.repo_id == repo.id).delete(synchronize_session=False)
    db.query(ActionDraft).filter(ActionDraft.repo_id == repo.id).delete(synchronize_session=False)
    db.query(AuditLog).filter(AuditLog.repo_id == repo.id).delete(synchronize_session=False)
    db.query(ChatFeedback).filter(ChatFeedback.repo_id == repo.id).delete(synchronize_session=False)

    # ── 第五步：删"可能只挂在会话/会话记录下"的表（三个条件任一命中）──
    db.query(MemoryCandidate).filter(
        or_(
            MemoryCandidate.repo_id == repo.id,
            MemoryCandidate.conversation_id.in_(conversation_ids) if conversation_ids else False,
            MemoryCandidate.session_id.in_(session_ids) if session_ids else False,
        )
    ).delete(synchronize_session=False)
    db.query(RecallEvent).filter(
        or_(
            RecallEvent.repo_id == repo.id,
            RecallEvent.conversation_id.in_(conversation_ids) if conversation_ids else False,
        )
    ).delete(synchronize_session=False)

    # ── 第六步：删工作流 → 再删任务（任务依赖 workflow_id）──
    if workflow_ids:
        db.query(AgentTaskRun).filter(AgentTaskRun.workflow_id.in_(workflow_ids)).delete(synchronize_session=False)
    db.query(AgentWorkflowRun).filter(
        or_(
            AgentWorkflowRun.repo_id == repo.id,
            AgentWorkflowRun.conversation_id.in_(conversation_ids) if conversation_ids else False,
        )
    ).delete(synchronize_session=False)

    # ── 第七步：删消息 → 会话记录 → 记忆 → 证据 → 运行记录 → 会话 ──
    db.query(ChatMessage).filter(
        or_(
            ChatMessage.repo_id == repo.id,
            ChatMessage.conversation_id.in_(conversation_ids) if conversation_ids else False,
        )
    ).delete(synchronize_session=False)
    db.query(ChatSession).filter(
        or_(
            ChatSession.repo_id == repo.id,
            ChatSession.conversation_id.in_(conversation_ids) if conversation_ids else False,
        )
    ).delete(synchronize_session=False)
    db.query(ConversationMemory).filter(
        or_(
            ConversationMemory.repo_id == repo.id,
            ConversationMemory.conversation_id.in_(conversation_ids) if conversation_ids else False,
        )
    ).delete(synchronize_session=False)
    db.query(ThreadMemory).filter(ThreadMemory.repo_id == repo.id).delete(synchronize_session=False)
    db.query(EvidenceItem).filter(
        or_(
            EvidenceItem.repo_id == repo.id,
            EvidenceItem.conversation_id.in_(conversation_ids) if conversation_ids else False,
        )
    ).delete(synchronize_session=False)
    db.query(AgentRun).filter(
        or_(
            AgentRun.repo_id == repo.id,
            AgentRun.conversation_id.in_(conversation_ids) if conversation_ids else False,
        )
    ).delete(synchronize_session=False)
    db.query(Conversation).filter(Conversation.repo_id == repo.id).delete(synchronize_session=False)

    # ── 第八步：删 GitHub 同步来的三类事实 ──
    db.query(Issue).filter(Issue.repo_id == repo.id).delete(synchronize_session=False)
    db.query(PullRequest).filter(PullRequest.repo_id == repo.id).delete(synchronize_session=False)
    db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo.id).delete(synchronize_session=False)

    # ── 第九步：从所有 workspace 里摘掉这个 repo_id（workspace 本身保留）──
    for workspace in db.query(Workspace).all():
        next_repo_ids = [item for item in (workspace.repo_ids or []) if str(item) != str(repo.id)]
        if next_repo_ids != (workspace.repo_ids or []):
            workspace.repo_ids = next_repo_ids

    db.delete(repo)
    db.commit()

    # ── 第十步：数据库提交后再清外部资源 ──
    # 顺序很重要：如果先删 Milvus 再 commit 失败，会出现"记录还在但向量没了"
    # strict=False：Milvus 挂了也不该让删除接口失败
    milvus_deleted = delete_milvus_documents(repo_id=repo.id, strict=False)
    removed_files = 0
    file_cleanup_errors: list[str] = []
    for path in stored_paths:
        try:
            if path.exists() and path.is_file():
                path.unlink()
                removed_files += 1
        except OSError as exc:
            # 单个文件删不掉不影响整体，收集起来在响应里报告
            file_cleanup_errors.append(f"{path}: {exc}")

    return {
        "repo_id": str(repo_id),
        "deleted": True,
        "cleanup": {
            "milvus_deleted": milvus_deleted,
            "removed_files": removed_files,
            "file_errors": file_cleanup_errors,
        },
    }


@router.get("/{repo_id}/sync-status")
async def sync_status(repo_id: UUID, db: Session = Depends(get_db)) -> dict:
    """同步状态。只看 last_sync_at 有没有值 —— 不区分成功/失败（失败会留 last_sync_error）。"""
    repo = db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    return {
        "repo_id": str(repo.id),
        "status": "completed" if repo.last_sync_at else "not_started",
        "last_sync_at": repo.last_sync_at,
        "message": "最近一次同步已成功完成。" if repo.last_sync_at else "仓库还没有同步过。",
    }
