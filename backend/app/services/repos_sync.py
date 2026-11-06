"""
仓库同步核心
============

【业务定位】
    DevFlow AI 要回答"这个 PR 能不能合并""这个 Issue 该谁处理"，
    但这些判断都依赖 GitHub 上的实时事实。系统不能每次都实时查 GitHub
    （慢、有额度限制、断网就废），所以必须先把事实"搬"到本地。
    本文件就是这个【搬运工】：它是整个系统所有研发事实的唯一入口。

    业务上的一句话：把"散落在 GitHub 的协作信息"变成"本地可查询、可检索、
    可交叉分析的结构化数据"。

【数据流向】
    上游（谁把数据给它）：
        api/routes/repos.py     → 连接仓库 / 手动刷新
        api/routes/webhooks.py  → GitHub 事件推送
        services/scheduler.py   → 定时兜底（每 300 秒）
        ↓ 都汇聚到 sync_repository()
        services/github/*.py    → 实际发 HTTP 请求，返回 GitHub 原始 JSON

    下游（它把数据给谁）：
        PostgreSQL:
            repositories          ← 更新 last_sync_at
            issues                ← Issue 事实
            pull_requests         ← PR 本体
            pr_files              ← PR 改动的文件（整组替换）
            pr_review_comments    ← PR 行级评论（整组替换）
            workflow_runs         ← CI 运行 + jobs(JSON) + 失败日志
        ↓ 之后被这些模块消费
        api/routes/issues.py / pull_requests.py / ci.py  → 前端展示（只读）
        services/rag/indexing.py                          → 进向量库供 Agent 检索
        services/agents/*                                 → 作为 Agent 工具的数据源

【核心业务规则】
    1. 同一个 GitHub 对象同步多次，本地只留一行，内容以最新为准（幂等）。
    2. 判断"是不是同一个对象"必须用 `repo_id + github_issue_id` 双条件，
       只用其中一个会导致多仓库串线（两个仓库都有 #42）。
    3. 四类入口共用这一个函数 —— "入口可以增加，同步规则必须收口"。
    4. 本链路只负责"结构化事实入库"，不负责向量化（那是 RAG 的事）。

【技术要点】
    - Issue/PR/Run 本体：逐条 upsert
    - PR 的 files/comments：整组删除再插入（因为子资源会被删除，增量感知不到）
    - CI 日志：只对 conclusion=failure 的 Run 拉取（省流量省存储）

本文件【不负责】什么：
    - 不做 HTTP 请求（交给 services/github/）
    - 不做 token 加密（连接仓库时已加密，这里只解密）
    - 不做 RAG 向量化（那是下游 indexing.py 的事）
"""

from datetime import datetime, timezone
from uuid import UUID

import httpx
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import decrypt_token
from app.db.models import Issue, PRFile, PRReviewComment, PullRequest, Repository, WorkflowRun
from app.schemas.repos import RepoSyncRequest, RepoSyncResponse
from app.services.github.actions import get_workflow_run_logs, list_workflow_run_jobs, list_workflow_runs
from app.services.github.issues import list_issues
from app.services.github.pull_requests import list_pull_request_files, list_pull_request_review_comments, list_pull_requests


# ════════════════════════════════════════════════════════════════
# 两个小工具函数
# ════════════════════════════════════════════════════════════════

def parse_github_datetime(value: str | None) -> datetime | None:
    """把 GitHub 的 ISO 8601 时间字符串转成 Python datetime。

    输入样例："2026-10-18T05:30:00Z"

    为什么需要 .replace("Z", "+00:00")：
        GitHub 用 "Z" 表示 UTC（Zulu time），但 Python 的 fromisoformat()
        在旧版本里不认 "Z"，必须换成等价的 "+00:00" 时区偏移写法。

    为什么空值返回 None：
        不是所有字段都有值（比如未关闭的 Issue 没有 closed_at），
        对应数据库列是 nullable 的。
    """
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def login_of(user: dict | None) -> str | None:
    """从 GitHub 的嵌套 user 对象里取出登录名。

    GitHub 返回的作者信息长这样：{"login": "alice", "id": 123, ...}
    项目里只存 login（用户名），不存 id。

    为什么用 .get("login") 而不是 user["login"]：
        user 可能是 None（比如已注销账号），用 [] 会抛 TypeError。
    """
    return user.get("login") if user else None


# ════════════════════════════════════════════════════════════════
# 核心同步函数
# ════════════════════════════════════════════════════════════════

async def sync_repository(db: Session, repo: Repository, payload: RepoSyncRequest) -> RepoSyncResponse:
    """同步一个仓库的三类资源（Issue / PR / CI），并写库。

    参数：
        db      → 数据库会话（调用方负责提交/关闭）
        repo    → 已存在的仓库记录（必须有 repo.id，作为所有子表的外键）
        payload → 同步参数：要同步哪几类、每类拉多少条

    返回：
        RepoSyncResponse，其中 synced 记录"本次处理了多少个远端对象"
        （注意：是处理数量，不是新增行数）。

    异常：
        仓库类型不支持 → 400
        GitHub 接口报错（Issue/PR） → 400，并把 GitHub 的原始错误带出来
        CI 接口报错 → 静默降级（继续同步，不中断）
    """

    # ── 前置校验：只支持 GitHub 系 ──
    if repo.provider not in {"github", "github_compatible"}:
        raise HTTPException(status_code=400, detail=f"{repo.provider} sync is not implemented yet")

    # 计数器：返回给前端显示"这次同步了多少条"
    synced = {"issues": 0, "pull_requests": 0, "workflow_runs": 0}

    # ── 取 token：仓库级优先，没有则匿名 ──
    # 为什么是仓库级优先：
    #   不同仓库可能属于不同组织，需要不同权限。
    #   github_token_encrypted 是连接仓库时用 Fernet 加密存进去的，这里解密。
    # 为什么可能是 None：
    #   公开仓库允许匿名读（额度 60 次/小时），不配 token 也能跑演示。
    token = decrypt_token(repo.github_token_encrypted, settings.token_encryption_key) if repo.github_token_encrypted else None

    # 仓库可能指定了企业版 API 地址（github_compatible）
    base_url = repo.api_base_url

    # ════════════════════════════════════════════════════════════
    # 第一部分：Issue
    # ════════════════════════════════════════════════════════════
    if payload.sync_issues:
        try:
            remote_issues = await list_issues(repo.owner, repo.name, token=token, limit=payload.limit, base_url=base_url)
        except httpx.HTTPStatusError as exc:
            # 这里【不】静默降级：Issue 是核心数据，拉不到应该让用户知道原因
            raise HTTPException(status_code=400, detail=f"GitHub issues sync failed: {exc.response.status_code} {exc.response.text}") from exc

        for item in remote_issues:
            # ── 【业务】幂等判断：这个 Issue 之前同步过吗？ ──
            # 业务规则：同一个 GitHub 对象同步 N 次，本地只能有 1 行。
            # 两个条件缺一不可：
            #   repo_id          → 锁定是哪个仓库（业务上的"仓库隔离"，否则两个仓库的 #42 会串线）
            #   github_issue_id  → 锁定是哪个 GitHub 对象（不是用户看到的 number）
            # 用 GitHub 原始 ID 而不是 number，是因为 number 可能因仓库迁移而变化，
            # 而 github_issue_id 是永不变的——这是"同一对象"的唯一可靠判据。
            issue = db.query(Issue).filter(Issue.repo_id == repo.id, Issue.github_issue_id == item.get("id")).one_or_none()

            if issue is None:
                # 首次见到 → 新建一条（只填必填字段，其余下面统一覆盖）
                issue = Issue(repo_id=repo.id, github_issue_id=item.get("id"), number=item.get("number", 0), title=item.get("title") or "")
                db.add(issue)

            # ── 字段映射：外部 JSON → 内部字段 ──
            # 这里采用"无条件覆盖"策略：不管新建还是更新，所有字段都按最新值写一遍。
            # 好处是逻辑简单、不会漏字段；代价是即使值没变也会产生 UPDATE 语句。
            issue.number = item.get("number", issue.number)          # 用户看到的 #42
            issue.title = item.get("title") or issue.title           # 用 or 兜底：空标题不覆盖旧值
            issue.body = item.get("body")
            issue.state = item.get("state") or "open"                # open / closed
            # labels 是嵌套结构 [{"name":"bug"}]，要拍平成 ["bug"]
            issue.labels = [label.get("name") for label in item.get("labels", []) if label.get("name")]
            issue.author = login_of(item.get("user"))                # 嵌套对象取 login
            issue.assignees = [login_of(user) for user in item.get("assignees", []) if login_of(user)]
            issue.created_at = parse_github_datetime(item.get("created_at"))
            issue.updated_at = parse_github_datetime(item.get("updated_at"))
            issue.closed_at = parse_github_datetime(item.get("closed_at"))
            synced["issues"] += 1

    # ════════════════════════════════════════════════════════════
    # 第二部分：Pull Request（本体 + 两个子资源）
    # ════════════════════════════════════════════════════════════
    if payload.sync_pull_requests:
        try:
            remote_prs = await list_pull_requests(repo.owner, repo.name, token=token, limit=payload.limit, base_url=base_url)
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=400, detail=f"GitHub pull requests sync failed: {exc.response.status_code} {exc.response.text}") from exc

        for item in remote_prs:
            # PR 本体也走"按 GitHub ID 幂等更新"
            pr = db.query(PullRequest).filter(PullRequest.repo_id == repo.id, PullRequest.github_pr_id == item.get("id")).one_or_none()
            if pr is None:
                pr = PullRequest(repo_id=repo.id, github_pr_id=item.get("id"), number=item.get("number", 0), title=item.get("title") or "")
                db.add(pr)
                # ⚠️ flush 是必须的：下面要拿 pr.id 作为 files/comments 的外键，
                # 但新建对象的 id 要 flush 之后才由数据库生成。
                db.flush()

            pr.number = item.get("number", pr.number)
            pr.title = item.get("title") or pr.title
            pr.body = item.get("body")
            pr.state = item.get("state") or "open"
            pr.author = login_of(item.get("user"))
            # base/head 是嵌套对象 {"ref": "main"}，用 (item.get("base") or {}) 防止 None 报错
            pr.base_branch = (item.get("base") or {}).get("ref")
            pr.head_branch = (item.get("head") or {}).get("ref")
            pr.merged_at = parse_github_datetime(item.get("merged_at"))
            pr.created_at = parse_github_datetime(item.get("created_at"))
            pr.updated_at = parse_github_datetime(item.get("updated_at"))

            # ── 【业务】子资源 1：PR 改动的文件（整组替换）──
            # 业务规则：files 是"当前 PR 的完整改动快照"，不是增量日志。
            # 为什么是"先全删再全插"而不是逐条更新：
            #   PR 可能 rebase，导致某些文件从"改动列表"里消失。
            #   增量更新无法感知"消失"，旧文件会永远留在库里，
            #   最终导致 PR 审查 Agent 看到根本不存在的改动。
            # 代价：如果这次拉取失败（files=[]），就会把原有数据删光，留下空快照。
            db.query(PRFile).filter(PRFile.pr_id == pr.id).delete()
            try:
                files = await list_pull_request_files(repo.owner, repo.name, pr.number, token=token, base_url=base_url)
            except httpx.HTTPStatusError:
                # 子资源失败不阻断主流程，退化成空列表
                files = []
            for file_item in files:
                db.add(
                    PRFile(
                        pr_id=pr.id,
                        filename=file_item.get("filename") or "",
                        status=file_item.get("status") or "modified",   # added / modified / removed / renamed
                        additions=file_item.get("additions") or 0,
                        deletions=file_item.get("deletions") or 0,
                        patch=file_item.get("patch"),                    # 具体 diff 文本（大文件可能为 None）
                    )
                )

            # ── 子资源 2：PR 的行级 Review 评论（同样整组替换）──
            # 注意：这里拿的是"针对某一行 diff 的评论"，
            # 不包含普通 Issue 评论，也不包含 Review 的整体结论。
            db.query(PRReviewComment).filter(PRReviewComment.pr_id == pr.id).delete()
            try:
                review_comments = await list_pull_request_review_comments(repo.owner, repo.name, pr.number, token=token, base_url=base_url)
            except httpx.HTTPStatusError:
                review_comments = []
            for comment_item in review_comments:
                db.add(
                    PRReviewComment(
                        pr_id=pr.id,
                        github_comment_id=comment_item.get("id"),
                        body=comment_item.get("body"),
                        path=comment_item.get("path"),                  # 评论所在文件
                        # 行号：新版 API 用 line，旧版/已过期的评论只有 original_line
                        line=comment_item.get("line") or comment_item.get("original_line"),
                        author=login_of(comment_item.get("user")),
                        created_at=parse_github_datetime(comment_item.get("created_at")),
                        updated_at=parse_github_datetime(comment_item.get("updated_at")),
                    )
                )
            synced["pull_requests"] += 1

    # ════════════════════════════════════════════════════════════
    # 第三部分：CI（Workflow Run + Jobs + 失败日志）
    # ════════════════════════════════════════════════════════════
    if payload.sync_workflow_runs:
        try:
            remote_runs = await list_workflow_runs(repo.owner, repo.name, token=token, limit=payload.limit, base_url=base_url)
        except httpx.HTTPStatusError:
            # CI 数据是"锦上添花"，拉不到不阻断整个同步
            remote_runs = []

        for item in remote_runs:
            run = db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo.id, WorkflowRun.github_run_id == item.get("id")).one_or_none()
            if run is None:
                run = WorkflowRun(repo_id=repo.id, github_run_id=item.get("id"), name=item.get("name") or "workflow")
                db.add(run)

            run.name = item.get("name") or run.name
            run.status = item.get("status") or "unknown"        # queued / in_progress / completed
            run.conclusion = item.get("conclusion")              # success / failure / cancelled / None
            run.html_url = item.get("html_url")
            run.created_at = parse_github_datetime(item.get("created_at"))
            run.updated_at = parse_github_datetime(item.get("updated_at"))

            # ── Jobs 与 Steps：整体存成一个 JSON 字段 ──
            # 为什么用 JSON 而不是单独建 job / step 两张表：
            #   jobs 只在"展示 CI 详情"时整体读取，不需要按 job 字段查询，
            #   建两张表会增加大量写入却没有查询收益。
            if run.github_run_id:
                try:
                    jobs = await list_workflow_run_jobs(repo.owner, repo.name, run.github_run_id, token=token, base_url=base_url)
                    run.jobs = [
                        {
                            "id": job.get("id"),
                            "name": job.get("name"),
                            "status": job.get("status"),
                            "conclusion": job.get("conclusion"),
                            "html_url": job.get("html_url"),
                            "started_at": job.get("started_at"),
                            "completed_at": job.get("completed_at"),
                            # 嵌套的 steps 也一并整理成精简结构
                            "steps": [
                                {
                                    "name": step.get("name"),
                                    "status": step.get("status"),
                                    "conclusion": step.get("conclusion"),
                                    "number": step.get("number"),
                                }
                                for step in job.get("steps", [])
                            ],
                        }
                        for job in jobs
                    ]
                except httpx.HTTPStatusError:
                    # 拉不到 jobs 就保留旧值（注意：是保留，不是清空）
                    run.jobs = run.jobs or []

            # ── 失败日志：只为失败的 Run 拉取 ──
            # 为什么只拉失败的：
            #   日志包可能几 MB，成功的 Run 拉下来毫无用处，纯浪费流量和存储。
            if run.conclusion == "failure" and run.github_run_id:
                try:
                    run.logs_text = await get_workflow_run_logs(repo.owner, repo.name, run.github_run_id, token=token, base_url=base_url)
                except httpx.HTTPStatusError:
                    # 拉日志失败不影响主体同步，保留原有日志
                    run.logs_text = run.logs_text
            synced["workflow_runs"] += 1

    # ════════════════════════════════════════════════════════════
    # 收尾：更新时间戳 + 提交
    # ════════════════════════════════════════════════════════════
    # last_sync_at 是"基础同步状态"的唯一依据：
    #   /sync-status 接口靠它判断是 completed 还是 not_started。
    # ⚠️ 它只表示"主流程曾经跑到提交点"，不代表每类资源都成功。
    repo.last_sync_at = datetime.now(timezone.utc)

    # flush：把待写入的变更推给数据库（生成 SQL），但不提交事务
    db.flush()
    # commit：真正提交。这一行是整条同步链路的"原子提交点"——
    # 要么全部写入成功，要么（中途抛异常时）全部回滚。
    db.commit()

    return RepoSyncResponse(repo_id=repo.id, status="completed", synced=synced)


# ════════════════════════════════════════════════════════════════
# 后台任务用的包装函数
# ════════════════════════════════════════════════════════════════

async def sync_repository_by_id(repo_id: UUID, payload: RepoSyncRequest) -> RepoSyncResponse | None:
    """按 repo_id 同步——供【请求之外】的场景使用（定时任务、Webhook 后台任务）。

    为什么需要这个包装（而不是直接用 sync_repository）：
        定时任务和 Webhook 已经离开了 FastAPI 请求上下文，
        拿不到 Depends(get_db) 提供的那个 Session
        （请求一结束它就被关闭了，再用会报 "Session is closed"）。

        所以这里自己 new 一个 Session，用完自己关。

    参数：
        repo_id → 仓库主键（而不是 Repository 对象，因为跨请求传对象不安全）
        payload → 同步参数

    返回：
        同步结果；仓库不存在时返回 None（可能已被删除）。
    """
    # 局部 import：避免模块级循环导入
    # （db.session 会在初始化时读 settings，而本模块也在被 main.py 早期导入）
    from app.db.session import SessionLocal

    db = SessionLocal()
    try:
        repo = db.get(Repository, repo_id)
        if repo is None:
            # 仓库可能已被用户删除，静默返回即可
            return None
        # 真正的业务逻辑仍然复用同一个核心函数 —— 这就是"收口"的价值
        return await sync_repository(db, repo, payload)
    finally:
        # 无论成功失败都关闭，防止连接泄漏
        db.close()
