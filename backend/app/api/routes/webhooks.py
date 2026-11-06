"""
GitHub Webhook 路由
===================

【路由清单】
    POST /github    GitHub 事件推送入口（issues / pull_request / workflow_run / push / check_*）

【数据流向】
    GitHub → 本文件 github_webhook()
      → verify_signature()      验签（HMAC-SHA256）
      → find_repo_by_full_name() 找到本地仓库
      → 后台任务 sync_repository_by_id()  → 和手动刷新走同一个同步核心

【要点】
    1. 这是【第四个同步入口】（另外三个：连接仓库 / 手动刷新 / 定时任务）。
       四个入口都收口到 services/repos_sync.py，规则只维护一份。
    2. 必须"先返回 200，再后台同步"：GitHub 只等 10 秒，超时就标记投递失败并重试，
       而一次完整同步可能几十秒。
    3. 未配置 webhook_secret 时跳过验签（本地开发方便，生产必须配）。
"""

import hashlib
import hmac

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Repository
from app.db.session import SessionLocal
from app.schemas.repos import RepoSyncRequest
from app.services.repos_sync import sync_repository_by_id

router = APIRouter()


def verify_signature(body: bytes, signature: str | None) -> None:
    """校验 GitHub 的 X-Hub-Signature-256。

    没配 secret 就直接放行 —— 本地没配时也能测通。
    生产环境必须配 GITHUB_WEBHOOK_SECRET，否则任何人都能伪造推送触发同步。
    """
    if not settings.github_webhook_secret:
        return
    if not signature or not signature.startswith("sha256="):
        raise HTTPException(status_code=401, detail="缺少 GitHub webhook 签名")
    expected = hmac.new(settings.github_webhook_secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    actual = signature.removeprefix("sha256=")
    # 用 compare_digest 而不是 == ：防时序攻击（逐字节比较会泄漏"前几位对了"）
    if not hmac.compare_digest(expected, actual):
        raise HTTPException(status_code=401, detail="GitHub webhook 签名无效")


def find_repo_by_full_name(full_name: str) -> Repository | None:
    """按 owner/repo 查本地仓库。

    这里自己开 Session：webhook 是外部调用，没有登录用户，
    用不了 get_db 依赖注入那套。
    """
    db: Session = SessionLocal()
    try:
        return db.query(Repository).filter(Repository.full_name == full_name).one_or_none()
    finally:
        db.close()


@router.post("/github")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_github_event: str | None = Header(default=None),
    x_hub_signature_256: str | None = Header(default=None),
) -> dict:
    """接收 GitHub 事件。永远返回 200（除非验签失败），让 GitHub 别重试。"""
    # 必须先读原始 body 再验签 —— 验签算的就是原始字节，不能先 json 解析
    body = await request.body()
    verify_signature(body, x_hub_signature_256)
    payload = await request.json()

    repository = payload.get("repository") or {}
    full_name = repository.get("full_name")
    if not full_name:
        # 不是仓库相关事件（比如 ping），照常 200
        return {"accepted": False, "reason": "payload.repository.full_name is missing"}

    repo = find_repo_by_full_name(full_name)
    if repo is None:
        # 没连接过的仓库：不报错，否则 GitHub 会一直重投
        return {"accepted": False, "reason": f"{full_name} is not connected"}

    # 只对这 6 类事件触发同步，其余（star / fork / release）忽略 —— 同步了也没数据变化
    if x_github_event in {"issues", "pull_request", "workflow_run", "push", "check_suite", "check_run"}:
        background_tasks.add_task(sync_repository_by_id, repo.id, RepoSyncRequest(limit=settings.auto_sync_limit))
        return {"accepted": True, "event": x_github_event, "repo_id": str(repo.id), "action": "sync_scheduled"}

    return {"accepted": True, "event": x_github_event, "repo_id": str(repo.id), "action": "ignored"}
