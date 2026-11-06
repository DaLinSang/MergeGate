"""
动作草稿路由 —— 【AI 写 GitHub 的唯一合法入口】
==============================================

【路由清单】
    POST /action-drafts                    创建草稿（AI 或人提交，状态 pending_confirmation）
    GET  /action-drafts                    列草稿（可按 repo / status 过滤）
    GET  /action-drafts/audit-logs         查审计日志（需 settings:manage）
    GET  /action-drafts/{draft_id}         查单个草稿
    POST /action-drafts/{draft_id}/cancel  取消草稿（只能取消待确认的）
    POST /action-drafts/{draft_id}/confirm 确认执行 → 真的写 GitHub

【数据流向】
    写操作全链路（数据流 7）：
      AI 想改 GitHub
        → services/agents/safety_agent.py 生成草稿
        → POST /action-drafts（本文件）→ action_drafts 表（pending_confirmation）
        → 前端 Drafts 页展示，人工点"确认"
        → POST /{draft_id}/confirm（本文件）
        → _execute_github_draft()
        → services/github/client.py → GitHub API
        → 回写 status=executed + execution_result + 审计日志

【要点】
    ★ 这是全项目最重要的安全闸门：没有人工点"确认"，AI 永远改不了 GitHub。
    1. 状态机单向：pending_confirmation → executed / failed / cancelled，不可逆。
    2. 权限分两级：draft:create 能建/取消，draft:approve 才能确认执行。
       这样"能提草稿的人"和"能批准的人"可以分开。
    3. 执行失败不抛 500：把 status 记成 failed + error_message，让前端能显示原因，
       同时写一条 failed 审计。抛异常的话审计就丢了。
    4. 成功和失败都写审计 —— 审计的目的是"可追溯"，失败记录同样有价值。
"""

from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import decrypt_token
from app.db.models import ActionDraft, AuditLog, Issue, PullRequest, Repository, User
from app.db.session import get_db
from app.schemas.action_drafts import ActionDraftConfirmResponse, ActionDraftCreateRequest, ActionDraftResponse, AuditLogResponse
from app.services.github.client import GitHubClient
from app.services.permissions import require_permission, write_audit_log

router = APIRouter()


def _target_number(db: Session, draft: ActionDraft) -> int | None:
    """找出这条草稿要操作的 issue/PR 编号。

    两种来源，按可靠性排序：
      1. meta 里直接存的 target_number（AI 生成草稿时写入，最直接）
      2. target_id（内部 UUID）→ 反查 Issue / PullRequest 拿 number
    返回 None 表示定位不到，调用方要报错（否则会往错误的对象上写东西）。
    """
    meta = draft.meta or {}
    raw = meta.get("target_number") or meta.get("issue_number") or meta.get("pr_number")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None
    if draft.target_id and draft.target_type == "issue":
        issue = db.get(Issue, draft.target_id)
        return issue.number if issue else None
    if draft.target_id and draft.target_type in {"pull_request", "pr"}:
        pr = db.get(PullRequest, draft.target_id)
        return pr.number if pr else None
    return None


async def _execute_github_draft(db: Session, draft: ActionDraft, repo: Repository) -> dict:
    """把草稿真正写进 GitHub。按 draft_type 分发到不同 API。

    这里是【唯一】会写 GitHub 的地方。任何新增的写操作都必须在这里加分支，
    并且必须先经过 action_drafts 的状态机 —— 不允许绕过。
    """
    # token 在连接仓库时加密存库，这里解密。解不出来就是 None（会被 GitHub 拒）
    token = decrypt_token(repo.github_token_encrypted, settings.token_encryption_key) if repo.github_token_encrypted else None
    client = GitHubClient(token=token, base_url=repo.api_base_url)
    draft_type = draft.draft_type.lower()
    issue_number = _target_number(db, draft)

    if draft_type in {"issue_comment", "pr_comment", "comment"}:
        if issue_number is None:
            raise RuntimeError("评论草稿缺少 issue/pr number")
        return await client.create_issue_comment(repo.owner, repo.name, issue_number, draft.content)

    if draft_type == "create_issue":
        # 标题兜底：用正文第一行截 120 字。AI 有时只给正文不给标题
        title = draft.title or (draft.content.splitlines()[0][:120] if draft.content else "DevFlow AI Issue")
        return await client.create_issue(repo.owner, repo.name, title, draft.content)

    if draft_type == "close_issue":
        if issue_number is None:
            raise RuntimeError("关闭 Issue 草稿缺少 issue number")
        return await client.close_issue(repo.owner, repo.name, issue_number)

    if draft_type in {"label", "add_label", "add_labels"}:
        if issue_number is None:
            raise RuntimeError("Label 草稿缺少 issue/pr number")
        # labels 优先从 meta 取；没有就把 title 当成单个标签（历史写法兼容）
        labels = draft.meta.get("labels") if draft.meta else None
        if not isinstance(labels, list) or not labels:
            labels = [draft.title] if draft.title else []
        if not labels:
            raise RuntimeError("Label 草稿缺少 labels")
        return await client.add_issue_labels(repo.owner, repo.name, issue_number, [str(item) for item in labels])

    if draft_type == "send_report":
        # 报告目前只在站内标记"已发送"，没接外部渠道（飞书/邮件）
        return {"status": "recorded", "message": "报告草稿已在 DevFlow AI 中标记为已发送，未接入外部消息渠道。"}

    raise RuntimeError(f"暂不支持执行草稿类型：{draft.draft_type}")


@router.post("", response_model=ActionDraftResponse)
async def create_action_draft(
    payload: ActionDraftCreateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("draft:create")),
) -> ActionDraft:
    """创建草稿。注意：这里【不会】碰 GitHub，只是落一条待确认记录。"""
    repo = db.get(Repository, payload.repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    draft = ActionDraft(
        repo_id=payload.repo_id,
        draft_type=payload.draft_type,
        target_type=payload.target_type,
        target_id=payload.target_id,
        title=payload.title,
        content=payload.content,
        risk_level=payload.risk_level,
        created_by=user.id,
        meta=payload.metadata,
    )
    db.add(draft)
    # 建草稿也要审计：谁在什么时候想做什么，是事后追责的依据
    write_audit_log(
        db,
        user=user,
        repo_id=payload.repo_id,
        action="action_draft.create",
        target_type=payload.target_type,
        target_id=str(payload.target_id) if payload.target_id else None,
        request_json=payload.model_dump(mode="json"),
    )
    db.commit()
    db.refresh(draft)
    return draft


@router.get("", response_model=list[ActionDraftResponse])
async def list_action_drafts(
    repo_id: UUID | None = Query(default=None),
    status: str | None = Query(default=None),
    db: Session = Depends(get_db),
    _user: User = Depends(require_permission("repo:read")),
) -> list[ActionDraft]:
    """列草稿。前端 Drafts 页默认按 status=pending_confirmation 过滤。"""
    query = db.query(ActionDraft)
    if repo_id:
        query = query.filter(ActionDraft.repo_id == repo_id)
    if status:
        query = query.filter(ActionDraft.status == status)
    return query.order_by(ActionDraft.created_at.desc()).limit(100).all()


@router.get("/audit-logs", response_model=list[AuditLogResponse])
async def list_audit_logs(
    repo_id: UUID | None = Query(default=None),
    db: Session = Depends(get_db),
    _user: User = Depends(require_permission("settings:manage")),
) -> list[AuditLog]:
    """查审计日志。要 settings:manage 权限 —— 审计记录里有敏感操作细节。"""
    query = db.query(AuditLog)
    if repo_id:
        query = query.filter(AuditLog.repo_id == repo_id)
    return query.order_by(AuditLog.created_at.desc()).limit(100).all()


@router.get("/{draft_id}", response_model=ActionDraftResponse)
async def get_action_draft(
    draft_id: UUID,
    db: Session = Depends(get_db),
    _user: User = Depends(require_permission("repo:read")),
) -> ActionDraft:
    """查单个草稿详情（前端确认弹窗用）。"""
    draft = db.get(ActionDraft, draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail="未找到安全草稿")
    return draft


@router.post("/{draft_id}/cancel", response_model=ActionDraftResponse)
async def cancel_action_draft(
    draft_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("draft:create")),
) -> ActionDraft:
    """取消草稿。只有 pending_confirmation 能取消（已执行的无法撤销）。"""
    draft = db.get(ActionDraft, draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail="未找到安全草稿")
    # 状态机保护：executed / failed 都不可回退
    if draft.status != "pending_confirmation":
        raise HTTPException(status_code=400, detail="只有待确认草稿可以取消")
    draft.status = "cancelled"
    write_audit_log(db, user=user, repo_id=draft.repo_id, action="action_draft.cancel", target_type=draft.target_type, target_id=str(draft.id))
    db.commit()
    db.refresh(draft)
    return draft


@router.post("/{draft_id}/confirm", response_model=ActionDraftConfirmResponse)
async def confirm_action_draft(
    draft_id: UUID,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("draft:approve")),
) -> dict:
    """★ 人工确认 → 真的执行。这是 AI 改 GitHub 的最后一道闸门。

    成功和失败都：
      - 更新草稿状态（executed / failed）
      - 记 confirmed_by 和 executed_at（谁批的、什么时候）
      - 写一条审计（success / failed）
    """
    draft = db.get(ActionDraft, draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail="未找到安全草稿")
    if draft.status != "pending_confirmation":
        raise HTTPException(status_code=400, detail="只有待确认草稿可以执行")
    repo = db.get(Repository, draft.repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")

    try:
        result = await _execute_github_draft(db, draft, repo)
        draft.status = "executed"
        draft.confirmed_by = user.id
        draft.executed_at = datetime.now(timezone.utc)
        draft.execution_result = result
        audit = write_audit_log(
            db,
            user=user,
            repo_id=draft.repo_id,
            action=f"action_draft.confirm.{draft.draft_type}",
            target_type=draft.target_type,
            target_id=str(draft.target_id or draft.id),
            status="success",
            request_json={"draft_id": str(draft.id), "draft_type": draft.draft_type},
            result_json=result if isinstance(result, dict) else {"result": result},
        )
    except Exception as exc:
        # 【关键】不 raise：把失败落成状态 + 审计，前端才能显示"为什么失败"
        draft.status = "failed"
        draft.confirmed_by = user.id
        draft.executed_at = datetime.now(timezone.utc)
        draft.error_message = str(exc)
        audit = write_audit_log(
            db,
            user=user,
            repo_id=draft.repo_id,
            action=f"action_draft.confirm.{draft.draft_type}",
            target_type=draft.target_type,
            target_id=str(draft.target_id or draft.id),
            status="failed",
            request_json={"draft_id": str(draft.id), "draft_type": draft.draft_type},
            result_json={"error": str(exc)},
        )
    db.commit()
    db.refresh(draft)
    db.refresh(audit)
    return {"draft": draft, "audit_id": audit.id}
