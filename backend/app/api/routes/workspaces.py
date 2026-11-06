"""
Workspace（多仓库集合）路由
==========================

【路由清单】
    POST /workspaces                    新建 workspace（把多个仓库归成一组）
    GET  /workspaces                    列出全部 workspace
    POST /workspaces/multi-repo-report  生成多仓库汇总周报（并逐仓库入库）

【数据流向】
    POST /multi-repo-report
      前端"多仓库周报"
        → _repo_ids_from_payload()  合并显式 repo_ids + workspace 里的 repo_ids 并去重
        → 逐仓库查 issues / pull_requests / workflow_runs
        → 纯模板拼 Markdown（不调 LLM）
        → 对**每个仓库**都 add_document(source_type="weekly_report")
        → write_audit_log() + commit

【要点】
    1. 权限分级：建 workspace 要 settings:manage，看要 repo:read，跑多仓报告要 agent:run。
    2. 多仓周报是"同一份报告写进每个仓库的 RAG 空间" —— 因为检索是按 repo_id 隔离的，
       不这么写的话，在仓库 A 里问"整体情况"检索不到。
    3. add_document(commit=False) 是为了和审计日志共用一次 commit（一个事务）。
    4. 时间过滤仍在 Python 里做，理由同 reports.py。
"""

from datetime import datetime, time, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models import Issue, PullRequest, Repository, User, WorkflowRun, Workspace
from app.db.session import get_db
from app.schemas.workspaces import MultiRepoReportRequest, MultiRepoReportResponse, WorkspaceCreateRequest, WorkspaceResponse
from app.services.permissions import require_permission, write_audit_log
from app.services.rag.vector_store import add_document

router = APIRouter()


def _bounds(payload: MultiRepoReportRequest) -> tuple[datetime, datetime]:
    """日期区间 → 带时区的闭区间。"""
    return (
        datetime.combine(payload.start_date, time.min, tzinfo=timezone.utc),
        datetime.combine(payload.end_date, time.max, tzinfo=timezone.utc),
    )


def _normalize(value: datetime | None) -> datetime | None:
    """补齐 UTC 时区，避免 naive 与 aware 比较报 TypeError。"""
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _in_range(value: datetime | None, start_at: datetime, end_at: datetime) -> bool:
    normalized = _normalize(value)
    return normalized is not None and start_at <= normalized <= end_at


def _repo_ids_from_payload(db: Session, payload: MultiRepoReportRequest) -> list[UUID]:
    """合并两种来源的仓库 id 并保序去重。

    来源 A：payload.repo_ids（前端勾选的）
    来源 B：workspace.repo_ids（预设分组里的）
    两者可能重叠，所以要去重；用 set 记录已见 + list 保序。
    """
    repo_ids = list(payload.repo_ids)
    if payload.workspace_id:
        workspace = db.get(Workspace, payload.workspace_id)
        if workspace is None:
            raise HTTPException(status_code=404, detail="未找到 workspace")
        repo_ids.extend(UUID(str(item)) for item in (workspace.repo_ids or []))
    seen: set[UUID] = set()
    output = []
    for repo_id in repo_ids:
        if repo_id not in seen:
            seen.add(repo_id)
            output.append(repo_id)
    return output


@router.post("", response_model=WorkspaceResponse)
async def create_workspace(
    payload: WorkspaceCreateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("settings:manage")),
) -> Workspace:
    """新建 workspace。会先校验所有 repo_id 都真实存在。"""
    # 一次查出存在的仓库，再和请求里的比 —— 比逐个 get() 少查 N 次
    known_repo_ids = {repo.id for repo in db.query(Repository).filter(Repository.id.in_(payload.repo_ids)).all()} if payload.repo_ids else set()
    missing = [str(repo_id) for repo_id in payload.repo_ids if repo_id not in known_repo_ids]
    if missing:
        # 把缺的一次性全列出来，方便前端一次性提示
        raise HTTPException(status_code=404, detail=f"仓库不存在：{', '.join(missing)}")
    workspace = Workspace(
        name=payload.name,
        description=payload.description,
        repo_ids=[str(repo_id) for repo_id in payload.repo_ids],
        created_by=user.id,
    )
    db.add(workspace)
    # 审计日志和业务写入同一个事务：要么都成功，要么都不留痕
    write_audit_log(db, user=user, action="workspace.create", request_json=payload.model_dump(mode="json"))
    db.commit()
    db.refresh(workspace)
    return workspace


@router.get("", response_model=list[WorkspaceResponse])
async def list_workspaces(
    db: Session = Depends(get_db),
    _user: User = Depends(require_permission("repo:read")),
) -> list[Workspace]:
    """列出全部 workspace，最近更新的在前。"""
    return db.query(Workspace).order_by(Workspace.updated_at.desc()).all()


@router.post("/multi-repo-report", response_model=MultiRepoReportResponse)
async def multi_repo_report(
    payload: MultiRepoReportRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("agent:run")),
) -> dict:
    """多仓库汇总周报：模板拼 Markdown + 逐仓库入库 + 审计。"""
    repo_ids = _repo_ids_from_payload(db, payload)
    if not repo_ids:
        raise HTTPException(status_code=400, detail="至少选择一个仓库")
    start_at, end_at = _bounds(payload)
    repos = db.query(Repository).filter(Repository.id.in_(repo_ids)).all()

    summaries = []
    totals = {"open_issues": 0, "open_prs": 0, "merged_prs": 0, "failed_ci": 0, "repos": len(repos)}

    # ── 逐仓库统计 ──
    for repo in repos:
        issues = db.query(Issue).filter(Issue.repo_id == repo.id).all()
        prs = db.query(PullRequest).filter(PullRequest.repo_id == repo.id).all()
        runs = db.query(WorkflowRun).filter(WorkflowRun.repo_id == repo.id).all()

        # 区间内活跃的（任一动作落在区间）
        ranged_issues = [item for item in issues if any(_in_range(value, start_at, end_at) for value in [item.created_at, item.updated_at, item.closed_at])]
        ranged_prs = [item for item in prs if any(_in_range(value, start_at, end_at) for value in [item.created_at, item.updated_at, item.merged_at])]
        ranged_runs = [item for item in runs if any(_in_range(value, start_at, end_at) for value in [item.created_at, item.updated_at])]

        # 注意：open_* 用的是全量（不是区间内）—— "当前还开着多少"是瞬时状态，不是区间统计
        open_issues = len([item for item in issues if item.state == "open"])
        open_prs = len([item for item in prs if item.state == "open"])
        merged_prs = len([item for item in ranged_prs if item.merged_at])
        failed_ci = len([item for item in ranged_runs if item.conclusion == "failure"])

        totals["open_issues"] += open_issues
        totals["open_prs"] += open_prs
        totals["merged_prs"] += merged_prs
        totals["failed_ci"] += failed_ci

        summaries.append(
            {
                "repo_id": str(repo.id),
                "repo_name": repo.full_name,
                "range_issues": len(ranged_issues),
                "range_prs": len(ranged_prs),
                "open_issues": open_issues,
                "open_prs": open_prs,
                "merged_prs": merged_prs,
                "failed_ci": failed_ci,
                # 三级风险：有失败 CI 或 PR 积压 >5 → high；issue >5 → medium；否则 low
                "risk_level": "high" if failed_ci or open_prs > 5 else "medium" if open_issues > 5 else "low",
            }
        )

    # ── 拼 Markdown（纯模板，不调 LLM）──
    risk_repos = [item for item in summaries if item["risk_level"] == "high"]
    lines = [
        f"# 多仓库研发周报 {payload.start_date} - {payload.end_date}",
        "",
        f"- 覆盖仓库：{len(repos)} 个",
        f"- Open Issues：{totals['open_issues']}",
        f"- Open PRs：{totals['open_prs']}",
        f"- 本周期 merged PRs：{totals['merged_prs']}",
        f"- 失败 CI：{totals['failed_ci']}",
        "",
        "## 仓库明细",
    ]
    for item in summaries:
        lines.append(
            f"- {item['repo_name']}：open issue {item['open_issues']}，open PR {item['open_prs']}，merged {item['merged_prs']}，失败 CI {item['failed_ci']}，风险 {item['risk_level']}"
        )
    lines.extend(["", "## 重点关注", *(f"- {item['repo_name']} 需要优先处理失败 CI 或 PR backlog。" for item in risk_repos)])
    if not risk_repos:
        lines.append("- 暂无高风险仓库，建议继续关注 PR 停留时间和评审积压。")
    report_markdown = "\n".join(lines) + "\n"

    # ── 同一份报告写进每个仓库的 RAG 空间 ──
    # 检索按 repo_id 隔离，不逐仓库写的话，在单个仓库里问"整体情况"检索不到
    for repo in repos:
        add_document(
            db,
            repo.id,
            "weekly_report",
            None,
            f"多仓库研发周报 {payload.start_date} - {payload.end_date}",
            report_markdown,
            {
                "display_title": f"多仓库研发周报 {payload.start_date} - {payload.end_date}",
                "start_date": str(payload.start_date),
                "end_date": str(payload.end_date),
                "workspace_id": str(payload.workspace_id) if payload.workspace_id else None,
                "repo_count": len(repos),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            # commit=False：下面和审计日志一起提交，保证"入库 + 审计"原子
            commit=False,
        )
    write_audit_log(db, user=user, action="workspace.multi_repo_report", request_json=payload.model_dump(mode="json"), result_json=totals)
    db.commit()
    return {"report_markdown": report_markdown, "metrics": totals, "repo_summaries": summaries}
