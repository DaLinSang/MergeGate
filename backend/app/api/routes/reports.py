"""
报告路由
========

【路由清单】
    POST /weekly       生成研发周报（并自动存进 RAG，可被后续对话检索到）
    POST /repo-health  仓库健康度速览（demo 级指标，不调 LLM）

【数据流向】
    POST /weekly
      前端"生成周报"
        → 本文件 weekly_report()
        → 查 issues / pull_requests / workflow_runs（PostgreSQL）
        → services/agents/report_agent.py::ReportAgent.run()（纯模板，不调 LLM）
        → services/rag/vector_store.py::add_document(source_type="weekly_report")
        → 同时写 Milvus + PostgreSQL

    POST /repo-health
      → 三条 count 查询 → 直接返回（无 LLM、无落库）

【要点】
    1. 周报生成后【立刻向量化入库】，所以之后问"上周做了什么"能被 RAG 检索到 ——
       这是"沉淀"这一步的典型实现。
    2. 时间过滤在 Python 里做（不在 SQL 里）：因为判断条件是"created_at / updated_at /
       closed_at 任一落在区间内"，SQL 里写要三个 OR 加时区处理，不如内存过滤清楚。
       数据量是单仓库级别，可以接受。
"""

from datetime import date, datetime, time, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models import Issue, PullRequest, Repository, WorkflowRun
from app.db.session import get_db
from app.schemas.reports import WeeklyReportRequest, WeeklyReportResponse
from app.services.agents.report_agent import ReportAgent
from app.services.rag.vector_store import add_document

router = APIRouter()


def _range_bounds(start_date: date, end_date: date) -> tuple[datetime, datetime]:
    """把日期区间变成带时区的闭区间 [当天 00:00:00, 当天 23:59:59.999999]。"""
    return (
        datetime.combine(start_date, time.min, tzinfo=timezone.utc),
        datetime.combine(end_date, time.max, tzinfo=timezone.utc),
    )


def _normalize_dt(value: datetime | None) -> datetime | None:
    """统一成带时区的 datetime。

    GitHub 同步来的时间可能是不带时区的（naive），
    和带时区的 start_at 比较会抛 TypeError，所以先补齐 UTC。
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _in_range(value: datetime | None, start_at: datetime, end_at: datetime) -> bool:
    normalized = _normalize_dt(value)
    return normalized is not None and start_at <= normalized <= end_at


def _issue_in_range(issue: Issue, start_at: datetime, end_at: datetime) -> bool:
    """Issue 只要"创建 / 更新 / 关闭"任一动作发生在区间内，就算本周活跃。"""
    return any(_in_range(value, start_at, end_at) for value in [issue.created_at, issue.updated_at, issue.closed_at])


def _pr_in_range(pr: PullRequest, start_at: datetime, end_at: datetime) -> bool:
    """PR 同理，看"创建 / 更新 / 合入"。"""
    return any(_in_range(value, start_at, end_at) for value in [pr.created_at, pr.updated_at, pr.merged_at])


@router.post("/weekly", response_model=WeeklyReportResponse)
async def weekly_report(payload: WeeklyReportRequest, db: Session = Depends(get_db)) -> dict:
    """生成周报：取数 → 模板渲染 → 入库（可被 RAG 检索）。"""
    repo = db.get(Repository, payload.repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")

    start_at, end_at = _range_bounds(payload.start_date, payload.end_date)

    # 取全量再内存过滤（理由见文件头要点 2）
    issues = [
        item
        for item in db.query(Issue).filter(Issue.repo_id == payload.repo_id).all()
        if _issue_in_range(item, start_at, end_at)
    ]
    prs = [
        item
        for item in db.query(PullRequest).filter(PullRequest.repo_id == payload.repo_id).all()
        if _pr_in_range(item, start_at, end_at)
    ]
    runs = [
        item
        for item in db.query(WorkflowRun).filter(WorkflowRun.repo_id == payload.repo_id).all()
        if _in_range(item.created_at, start_at, end_at) or _in_range(item.updated_at, start_at, end_at)
    ]

    # 派生指标：这些数字直接决定周报里"合并了多少 / 失败多少次"
    merged_prs = [item for item in prs if item.merged_at]
    open_prs = [item for item in prs if item.state == "open"]
    failed_ci = [item for item in runs if item.conclusion == "failure"]

    result = await ReportAgent().run(
        {
            "issues": [{"number": item.number, "title": item.title, "state": item.state, "closed_at": item.closed_at} for item in issues],
            "pull_requests": [
                {"number": item.number, "title": item.title, "state": item.state, "merged_at": item.merged_at}
                for item in prs
            ],
            "workflow_runs": [{"name": item.name, "conclusion": item.conclusion} for item in runs],
            "metrics": {
                "issues": len(issues),
                "pull_requests": len(prs),
                "merged_prs": len(merged_prs),
                "open_prs": len(open_prs),
                "failed_ci": len(failed_ci),
            },
        },
        {
            "repo_id": str(payload.repo_id),
            "repo_name": repo.full_name,
            "start_date": str(payload.start_date),
            "end_date": str(payload.end_date),
        },
    )

    # 【关键】周报生成后立刻入库，否则"上周做了什么"这类问题检索不到
    add_document(
        db,
        payload.repo_id,
        "weekly_report",
        None,  # source_id 为 None：周报是"凭空生成"的，没有对应的 GitHub 对象
        f"研发周报 {payload.start_date} - {payload.end_date}",
        result["report_markdown"],
        {
            "display_title": f"研发周报 {payload.start_date} - {payload.end_date}",
            "start_date": str(payload.start_date),
            "end_date": str(payload.end_date),
            "repo_name": repo.full_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return result


@router.post("/repo-health")
async def repo_health(payload: WeeklyReportRequest, db: Session = Depends(get_db)) -> dict:
    """仓库健康度速览：三个 count + 一句固定建议。

    注意 merged_prs_7d 恒为 0 —— 这是 demo 占位，还没实现真正的 7 日窗口统计。
    """
    open_issues = db.query(Issue).filter(Issue.repo_id == payload.repo_id, Issue.state == "open").count()
    open_prs = db.query(PullRequest).filter(PullRequest.repo_id == payload.repo_id, PullRequest.state == "open").count()
    failed_ci = db.query(WorkflowRun).filter(WorkflowRun.repo_id == payload.repo_id, WorkflowRun.conclusion == "failure").count()
    return {
        "summary": "仓库健康度以 demo 指标估算，建议结合近期失败 CI 和高风险 PR 判断。",
        "metrics": {"open_issues": open_issues, "open_prs": open_prs, "failed_ci_runs": failed_ci, "merged_prs_7d": 0},
        "risks": ["存在失败 CI，需要排查"] if failed_ci else [],
    }
