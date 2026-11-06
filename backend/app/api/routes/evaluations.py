"""
评测路由
========

【路由清单】（8 个）
    GET  /evaluations/suites                    列可用评测集（含 ready 状态）
    POST /evaluations/run                       跑基础评测（同步，快）
    POST /evaluations/rag/run                   跑 RAGAS 评测（后台任务，慢）
    POST /evaluations/{eval_id}/rescore         只重跑"失败的 Judge 指标"（复用原现场）
    GET  /evaluations/compare                   两次评测逐题对比（回归门禁）
    POST /evaluations/{eval_id}/issue-draft     把评测失败整理成 Issue 草稿
    GET  /evaluations                           列历史评测
    GET  /evaluations/{eval_id}                 查单次评测详情

【数据流向】
    POST /rag/run
      前端"跑评测"
        → 三种用例来源判定
            ① payload.cases   → 自定义用例（算 sha256 记版本）
            ② payload.suite_id → 从仓库文档加载固定题库
            ③ 都没有           → AUTO_SMOKE（用当前文档临时出题）
        → eval_runs(result_json=queued)  立即返回 eval_id
        → 后台 _execute_rag_eval_job()
            → services/evals/ragas_runner.run_rag_quality_eval()
            → 每阶段 persist_progress() 回写 result_json（前端轮询看进度）

    GET /compare（发布门禁）
      → _eval_comparison()  比 4 项一致性 + 指标 delta + 逐题 improved/regressed

【要点】
    1. 【为什么异步】一次 RAGAS 评测要跑几十道题、每题调 2~3 次 LLM，
       耗时几分钟到几十分钟，绝不能阻塞 HTTP 请求。所以返回 queued + 后台跑。
    2. 【进度靠 result_json】没有单独的进度表，直接把整个 job 状态写进
       eval_runs.result_json —— 前端轮询 GET /{eval_id} 就能看到 stage/message。
    3. 【后台必须自己开 Session】请求的 db 在响应返回后就关闭了，
       所以 _execute_rag_eval_job 里用 SessionLocal() 新建。
    4. 【rescore 的设计意图】Judge 调用可能偶发失败，整轮重跑太贵。
       rescore 复用已存的回答和检索现场，只重算失败的指标。
    5. 【compare 的 comparable】评测集 hash / ragas 版本 / judge 模型 / top_k
       任一不同就不可比 —— 直接比分数会得出错误结论。
    6. 【issue-draft 也走草稿】评测失败要提 Issue，同样不能直接写 GitHub，
       必须走 ActionDraft 人工确认。
"""

import asyncio
import hashlib
import json
import logging
from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.models import ActionDraft, EvalRun, Repository, User
from app.db.session import SessionLocal, get_db
from app.schemas.evals import EvalIssueDraftRequest, EvalRunRequest, EvalRunResponse
from app.services.evals.ragas_runner import (
    REQUIRED_RAGAS_METRICS,
    RagEvalCase,
    rescore_rag_result,
    run_rag_quality_eval,
)
from app.services.evals.suites import AUTO_SMOKE_SUITE_ID, list_eval_suites, load_eval_suite
from app.services.evals.runner import run_basic_eval
from app.services.permissions import require_permission, write_audit_log

router = APIRouter()
logger = logging.getLogger(__name__)


def _request_case(item) -> RagEvalCase:
    """请求里的用例 → 内部 RagEvalCase。逐字段搬运，没有转换逻辑。"""
    return RagEvalCase(
        id=item.id,
        question=item.question,
        reference=item.reference,
        expected_source_ids=item.expected_source_ids,
        source_type=item.source_type,
        required_tools=item.required_tools,
        expected_tool_calls=item.expected_tool_calls,
        kind=item.kind,
        must_include=item.must_include,
        must_not_include=item.must_not_include,
        answer_regex=item.answer_regex,
        source_file=item.source_file,
    )


def _custom_evaluation_set(payload: EvalRunRequest) -> dict:
    """给自定义用例算一个内容指纹。

    为什么需要 sha256：两次评测只有题库一致才可比。
    排序键 + 紧凑分隔符，保证同样的用例永远算出同样的 hash。
    """
    serialized = [item.model_dump(mode="json") for item in payload.cases or []]
    canonical = json.dumps(serialized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "id": payload.name or "custom-cases",
        "name": payload.name or "custom-cases",
        "label": payload.name or "自定义评测集",
        "description": "通过 API 提交的固定评测用例",
        "source": "api_cases",
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "case_count": len(serialized),
    }


@router.get("/suites")
async def get_eval_suites(
    repo_id: UUID = Query(...),
    db: Session = Depends(get_db),
    _user=Depends(require_permission("eval:read")),
) -> list[dict]:
    """列评测集。每项带 ready 标记（源文档是否已索引完）。"""
    try:
        return list_eval_suites(db, repo_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/run", response_model=EvalRunResponse)
async def run_eval(
    payload: EvalRunRequest,
    db: Session = Depends(get_db),
    _user=Depends(require_permission("eval:read")),
) -> EvalRunResponse:
    """基础评测（同步）。跑的是纯规则指标，不调 LLM，所以很快。"""
    result = await run_basic_eval(db, str(payload.repo_id) if payload.repo_id else None)
    eval_run = EvalRun(name=payload.name, result_json=result)
    db.add(eval_run)
    db.commit()
    db.refresh(eval_run)
    return EvalRunResponse(
        eval_id=str(eval_run.id),
        status="completed",
        result=result,
    )


@router.post("/rag/run", response_model=EvalRunResponse)
async def run_rag_eval(
    payload: EvalRunRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _user=Depends(require_permission("eval:read")),
) -> EvalRunResponse:
    """RAGAS 评测（异步）。立即返回 queued + eval_id，前端轮询进度。"""
    if payload.repo_id is None:
        raise HTTPException(status_code=400, detail="RAG Eval 需要 repo_id")
    if db.get(Repository, payload.repo_id) is None:
        raise HTTPException(status_code=404, detail="未找到仓库")
    # 两种来源二选一，同时传会让人分不清用哪个
    if payload.cases is not None and payload.suite_id:
        raise HTTPException(status_code=400, detail="cases 与 suite_id 不能同时提交")

    # ── 三种用例来源 ──
    if payload.cases is not None:
        cases = [_request_case(item) for item in payload.cases]
        evaluation_set = _custom_evaluation_set(payload)
    elif payload.suite_id:
        try:
            suite = load_eval_suite(db, payload.repo_id, payload.suite_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # AUTO_SMOKE 的用例是"运行时从文档临时生成"的，所以 cases 传 None 让 runner 自己出题
        cases = None if payload.suite_id == AUTO_SMOKE_SUITE_ID else suite.cases
        evaluation_set = suite.metadata
    else:
        # 什么都没传 → 自动冒烟：拿当前仓库文档现场出题
        cases = None
        evaluation_set = {
            "id": AUTO_SMOKE_SUITE_ID,
            "name": AUTO_SMOKE_SUITE_ID,
            "label": "自动文档冒烟测试",
            "description": "未指定固定评测集，使用当前仓库文档临时出题。",
            "source": "auto_smoke",
            "sha256": None,  # 临时出题，没有稳定指纹
            "case_count": None,
        }

    # 先落一条 queued 记录，前端立刻能拿到 eval_id 开始轮询
    result = _job_result(
        status="queued",
        stage="queued",
        message="评测任务已提交，正在等待后台执行。",
        repo_id=str(payload.repo_id),
        evaluation_set=evaluation_set,
    )
    eval_run = EvalRun(name=payload.name or "rag_eval", result_json=result)
    db.add(eval_run)
    db.commit()
    db.refresh(eval_run)
    background_tasks.add_task(
        _execute_rag_eval_job,
        str(eval_run.id),
        str(payload.repo_id),
        cases,
        payload.top_k,
        payload.run_judge,
        evaluation_set,
    )
    return EvalRunResponse(eval_id=str(eval_run.id), status="queued", result=result)


async def _execute_rag_eval_job(
    eval_id: str,
    repo_id: str,
    cases: list[RagEvalCase] | None,
    limit: int,
    run_judge: bool,
    evaluation_set: dict,
) -> None:
    """后台跑 RAGAS 评测，边跑边把进度写回 eval_runs.result_json。"""
    # 必须新建 Session：请求的 db 已经随响应关闭了
    db = SessionLocal()
    # Yield once after Starlette sends the queued response so the client receives
    # the Eval ID before imports, index loading, or model setup can block the loop.
    await asyncio.sleep(0)

    async def persist_progress(progress: dict) -> None:
        """进度回调：把 runner 报的 stage/message 写进 result_json。

        status 固定写成 running —— 只要还在回调，就说明没跑完。
        """
        eval_run = db.get(EvalRun, UUID(eval_id))
        if eval_run is None:
            return
        eval_run.result_json = _job_result(
            status="running",
            stage=str(progress.get("stage") or "running"),
            message=str(progress.get("message") or "评测正在后台运行。"),
            repo_id=repo_id,
            evaluation_set=evaluation_set,
            # 其余字段（进度计数、当前题号等）原样透传
            **{
                key: value
                for key, value in progress.items()
                if key not in {"stage", "message"}
            },
        )
        db.add(eval_run)
        db.commit()

    try:
        await persist_progress({"stage": "starting", "message": "后台任务已启动，正在加载仓库上下文。"})
        result = await run_rag_quality_eval(
            db,
            repo_id,
            cases=cases,
            limit=limit,
            run_judge=run_judge,
            progress_callback=persist_progress,
            evaluation_set=evaluation_set,
        )
        result["progress"] = {
            "stage": "completed",
            "message": "评测已完成。",
            "updated_at": datetime.now(UTC).isoformat(),
        }
    except Exception as exc:
        # 后台任务失败必须落库，否则前端会一直显示 running
        logger.exception("RAGAS background evaluation failed: eval_id=%s", eval_id)
        result = _job_result(
            status="error",
            stage="error",
            message=f"评测运行失败：{exc}",
            repo_id=repo_id,
            evaluation_set=evaluation_set,
        )
        # 补上最小结构，让前端不会因为缺字段崩掉
        result.update(
            {
                "passed": False,
                "ragas": {
                    "requested": run_judge,
                    "available": False,
                    "passed": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "metrics": {},
                },
                "cases": [],
            }
        )

    try:
        eval_run = db.get(EvalRun, UUID(eval_id))
        if eval_run is not None:
            eval_run.result_json = result
            db.add(eval_run)
            db.commit()
    finally:
        db.close()


def _job_result(
    *,
    status: str,
    stage: str,
    message: str,
    repo_id: str | None = None,
    evaluation_set: dict | None = None,
    **progress: object,
) -> dict:
    """构造一条"任务状态"结构的 result_json。

    这是所有阶段（queued / running / error）的统一骨架，
    保证前端任何时候读到的字段都是齐的（不会因为阶段不同缺字段）。
    """
    return {
        "kind": "chatagent_ragas",
        "status": status,
        "execution_status": status,
        # 刚提交时质量状态未知，所以是 incomplete
        "quality_status": "incomplete",
        "evaluation_complete": False,
        "passed": False,
        "target": {"name": "ChatAgent", "repo_id": repo_id} if repo_id else {"name": "ChatAgent"},
        "evaluation_set": evaluation_set or {},
        "progress": {
            "stage": stage,
            "message": message,
            "updated_at": datetime.now(UTC).isoformat(),
            **progress,
        },
    }


@router.post("/{eval_id}/rescore", response_model=EvalRunResponse)
async def rescore_eval(
    eval_id: UUID,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _user=Depends(require_permission("eval:read")),
) -> EvalRunResponse:
    """只重跑失败的 Judge 指标（复用原评测的回答和检索现场）。

    触发条件：原评测里必须有"报错的指标"或"缺失的指标"，否则没意义。
    """
    source = db.get(EvalRun, eval_id)
    if source is None:
        raise HTTPException(status_code=404, detail="未找到 Eval run")
    source_result = source.result_json or {}
    if not source_result.get("cases"):
        raise HTTPException(status_code=400, detail="原评测没有可用于重评分的逐题现场")

    # 统计两类"需要重跑"的题：judge 报错的、指标缺失的
    error_count = sum(bool(item.get("ragas_errors")) for item in source_result.get("cases") or [])
    missing_count = sum(
        any(name not in (item.get("ragas") or {}) for name in ("context_precision", "context_recall", "faithfulness", "answer_relevancy", "agent_goal_accuracy"))
        for item in source_result.get("cases") or []
    )
    if error_count == 0 and missing_count == 0:
        raise HTTPException(status_code=400, detail="原评测没有失败或缺失的 Judge 指标")

    repo_id = str((source_result.get("target") or {}).get("repo_id") or "")
    evaluation_set = dict(source_result.get("evaluation_set") or {})
    # 新建一条评测记录（不覆盖原来的），用 rescore_of 关联
    queued = _job_result(
        status="queued",
        stage="queued",
        message=f"已提交重评分任务，将复用 Eval {str(eval_id)[:8]} 的 ChatAgent 现场。",
        repo_id=repo_id or None,
        evaluation_set=evaluation_set,
        rescore_of=str(eval_id),
    )
    target = EvalRun(name=f"rescore-{source.name}", result_json=queued)
    db.add(target)
    db.commit()
    db.refresh(target)
    background_tasks.add_task(
        _execute_rescore_job,
        str(target.id),
        str(source.id),
        source_result,
        repo_id,
        evaluation_set,
    )
    return EvalRunResponse(eval_id=str(target.id), status="queued", result=queued)


async def _execute_rescore_job(
    target_eval_id: str,
    source_eval_id: str,
    source_result: dict,
    repo_id: str,
    evaluation_set: dict,
) -> None:
    """后台重评分。流程和 _execute_rag_eval_job 一样，只是调 rescore_rag_result。"""
    db = SessionLocal()
    await asyncio.sleep(0)

    async def persist_progress(progress: dict) -> None:
        row = db.get(EvalRun, UUID(target_eval_id))
        if row is None:
            return
        row.result_json = _job_result(
            status="running",
            stage=str(progress.get("stage") or "rescore"),
            message=str(progress.get("message") or "正在重试异常评分。"),
            repo_id=repo_id or None,
            evaluation_set=evaluation_set,
            rescore_of=source_eval_id,
            **{key: value for key, value in progress.items() if key not in {"stage", "message"}},
        )
        db.add(row)
        db.commit()

    try:
        result = await rescore_rag_result(
            source_result,
            eval_id=source_eval_id,
            progress_callback=persist_progress,
        )
        result["progress"] = {
            "stage": "completed",
            "message": "异常指标重评分已完成；ChatAgent 回答和检索现场均未重新运行。",
            "updated_at": datetime.now(UTC).isoformat(),
        }
    except Exception as exc:
        logger.exception("RAGAS rescore failed: eval_id=%s", target_eval_id)
        result = _job_result(
            status="error",
            stage="error",
            message=f"重评分失败：{exc}",
            repo_id=repo_id or None,
            evaluation_set=evaluation_set,
            rescore_of=source_eval_id,
        )
        result["ragas"] = {
            "requested": True,
            "available": False,
            "complete": False,
            "passed": None,
            "error": f"{type(exc).__name__}: {exc}",
            "metrics": {},
        }
        # 失败时把原用例带回来，至少前端还能看到原始现场
        result["cases"] = source_result.get("cases") or []
    try:
        row = db.get(EvalRun, UUID(target_eval_id))
        if row is not None:
            row.result_json = result
            db.add(row)
            db.commit()
    finally:
        db.close()


def _eval_comparison(base: EvalRun, candidate: EvalRun) -> dict:
    """两次评测对比。这是"发布门禁"的核心算法。

    先判"可比性"（4 项一致性），不可比就直接标 comparable=False，
    因为拿不同题库/不同 judge 的分数对比会得出完全错误的结论。
    """
    base_result = base.result_json or {}
    candidate_result = candidate.result_json or {}
    base_set = base_result.get("evaluation_set") or {}
    candidate_set = candidate_result.get("evaluation_set") or {}

    # ── 四项一致性检查 ──
    reasons: list[str] = []
    if base_set.get("sha256") != candidate_set.get("sha256"):
        reasons.append("评测集 Hash 不一致")
    if (base_result.get("ragas") or {}).get("version") != (candidate_result.get("ragas") or {}).get("version"):
        reasons.append("Ragas 版本不一致")
    if (base_result.get("ragas") or {}).get("judge_model") != (candidate_result.get("ragas") or {}).get("judge_model"):
        reasons.append("Judge 模型不一致")
    if (base_result.get("pipeline") or {}).get("top_k") != (candidate_result.get("pipeline") or {}).get("top_k"):
        reasons.append("Top-K 不一致")

    # ── 聚合指标 delta ──
    base_metrics = base_result.get("summary_metrics") or {}
    candidate_metrics = candidate_result.get("summary_metrics") or {}
    metric_deltas = []
    for name in sorted(set(base_metrics) | set(candidate_metrics)):
        left = base_metrics.get(name)
        right = candidate_metrics.get(name)
        # 非数值跳过（比如 error 字符串）
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            continue
        metric_deltas.append(
            {"name": name, "base": float(left), "candidate": float(right), "delta": round(float(right) - float(left), 4)}
        )

    # ── 逐题 delta ──
    base_cases = {str(item.get("id")): item for item in base_result.get("cases") or []}
    candidate_cases = {str(item.get("id")): item for item in candidate_result.get("cases") or []}
    case_deltas = []
    for case_id in sorted(set(base_cases) | set(candidate_cases)):
        left = base_cases.get(case_id) or {}
        right = candidate_cases.get(case_id) or {}
        # 单题"通过"的定义：硬规则过 + 没有 judge 报错
        left_passed = bool((left.get("hard_rules") or {}).get("passed")) and not bool(left.get("ragas_errors"))
        right_passed = bool((right.get("hard_rules") or {}).get("passed")) and not bool(right.get("ragas_errors"))
        state = "unchanged"
        if not left_passed and right_passed:
            state = "improved"
        elif left_passed and not right_passed:
            state = "regressed"
        case_deltas.append(
            {"id": case_id, "base_passed": left_passed, "candidate_passed": right_passed, "state": state}
        )

    return {
        "base_eval_id": str(base.id),
        "candidate_eval_id": str(candidate.id),
        "comparable": not reasons,
        "incompatibility_reasons": reasons,
        "metric_deltas": metric_deltas,
        "case_deltas": case_deltas,
        "improved_cases": sum(item["state"] == "improved" for item in case_deltas),
        "regressed_cases": sum(item["state"] == "regressed" for item in case_deltas),
        "candidate_quality_status": candidate_result.get("quality_status"),
        "candidate_complete": bool(candidate_result.get("evaluation_complete")),
        # 门禁：可比 + 候选版本 passed，缺一不可
        "release_gate_passed": bool(not reasons and candidate_result.get("passed")),
    }


@router.get("/compare")
async def compare_evals(
    base_eval_id: UUID = Query(...),
    candidate_eval_id: UUID = Query(...),
    db: Session = Depends(get_db),
    _user=Depends(require_permission("eval:read")),
) -> dict:
    """对比两次评测（回归门禁用）。"""
    base = db.get(EvalRun, base_eval_id)
    candidate = db.get(EvalRun, candidate_eval_id)
    if base is None or candidate is None:
        raise HTTPException(status_code=404, detail="未找到用于对比的 Eval run")
    return _eval_comparison(base, candidate)


def _issue_markdown(eval_run: EvalRun) -> str:
    """把评测结果渲染成一份可复现的 Issue 正文。

    必须包含"复现环境"（eval id / 题库 hash / ragas 版本 / judge 模型）——
    没有这些，别人拿到 Issue 也复现不出来。
    """
    result = eval_run.result_json or {}
    suite = result.get("evaluation_set") or {}
    ragas = result.get("ragas") or {}
    lines = [
        "## 复现环境",
        "",
        f"- Eval ID：`{eval_run.id}`",
        f"- 评测集：`{suite.get('name') or '-'}`",
        f"- 评测集 Hash：`{suite.get('sha256') or '-'}`",
        f"- Ragas：`{ragas.get('version') or '-'}`",
        f"- Judge：`{ragas.get('judge_model') or '-'}`",
        f"- 运行状态：`{result.get('status') or '-'}`",
        f"- 质量状态：`{result.get('quality_status') or '-'}`",
        "",
        "## 聚合结果",
        "",
        "| 指标 | 分数 | 门槛 |",
        "| --- | ---: | ---: |",
    ]
    # 只列门禁指标（REQUIRED_RAGAS_METRICS），其他指标不进 Issue（噪音）
    for name, detail in (ragas.get("metrics") or {}).items():
        if name not in REQUIRED_RAGAS_METRICS:
            continue
        score = detail.get("score")
        threshold = detail.get("threshold")
        # score 为 None 说明这一项算失败了，明确写 ERROR 而不是留空
        lines.append(
            f"| {name} | {score if score is not None else 'ERROR'} | {threshold if threshold is not None else '-'} |"
        )
    lines.extend(["", "## 失败用例", ""])
    for case in result.get("cases") or []:
        # 只列"非 passed"的诊断
        diagnoses = [item for item in case.get("diagnoses") or [] if item.get("category") != "passed"]
        if not diagnoses and not case.get("ragas_errors"):
            continue
        lines.extend(
            [
                f"### {case.get('id') or 'unknown'}",
                "",
                f"- 问题：{case.get('question') or '-'}",
                f"- 标准答案：{case.get('reference') or '-'}",
                f"- 实际答案：{case.get('response') or '-'}",
            ]
        )
        if case.get("ragas_errors"):
            lines.append(f"- Judge 错误：`{json.dumps(case['ragas_errors'], ensure_ascii=False)}`")
        for diagnosis in diagnoses:
            lines.append(f"- {diagnosis.get('title')}：{diagnosis.get('action')}")
        lines.append("")
    lines.extend(
        [
            "## 验收标准",
            "",
            "- 使用同一评测集版本重新运行。",
            "- Judge 指标计算零错误。",
            "- 所有硬规则通过，所有门禁指标达到当前阈值。",
            "- 与本 Eval 做逐题对比，不新增回归用例。",
        ]
    )
    return "\n".join(lines)


@router.post("/{eval_id}/issue-draft")
async def create_eval_issue_draft(
    eval_id: UUID,
    payload: EvalIssueDraftRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("draft:create")),
) -> dict:
    """把评测失败整理成 Issue 草稿（仍需人工确认才真的建 Issue）。"""
    eval_run = db.get(EvalRun, eval_id)
    if eval_run is None:
        raise HTTPException(status_code=404, detail="未找到 Eval run")
    result = eval_run.result_json or {}
    repo_id = (result.get("target") or {}).get("repo_id")
    # 老评测可能没存 repo_id（早期版本），没法定位仓库就报错
    if not repo_id:
        raise HTTPException(status_code=400, detail="该历史评测没有保存仓库信息，无法创建 Issue 草稿")
    repo = db.get(Repository, UUID(str(repo_id)))
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到评测对应仓库")
    suite_label = (result.get("evaluation_set") or {}).get("label") or "RAGAS"
    title = payload.title or f"RAG 质量回归：{suite_label} 未通过"
    draft = ActionDraft(
        repo_id=repo.id,
        draft_type="create_issue",
        target_type="repository",
        title=title,
        content=_issue_markdown(eval_run),
        # medium：建 Issue 不改代码，风险不高
        risk_level="medium",
        created_by=user.id,
        meta={"eval_id": str(eval_run.id), "quality_status": result.get("quality_status")},
    )
    db.add(draft)
    write_audit_log(
        db,
        user=user,
        repo_id=repo.id,
        action="eval.issue_draft.create",
        target_type="eval_run",
        target_id=str(eval_run.id),
        request_json={"eval_id": str(eval_run.id), "title": title},
    )
    db.commit()
    db.refresh(draft)
    return {"draft_id": str(draft.id), "status": draft.status, "title": draft.title}


@router.get("")
async def list_evals(
    limit: int = 20,
    repo_id: UUID | None = None,
    db: Session = Depends(get_db),
    _user=Depends(require_permission("eval:read")),
) -> list[dict]:
    """列历史评测。

    注意：repo_id 过滤是在内存里做的（先取 100 条再筛）——
    因为 repo_id 存在 result_json 里，SQL 查不了。
    """
    rows = db.query(EvalRun).order_by(EvalRun.created_at.desc()).limit(100).all()
    if repo_id is not None:
        rows = [
            row
            for row in rows
            if str(((row.result_json or {}).get("target") or {}).get("repo_id") or "") == str(repo_id)
        ]
    # 双保险：limit 再夹一次（1~100）
    rows = rows[: max(1, min(limit, 100))]
    return [
        {
            "eval_id": str(row.id),
            "name": row.name,
            "status": str((row.result_json or {}).get("status") or "completed"),
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "metrics": _summary_metrics(row.result_json or {}),
            "result": row.result_json,
        }
        for row in rows
    ]


@router.get("/{eval_id}")
async def get_eval(
    eval_id: str,
    db: Session = Depends(get_db),
    _user=Depends(require_permission("eval:read")),
) -> dict:
    """查单次评测。前端轮询这个接口看进度。"""
    try:
        eval_uuid = UUID(eval_id)
    except ValueError as exc:
        # 这里用 str 接参数再手动转 UUID：为了给出 400 而不是 422
        raise HTTPException(status_code=400, detail="Eval id 无效") from exc
    eval_run = db.get(EvalRun, eval_uuid)
    if eval_run is None:
        raise HTTPException(status_code=404, detail="未找到 Eval run")
    result = eval_run.result_json or {}
    return {
        "eval_id": str(eval_run.id),
        "name": eval_run.name,
        "status": str(result.get("status") or "completed"),
        "result": result,
    }


def _summary_metrics(result: dict) -> dict[str, float]:
    """提取数值型指标。兼容两种结构：summary_metrics 子字典 / 顶层平铺。"""
    summary = result.get("summary_metrics")
    metrics = {
        key: float(value)
        for key, value in (summary.items() if isinstance(summary, dict) else result.items())
        # bool 是 int 的子类，必须显式排除，否则 True/False 会被当成 1/0
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    return metrics
