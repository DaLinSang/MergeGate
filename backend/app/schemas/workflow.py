"""
多 Agent 工作流契约
================

【业务定位】"复杂工程问题先拆再合"的完整数据结构。如"这个 PR 能不能合"需多维度取证。
【数据流向】PlannerAgent（出 spec） → Orchestrator（执行） → Observer（校验） → Synthesis（合成）
【核心规则】任务图显式可见；依赖全成功才 ready；证据不足可 Replan 但最多 2 轮。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


WorkflowTaskStatus = Literal["pending", "running", "success", "skipped", "error"]   # 任务状态
WorkflowFindingSeverity = Literal["info", "warning", "blocker"]                     # 问题严重程度


def _workflow_id() -> str:
    """生成工作流 ID（短一点，日志里好找）。"""
    return f"workflow_{uuid.uuid4().hex[:12]}"


class WorkflowClaim(BaseModel):
    """任务的断言：声明这个任务打算验证什么（让任务可审计）。"""

    target_type: str                                        # 目标类型
    target_id: str | None = None                            # 目标 ID
    allowed_sources: list[str] = Field(default_factory=list)  # 允许的证据来源
    write_intent: bool = False                              # 是否打算写数据（True 必须走草稿确认）
    reason: str = ""                                        # 为什么需要这个任务


class WorkflowTask(BaseModel):
    """工作流中的一个子任务。"""

    id: str                                                 # 任务 ID，如 t1
    agent_name: str                                         # 由哪个 Agent 执行
    task_type: str                                          # 任务类型
    objective: str                                          # 目标（一句话）
    dependencies: list[str] = Field(default_factory=list)   # 依赖的任务 ID（全成功才 ready）
    claim: WorkflowClaim                                    # 任务的断言
    input_hint: dict[str, Any] = Field(default_factory=dict)  # 输入提示
    critical: bool = False                                  # 是否关键任务（失败则整体失败）


class WorkflowSpec(BaseModel):
    """工作流规格（PlannerAgent 产出，执行前就能看到的计划）。"""

    workflow_id: str = Field(default_factory=_workflow_id)  # 工作流 ID
    goal: str                                               # 用户原始目标
    trigger_message: str                                    # 触发消息
    acceptance_criteria: list[str] = Field(default_factory=list)  # 验收标准
    tasks: list[WorkflowTask] = Field(default_factory=list)  # 任务列表（任务图）
    max_parallel_tasks: int = Field(default=3, ge=1, le=8)  # 最大并行数
    requires_human_confirmation: bool = False               # 是否需人工确认后执行
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))  # 创建时间


class WorkflowTaskResult(BaseModel):
    """单个子任务的执行结果。"""

    task_id: str                                            # 对应 WorkflowTask.id
    agent_name: str                                         # 执行者
    task_type: str                                          # 任务类型
    status: WorkflowTaskStatus                              # 执行状态
    summary: str = ""                                       # 结果摘要
    output: dict[str, Any] = Field(default_factory=dict)    # 完整结构化输出
    evidence_count: int = 0                                 # 证据条数（判断"够不够"）
    confidence: float | None = Field(default=None, ge=0, le=1)  # 置信度
    citations: list[dict[str, Any]] = Field(default_factory=list)  # 引用来源
    error: str | None = None                                # 错误信息
    started_at: datetime | None = None                      # 开始时间
    completed_at: datetime | None = None                    # 完成时间


class WorkflowFinding(BaseModel):
    """ObserverAgent 发现的一个问题。"""

    finding_type: str                                       # conflict / evidence_gap / low_confidence / merge_gate
    severity: WorkflowFindingSeverity                       # 严重程度
    message: str                                            # 问题描述
    task_ids: list[str] = Field(default_factory=list)       # 涉及哪些任务
    recommendation: str = ""                                # 建议怎么处理


class WorkflowObservation(BaseModel):
    """ObserverAgent 的校验结论（质量闸门：证据不足会触发 Replan）。"""

    observer_agent: str = "observer_agent"                  # 校验者
    findings: list[WorkflowFinding] = Field(default_factory=list)  # 发现的问题
    overall_confidence: float = Field(default=0.0, ge=0, le=1)     # 整体置信度
    human_review_required: bool = False                     # 是否需人工复核
    summary: str = ""                                       # 校验摘要


class WorkflowReport(BaseModel):
    """完整工作流报告（落进 agent_workflow_runs 表的内容）。"""

    spec: WorkflowSpec                                      # 执行用的计划
    task_results: list[WorkflowTaskResult]                  # 各子任务结果
    observation: WorkflowObservation                        # 校验结论
    final_answer: str                                       # 最终综合答案（模板拼装，不调 LLM）
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)  # 工具调用轨迹
    citations: list[dict[str, Any]] = Field(default_factory=list)   # 汇总引用
    metrics: dict[str, Any] = Field(default_factory=dict)   # 耗时/token/重规划次数
    agent_events: list[dict[str, Any]] = Field(default_factory=list)  # 细粒度事件
