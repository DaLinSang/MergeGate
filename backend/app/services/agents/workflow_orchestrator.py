"""
工作流编排器
===========

【业务定位】多 Agent 协作的执行引擎：按任务图并行调度、观察结果、必要时重新规划、最后综合。
【数据流向】chat.py（工具 workflow.run_engineering_review）→ 本类 → Planner/专用 Agent/Observer/Synthesis
【核心规则】任务图显式；依赖全成功才 ready；单个失败只影响后继（标 skipped）；重规划最多 2 轮。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings
from app.schemas.workflow import WorkflowObservation, WorkflowReport, WorkflowSpec, WorkflowTask, WorkflowTaskResult
from app.services.agents.base import BaseAgent
from app.services.agents.observer_agent import ObserverAgent
from app.services.agents.planner_agent import PlannerAgent
from app.services.agents.synthesis_agent import SynthesisAgent


# 任务执行器签名：接收（任务、spec、上下文），返回结果
WorkflowTaskRunner = Callable[[WorkflowTask, WorkflowSpec, dict[str, Any]], Awaitable[WorkflowTaskResult | dict[str, Any]]]


class WorkflowOrchestrator(BaseAgent):
    """协调 Planner、专用 Agent、Observer、Synthesis 完成工程决策。"""

    name = "workflow_orchestrator"
    description = "协调 Planner、专用 Agent、Observer 和 Synthesis 完成工程决策。"
    available_tools = ["run_engineering_workflow"]

    def __init__(
        self,
        task_runners: dict[str, WorkflowTaskRunner],
        planner: PlannerAgent | None = None,
        observer: ObserverAgent | None = None,
        synthesizer: SynthesisAgent | None = None,
    ) -> None:
        """注入依赖。

        参数：
            task_runners → 任务类型到执行函数的映射（由 chat.py 构造，闭包持有 db/repo）
            planner/observer/synthesizer → 可替换，默认用标准实现（便于测试）
        """
        self.task_runners = task_runners
        self.planner = planner or PlannerAgent()
        self.observer = observer or ObserverAgent()
        self.synthesizer = synthesizer or SynthesisAgent()

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """入口：先规划，再执行。"""
        spec_data = await self.planner.run(input_data, context)
        return await self.run_spec(spec_data, context)

    async def run_spec(
        self,
        spec_data: WorkflowSpec | dict[str, Any],
        context: dict[str, Any],
        *,
        allow_replan: bool = True,
    ) -> dict[str, Any]:
        """执行一个已确定的 spec（计划模式下用户确认后也走这里）。

        主循环：
            执行任务图 → Observer 校验 → 有可修复问题则 Replan → 重复
            最多 1 + max_replans 轮（max_replans 上限 2）
        """
        started = time.perf_counter()
        spec = spec_data if isinstance(spec_data, WorkflowSpec) else WorkflowSpec.model_validate(spec_data)
        current_spec = spec
        task_results: list[WorkflowTaskResult] = []
        observations: list[WorkflowObservation] = []
        round_records: list[dict[str, Any]] = []        # 每一轮的快照（给前端画时间线）
        replan_events: list[dict[str, Any]] = []
        max_replans = self._max_replans(context, allow_replan)

        for attempt in range(max_replans + 1):
            # 执行本轮任务图（带上历史结果，已成功的任务不会重跑）
            round_results = await self._run_task_graph(current_spec, context, previous_results=task_results)
            task_results.extend(round_results)
            observation = await self._observe(current_spec, task_results, context)
            observations.append(observation)
            round_records.append({"spec": current_spec, "results": round_results, "observation": observation})

            if attempt >= max_replans:
                break

            replan_reason = self._replan_reason(task_results, observation)
            if replan_reason is None:
                break                                   # 没有可修复的问题，不用重规划
            if not hasattr(self.planner, "replan"):
                break                                   # Planner 不支持重规划

            next_spec_data = await self.planner.replan(
                {
                    "spec": current_spec.model_dump(mode="json"),
                    "task_results": [item.model_dump(mode="json") for item in task_results],
                    "observation": observation.model_dump(mode="json"),
                    "reason": replan_reason,
                    "replan_iteration": attempt + 1,
                },
                context,
            )
            next_spec = self._normalize_replanned_spec(
                original_spec=spec,
                current_spec=current_spec,
                next_spec=WorkflowSpec.model_validate(next_spec_data),
            )
            # 防死循环：新旧任务图一样就没必要再跑一轮
            if self._same_task_graph(current_spec, next_spec):
                break

            replan_events.append(
                {
                    "type": "workflow_replan",
                    "workflow_id": spec.workflow_id,
                    "iteration": attempt + 1,
                    "reason": replan_reason,
                    "from_task_count": len(current_spec.tasks),
                    "to_task_count": len(next_spec.tasks),
                    "added_tasks": [
                        task.id
                        for task in next_spec.tasks
                        if task.id not in {existing.id for existing in current_spec.tasks}
                    ],
                }
            )
            current_spec = next_spec

        # 用最后一轮的观察结果做综合
        observation = observations[-1] if observations else WorkflowObservation(summary="Workflow did not run.")
        synthesis = await self.synthesizer.run(
            {
                "spec": current_spec.model_dump(mode="json"),
                "task_results": [item.model_dump(mode="json") for item in task_results],
                "observation": observation.model_dump(mode="json"),
            },
            context,
        )
        report = WorkflowReport(
            spec=current_spec,
            task_results=task_results,
            observation=observation,
            final_answer=str(synthesis.get("answer") or ""),
            tool_calls=self._tool_calls(current_spec, task_results, replan_events),
            citations=self._citations(task_results),
            # 指标：耗时、任务数、重规划次数、成败统计（前端展示 + 评测用）
            metrics={
                "duration_ms": int((time.perf_counter() - started) * 1000),
                "task_count": len(task_results),
                "plan_task_count": len(current_spec.tasks),
                "execution_rounds": len(observations),
                "replan_count": len(replan_events),
                "success_count": len([item for item in task_results if item.status == "success"]),
                "error_count": len([item for item in task_results if item.status == "error"]),
                "skipped_count": len([item for item in task_results if item.status == "skipped"]),
                "observer_findings": len(observation.findings),
            },
            agent_events=self._agent_events(round_records, replan_events),
        )
        return report.model_dump(mode="json")

    async def _observe(
        self,
        spec: WorkflowSpec,
        task_results: list[WorkflowTaskResult],
        context: dict[str, Any],
    ) -> WorkflowObservation:
        """调 Observer 做质量校验。"""
        observation_data = await self.observer.run(
            {
                "spec": spec.model_dump(mode="json"),
                "task_results": [item.model_dump(mode="json") for item in task_results],
            },
            context,
        )
        return WorkflowObservation.model_validate(observation_data)

    async def _run_task_graph(
        self,
        spec: WorkflowSpec,
        context: dict[str, Any],
        previous_results: list[WorkflowTaskResult] | None = None,
    ) -> list[WorkflowTaskResult]:
        """按依赖拓扑执行任务图（核心调度逻辑）。

        算法：
            1. 把已成功的任务从 pending 移除（重规划时不会重跑）
            2. 依赖失败的任务直接标 skipped（不执行）
            3. 依赖全部完成的任务进入 ready，按 max_parallel_tasks 分批并行
            4. 循环直到 pending 为空
        """
        pending = {task.id: task for task in spec.tasks}
        # 已成功的任务（重规划时跳过，避免重复消耗）
        completed: dict[str, WorkflowTaskResult] = {
            result.task_id: result
            for result in previous_results or []
            if result.status == "success"
        }
        # 失败/跳过的任务（其下游要跟着跳过）
        blocked_dependencies: dict[str, WorkflowTaskResult] = {
            result.task_id: result
            for result in previous_results or []
            if result.status in {"error", "skipped"}
        }
        for task_id in completed:
            pending.pop(task_id, None)
        results: list[WorkflowTaskResult] = []

        while pending:
            # ── 第一步：把依赖失败的任务标 skipped（级联跳过）──
            blocked = [
                task
                for task in pending.values()
                if any(dependency in blocked_dependencies for dependency in task.dependencies)
            ]
            for task in blocked:
                failed_dependencies = [
                    dependency
                    for dependency in task.dependencies
                    if dependency in blocked_dependencies
                ]
                result = self._dependency_blocked_result(task, failed_dependencies)
                blocked_dependencies[task.id] = result
                results.append(result)
                pending.pop(task.id, None)
            if not pending:
                break

            # ── 第二步：找出依赖全部完成的 ready 任务 ──
            ready = [
                task
                for task in pending.values()
                if all(dependency in completed for dependency in task.dependencies)
            ]
            # 有 pending 但没有 ready → 依赖成环或引用了不存在的任务
            if not ready:
                for task in pending.values():
                    result = self._error_result(task, "任务依赖存在循环或缺失依赖。", error_kind="dependency_cycle_or_missing")
                    blocked_dependencies[task.id] = result
                    results.append(result)
                break

            # ── 第三步：分批并行执行 ──
            batch = ready[: spec.max_parallel_tasks]
            batch_results = await asyncio.gather(
                *(self._run_single_task(task, spec, context) for task in batch),
                return_exceptions=False,
            )
            for task, result in zip(batch, batch_results):
                if result.status == "success":
                    completed[task.id] = result
                else:
                    blocked_dependencies[task.id] = result
                results.append(result)
                pending.pop(task.id, None)

        return results

    async def _run_single_task(
        self,
        task: WorkflowTask,
        spec: WorkflowSpec,
        context: dict[str, Any],
    ) -> WorkflowTaskResult:
        """执行单个任务：查 runner → 带超时执行 → 异常转 error。"""
        started_at = datetime.now(timezone.utc)
        runner = self.task_runners.get(task.task_type)
        # 没注册执行器：标 skipped 而不是报错（该任务类型只是没实现）
        if runner is None:
            return WorkflowTaskResult(
                task_id=task.id,
                agent_name=task.agent_name,
                task_type=task.task_type,
                status="skipped",
                summary=f"没有为 {task.task_type} 注册任务执行器。",
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )
        try:
            # wait_for 保证单任务不会无限挂起
            raw_result = await asyncio.wait_for(
                runner(task, spec, context),
                timeout=self._task_timeout_seconds(context),
            )
            result = raw_result if isinstance(raw_result, WorkflowTaskResult) else WorkflowTaskResult.model_validate(raw_result)
            result.started_at = result.started_at or started_at
            result.completed_at = result.completed_at or datetime.now(timezone.utc)
            return result
        except TimeoutError:
            return self._error_result(
                task,
                f"{task.agent_name} 执行超过 {self._task_timeout_seconds(context):g} 秒。",
                started_at=started_at,
                error_kind="timeout",
            )
        except Exception as exc:
            # 任何异常都转成 error 结果，不让整个工作流崩掉
            return self._error_result(task, str(exc), started_at=started_at, error_kind="task_runtime_error")

    def _error_result(
        self,
        task: WorkflowTask,
        error: str,
        started_at: datetime | None = None,
        *,
        error_kind: str = "task_runtime_error",
    ) -> WorkflowTaskResult:
        """构造一个 error 状态的结果。"""
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="error",
            summary=f"{task.agent_name} 执行失败。",
            output={"error_kind": error_kind},
            error=error,
            started_at=started_at or datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

    def _dependency_blocked_result(self, task: WorkflowTask, failed_dependencies: list[str]) -> WorkflowTaskResult:
        """构造一个因依赖失败而 skipped 的结果。"""
        return WorkflowTaskResult(
            task_id=task.id,
            agent_name=task.agent_name,
            task_type=task.task_type,
            status="skipped",
            summary=f"{task.agent_name} 未执行，因为依赖任务未成功：{', '.join(failed_dependencies)}。",
            output={"error_kind": "dependency_failed", "failed_dependencies": failed_dependencies},
            confidence=0.0,
            completed_at=datetime.now(timezone.utc),
        )

    def _task_timeout_seconds(self, context: dict[str, Any]) -> float:
        """单任务超时（默认 90 秒，可由 context 覆盖）。"""
        try:
            value = float(context.get("workflow_task_timeout_seconds", settings.workflow_task_timeout_seconds))
        except (TypeError, ValueError):
            value = settings.workflow_task_timeout_seconds
        return max(0.001, value)

    def _max_replans(self, context: dict[str, Any], allow_replan: bool) -> int:
        """允许几轮重规划。硬上限 2，防止无限循环烧钱。"""
        if not allow_replan or context.get("disable_replan"):
            return 0
        try:
            requested = int(context.get("max_replans", 1))
        except (TypeError, ValueError):
            requested = 1
        return max(0, min(2, requested))

    def _replan_reason(self, results: list[WorkflowTaskResult], observation: WorkflowObservation) -> str | None:
        """判断要不要重规划，返回原因（None 表示不需要）。

        只有"补任务能解决"的问题才触发重规划：
            task_error / task_skipped → 补上下文
            evidence_gap / low_confidence → 补检索
            missing_safety_review → 补安全审查
        像 cross_agent_conflict 这种是事实冲突，补任务解决不了，不重规划。
        """
        fixable_findings = {
            "task_error",
            "task_skipped",
            "evidence_gap",
            "low_confidence",
            "missing_safety_review",
            "owner_needs_review",
            "team_profile_missing",
        }
        for finding in observation.findings:
            if finding.finding_type in fixable_findings:
                return f"{finding.finding_type}: {finding.recommendation or finding.message}"
        if any(result.status == "error" for result in results):
            return "task_error: at least one workflow task failed"
        low_confidence_results = [
            result
            for result in results
            if result.status == "success" and result.confidence is not None and result.confidence < 0.45
        ]
        if low_confidence_results:
            return "low_confidence: one or more successful task results have weak confidence"
        if results and observation.overall_confidence < 0.45:
            return "low_confidence: observer confidence is below the workflow threshold"
        return None

    def _normalize_replanned_spec(
        self,
        *,
        original_spec: WorkflowSpec,
        current_spec: WorkflowSpec,
        next_spec: WorkflowSpec,
    ) -> WorkflowSpec:
        """重规划后修正 spec：ID 不变、空字段沿用上一轮、安全要求只能收紧不能放松。"""
        normalized = next_spec.model_copy(deep=True)
        normalized.workflow_id = original_spec.workflow_id          # ID 必须保持一致（日志串联）
        normalized.goal = normalized.goal or current_spec.goal
        normalized.trigger_message = normalized.trigger_message or current_spec.trigger_message
        if not normalized.acceptance_criteria:
            normalized.acceptance_criteria = list(current_spec.acceptance_criteria)
        normalized.max_parallel_tasks = current_spec.max_parallel_tasks
        # 人工确认要求做"或"运算：一旦要求过，就不能被重规划取消
        normalized.requires_human_confirmation = (
            current_spec.requires_human_confirmation or normalized.requires_human_confirmation
        )
        return normalized

    def _same_task_graph(self, left: WorkflowSpec, right: WorkflowSpec) -> bool:
        """比较两个任务图是否相同（只看 id/type/依赖，不看文案）。"""
        def signature(spec: WorkflowSpec) -> list[tuple[str, str, tuple[str, ...]]]:
            return [(task.id, task.task_type, tuple(task.dependencies)) for task in spec.tasks]

        return signature(left) == signature(right)

    def _tool_calls(
        self,
        spec: WorkflowSpec,
        results: list[WorkflowTaskResult],
        replan_events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """把每个子任务和每次重规划都包装成"工具调用"，供前端展示轨迹。"""
        calls = [
            {
                "tool_name": f"workflow.{result.task_type}",
                "arguments": {
                    "workflow_id": spec.workflow_id,
                    "task_id": result.task_id,
                    "agent": result.agent_name,
                },
                "status": result.status,
                "error": result.error,
            }
            for result in results
        ]
        calls.extend(
            {
                "tool_name": "planner_agent.replan_workflow_spec",
                "arguments": {
                    "workflow_id": spec.workflow_id,
                    "iteration": event.get("iteration"),
                    "reason": event.get("reason"),
                },
                "status": "success",
                "error": None,
            }
            for event in replan_events
        )
        return calls

    def _citations(self, results: list[WorkflowTaskResult]) -> list[dict[str, Any]]:
        """汇总所有子任务的引用并去重（同一来源只保留一条）。"""
        citations: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for result in results:
            for citation in result.citations:
                key = (
                    str(citation.get("type") or ""),
                    str(citation.get("id") or citation.get("source_id") or ""),
                    str(citation.get("title") or ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                citations.append(citation)
        return citations

    def _agent_events(
        self,
        round_records: list[dict[str, Any]],
        replan_events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """生成前端时间线事件：每轮的计划 → 各任务结果 → Observer 结论。"""
        events: list[dict[str, Any]] = []
        for index, record in enumerate(round_records):
            # 第二轮开始，先插入一次重规划事件
            if index > 0 and index - 1 < len(replan_events):
                events.append(replan_events[index - 1])
            spec = record["spec"]
            events.append(
                {
                    "type": "workflow_spec",
                    "workflow_id": spec.workflow_id,
                    "goal": spec.goal,
                    "iteration": index,
                    "is_replan": index > 0,
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
            )
            for result in record["results"]:
                # 把 Skill 运行信息透传到前端（用于显示"用了哪个技能"）
                skill_runtime = result.output.get("_skill_runtime") if isinstance(result.output, dict) else None
                events.append(
                    {
                        "type": "workflow_task_result",
                        "workflow_id": spec.workflow_id,
                        "iteration": index,
                        "task_id": result.task_id,
                        "agent_name": result.agent_name,
                        "task_type": result.task_type,
                        "status": result.status,
                        "summary": result.summary,
                        "confidence": result.confidence,
                        "evidence_count": result.evidence_count,
                        **(
                            {
                                key: value
                                for key, value in skill_runtime.items()
                                if key
                                in {
                                    "skill_name",
                                    "skill_title",
                                    "skill_version",
                                    "skill_steps",
                                    "skill_activation_mode",
                                    "skill_activation_reason",
                                    "skill_instruction_digest",
                                    "validation",
                                }
                            }
                            if isinstance(skill_runtime, dict)
                            else {}
                        ),
                    }
                )
            observation = record["observation"]
            events.append(
                {
                    "type": "workflow_observer_result",
                    "workflow_id": spec.workflow_id,
                    "iteration": index,
                    "is_final": index == len(round_records) - 1,      # 只有最后一轮是最终结论
                    "overall_confidence": observation.overall_confidence,
                    "human_review_required": observation.human_review_required,
                    "findings": [finding.model_dump(mode="json") for finding in observation.findings],
                }
            )
        return events
