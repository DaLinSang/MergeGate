"""
观察 Agent（Observer）
====================

【业务定位】多 Agent 工作流的"质量闸门"：检查各 Agent 结论是否冲突、证据是否够、置信度是否够。
【数据流向】workflow_orchestrator.py 在任务执行后调用；输出 WorkflowObservation 决定是否 Replan。
【核心规则】纯规则判断，不调 LLM。发现 blocker 会触发重新规划；置信度按 blocker/warning 数量扣分。
"""

from __future__ import annotations

from typing import Any

from app.schemas.workflow import WorkflowFinding, WorkflowObservation, WorkflowSpec, WorkflowTaskResult
from app.services.agents.base import BaseAgent


class ObserverAgent(BaseAgent):
    """审查工作流输出的冲突、证据缺口、低置信度和人工确认门槛。"""

    name = "observer_agent"
    description = "审查工作流任务输出中的冲突、证据缺口、低置信度和人工确认门槛。"
    available_tools = ["inspect_workflow_outputs"]

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """入口：跑 6 类检查 → 汇总成 observation。"""
        spec = WorkflowSpec.model_validate(input_data["spec"])
        results = [WorkflowTaskResult.model_validate(item) for item in input_data.get("task_results") or []]
        findings: list[WorkflowFinding] = []

        # 6 类检查，每类只负责一件事
        findings.extend(self._status_findings(results))          # 任务失败/被跳过
        findings.extend(self._evidence_findings(results))        # 有结论但没证据
        findings.extend(self._confidence_findings(results))      # 置信度太低
        findings.extend(self._merge_gate_findings(results))      # 能否合并的门禁
        findings.extend(self._ownership_findings(results, context))  # 负责人是否可信
        findings.extend(self._safety_findings(results, spec))    # 写操作是否需人工确认

        # 置信度计算：各任务平均分，再按问题数量扣分
        confidence_values = [item.confidence for item in results if item.confidence is not None and item.status == "success"]
        base_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.55
        blocker_penalty = 0.2 * len([item for item in findings if item.severity == "blocker"])
        warning_penalty = 0.05 * len([item for item in findings if item.severity == "warning"])
        # 夹在 0.1~0.95：再差也留一点，再好也不给满分（避免过度自信）
        overall_confidence = round(max(0.1, min(0.95, base_confidence - blocker_penalty - warning_penalty)), 2)

        observation = WorkflowObservation(
            findings=findings,
            overall_confidence=overall_confidence,
            # 有 blocker 或 spec 要求人工确认 → 必须人工复核
            human_review_required=spec.requires_human_confirmation or any(item.severity == "blocker" for item in findings),
            summary=self._summary(findings, overall_confidence),
        )
        return observation.model_dump(mode="json")

    def _status_findings(self, results: list[WorkflowTaskResult]) -> list[WorkflowFinding]:
        """检查任务状态：error → blocker；skipped → warning（覆盖不完整）。"""
        findings = []
        for result in results:
            if result.status == "error":
                findings.append(
                    WorkflowFinding(
                        finding_type="task_error",
                        severity="blocker",
                        message=f"{result.agent_name} 执行失败：{result.error or '未知错误'}",
                        task_ids=[result.task_id],
                        recommendation="先修复任务失败，或用更窄范围重新运行，再信任综合结论。",
                    )
                )
            elif result.status == "skipped":
                findings.append(
                    WorkflowFinding(
                        finding_type="task_skipped",
                        severity="warning",
                        message=f"{result.agent_name} 被跳过，因此工作流覆盖不完整。",
                        task_ids=[result.task_id],
                        recommendation="如果该领域会影响决策，请先连接或同步缺失的仓库信号。",
                    )
                )
        return findings

    def _evidence_findings(self, results: list[WorkflowTaskResult]) -> list[WorkflowFinding]:
        """检查"有结论但零证据"的情况。repo_health / safety_review 不要求证据。"""
        findings = []
        for result in results:
            if result.status != "success" or result.task_type in {"repo_health", "safety_review"}:
                continue
            if result.evidence_count <= 0 and not result.citations:
                findings.append(
                    WorkflowFinding(
                        finding_type="evidence_gap",
                        severity="warning",
                        message=f"{result.agent_name} 没有返回明确证据或引用。",
                        task_ids=[result.task_id],
                        recommendation="在做高风险决策前，优先同步相关 Issue、PR 文件、CI 日志，并实时搜索当前代码。",
                    )
                )
        return findings

    def _confidence_findings(self, results: list[WorkflowTaskResult]) -> list[WorkflowFinding]:
        """置信度低于 0.5 视为不可靠。"""
        findings = []
        for result in results:
            if result.status == "success" and result.confidence is not None and result.confidence < 0.5:
                findings.append(
                    WorkflowFinding(
                        finding_type="low_confidence",
                        severity="warning",
                        message=f"{result.agent_name} 的置信度偏低（{result.confidence:.2f}）。",
                        task_ids=[result.task_id],
                        recommendation="行动前请补充更多上下文，或检查底层证据。",
                    )
                )
        return findings

    def _merge_gate_findings(self, results: list[WorkflowTaskResult]) -> list[WorkflowFinding]:
        """合并门禁：CI 失败 / PR 阻塞项 / 两者结论冲突。"""
        findings = []
        pr_result = self._by_type(results, "pr_review")
        ci_result = self._by_type(results, "ci_debug")

        # CI 判定阻塞合并
        if ci_result and bool(ci_result.output.get("is_merge_blocking")):
            findings.append(
                WorkflowFinding(
                    finding_type="ci_merge_gate",
                    severity="blocker",
                    message="CI Agent 将最新失败 workflow 标记为阻塞合并。",
                    task_ids=[ci_result.task_id],
                    recommendation="批准合并或发布前，请先修复或解释 CI 失败。",
                )
            )
        # PR 审查有阻塞项
        if pr_result and pr_result.output.get("blocking_issues"):
            findings.append(
                WorkflowFinding(
                    finding_type="pr_blocking_issues",
                    severity="blocker",
                    message="PR 审查发现了阻塞问题。",
                    task_ids=[pr_result.task_id],
                    recommendation="解决 PR 阻塞问题后重新运行工作流。",
                )
            )
        # 跨 Agent 冲突：PR 说能合，CI 说挂了 —— 这是最需要人工介入的情况
        if pr_result and ci_result and bool(ci_result.output.get("is_merge_blocking")):
            if self._looks_like_merge_approval(str(pr_result.output.get("merge_recommendation") or "")):
                findings.append(
                    WorkflowFinding(
                        finding_type="cross_agent_conflict",
                        severity="blocker",
                        message="PR Agent 倾向合并，但 CI Agent 报告了阻塞合并的失败。",
                        task_ids=[pr_result.task_id, ci_result.task_id],
                        recommendation="在失败解决前，让 CI 门禁优先于 PR 合并建议。",
                    )
                )
        return findings

    def _ownership_findings(self, results: list[WorkflowTaskResult], context: dict[str, Any]) -> list[WorkflowFinding]:
        """检查 Issue 推荐的负责人是否在团队成员名单里。"""
        issue_result = self._by_type(results, "issue_analysis")
        if not issue_result:
            return []
        owner = str(issue_result.output.get("suggested_owner") or "").strip().lower()
        if not owner:
            return []
        team_members = context.get("team_members") or []
        known_names = {
            str(member.get("name") or "").strip().lower()
            for member in team_members
            if isinstance(member, dict) and str(member.get("name") or "").strip()
        }
        unassigned_markers = {"unassigned", "待分配"}
        # 负责人是"待分配"或不在名单里 → 推荐不可信
        if owner in unassigned_markers or (known_names and owner not in known_names):
            return [
                WorkflowFinding(
                    finding_type="owner_needs_review",
                    severity="warning",
                    message="Issue 负责人建议为空，或不在已配置团队成员中。",
                    task_ids=[issue_result.task_id],
                    recommendation="依赖自动负责人分配前，请先更新团队成员画像。",
                )
            ]
        # 完全没配团队 → 推荐无法校验
        if not known_names:
            return [
                WorkflowFinding(
                    finding_type="team_profile_missing",
                    severity="warning",
                    message="当前还没有配置用于负责人路由的 DevFlow 团队成员。",
                    task_ids=[issue_result.task_id],
                    recommendation="请添加团队成员画像，让负责人推荐更可解释。",
                )
            ]
        return []

    def _safety_findings(self, results: list[WorkflowTaskResult], spec: WorkflowSpec) -> list[WorkflowFinding]:
        """安全检查：需要人工确认却没跑 SafetyAgent，或 SafetyAgent 要求确认。"""
        safety_result = self._by_type(results, "safety_review")
        # 该有安全审查却没有 → blocker（不能跳过安全环节）
        if not safety_result and spec.requires_human_confirmation:
            return [
                WorkflowFinding(
                    finding_type="missing_safety_review",
                    severity="blocker",
                    message="工作流需要人工确认，但没有运行 SafetyAgent 任务。",
                    recommendation="在起草评论、标签、创建/关闭 Issue 或外部发送前，请先运行 SafetyAgent。",
                )
            ]
        # SafetyAgent 说需要确认 → blocker，只能出草稿
        if safety_result and safety_result.output.get("requires_confirmation"):
            return [
                WorkflowFinding(
                    finding_type="human_confirmation_required",
                    severity="blocker",
                    message="SafetyAgent 要求用户对本次写入类操作进行人工确认。",
                    task_ids=[safety_result.task_id],
                    recommendation="只返回草稿；MVP 模式下不要执行外部动作。",
                )
            ]
        return []

    def _by_type(self, results: list[WorkflowTaskResult], task_type: str) -> WorkflowTaskResult | None:
        """按任务类型找第一个成功的结果。"""
        return next((item for item in results if item.task_type == task_type and item.status == "success"), None)

    def _looks_like_merge_approval(self, value: str) -> bool:
        """判断 PR Agent 的结论是不是"同意合并"（排除"修改后可合"这种有条件同意）。"""
        lowered = value.lower()
        return any(token in lowered for token in ["approve", "merge", "建议合并", "建议合入"]) and not any(
            token in lowered for token in ["after", "blocked", "暂缓", "修复后", "修改后"]
        )

    def _summary(self, findings: list[WorkflowFinding], confidence: float) -> str:
        """一句话总结校验结果。"""
        blockers = len([item for item in findings if item.severity == "blocker"])
        warnings = len([item for item in findings if item.severity == "warning"])
        if blockers:
            return f"Observer 发现 {blockers} 个阻塞项和 {warnings} 个警告；置信度 {confidence:.2f}。"
        if warnings:
            return f"Observer 发现 {warnings} 个警告；置信度 {confidence:.2f}。"
        return f"Observer 没有发现跨 Agent 阻塞项；置信度 {confidence:.2f}。"
