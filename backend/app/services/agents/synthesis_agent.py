"""
综合 Agent（Synthesis）
=====================

【业务定位】多 Agent 工作流的最后一环：把 spec + 各任务结果 + Observer 校验，拼成决策备忘录。
【数据流向】workflow_orchestrator.py 最后一步调用；结果写入 WorkflowReport.final_answer。
【核心规则】纯模板拼装，**不调 LLM** —— 前面 Agent 已产出结构化结论，这里只需排版，调模型只会引入幻觉。
"""

from __future__ import annotations

from typing import Any

from app.schemas.workflow import WorkflowObservation, WorkflowSpec, WorkflowTaskResult
from app.services.agents.base import BaseAgent


class SynthesisAgent(BaseAgent):
    """把多 Agent 结果综合成工程决策备忘录。"""

    name = "synthesis_agent"
    description = "将多 Agent 工作流综合成一份工程决策备忘录。"
    available_tools = ["synthesize_workflow_report"]

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """入口：解析输入 → 拼装答案。"""
        spec = WorkflowSpec.model_validate(input_data["spec"])
        results = [WorkflowTaskResult.model_validate(item) for item in input_data.get("task_results") or []]
        observation = WorkflowObservation.model_validate(input_data["observation"])
        answer = self._build_answer(spec, results, observation)
        return {"answer": answer}

    def _build_answer(self, spec: WorkflowSpec, results: list[WorkflowTaskResult], observation: WorkflowObservation) -> str:
        """按固定小节拼装 Markdown 备忘录。"""
        # 按严重程度分流：blocker 决定能不能推进，warning 只提示补材料
        blockers = [item for item in observation.findings if item.severity == "blocker"]
        warnings = [item for item in observation.findings if item.severity == "warning"]
        decision = self._decision(blockers, warnings, results)

        lines = [
            "## 多 Agent 协作结论",
            "",
            f"目标：{spec.goal}",
            f"总体判断：{decision}",
            f"Observer 置信度：{observation.overall_confidence:.2f}",
            "",
            "## 协作任务图",
        ]
        # 展示任务图，让用户看到"系统打算怎么查"
        for task in spec.tasks:
            deps = f"，依赖：{', '.join(task.dependencies)}" if task.dependencies else ""
            lines.append(f"- {task.id}: {task.agent_name} 负责 {task.objective}{deps}")

        lines.extend(["", "## Agent 输出摘要"])
        for result in results:
            lines.append(f"- {result.agent_name} [{result.status}]: {result.summary or '无摘要。'}")

        lines.extend(["", "## Observer 检查"])
        if observation.findings:
            for finding in observation.findings:
                lines.append(f"- {finding.severity.upper()}: {finding.message} 建议：{finding.recommendation}")
        else:
            lines.append("- 未发现跨 Agent 结论冲突。")

        lines.extend(["", "## 决策依据与待确认项"])
        lines.extend(f"- {item}" for item in self._decision_basis(results, observation))

        next_steps = self._next_steps(blockers, warnings, results)
        lines.extend(["", "## 下一步"])
        lines.extend(f"- {item}" for item in next_steps)
        return "\n".join(lines).strip()

    def _decision(
        self,
        blockers: list[Any],
        warnings: list[Any],
        results: list[WorkflowTaskResult],
    ) -> str:
        """给出一句话总体判断（优先级：阻塞项 > PR 建议 > 警告 > 通过）。"""
        if blockers:
            # CI 阻塞或跨 Agent 冲突：最严重，直接建议暂缓
            if any(item.finding_type in {"ci_merge_gate", "cross_agent_conflict"} for item in blockers):
                return "暂缓合并或发布，先处理 CI/跨 Agent 冲突。"
            # 需要人工确认：只出草稿，不执行
            if any(item.finding_type == "human_confirmation_required" for item in blockers):
                return "只生成草稿，等待人工确认后再执行外部写操作。"
            return "存在阻塞项，当前不建议直接推进。"
        # 没有阻塞项时，优先采纳 PR Agent 的建议（它最懂这次改动）
        pr_result = self._by_type(results, "pr_review")
        if pr_result and pr_result.output.get("merge_recommendation"):
            return f"按 PR Agent 建议推进：{pr_result.output.get('merge_recommendation')}。"
        if warnings:
            return "可以继续推进，但需要补齐 Observer 标出的证据或团队信息。"
        return "可以继续推进，未发现明显跨域阻塞。"

    def _next_steps(
        self,
        blockers: list[Any],
        warnings: list[Any],
        results: list[WorkflowTaskResult],
    ) -> list[str]:
        """从各 Agent 的输出里抽取可执行的下一步（去重，最多 6 条）。"""
        steps: list[str] = []
        ci_result = self._by_type(results, "ci_debug")
        pr_result = self._by_type(results, "pr_review")
        issue_result = self._by_type(results, "issue_analysis")

        # CI 阻塞：优先给排查步骤
        if ci_result and ci_result.output.get("is_merge_blocking"):
            debug_steps = [str(item) for item in ci_result.output.get("debug_steps") or [] if str(item).strip()]
            steps.extend(debug_steps[:3] or ["先定位并修复最新失败 CI。"])
        # PR：阻塞项 + 建议补的测试
        if pr_result:
            blocking = [str(item) for item in pr_result.output.get("blocking_issues") or [] if str(item).strip()]
            tests = [str(item) for item in pr_result.output.get("test_suggestions") or [] if str(item).strip()]
            steps.extend(blocking[:2])
            steps.extend(tests[:2])
        # Issue：检查清单或行动项
        if issue_result:
            actions = [str(item) for item in issue_result.output.get("checklist") or issue_result.output.get("action_items") or [] if str(item).strip()]
            steps.extend(actions[:2])
        # Observer 的建议也纳入
        for finding in [*blockers, *warnings]:
            if finding.recommendation:
                steps.append(str(finding.recommendation))
        if not steps:
            steps.append("把本次 workflow 结论回填到 PR/Issue 描述或团队同步中。")
        # dict.fromkeys 去重且保持顺序
        return list(dict.fromkeys(steps))[:6]

    def _decision_basis(self, results: list[WorkflowTaskResult], observation: WorkflowObservation) -> list[str]:
        """列出"凭什么这么判断"：每个 Agent 的摘要 + 证据数 + 置信度。"""
        basis: list[str] = []
        for result in results:
            confidence = f"，置信度 {result.confidence:.2f}" if result.confidence is not None else ""
            evidence = f"，证据 {result.evidence_count} 条" if result.evidence_count else ""
            basis.append(f"{result.agent_name}：{result.summary or '无摘要'}{evidence}{confidence}。")
        if observation.findings:
            for finding in observation.findings:
                basis.append(f"待确认：{finding.message} 处理建议：{finding.recommendation}")
        else:
            # 即使没有发现，也要提醒：最终合入仍需人工按流程确认
            basis.append("待确认：最终合入、评论或外部发送仍需要人工按团队流程确认。")
        return basis[:8]

    def _by_type(self, results: list[WorkflowTaskResult], task_type: str) -> WorkflowTaskResult | None:
        """按任务类型找第一个成功的结果。"""
        return next((item for item in results if item.task_type == task_type and item.status == "success"), None)
