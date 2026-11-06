"""
PR 审查 Agent
============

【业务定位】
    读 PR 的 diff + 评论 + CI 状态，产出结构化的审查结论：
    P1/P2/P3 分级发现、阻塞项、测试建议、合并建议。

    【和 CI 排障 Agent 一样采用"规则兜底 + LLM 增强"】
        先用规则算一份（_fallback_review），再让 LLM 审一次，逐字段合并。
        原因：diff 可能有几万行，模型读不全；
             但"哪些文件敏感""有没有测试"这类判断，正则比模型更可靠。

【数据流向】
    上游（谁调它）：
        services/agents/workflow_orchestrator.py  工作流里的 pr_review 任务
        services/evals/runner.py                  评测
    下游（它调谁）：
        rag/chunking.chunk_pr_files()      把 diff 切片（喂给 LLM）
        llm.client.chat_json()             调模型（用 PR_REVIEW_PROMPT）
        schemas.analysis.PRReviewOutput    输出契约校验

【核心规则】
    ★ **P1/P2 必须 blocking=true，且不能"建议合入"**。
      这条规则在三个地方强制：
          ① prompt 里写死（PR_REVIEW_PROMPT）
          ② _finding() 里按 severity 自动推导
          ③ _merge_with_fallback() 里做一致性修正
      三重保障，是因为"列出 P1 却说建议合入"是最危险的输出矛盾。

【本文件函数地图】
    run()                          ★ 主流程：切片 → 规则兜底 → LLM → 合并 → 校验
    _fallback_review()             ★ 纯规则审查（五个检查维度）
        ├ _finding()                   构造一条分级发现
        ├ _choose_recommendation()     ★ 七级判定合并建议
        ├ _test_suggestions()          按技术栈给测试建议
        └ _estimate_confidence()       置信度
    _merge_with_fallback()         ★ LLM + 规则合并（含一致性修正）
        ├ _normalize_review_findings() 归一化模型返回的发现
        └ _blocking_issues_from_findings() 从发现反推阻塞项
"""

from typing import Any

from app.schemas.analysis import PRReviewOutput
from app.services.agents.base import BaseAgent
from app.services.llm.client import LLMClient
from app.services.llm.prompts import PR_REVIEW_PROMPT
from app.services.rag.chunking import chunk_pr_files


class PRReviewAgent(BaseAgent):
    """PR 审查 Agent（规则兜底 + LLM 增强）。

    【为什么 diff 要切片】
        一个 PR 可能改 50 个文件、diff 十几万字符。
        全塞给模型会超上下文窗口。chunk_pr_files 按文件切、overlap=0
        （diff 的行号有意义，不能重叠）。
    """

    name = "pr_review_agent"
    description = "总结 PR diff，并生成风险点、检查清单和测试建议。"
    available_tools = ["get_pull_request_detail", "get_pull_request_files", "chunk_diff"]

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """审查一个 PR。

        【调用链】追这个函数会进入：
            1. chunk_pr_files()      → diff 切片（rag/chunking）
            2. _fallback_review()    → 纯规则审查（永远成功）
            3. llm.chat_json()       → LLM 审查
            4. _merge_with_fallback()→ 合并 + 一致性修正
            5. PRReviewOutput.model_validate → 契约校验
        【被谁调用】workflow_orchestrator（pr_review 任务）、evals/runner

        【数据流向】
            入：input_data（PR 的 number/title/body/state/author/files/comments）、
                context（ci_summary / code_references / project_documents / conversation_evidence）
            出：PRReviewOutput 的 dict

        【★ 三层降级（和 CIDebugAgent 同构）】
            LLM 空 → fallback
            LLM 校验失败 → fallback（except 里 pass，静默）
            LLM 正常 → 合并结果

        【为什么把 diff_chunks 单独传】
            模型的输入分两块：
                pull_request  → PR 元信息（标题、描述、评论）
                diff_chunks   → 代码改动（可能很长）
            分开传便于模型理解结构，也便于 prompt 里分别引用。
        """
        diff_chunks = chunk_pr_files(input_data.get("files", []))
        fallback = self._fallback_review(input_data, context, diff_chunks)
        llm_output = await self.llm.chat_json(PR_REVIEW_PROMPT, {"pull_request": input_data, "diff_chunks": diff_chunks, "context": context})
        if llm_output:
            try:
                return PRReviewOutput.model_validate(self._merge_with_fallback(llm_output, fallback)).model_dump()
            except Exception:
                pass
        return fallback

    def _fallback_review(self, input_data: dict[str, Any], context: dict[str, Any], diff_chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """纯规则审查（★ 五个检查维度）。

        【调用链】追这个函数会进入：
            1. 统计文件/行数/敏感文件/测试文件
            2. 五个 if 分支各产出一条 finding
            3. _choose_recommendation() / _test_suggestions() / _estimate_confidence()
        【被谁调用】run()

        【数据流向】
            入：input_data、context、diff_chunks
            出：PRReviewOutput 的 dict

        【★ 五个检查维度，按严重程度排列】
            ① 失败 CI          → P1（必须修，否则合进去就挂）
            ② 敏感文件变更      → P2（auth/security/migration/payment 等）
            ③ 未处理的 review 评论 → P2（有人提了意见没回应）
            ④ Diff 过大        → P3（建议分块 review，不阻塞）
            ⑤ 缺测试覆盖       → P2（有敏感文件或改动 >250 行时才算）

        【为什么"缺测试"是有条件的】
            改一个 typo 也要测试吗？不。所以只在"敏感文件 或 改动量大"时才提。
            这就是"规则"相对于"模型"的价值 —— 可以有精确的触发条件。

        【★ 兜底：一条 finding 都没有时补一条 P3】
            不能返回空的 review_findings —— 前端会显示空白，用户以为坏了。
            补一条"未发现明确阻断项"既诚实又完整。
        """
        files = input_data.get("files", [])
        comments = input_data.get("comments", [])
        filenames = [item.get("filename", "") for item in files]
        total_additions = sum(int(item.get("additions") or 0) for item in files)
        total_deletions = sum(int(item.get("deletions") or 0) for item in files)

        # ── 分类：敏感文件 / 测试文件 ──
        risky_files = [
            name
            for name in filenames
            if any(token in name.lower() for token in ["auth", "security", "migration", "schema", "payment", "permission", "config", "ci", "workflow"])
        ]
        test_files = [
            name
            for name in filenames
            if any(token in name.lower() for token in ["test", "spec", "__tests__", "pytest", "playwright"])
        ]
        total_patch_chars = sum(len(item.get("patch") or "") for item in files)

        risk_points = []
        blocking_issues = []
        review_findings = []

        # CI 摘要（由调用方传进来，这里只读）
        ci_summary = context.get("ci_summary") or {}
        failed_ci = int(ci_summary.get("failed_runs") or 0)

        # ── 检查 ①：失败 CI（P1，最严重）──
        if failed_ci:
            message = f"关联仓库仍有 {failed_ci} 个失败 CI，需要先确认是否由当前 PR 引入或阻塞合并。"
            blocking_issues.append(message)
            review_findings.append(self._finding("P1", "失败 CI 阻塞合并", message, "先修复或解释失败 CI，再重新评估合入。"))

        # ── 检查 ②：敏感文件（P2）──
        if risky_files:
            message = f"高敏感文件需要重点审查：{', '.join(risky_files[:5])}"
            risk_points.append(message)
            review_findings.append(self._finding("P2", "敏感文件变更需要重点验证", message, "补充代码审查记录、回归测试或人工验证说明。"))

        # ── 检查 ③：未处理的 review 评论（P2）──
        if comments:
            message = f"已有 {len(comments)} 条 review comment，需要确认是否都已处理。"
            risk_points.append(message)
            blocking_issues.append(message)
            review_findings.append(self._finding("P2", "Review 评论尚需确认", message, "逐条回应并确认评论已解决后再放行。"))

        # ── 检查 ④：Diff 过大（P3，不阻塞）──
        # 两个阈值：字符数 >12000 或 切片数 >8（说明文件多）
        if total_patch_chars > 12000 or len(diff_chunks) > 8:
            message = "Diff 较大，建议按模块分块 Review，避免遗漏跨文件影响。"
            risk_points.append(message)
            review_findings.append(self._finding("P3", "Diff 体量较大", message, "按模块拆分审查路径，必要时补充分阶段验证。", blocking=False))

        # ── 检查 ⑤：缺测试覆盖（P2，有条件）──
        # 条件：有改动 + 没有测试文件 + （敏感文件 或 改动 >250 行）
        if files and not test_files and (risky_files or total_additions + total_deletions > 250):
            message = "当前变更缺少明显测试文件，合并前需要补充自动化测试或明确手工验证记录。"
            blocking_issues.append(message)
            review_findings.append(self._finding("P2", "缺少测试覆盖证明", message, "补充自动化测试，或留下可复核的手工验证记录。"))

        # ── 兜底：不能返回空 ──
        if not risk_points:
            risk_points.append("需要确认边界条件和异常路径是否有测试覆盖")
        if not review_findings:
            review_findings.append(
                self._finding(
                    "P3",
                    "未发现明确阻断项",
                    "基于当前 diff、评论和 CI 摘要，未发现明确 P1/P2 阻断项。",
                    "按 checklist 完成最终人工确认。",
                    blocking=False,
                )
            )

        # ── 合并建议 + 计划 + 已执行步骤 ──
        merge_recommendation, recommendation_reason = self._choose_recommendation(input_data, files, risk_points, blocking_issues, failed_ci)

        plan = [
            "理解 PR 标题、描述、分支、变更文件和关联上下文。",
            "按模块拆分 diff 风险，识别敏感路径和大体量改动。",
            "读取相关源代码、项目文档和历史会话证据。",
            "检查 CI 状态、失败记录和现有 review comments。",
            "形成合入建议、阻塞项、测试建议和 review checklist。",
        ]
        # 已执行步骤：每条都带具体数字（让用户看到"真的扫了"）
        executed_steps = [
            f"已读取 PR #{input_data.get('number')}，状态 {input_data.get('state') or 'unknown'}，作者 {input_data.get('author') or 'unknown'}。",
            f"已扫描 {len(files)} 个变更文件，合计 +{total_additions} / -{total_deletions}。",
            f"已识别 {len(risky_files)} 个敏感文件、{len(test_files)} 个测试相关文件、{len(comments)} 条 review comment。",
            f"已纳入 {len(context.get('code_references') or [])} 条代码线索、{len(context.get('project_documents') or [])} 条项目文档、{failed_ci} 个失败 CI 信号。",
            f"已输出结论：{merge_recommendation}。",
        ]

        return PRReviewOutput(
            summary=f"PR #{input_data.get('number')} 修改了 {len(files)} 个文件，主要集中在 {', '.join(filenames[:3]) or '未提供文件'}。",
            plan=plan,
            executed_steps=executed_steps,
            merge_recommendation=merge_recommendation,
            recommendation_reason=recommendation_reason,
            review_findings=review_findings,
            key_changes=[f"更新 {name}" for name in filenames[:5]],
            risk_points=risk_points if files else ["缺少 changed files，风险判断有限"],
            blocking_issues=blocking_issues,
            review_checklist=["确认接口兼容性", "确认错误处理路径", "确认测试覆盖关键分支", "确认 review comments 已回应"],
            test_suggestions=self._test_suggestions(filenames, risky_files, failed_ci),
            # 有敏感文件就只列敏感文件（更重要），否则列全部
            files_need_attention=(risky_files or filenames)[:5],
            # 只取前 5 条评论正文（太多会刷屏）
            review_comments=[str(comment.get("body") or "").strip() for comment in comments[:5] if str(comment.get("body") or "").strip()],
            confidence=self._estimate_confidence(files, context, failed_ci, blocking_issues),
        ).model_dump()

    def _merge_with_fallback(self, output: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
        """把 LLM 结果合并进规则结果（★ 含一致性修正）。

        【调用链】_normalize_review_findings() / _blocking_issues_from_findings() → 本函数
        【被谁调用】run()

        【数据流向】
            入：output（LLM）、fallback（规则）
            出：合并后的 dict

        【四步合并】
            ① 基础合并 + 空值兜底（列表/文本分别判空）
            ② 归一化 review_findings（模型返回的格式可能不标准）
            ③ ★ 从 findings 反推 blocking_issues（保证两者一致）
            ④ ★ 一致性修正：有阻塞项却写"建议合入" → 改成"修改后合入"

        【★ 第 ③ 步为什么必要】
            模型可能只给了 review_findings（P1 发现），但 blocking_issues 是空的。
            这样前端"阻塞项"区域是空的，用户会以为"没有阻塞" —— 严重误导。
            所以从 findings 里把所有 P1/P2/blocking 的项反推成阻塞项。

        【★ 第 ④ 步为什么必要】
            prompt 里已经要求"有 P1/P2 不能建议合入"，但模型可能不遵守。
            代码层强制修正，保证输出永远不会自相矛盾。
        """
        # ① 基础合并
        merged = {**fallback, **output}

        # ① 列表字段空值兜底
        for key in [
            "plan",
            "executed_steps",
            "review_findings",
            "key_changes",
            "risk_points",
            "blocking_issues",
            "review_checklist",
            "test_suggestions",
            "files_need_attention",
            "review_comments",
        ]:
            if not merged.get(key):
                merged[key] = fallback[key]

        # ① 文本字段空值兜底
        for key in ["merge_recommendation", "recommendation_reason", "summary"]:
            if not str(merged.get(key) or "").strip():
                merged[key] = fallback[key]

        # ② 归一化 findings（模型可能返回不标准的结构）
        merged["review_findings"] = self._normalize_review_findings(merged.get("review_findings"), fallback)

        # ③ 从 findings 反推阻塞项
        merged["blocking_issues"] = self._blocking_issues_from_findings(merged.get("blocking_issues"), merged["review_findings"])

        # ④ ★ 一致性修正：有阻塞项就不能"建议合入"
        if merged["blocking_issues"] and merged.get("merge_recommendation") == "建议合入":
            merged["merge_recommendation"] = "修改后合入"
            merged["recommendation_reason"] = "存在 P1/P2 审查发现，完成修复或确认后才能放行。"
        return merged

    def _finding(
        self,
        severity: str,
        title: str,
        evidence: str,
        required_action: str,
        *,
        blocking: bool | None = None,
    ) -> dict[str, Any]:
        """构造一条分级发现。

        【调用链】被 _fallback_review() / _normalize_review_findings() 调用

        【★ blocking 的默认值由 severity 推导】
            P1/P2 → True（必须修）
            P3    → False（建议）
            显式传 blocking 可以覆盖（P3 也可以标成阻塞，但很少用）。

        这是"P1/P2 必须 blocking"这条规则的第二重保障。
        """
        return {
            "severity": severity,
            "title": title,
            "evidence": evidence,
            "required_action": required_action,
            "blocking": severity in {"P1", "P2"} if blocking is None else blocking,
        }

    def _normalize_review_findings(self, items: Any, fallback: dict[str, Any]) -> list[dict[str, Any]]:
        """归一化模型返回的 findings（★ 容错各种字段命名）。

        【调用链】_finding() → 本函数
        【被谁调用】_merge_with_fallback()

        【数据流向】
            入：items（模型返回的，可能不标准）、fallback
            出：标准格式的 findings 列表

        【★ 三处容错，都来自真实经验】
            ① severity 不合法（模型写了 "high"/"严重"）→ 降级成 P3
               为什么不猜？猜错更危险（把 P3 当 P1 会误报阻塞）。
            ② 字段名不统一（title/summary/message 都可能是标题）
               三个都试，取第一个有值的。
            ③ title 和 evidence 都为空 → 丢掉（这条没有任何信息量）

        【兜底：归一化后一条都没有 → 用 fallback 的】
            宁可给规则的发现，也不要给空的。
        """
        source = items if isinstance(items, list) and items else fallback.get("review_findings") or []
        normalized: list[dict[str, Any]] = []

        for item in source:
            if not isinstance(item, dict):
                continue

            # ① severity 容错
            severity = str(item.get("severity") or "").strip().upper()
            if severity not in {"P1", "P2", "P3"}:
                severity = "P3"

            # ② 字段名容错
            title = str(item.get("title") or item.get("summary") or item.get("message") or "").strip()
            evidence = str(item.get("evidence") or item.get("reason") or item.get("description") or "").strip()
            required_action = str(item.get("required_action") or item.get("action") or item.get("recommendation") or "").strip()

            # ③ 完全没信息 → 丢掉
            if not title and not evidence:
                continue

            # blocking 默认按 severity 推导；模型显式给了就尊重（但必须是 True）
            raw_blocking = item.get("blocking")
            blocking = severity in {"P1", "P2"} if raw_blocking is None else raw_blocking is True

            normalized.append(
                self._finding(
                    severity,
                    # title 为空就用 evidence 前 80 字（保证有标题）
                    title or evidence[:80],
                    evidence or title,
                    # 处理动作为空就给一句默认的
                    required_action or "请补充可复核的处理说明。",
                    blocking=blocking,
                )
            )
        return normalized or fallback.get("review_findings") or []

    def _blocking_issues_from_findings(self, blocking_issues: Any, review_findings: list[dict[str, Any]]) -> list[str]:
        """从 findings 反推阻塞项（保证"阻塞项"和"分级发现"一致）。

        【调用链】被 _merge_with_fallback() 调用

        【数据流向】
            入：blocking_issues（可能是模型给的）、review_findings（已归一化）
            出：合并后的阻塞项列表（去重保序）

        【★ 为什么要做这一步】
            模型常见的问题：只给 findings 不给 blocking_issues。
            结果前端"阻塞项"区域空白，但明明有 3 个 P1 —— 用户会误判"可以合"。

        【message 的拼法】
            "标题：处理动作"（两者都有时）
            只有标题 → 标题
            只有动作 → 动作
            都没有 → 跳过（不产生空字符串）
        """
        rows = [str(item).strip() for item in blocking_issues or [] if str(item).strip()]
        for finding in review_findings:
            # P1/P2 或显式 blocking → 都算阻塞
            if finding.get("severity") in {"P1", "P2"} or finding.get("blocking"):
                action = str(finding.get("required_action") or "").strip()
                title = str(finding.get("title") or "").strip()
                message = f"{title}：{action}" if title and action else title or action
                if message:
                    rows.append(message)
        # 去重保序（模型给的 + 反推的可能重复）
        return list(dict.fromkeys(rows))

    def _choose_recommendation(
        self,
        input_data: dict[str, Any],
        files: list[dict[str, Any]],
        risk_points: list[str],
        blocking_issues: list[str],
        failed_ci: int,
    ) -> tuple[str, str]:
        """决定合并建议（★ 七级判定，从上到下命中即返回）。

        【调用链】被 _fallback_review() 调用

        【数据流向】
            入：PR 元信息、文件、风险点、阻塞项、失败 CI 数
            出：(建议, 理由)

        【★ 七级顺序，从"最确定"到"最宽松"】
            ① 已关闭且未合并        → 拒绝（没有可合的东西）
            ② 标题/描述有 WIP 信号  → 暂缓（作者自己说别合）
            ③ 有失败 CI             → 暂缓（质量门禁没过）
            ④ 没有 changed files    → 暂缓（不知道改了什么）
            ⑤ 有阻塞项              → 修改后合入
            ⑥ 有明确风险点          → 修改后合入
            ⑦ 都没有                → 建议合入

        【★ 为什么"作者说 WIP"要优先于"有失败 CI"】
            WIP 是作者主动的意图（"还没写完"）。
            即使 CI 通过了，也不该合 —— 尊重作者意图。
            所以 WIP 判定放在 CI 之前。

        【为什么 ④"没有文件"也是暂缓而不是拒绝】
            可能是 API 没返回 files（网络问题），不代表 PR 有问题。
            暂缓让人工看一眼，比直接拒绝稳妥。
        """
        state = str(input_data.get("state") or "").lower()
        # 标题 + 描述拼起来找信号词
        text = f"{input_data.get('title', '')} {input_data.get('body', '')}".lower()

        if state == "closed" and not input_data.get("merged_at"):
            return "拒绝", "PR 已关闭且没有合并记录，当前不建议继续合入。"

        if any(token in text for token in ["do not merge", "wip", "draft", "blocked", "暂缓", "不要合并"]):
            return "暂缓", "PR 描述中存在 WIP/blocked/暂缓信号，需要等待作者更新。"

        if failed_ci:
            return "暂缓", "存在失败 CI 信号，合并前需要先确认根因并恢复质量门禁。"

        if not files:
            return "暂缓", "缺少 changed files，无法判断真实影响范围。"

        if blocking_issues:
            return "修改后合入", "存在可修复的阻塞项，完成测试或说明补充后可重新评估。"

        # ⑥ 风险点里出现这三类关键词 → 要改
        if risk_points and any(token in " ".join(risk_points) for token in ["高敏感", "较大", "review comment"]):
            return "修改后合入", "PR 有明确风险点，建议先完成重点 review 和验证再合入。"

        return "建议合入", "未发现明显阻塞项，按 checklist 完成最终确认后可以合入。"

    def _test_suggestions(self, filenames: list[str], risky_files: list[str], failed_ci: int) -> list[str]:
        """按技术栈给测试建议。

        【调用链】被 _fallback_review() 调用

        【规则】
            前端文件（frontend/.tsx/.ts/next/react）→ 加前端类型检查/构建
            后端文件（backend/.py/api/db）          → 加 pytest + 接口/DB 边界
            有敏感文件                              → 加回归测试或人工验证
            有失败 CI                               → 先修 CI 再跑全流水线

        【为什么第一条永远是"运行相关单元测试和集成测试"】
            这是最基础的，任何 PR 都适用。
        """
        suggestions = ["运行相关单元测试和集成测试"]
        lowered = " ".join(filenames).lower()

        if any(token in lowered for token in ["frontend", ".tsx", ".ts", "next", "react"]):
            suggestions.append("运行前端类型检查/构建，并覆盖关键页面交互")
        if any(token in lowered for token in ["backend", ".py", "api", "db"]):
            suggestions.append("运行后端 pytest，并覆盖接口或数据库边界")
        if risky_files:
            suggestions.append("对敏感文件补充回归测试或人工验证记录")
        if failed_ci:
            suggestions.append("先修复失败 CI，再重新运行完整流水线")

        # 去重（"运行单元测试"可能被上面的分支重复加）
        return list(dict.fromkeys(suggestions))

    def _estimate_confidence(self, files: list[dict[str, Any]], context: dict[str, Any], failed_ci: int, blocking_issues: list[str]) -> float:
        """按"证据充足度"算置信度（0.38~0.88）。

        【调用链】被 _fallback_review() 调用

        【加分表】
            基础              0.48
            有变更文件        +0.12   ← 最重要（没文件等于没审）
            有代码线索        +0.08
            有项目文档        +0.06
            有会话证据        +0.05
            有失败 CI         +0.04
            有阻塞项          +0.04

        【为什么上限 0.88（比 CI Agent 的 0.86 高一点）】
            PR 审查有明确的 diff 可读，确定性比"从日志猜根因"高一些。

        【为什么"有阻塞项"也加分】
            不是"问题多所以更确定"，而是"识别出了具体问题"说明证据充足。
            全空的审查反而置信度低（可能没读到东西）。
        """
        score = 0.48
        if files:
            score += 0.12
        if context.get("code_references"):
            score += 0.08
        if context.get("project_documents"):
            score += 0.06
        if context.get("conversation_evidence"):
            score += 0.05
        if failed_ci:
            score += 0.04
        if blocking_issues:
            score += 0.04
        return round(max(0.38, min(0.88, score)), 2)
