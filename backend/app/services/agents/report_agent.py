"""
周报生成 Agent
=============

【业务定位】
    把一段时间内的 Issue / PR / CI 数据，渲染成一份 Markdown 工程周报。

    【为什么这个 Agent 特别简单】
        它不调 LLM —— 纯模板填充。
        因为周报的结构是固定的（概览 / 完成 / 进行中 / 风险 / 建议），
        让模型写反而会"发挥"，把数字说错、把标题改乱。
        模板 + 真实数据 = 可信、可复现、零成本。

【数据流向】
    上游（谁调它）：
        services/agents/workflow_orchestrator.py  工作流里的 report 任务
        api/routes/reports.py                     手动生成周报
    下游（它调谁）：
        无 —— 输入数据已经由调用方准备好（三个 *_by_time_range 工具的结果）

【核心规则】
    1. 只用输入里的数据，不查库、不调模型（纯函数式，易测）。
    2. 数字优先用 metrics（调用方算好的），没有才用列表长度兜底。
    3. 每个列表最多展示 5~8 条（周报要能读，不是数据导出）。

【本文件函数地图】
    ReportAgent.run()  ★ 唯一方法：分类 → 统计 → 拼 Markdown
"""

from typing import Any

from app.services.agents.base import BaseAgent


class ReportAgent(BaseAgent):
    """周报生成 Agent（纯模板，不调 LLM）。"""

    name = "report_agent"
    description = "根据仓库活动生成工程周报。"
    available_tools = ["list_issues_by_time_range", "list_prs_by_time_range", "list_failed_workflow_runs"]

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """渲染周报。

        【调用链】无 —— 纯字符串拼接
        【被谁调用】workflow_orchestrator（report 任务）、api/routes/reports.py

        【数据流向】
            入：input_data.{issues, pull_requests, workflow_runs, metrics}
                context.{repo_name, repo_id, start_date, end_date}
            出：{"report_markdown": "..."}（前端直接渲染 Markdown）

        【为什么 metrics 优先于 len()】
            调用方可能按"全部数据"算指标，而传进来的列表只是"前 N 条样例"。
            这时用 len() 会算少。metrics 是权威数字，列表长度只是兜底。
        """
        issues = input_data.get("issues", [])
        prs = input_data.get("pull_requests", [])
        ci_runs = input_data.get("workflow_runs", [])
        metrics = input_data.get("metrics", {})

        # ── 步骤 1：分四类（周报的骨架就是这四类）──
        failed_ci = [run for run in ci_runs if run.get("conclusion") == "failure"]
        merged_prs = [pr for pr in prs if pr.get("merged_at")]        # 有 merged_at = 已合并
        open_prs = [pr for pr in prs if pr.get("state") == "open"]
        closed_issues = [issue for issue in issues if issue.get("state") == "closed"]

        # ── 步骤 2：取标题和日期范围（都有兜底，避免出现 "Repo " 这种空值）──
        repo_name = context.get("repo_name") or f"Repo {context.get('repo_id', '')}".strip()
        date_range = f"{context.get('start_date', '')} 至 {context.get('end_date', '')}".strip()

        # 最多列 8 个编号，没有就写"暂无"（不要留空，读者会以为出错了）
        merged_pr_numbers = ", ".join(f"#{pr.get('number')}" for pr in merged_prs[:8]) or "暂无"
        closed_issue_numbers = ", ".join(f"#{issue.get('number')}" for issue in closed_issues[:8]) or "暂无"

        # ── 步骤 3：拼 Markdown（六段式固定结构）──
        markdown = (
            f"# {repo_name} 研发周报\n\n"
            f"> 时间范围：{date_range}\n\n"
            "## 数据概览\n"
            f"- Issues：{metrics.get('issues', len(issues))} 个\n"
            f"- PRs：{metrics.get('pull_requests', len(prs))} 个，其中 Merged PR {metrics.get('merged_prs', len(merged_prs))} 个，Open PR {metrics.get('open_prs', len(open_prs))} 个\n"
            f"- 失败 CI：{metrics.get('failed_ci', len(failed_ci))} 次\n\n"
            "## 本周完成\n"
            f"- 合并 PR {len(merged_prs)} 个：{merged_pr_numbers}。\n"
            f"- 关闭 Issue {len(closed_issues)} 个：{closed_issue_numbers}。\n\n"
            "## 进行中\n"
            f"- 当前仍有 {len(open_prs)} 个 Open PR 需要跟进。\n"
            + "\n".join(f"- #{pr.get('number')} {pr.get('title')}" for pr in open_prs[:5])
            + ("\n" if open_prs else "- 暂无 Open PR。\n")
            + "\n"
            "## 风险与阻塞\n"
            f"- 失败 CI {len(failed_ci)} 次，建议优先排查重复失败的 workflow。\n"
            + ("- 本期没有同步到失败 CI。\n" if not failed_ci else "")
            + "\n"
            "## 需要关注的 PR\n"
            + "\n".join(f"- #{pr.get('number')} {pr.get('title')}" for pr in prs[:5])
            + "\n\n## 失败 CI\n"
            + "\n".join(f"- {run.get('name')}: {run.get('conclusion')}" for run in failed_ci[:5])
            + "\n\n## 下周建议\n"
            "- 补充关键模块测试覆盖。\n- 对高风险 PR 提前拆分 Review。\n"
        )
        return {"report_markdown": markdown}
