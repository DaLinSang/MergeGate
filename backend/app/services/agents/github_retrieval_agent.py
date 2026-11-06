"""
GitHub 数据检索 Agent
====================

【业务定位】
    把 services/github/ 下的 7 个取数函数，包成一个"按 task 分发"的 Agent 接口。
    让工作流/工具调用方只需传一个 task 名，不用自己拼多个请求。

【数据流向】
    上游（谁调它）：
        services/agents/workflow_orchestrator.py  工作流里的 github_fetch 任务
        services/agents/chat_agent.py             需要拉 GitHub 数据时
    下游（它调谁）：
        services/github/issues.py          list_issues
        services/github/pull_requests.py   list_pull_requests / _files / _review_comments
        services/github/actions.py         list_workflow_runs / _run_jobs / _run_logs

【核心规则】
    1. 纯转发，不做任何业务加工（原始结构直接返回，由上游决定怎么映射）。
    2. task 不认识时，退化成"仓库快照"（issues + PRs + runs 各取一批），
       保证任何情况下都有数据返回，不会返回空。

【本文件函数地图】
    GitHubRetrievalAgent.run()  ★ 按 task 分发到 7 个取数函数之一
"""

from typing import Any

from app.services.agents.base import BaseAgent
from app.services.github.actions import get_workflow_run_logs, list_workflow_run_jobs, list_workflow_runs
from app.services.github.issues import list_issues
from app.services.github.pull_requests import list_pull_request_files, list_pull_request_review_comments, list_pull_requests


class GitHubRetrievalAgent(BaseAgent):
    """GitHub 取数 Agent（7 种 task 的分发器）。

    【为什么不做缓存】
        上层（repos_sync）已经把数据同步到本地库了。
        这个 Agent 是"实时取数"通道，用于同步之外的场景（比如刚创建的 PR 想立刻看）。
        加缓存会让"实时"这个语义变模糊。
    """

    name = "github_retrieval_agent"
    description = "检索 GitHub Issue、PR、diff、Review 评论、workflow run 和失败日志。"
    available_tools = [
        "list_issues",
        "list_pull_requests",
        "list_pull_request_files",
        "list_pull_request_review_comments",
        "list_workflow_runs",
        "list_workflow_run_jobs",
        "get_workflow_run_logs",
    ]

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """按 task 分发取数。

        【调用链】追这个函数会进入（取决于 task）：
            task="issues"              → github/issues.list_issues()
            task="pull_requests"       → github/pull_requests.list_pull_requests()
            task="pull_request_detail" → list_pull_request_files() + list_pull_request_review_comments()
            task="workflow_runs"       → github/actions.list_workflow_runs()
            task="workflow_run_failure"→ list_workflow_run_jobs() + get_workflow_run_logs()
            其他（默认）                → 三个 list_* 各取一批（仓库快照）

        【数据流向】
            入：input_data.{owner, repo, token, base_url, limit, task, number/run_id}
            出：dict，键名随 task 变化（issues / pull_requests / files+review_comments / ...）

        【为什么 owner 和 repo 用 [] 而不是 get()】
            这两个是必填参数，缺了没法发请求。
            用 [] 让它 KeyError 立刻暴露，而不是拼出 /repos/None/None 这种无效 URL。

        【为什么默认分支是"仓库快照"】
            task 拼错或没传时，返回三类数据的概览比返回空更有用 ——
            调用方至少能看到仓库里有什么。
        """
        owner = input_data["owner"]
        repo = input_data["repo"]
        token = input_data.get("token")
        base_url = input_data.get("base_url")
        limit = int(input_data.get("limit", 30))
        task = input_data.get("task", "repo_snapshot")

        # ── 单个资源 ──
        if task == "issues":
            return {"issues": await list_issues(owner, repo, token=token, limit=limit, base_url=base_url)}
        if task == "pull_requests":
            return {"pull_requests": await list_pull_requests(owner, repo, token=token, limit=limit, base_url=base_url)}

        # ── PR 详情：文件 + 行级评论（PR 审查 Agent 的核心输入）──
        if task == "pull_request_detail":
            number = int(input_data["number"])
            files = await list_pull_request_files(owner, repo, number, token=token, base_url=base_url)
            comments = await list_pull_request_review_comments(owner, repo, number, token=token, base_url=base_url)
            return {"files": files, "review_comments": comments}

        # ── CI ──
        if task == "workflow_runs":
            return {"workflow_runs": await list_workflow_runs(owner, repo, token=token, limit=limit, base_url=base_url)}
        if task == "workflow_run_failure":
            run_id = int(input_data["run_id"])
            # jobs 给"哪一步失败了"，logs 给"为什么失败" —— 两个都要
            jobs = await list_workflow_run_jobs(owner, repo, run_id, token=token, base_url=base_url)
            logs_text = await get_workflow_run_logs(owner, repo, run_id, token=token, base_url=base_url)
            return {"jobs": jobs, "logs_text": logs_text}

        # ── 默认：仓库快照（三类各取一批）──
        issues = await list_issues(owner, repo, token=token, limit=limit, base_url=base_url)
        pull_requests = await list_pull_requests(owner, repo, token=token, limit=limit, base_url=base_url)
        workflow_runs = await list_workflow_runs(owner, repo, token=token, limit=limit, base_url=base_url)
        return {"issues": issues, "pull_requests": pull_requests, "workflow_runs": workflow_runs}
