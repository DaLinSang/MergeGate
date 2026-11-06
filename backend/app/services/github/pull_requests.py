"""
GitHub Pull Request 接口
=======================

【业务定位】
    PR 数据的 HTTP 适配层：列 PR、查改动文件、查行级评论。

【数据流向】
    上游（谁调它）：
        services/repos_sync.py → 同步 pull_requests / pr_files / pr_review_comments 三张表
    下游（它调谁）：
        services/github/client.py → GitHubClient（统一 HTTP 层）

【核心规则】
    1. 只发请求取字段，不做业务判断。
    2. state 必须传 "all" —— 只要 open 会漏掉已合并/已关闭的 PR（见下方说明）。

【本文件函数地图】
    list_pull_requests()                  列 PR（★ state=all 的原因见函数内）
    list_pull_request_files()             查改动文件（含 diff patch）
    list_pull_request_review_comments()   查行级评论
"""

from app.services.github.client import GitHubClient


async def list_pull_requests(owner: str, repo: str, token: str | None = None, limit: int = 30, base_url: str | None = None) -> list[dict]:
    """列出一个仓库的 PR。

    【调用链】GitHubClient.get() → 本函数
    【被谁调用】services/repos_sync.py

    【数据流向】
        入：owner/repo、token、limit、base_url
        出：PR 数组（原始结构）

    【★ 为什么 state 必须是 "all"】
        GitHub 默认只返回 state=open 的 PR。
        但 PR 审查 Agent 最常分析的是"已合并的 PR"（学习历史决策），
        如果只同步 open，本地库里永远没有已合并的 PR —— 历史全丢。

        这和 Issue 的 state=all 是同一个道理（见 github/issues.py）。
    """
    client = GitHubClient(token, base_url)
    return await client.get(f"/repos/{owner}/{repo}/pulls", {"state": "all", "per_page": limit})


async def list_pull_request_files(owner: str, repo: str, number: int, token: str | None = None, base_url: str | None = None) -> list[dict]:
    """查 PR 改动了哪些文件（含每个文件的 diff patch）。

    【调用链】GitHubClient.get() → 本函数
    【被谁调用】services/repos_sync.py（写入 pr_files，整组替换）

    【数据流向】
        入：number（用户看到的 PR 编号，不是 GitHub 内部 ID）
        出：files 数组，每项含 filename / status / additions / deletions / patch

    【注意两个局限】
        ① per_page=100：改动超过 100 个文件的 PR 会被截断（GitHub 上限）
        ② patch 可能为空：超大文件的 diff 会被 GitHub 省略
           所以下游 pr_files.patch 允许 NULL
    """
    client = GitHubClient(token, base_url)
    return await client.get(f"/repos/{owner}/{repo}/pulls/{number}/files", {"per_page": 100})


async def list_pull_request_review_comments(owner: str, repo: str, number: int, token: str | None = None, base_url: str | None = None) -> list[dict]:
    """查 PR 的行级 Review 评论（针对某一行 diff 的评论）。

    【调用链】GitHubClient.get() → 本函数
    【被谁调用】services/repos_sync.py（写入 pr_review_comments，整组替换）

    【数据流向】
        入：number
        出：comments 数组，每项含 path / line / body / user / created_at

    【这里不含什么】
        - 不含普通 Issue 评论（那是另一个接口 /issues/{n}/comments）
        - 不含一次 Review 的整体结论（那是 /pulls/{n}/reviews）
        本项目只索引行级评论，因为它们的定位信息最精确。
    """
    client = GitHubClient(token, base_url)
    return await client.get(f"/repos/{owner}/{repo}/pulls/{number}/comments", {"per_page": 100})
