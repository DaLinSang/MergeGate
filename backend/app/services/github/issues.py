"""
GitHub Issue 读取
=================

【业务定位】
    调 GitHub 的 /issues 接口，把结果过滤成"纯 Issue 列表"。

【数据流向】
    上游（谁调它）：
        services/repos_sync.py（Issue 同步分支）
        services/agents/github_retrieval_agent.py（实时取数）
    下游（它调谁）：
        services/github/client.py → GitHubClient（统一 HTTP 层）

【核心规则】
    1. state 必须传 "all"（不能只拉 open，否则关闭状态永远同步不过来）。
    2. 必须过滤掉带 pull_request 字段的项（GitHub 把 PR 也当 Issue 返回）。

【本文件函数地图】
    list_issues()  ★ 唯一函数：拉列表 + 过滤 PR
"""

from app.services.github.client import GitHubClient


async def list_issues(owner: str, repo: str, token: str | None = None, limit: int = 30, base_url: str | None = None) -> list[dict]:
    """拉取指定仓库的 Issue 列表（已剔除 PR）。

    【调用链】GitHubClient.get() → 列表推导过滤 → 本函数
    【被谁调用】services/repos_sync.py / github_retrieval_agent.py

    参数：
        owner    → 仓库所属账号/组织，如 "openai"
        repo     → 仓库名，如 "openai-quickstart-python"
        token    → 明文 token（None 则用全局配置）
        limit    → 最多拉多少条（GitHub 单页上限 100）
        base_url → API 根地址（企业版 GitHub 时用）

    返回：
        GitHub 原始 JSON 的 list，每一项是一个 Issue 的 dict。

    【数据流向】
        入：owner/repo、token、limit、base_url
        出：Issue dict 列表（原始结构，未做任何字段映射）

    两个关键设计（都容易踩坑）：

    ① 为什么用 state="all" 而不是 "open"：
       如果只拉 open 的，Issue 在 GitHub 上被关闭后，本地数据库里那条
       永远还是 open —— 状态更新不过来。必须拉全部状态，
       靠 github_issue_id 匹配后覆盖本地字段。

    ② 为什么要过滤 "pull_request"：
       GitHub 的 /issues 接口【会把 PR 也当成 Issue 返回】，
       这是 GitHub API 的历史遗留设计。
       判断依据：PR 类型的 item 会多一个 pull_request 字段。
       如果不过滤，同一个 PR 会被存两遍（issues 表一份 + pull_requests 表一份）。

    【为什么这个文件只有几行】
        因为重复逻辑（URL 拼接、认证、超时、错误）都已经被 client.py 抽走了。
        这里只剩下"声明路径 + 挑字段"两件事——这正是分层的价值：
        新加一个 GitHub 资源，只需要写这么短的一个函数。
    """
    client = GitHubClient(token, base_url)
    items = await client.get(f"/repos/{owner}/{repo}/issues", {"state": "all", "per_page": limit})
    # 列表推导式过滤：只保留没有 pull_request 字段的项（即真正的 Issue）
    return [item for item in items if "pull_request" not in item]
