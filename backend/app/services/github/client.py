"""
GitHub HTTP 客户端
==================

【业务定位】
    业务上需要一个"能读也能写 GitHub"的统一出口。
    读：把 Issue/PR/CI 事实搬进本地（供 Agent 取证）。
    写：AI 生成的草稿经人工确认后，通过这里真正改 GitHub。
    所以它是"业务动作落到 GitHub 上"的唯一通道。

【数据流向】
    上游：
        services/github/issues.py / pull_requests.py / actions.py  → 读接口
        api/routes/repos.py                                        → 验证仓库存在
        api/routes/action_drafts.py                                → 写接口（人工确认后）
    下游：
        GitHub REST API

【核心业务规则】
    1. 仓库级 token 优先于全局 token（不同仓库可能属于不同组织）。
    2. 没配 token 时以匿名身份请求 —— 只能读公开仓库，额度 60 次/小时。
    3. 这里的写方法【只】给"人工确认后的草稿执行"用，Agent 不能直接调。

【技术要点】
    URL 拼接、认证头、API 版本、超时、重定向、错误抛出全部收口在此，
    资源函数（issues.py 等）因此只需 7-35 行。
"""

from typing import Any

import httpx

from app.core.config import settings


class GitHubClient:
    """GitHub REST API 客户端。

    典型用法：
        client = GitHubClient(token)                 # 用传入的 token
        data = await client.get("/repos/owner/repo") # 读
        await client.create_issue_comment(...)       # 写
    """

    def __init__(self, token: str | None = None, base_url: str | None = None) -> None:
        """初始化客户端。

        参数：
            token    → 明文 token。传 None 时回退到全局 settings.github_token
            base_url → API 根地址。传 None 时用 settings.github_api_base_url

        为什么 token 要支持"外部传入"：
            不同仓库可能属于不同账号/组织，需要不同的权限。
            调用方（repos_sync）会先解密该仓库自己的 token 再传进来，
            所以仓库级 token 的优先级高于全局 GITHUB_TOKEN。
        """
        self.token = token or settings.github_token
        # rstrip("/") 防止调用方传了尾部斜杠，导致拼出 "//repos/..." 这种双斜杠路径
        self.base_url = (base_url or settings.github_api_base_url).rstrip("/")

    def _headers(self) -> dict[str, str]:
        """构造每次请求都要带的 HTTP 头。

        三个头的用途：
            Accept                 → 告诉 GitHub 返回 v3 REST 的 JSON 格式
            X-GitHub-Api-Version   → 锁定 API 版本，避免 GitHub 升级后行为悄悄变化
            Authorization          → Bearer token 认证（没 token 就不加，以匿名身份请求）

        为什么没有 token 也能请求：
            公开仓库的读接口允许匿名访问，只是额度很低（60 次/小时）。
            这让项目在没有配 token 时也能跑起来做演示。
        """
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    # ────────────────────────────────────────────────────────────
    # 通用 HTTP 方法（读）
    # ────────────────────────────────────────────────────────────

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """发 GET 请求，返回解析后的 JSON。

        参数：
            path   → 以 / 开头的路径，如 "/repos/owner/repo"
            params → URL 查询参数，如 {"state": "all", "per_page": 30}

        raise_for_status() 的作用：
            4xx/5xx 时抛 httpx.HTTPStatusError。
            调用方（repos_sync）会捕获它，转成带上下文的 HTTPException，
            这样前端能看到"GitHub issues sync failed: 401 ..."这种明确信息，
            而不是一个光秃秃的 500。

        timeout=30：
            普通 JSON 接口 30 秒足够。太短会在网络差时误判失败。
        """
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(f"{self.base_url}{path}", headers=self._headers(), params=params)
            response.raise_for_status()
            return response.json()

    async def get_bytes(self, path: str, params: dict[str, Any] | None = None) -> bytes:
        """发 GET 请求，返回原始字节（不解码 JSON）。

        什么时候用：
            下载 CI 日志。GitHub 的 /actions/runs/{id}/logs 返回的是一个 zip 压缩包，
            不是 JSON，所以不能走 get()。

        timeout=60：
            日志包可能几 MB，比普通接口慢，所以放宽到 60 秒。
        """
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            response = await client.get(f"{self.base_url}{path}", headers=self._headers(), params=params)
            response.raise_for_status()
            return response.content

    # ────────────────────────────────────────────────────────────
    # 通用 HTTP 方法（写）
    # ────────────────────────────────────────────────────────────
    # ⚠️ 下面这两个方法【不会】被 Agent 直接调用。
    # 它们只在用户确认了 ActionDraft 之后，由 api/routes/action_drafts.py 执行。
    # 这是项目的安全设计：AI 只能生成草稿，写操作必须人工确认。

    async def post(self, path: str, payload: dict[str, Any]) -> Any:
        """发 POST 请求（创建资源），返回 JSON。"""
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.post(f"{self.base_url}{path}", headers=self._headers(), json=payload)
            response.raise_for_status()
            return response.json()

    async def patch(self, path: str, payload: dict[str, Any]) -> Any:
        """发 PATCH 请求（部分更新资源），返回 JSON。"""
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.patch(f"{self.base_url}{path}", headers=self._headers(), json=payload)
            response.raise_for_status()
            return response.json()

    # ────────────────────────────────────────────────────────────
    # 业务语义方法（对上面通用方法的薄封装）
    # ────────────────────────────────────────────────────────────
    # 这些方法本身不含逻辑，只是把"路径长什么样"这件事集中起来，
    # 避免各个调用方自己拼字符串拼错。

    async def get_repo(self, owner: str, repo: str) -> dict[str, Any]:
        """获取仓库元信息（描述、默认分支、clone 地址、GitHub ID）。

        用途：连接仓库时先调它，既能验证仓库真实存在，
        又能顺手拿到 description / default_branch / clone_url 存进数据库。
        """
        return await self.get(f"/repos/{owner}/{repo}")

    async def create_issue_comment(self, owner: str, repo: str, issue_number: int, body: str) -> dict[str, Any]:
        """在 Issue（或 PR）下发表评论。

        注意：GitHub 的 Issue 和 PR 共用同一套评论接口，
        所以这个方法是给两者通用的。
        """
        return await self.post(f"/repos/{owner}/{repo}/issues/{issue_number}/comments", {"body": body})

    async def create_issue(self, owner: str, repo: str, title: str, body: str) -> dict[str, Any]:
        """创建新 Issue。"""
        return await self.post(f"/repos/{owner}/{repo}/issues", {"title": title, "body": body})

    async def close_issue(self, owner: str, repo: str, issue_number: int) -> dict[str, Any]:
        """关闭 Issue。

        为什么用 PATCH 而不是 DELETE：
            GitHub 里"关闭"是状态变更（state=closed），不是删除。
            关闭后还能重新打开（state=open）。
        """
        return await self.patch(f"/repos/{owner}/{repo}/issues/{issue_number}", {"state": "closed"})

    async def add_issue_labels(self, owner: str, repo: str, issue_number: int, labels: list[str]) -> dict[str, Any]:
        """给 Issue 添加标签（追加，不会覆盖已有标签）。"""
        return await self.post(f"/repos/{owner}/{repo}/issues/{issue_number}/labels", {"labels": labels})
