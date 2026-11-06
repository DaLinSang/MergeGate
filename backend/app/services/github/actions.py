"""
GitHub Actions（CI）接口
=======================

【业务定位】
    GitHub CI 数据的 HTTP 适配层：列 workflow run、查 jobs、下载日志。

【数据流向】
    上游（谁调它）：
        services/repos_sync.py → 同步 workflow_runs 表
    下游（它调谁）：
        services/github/client.py → GitHubClient（统一 HTTP 层）

【核心规则】
    1. 本文件只负责"发请求 + 取字段"，不做任何业务判断（不决定存哪张表）。
    2. 日志接口返回的是 zip，必须解压；解压失败要能降级成纯文本。

【本文件函数地图】
    list_workflow_runs()        列 CI 运行
    list_workflow_run_jobs()    查某次运行的 jobs
    get_workflow_run_logs()     下载并解压日志（★ 唯一有逻辑的函数）
"""

from io import BytesIO
from zipfile import BadZipFile, ZipFile

from app.services.github.client import GitHubClient


async def list_workflow_runs(owner: str, repo: str, token: str | None = None, limit: int = 30, base_url: str | None = None) -> list[dict]:
    """列出一个仓库的 CI 运行记录。

    【调用链】GitHubClient.get() → 本函数
    【被谁调用】services/repos_sync.py

    【数据流向】
        入：owner/repo、token、limit、base_url
        出：GitHub 的 workflow_runs 数组（原始结构，未做任何映射）

    注意：GitHub 默认只返回最近 30 条。要更多得翻页（本项目按需取，不做全量）。
    """
    client = GitHubClient(token, base_url)
    payload = await client.get(f"/repos/{owner}/{repo}/actions/runs", {"per_page": limit})
    return payload.get("workflow_runs", [])


async def list_workflow_run_jobs(owner: str, repo: str, run_id: int, token: str | None = None, base_url: str | None = None) -> list[dict]:
    """查某次 CI 运行包含的所有 job（含每个 job 的 steps 和结论）。

    【调用链】GitHubClient.get() → 本函数
    【被谁调用】services/repos_sync.py（存进 workflow_runs.jobs 这个 JSON 列）

    【数据流向】
        入：run_id（GitHub 的运行 ID，不是我们的 UUID）
        出：jobs 数组

    【为什么 per_page 固定 100】
        一个 workflow 通常几个到十几个 job，100 足够覆盖；
        固定值可以避免调用方传错参数。
    """
    client = GitHubClient(token, base_url)
    payload = await client.get(f"/repos/{owner}/{repo}/actions/runs/{run_id}/jobs", {"per_page": 100})
    return payload.get("jobs", [])


async def get_workflow_run_logs(owner: str, repo: str, run_id: int, token: str | None = None, base_url: str | None = None, max_chars: int = 80000) -> str:
    """下载 CI 日志并解压成纯文本。

    【调用链】GitHubClient.get_bytes() → ZipFile 解压 → 本函数
    【被谁调用】services/repos_sync.py（存进 workflow_runs.logs_text）

    【数据流向】
        入：run_id、max_chars（字符上限）
        出：拼好的日志文本（每个文件前加 "### 文件名" 分隔）

    【★ 关键点：GitHub 的日志接口返回的是 ZIP】
        /actions/runs/{id}/logs 返回 application/zip，里面是每个 job 一个 .log 文件。
        所以必须解压 —— 而且要用 get_bytes（不是 get，因为不是 JSON）。

    【三个处理细节】
        ① 按文件名排序：保证多次下载的拼接顺序一致（便于 diff 和缓存）
        ② 跳过目录条目（name 以 "/" 结尾）：ZipFile 会把目录也列出来
        ③ errors="replace"：日志可能有非法 UTF-8 字节（二进制混进输出），
           用 replace 而不是抛异常，保证整份日志能读下来

    【为什么要 max_chars 上限】
        一次失败的 CI 日志可能几十 MB。全存进数据库会把表撑爆，
        而且 embedding 也用不上那么多。80000 字符足够覆盖错误现场。

    【解压失败怎么办】
        返回 BadZipFile 说明不是 zip（可能是 GitHub 直接返回了错误页文本），
        这时按纯文本解码返回 —— 至少让调用方看到"到底是什么内容"。
    """
    client = GitHubClient(token, base_url)
    payload = await client.get_bytes(f"/repos/{owner}/{repo}/actions/runs/{run_id}/logs")
    try:
        with ZipFile(BytesIO(payload)) as archive:
            parts: list[str] = []
            for name in sorted(archive.namelist()):
                # 跳过目录条目
                if name.endswith("/"):
                    continue
                raw = archive.read(name)
                text = raw.decode("utf-8", errors="replace")
                parts.append(f"### {name}\n{text}")
                # 累计超限就停（每加一个都要重算，因为文件大小差异很大）
                if sum(len(part) for part in parts) >= max_chars:
                    break
            return "\n\n".join(parts)[:max_chars]
    except BadZipFile:
        # 不是 zip → 按纯文本处理（可能是 GitHub 的错误响应）
        return payload.decode("utf-8", errors="replace")[:max_chars]
