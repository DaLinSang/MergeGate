"""
仓库接入与同步契约
================

【业务定位】定义"怎么接仓库"和"怎么触发同步"，是数据流 1 的入口契约。
【数据流向】frontend/app/repos/page.tsx → api/routes/repos.py → services/repos_sync.py
【核心规则】支持三种接入方式，所以 owner/repo 可选；首次同步限条数，防止拉爆大仓库。
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class RepoConnectRequest(BaseModel):
    """连接仓库的请求。三种方式任选：owner+repo / local_path / clone_parent_dir。"""

    owner: str | None = None                             # 账号/组织（方式①必填）
    repo: str | None = None                              # 仓库名（方式①必填）
    token: str | None = None                             # 本次使用的 token（加密后存库）
    github_token: str | None = None                      # 同上，兼容旧前端
    provider: str = "github"                             # github / github_compatible
    api_base_url: str | None = None                      # 企业版 API 地址
    local_path: str | None = None                        # 方式②：本地已有仓库路径
    clone_parent_dir: str | None = None                  # 方式③：克隆到哪个父目录
    demo_mode: bool = False                              # 演示模式（跳过真实校验）
    initial_sync_limit: int = Field(default=5, ge=1, le=50)  # 首次同步条数上限
    sync_issues: bool = True                             # 是否同步 Issue
    sync_pull_requests: bool = True                      # 是否同步 PR
    sync_workflow_runs: bool = True                      # 是否同步 CI


class RepoResponse(BaseModel):
    """仓库列表项。"""

    id: UUID                                # 内部 ID
    owner: str                              # 账号/组织
    name: str                               # 仓库名
    full_name: str                          # owner/name
    provider: str = "github"                # 平台类型
    api_base_url: str | None = None         # 企业版 API 地址
    clone_url: str | None = None            # clone 地址
    local_path: str | None = None           # 本地代码目录
    checkout_mode: str = "managed"          # managed（系统克隆）/ local（用本地目录）
    description: str | None = None          # 描述
    default_branch: str | None = None       # 默认分支
    last_sync_at: datetime | None = None    # 上次同步时间（判断同步状态的唯一依据）
    updated_at: datetime | None = None      # 记录更新时间

    model_config = {"from_attributes": True}


class RepoConnectResponse(BaseModel):
    """连接仓库的响应。"""

    repo_id: UUID                            # 仓库 ID
    full_name: str                           # owner/name
    provider: str                            # 平台类型
    api_base_url: str | None = None          # 企业版 API 地址
    clone_url: str | None = None             # clone 地址
    local_path: str | None = None            # 本地目录
    checkout_mode: str = "managed"           # 代码获取方式
    default_branch: str | None               # 默认分支
    connected: bool                          # 是否连接成功
    demo_mode: bool                          # 实际是否走了演示模式
    message: str                             # 给人看的提示
    sync_status: str = "not_started"         # completed / not_started
    synced: dict[str, int] | None = None     # 各类资源同步了多少条


class RepoSyncRequest(BaseModel):
    """手动同步的请求。可只刷新某一类数据。"""

    sync_issues: bool = True                 # 是否同步 Issue
    sync_pull_requests: bool = True          # 是否同步 PR
    sync_workflow_runs: bool = True          # 是否同步 CI
    limit: int = 30                          # 每类最多拉多少条


class RepoSyncResponse(BaseModel):
    """同步结果。synced 是"处理了几个对象"，不是"新增了几行"（同步是幂等的）。"""

    repo_id: UUID                            # 同步的仓库
    status: str                              # completed / failed
    synced: dict[str, int]                   # 各类资源的处理数量
