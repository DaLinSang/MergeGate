"""
数据模型（32 张表）
==================

【业务定位】
    这里是整个系统所有数据流的【终点站】。
    DevFlow AI 的业务主线是"定位 → 取证 → 判断 → 沉淀"，其中"沉淀"就落在这里：
    研发事实、对话过程、记忆、知识、安全审计，全部要有地方存。

    换句话说：前端看到的每个数字、Agent 引用的每条证据、评测用的每条轨迹，
    最终都来自这 32 张表。

【数据流向】
    上游（谁写入）：
        services/repos_sync.py        → repositories / issues / pull_requests
                                        / pr_files / pr_review_comments / workflow_runs
        services/code_analysis.py     → code_symbols / code_relations
        services/rag/document_processing.py → documents / knowledge_source_documents
        services/chat_memory.py       → chat_messages / chat_sessions / conversations
                                        / conversation_memories / thread_memories
        services/memory_hub.py        → memory_candidates / recall_events
        services/knowledge_graph.py   → knowledge_graph_edges
        services/evals/*              → eval_runs / analysis_results
        api/routes/action_drafts.py   → action_drafts / audit_logs
        api/routes/chat.py            → agent_runs

    下游（谁读取）：
        api/routes/*.py               → 所有只读接口
        services/agents/*             → Agent 的取证工具
        services/rag/*                → 检索与入库
        frontend/app/*                → 页面展示

【核心业务规则】
    1. 除了 users / workspaces / eval_runs，几乎所有表都挂在 repositories 下（repo_id 外键）。
       —— 这就是"仓库隔离"：任何查询都必须带 repo_id，否则会串仓库。
    2. 外部 ID（github_issue_id / github_pr_id / github_run_id）与内部 ID 分开存：
       内部 UUID 是主键（自己掌控），外部 ID 只用来做同步匹配。
    3. 大量用 JSON 字段：结构灵活但无法建索引，只适合"整体读写"的数据。

【技术要点】
    - Mapped[T] 是 Python 侧类型（给 IDE 看），mapped_column(...) 是数据库侧定义
    - 属性名 meta 映射到列名 metadata，因为 metadata 是 SQLAlchemy 的保留字
    - cascade="all, delete-orphan"：删父记录时自动删子记录（但 Milvus 向量不会自动删）

【怎么读这张表结构】
    - 类 docstring：【业务含义】/【谁写入】/【谁读取】/【对应数据流】
    - 每个字段行尾的 # 注释：这个字段存什么、取值范围、为什么这么设计
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, BigInteger, DateTime, Float, ForeignKey, Integer, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


# ════════════════════════════════════════════════════════════════
# 第一组：用户与仓库（系统的最外层身份）
# ════════════════════════════════════════════════════════════════

class User(Base):
    """用户表。

    【业务含义】系统里的操作者。当前是单用户演示模式，登录时自动创建。
    【谁写入】services/permissions.py 的 ensure_demo_user()
    【谁读取】权限校验 require_permission()

    注意：这是"课程项目级"的简化用户系统 —— 没有密码、没有登录态，
    靠"取第一个用户"来确定当前身份。
    """

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 主键（内部 UUID，不暴露给 GitHub）
    name: Mapped[str] = mapped_column(String(200), default="demo-user")                             # 显示名，演示模式固定为 demo-user
    role: Mapped[str] = mapped_column(String(80), default="owner", index=True)                      # 角色：owner / maintainer / developer / viewer（决定权限）
    github_token_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)                 # 加密后的 GitHub token（明文绝不入库，由 core/security.py 加解密）
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())  # 创建时间（数据库默认 now()）


class Repository(Base):
    """仓库表 —— 【全系统的根】。

    【业务含义】一个接入的 GitHub 仓库。几乎所有其他表都通过 repo_id 挂在它下面。
    【谁写入】api/routes/repos.py 的 connect_repo()；services/repos_sync.py 更新 last_sync_at
    【谁读取】所有业务接口

    【对应数据流】流 1（GitHub 事实 → 本地快照）的起点与终点
    """

    __tablename__ = "repositories"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 主键（内部 UUID，其他表靠它做仓库隔离）
    owner: Mapped[str] = mapped_column(String(200), index=True)                                      # 账号/组织，如 "openai"
    name: Mapped[str] = mapped_column(String(200), index=True)                                       # 仓库名，如 "openai-quickstart-python"
    full_name: Mapped[str] = mapped_column(String(400), unique=True, index=True)                     # "owner/name"，唯一索引保证同一仓库只接入一次
    provider: Mapped[str] = mapped_column(String(80), default="github")                              # 托管平台：github / github_compatible（企业版 GitLab 等可扩展）
    api_base_url: Mapped[str | None] = mapped_column(Text, nullable=True)                            # 企业版 API 根地址（公有云留空用默认）
    clone_url: Mapped[str | None] = mapped_column(Text, nullable=True)                               # git clone 地址（用于拉代码建图谱）
    local_path: Mapped[str | None] = mapped_column(Text, nullable=True)                              # 本地代码目录（工作区工具读源码的入口）
    checkout_mode: Mapped[str] = mapped_column(String(80), default="managed")                        # managed=系统克隆到 .data/repos；local=直接用用户本地目录
    github_token_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)                  # 仓库级 token（加密），优先级高于全局 GITHUB_TOKEN
    description: Mapped[str | None] = mapped_column(Text, nullable=True)                             # 仓库描述（GitHub 上填的那句）
    default_branch: Mapped[str | None] = mapped_column(String(200), nullable=True)                   # 默认分支，如 main（决定读代码时 checkout 哪个分支）
    github_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)                         # GitHub 侧的仓库数字 ID（跨改名稳定）
    last_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)    # 最后同步时间：/sync-status 靠它判断 completed 还是 not_started
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())  # 接入时间
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())  # 最后更新时间（任何字段改动自动刷新）

    # 关系属性：repo.issues 直接拿到该仓库的所有 Issue
    # cascade="all, delete-orphan"：删仓库时自动删子记录（但 Milvus 向量要手动清）
    issues: Mapped[list["Issue"]] = relationship(back_populates="repo", cascade="all, delete-orphan")                # 一对多：该仓库的所有 Issue
    pull_requests: Mapped[list["PullRequest"]] = relationship(back_populates="repo", cascade="all, delete-orphan")    # 一对多：该仓库的所有 PR
    workflow_runs: Mapped[list["WorkflowRun"]] = relationship(back_populates="repo", cascade="all, delete-orphan")    # 一对多：该仓库的所有 CI 运行


class Workspace(Base):
    """工作区表 —— 把多个仓库归成一组。

    【业务含义】做"跨仓库报告"时用（比如一个产品拆成前端/后端两个仓库）。
    【谁写入】api/routes/workspaces.py
    【谁读取】跨库汇总报告
    """

    __tablename__ = "workspaces"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 主键
    name: Mapped[str] = mapped_column(String(200), index=True)                                       # 工作区名称，如 "DevFlow 全栈"
    description: Mapped[str | None] = mapped_column(Text, nullable=True)                             # 工作区说明
    repo_ids: Mapped[list[str]] = mapped_column(JSON, default=list)                                   # 成员仓库 ID 列表（用 JSON 存，不建关联表：只整体读写，不需反向查）
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)  # 创建人（外键 users.id）
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)  # 创建时间（列表按它排序）
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())  # 最后更新时间


class ProjectIndexState(Base):
    """项目文档索引状态表（每仓库一行）。

    【业务含义】记录"这个仓库的项目文档扫描到哪一步了"，供前端显示进度条。
    【谁写入】services/project_indexing.py
    【谁读取】api/routes/project_index.py

    status 取值：missing（未扫描）/ building（扫描中）/ ready（完成）/ failed（失败）
    """

    __tablename__ = "project_index_states"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 主键
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), unique=True, index=True)  # 所属仓库（unique：一个仓库只有一条索引状态）
    status: Mapped[str] = mapped_column(String(40), default="missing", index=True)                   # 扫描状态：missing / building / ready / failed
    fingerprint: Mapped[str] = mapped_column(String(200), default="")                                # 文档集合指纹：内容没变就跳过重扫（省时间）
    last_scan_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)    # 上次扫描完成时间
    snoozed_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)   # 用户点"稍后提醒"的截止时间，之前不再自动扫描
    docs_indexed: Mapped[int] = mapped_column(Integer, default=0)                                    # 已索引文档数（进度条分子）
    docs_total: Mapped[int] = mapped_column(Integer, default=0)                                      # 待索引文档总数（进度条分母）
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)                           # 失败原因（前端展示）
    summary_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)                 # 扫描摘要（按类型统计的文档数等）
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())  # 创建时间
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())  # 最后更新时间

    repo: Mapped[Repository] = relationship()                                                        # 所属仓库对象（无反向属性）


# ════════════════════════════════════════════════════════════════
# 第二组：GitHub 协作事实（数据流 1 的产物）
# ════════════════════════════════════════════════════════════════

class Conversation(Base):
    """会话表 —— 一个仓库下可以有多个对话。

    【业务含义】"定位"这一步的落点：用户必须先选中"哪个仓库 + 哪个会话"，
    后续的「这个」「刚才那个」才有指代对象。
    【谁写入】api/routes/conversations.py、services/conversations.py
    【谁读取】chat.py（每次对话都要定位会话）
    """

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 主键（chat_messages / memories 靠它归属会话）
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)            # 所属仓库（会话不能跨仓库）
    title: Mapped[str] = mapped_column(String(300), default="默认会话")                              # 会话标题（通常取第一句提问）
    status: Mapped[str] = mapped_column(String(80), default="active", index=True)                    # 状态：active / archived
    message_count: Mapped[int] = mapped_column(Integer, default=0)                                   # 消息条数（列表页直接显示，不用 count 子查询）
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)  # 创建时间
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), index=True)  # 最后活跃时间（列表按它倒序）


class Issue(Base):
    """Issue 表。

    【业务含义】GitHub Issue 的本地快照。Issue 分析 Agent 的输入。
    【谁写入】services/repos_sync.py（同步时 upsert）
    【谁读取】api/routes/issues.py、issue_agent、RAG 入库

    【对应数据流】流 1（同步）→ 流 3（进 RAG）→ 流 4（被 Agent 检索）
    """

    __tablename__ = "issues"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 主键（内部 UUID）
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)            # 所属仓库
    github_issue_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)                   # GitHub 原始 ID：同步匹配用（不用 number，number 会因仓库迁移变化）
    number: Mapped[int] = mapped_column(Integer, index=True)                                         # 用户看到的编号，如 #42
    title: Mapped[str] = mapped_column(String(500))                                                  # 标题
    body: Mapped[str | None] = mapped_column(Text, nullable=True)                                    # 正文（进 RAG 的主要素材）
    state: Mapped[str] = mapped_column(String(50), default="open")                                   # 状态：open / closed
    labels: Mapped[list[str]] = mapped_column(JSON, default=list)                                    # 标签（GitHub 返回 [{name:bug}]，这里拍平成 ["bug"]）
    author: Mapped[str | None] = mapped_column(String(200), nullable=True)                           # 创建者 GitHub 登录名
    assignees: Mapped[list[str]] = mapped_column(JSON, default=list)                                 # 指派给谁（登录名列表）
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)      # GitHub 上的创建时间（不是本地入库时间）
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)      # GitHub 上的最后更新时间
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)       # 关闭时间（未关闭为 NULL，对应 schema 里的 Optional）

    repo: Mapped[Repository] = relationship(back_populates="issues")                                 # 反向关系：所属仓库


class PullRequest(Base):
    """PR 主表（本体）。

    【业务含义】PR 审查 Agent 的输入起点。真正的代码改动在 pr_files，评论在 pr_review_comments。
    【谁写入】services/repos_sync.py
    【谁读取】api/routes/pull_requests.py、pr_review_agent、RAG 入库

    设计要点：PR 拆成三张表（本体 / 文件 / 评论），因为它们是"一对多"关系，
    且子资源需要整组替换（见 pr_files 的注释）。
    """

    __tablename__ = "pull_requests"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 主键（pr_files / pr_review_comments 靠它归属）
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)            # 所属仓库
    github_pr_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)                      # GitHub 原始 ID：同步匹配用
    number: Mapped[int] = mapped_column(Integer, index=True)                                         # 用户看到的编号，如 #128
    title: Mapped[str] = mapped_column(String(500))                                                  # 标题
    body: Mapped[str | None] = mapped_column(Text, nullable=True)                                    # PR 说明（进 RAG）
    state: Mapped[str] = mapped_column(String(50), default="open")                                   # 状态：open / closed / merged
    author: Mapped[str | None] = mapped_column(String(200), nullable=True)                           # 提交者 GitHub 登录名
    base_branch: Mapped[str | None] = mapped_column(String(200), nullable=True)                      # 目标分支，如 main
    head_branch: Mapped[str | None] = mapped_column(String(200), nullable=True)                      # 来源分支（review 时要 checkout 它）
    merged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)       # 合并时间（未合并为 NULL）
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)      # GitHub 上的创建时间
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)      # GitHub 上的最后更新时间

    repo: Mapped[Repository] = relationship(back_populates="pull_requests")                          # 反向关系：所属仓库
    files: Mapped[list["PRFile"]] = relationship(back_populates="pr", cascade="all, delete-orphan")   # 一对多：改动的文件（整组替换）
    review_comments: Mapped[list["PRReviewComment"]] = relationship(back_populates="pr", cascade="all, delete-orphan")  # 一对多：行级评审评论


class PRFile(Base):
    """PR 改动的单个文件。

    【业务含义】PR 审查的核心材料：改了哪个文件、增删多少行、具体 diff 是什么。
    【谁写入】services/repos_sync.py —— 【整组替换】：先删光该 PR 的所有文件，再全量插入
    【谁读取】pr_review_agent（按文件分块审查）

    为什么整组替换：
        PR rebase 后某些文件可能从改动列表消失，增量更新无法感知"删除"。
    """

    __tablename__ = "pr_files"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 主键
    pr_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("pull_requests.id"), index=True)             # 所属 PR（整组替换按它先删后插）
    filename: Mapped[str] = mapped_column(String(500))                                               # 文件相对路径，如 "app/services/chat.py"
    status: Mapped[str] = mapped_column(String(50))                                                  # 变更类型：added / modified / removed / renamed
    additions: Mapped[int] = mapped_column(Integer, default=0)                                       # 新增行数
    deletions: Mapped[int] = mapped_column(Integer, default=0)                                       # 删除行数
    patch: Mapped[str | None] = mapped_column(Text, nullable=True)                                   # 具体 diff 文本（超大文件 GitHub 会返回空）

    pr: Mapped[PullRequest] = relationship(back_populates="files")                                   # 反向关系：所属 PR


class PRReviewComment(Base):
    """PR 的行级 Review 评论（针对某一行 diff）。

    【业务含义】历史评审意见 —— 判断"这个 PR 有没有人已经指出过问题"。
    【谁写入】services/repos_sync.py（整组替换）
    【谁读取】pr_review_agent、RAG 入库

    注意：这里不含普通 Issue 评论，也不含一次 Review 的整体结论。
    """

    __tablename__ = "pr_review_comments"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 主键
    pr_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("pull_requests.id"), index=True)             # 所属 PR（整组替换按它先删后插）
    github_comment_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)                 # GitHub 评论原始 ID（去重 / 回链用）
    body: Mapped[str | None] = mapped_column(Text, nullable=True)                                    # 评论内容（进 RAG）
    path: Mapped[str | None] = mapped_column(String(500), nullable=True)                             # 评论所在文件路径
    line: Mapped[int | None] = mapped_column(Integer, nullable=True)                                 # 评论所在行号（文件级评论为 NULL）
    author: Mapped[str | None] = mapped_column(String(200), nullable=True)                           # 评论者 GitHub 登录名
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)      # 评论创建时间
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)      # 评论最后编辑时间

    pr: Mapped[PullRequest] = relationship(back_populates="review_comments")                         # 反向关系：所属 PR


class WorkflowRun(Base):
    """CI 运行记录（GitHub Actions 的一次 workflow run）。

    【业务含义】CI 排障 Agent 的输入。回答"为什么这次构建失败了"。
    【谁写入】services/repos_sync.py
    【谁读取】api/routes/ci.py、ci_debug_agent、RAG 入库（只索引失败日志）

    设计要点：jobs 和 steps 用 JSON 整体存，不单独建表。
        因为只在"展示 CI 详情"时整体读取，不需要按 job 字段查询。
    """

    __tablename__ = "workflow_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 主键
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)            # 所属仓库
    github_run_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)                     # GitHub run 原始 ID：同步匹配用
    name: Mapped[str] = mapped_column(String(300))                                                   # 工作流名称，如 "CI"
    status: Mapped[str] = mapped_column(String(80))                                                  # 运行状态：queued / in_progress / completed
    conclusion: Mapped[str | None] = mapped_column(String(80), nullable=True)                        # 结论：success / failure / cancelled（未结束为 NULL）
    html_url: Mapped[str | None] = mapped_column(Text, nullable=True)                                # 跳转到 GitHub 的链接
    jobs: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)                           # 整个 jobs+steps 树存成一个 JSON（只整体展示，不按字段查）
    logs_text: Mapped[str | None] = mapped_column(Text, nullable=True)                               # 失败日志全文（只为 conclusion=failure 的 Run 拉取，省流量省存储）
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)      # GitHub 上的触发时间
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)      # GitHub 上的最后更新时间

    repo: Mapped[Repository] = relationship(back_populates="workflow_runs")                          # 反向关系：所属仓库


class AnalysisResult(Base):
    """分析结果归档表。

    【业务含义】保存"某次分析当时看到什么输入、得出什么结论、用的哪个模型"。
    【谁写入】各分析 Agent
    【谁读取】复盘、对比不同模型效果

    为什么要存 input_snapshot：
        模型和提示词会变。只存结果无法复现"当时为什么得出这个结论"。
        存下输入快照才能事后对比。
    """

    __tablename__ = "analysis_results"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 主键
    target_type: Mapped[str] = mapped_column(String(80), index=True)                                 # 分析对象类型：issue / pull_request / workflow_run
    target_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), index=True)                     # 分析对象 ID（多态，不建外键）
    analysis_type: Mapped[str] = mapped_column(String(100), index=True)                              # 分析类型标识，如 "issue_triage" / "pr_review"
    input_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)                       # 当时的输入快照（用于复现）
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)                          # 结构化结论
    model_name: Mapped[str | None] = mapped_column(String(200), nullable=True)                       # 用了哪个模型（区分 LLM 与纯规则）
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())  # 分析时间


# ════════════════════════════════════════════════════════════════
# 第三组：RAG 与知识（数据流 3 的产物）
# ════════════════════════════════════════════════════════════════

class Document(Base):
    """文档表 —— 【RAG 的唯一事实源】。

    【业务含义】所有进入 RAG 的知识，完整正文都存在这里。
    【谁写入】services/rag/vector_store.py 的 add_documents()
    【谁读取】services/rag/retrieval.py（关键词检索 BM25 直接查这张表）

    【对应数据流】流 3（入库）的终点；流 4（查询）中关键词分支的数据源

    ⚠️ 关键设计：Milvus 只是【索引副本】，这里才是真相。
        - Milvus 的 content 字段截断到 8192 字节，这里存完整正文
        - Milvus 挂了可以从这张表重建
        - 注意：这里【没有】embedding 列 —— 向量全在 Milvus，
          启动时 main.py 会主动 DROP 掉历史遗留的 embedding 列
    """

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 主键（Milvus 主键由它派生，见 stable_source_id）
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)            # 所属仓库（检索时必须带，做项目隔离）
    source_type: Mapped[str] = mapped_column(String(80), index=True)                                 # 来源类型：issue / pull_request / workflow_run / project_doc / knowledge_file / memory_note（code_file 刻意排除）
    source_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), index=True)                     # 源对象 ID（回查原始记录用）
    title: Mapped[str] = mapped_column(String(500))                                                  # 标题（切片时拼进正文，提升命中率）
    content: Mapped[str] = mapped_column(Text)                                                       # 完整正文（Milvus 只存截断版）
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)                     # 额外元数据（属性名用 meta，因 metadata 是 SQLAlchemy 保留字）
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())  # 入库时间


class KnowledgeBaseConfig(Base):
    """知识库配置表（每仓库一行）。

    【业务含义】每个仓库的 RAG 参数：用什么模型、切多大、检索几条、阈值多少。
    【谁写入】api/routes/rag.py 的 PATCH 配置接口
    【谁读取】services/rag/knowledge_base.py、retrieval.py

    ⚠️ 最重要的业务约束：
        所有共享同一个 Milvus collection 的仓库，必须用【同一个 embedding 模型和维度】。
        不同模型的向量空间不可比，混在一起检索结果是错的。
        services/rag/embeddings.py 的 validate_embedding_contract() 会强校验这一点。
    """

    __tablename__ = "knowledge_base_configs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 主键
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), unique=True, index=True)  # 所属仓库（unique：一仓库一份配置）
    embedding_provider: Mapped[str] = mapped_column(String(80), default="qwen_local")                # 向量化服务：qwen_local / qwen_api（改它必须重建索引）
    embedding_model: Mapped[str] = mapped_column(String(240), default="Qwen/Qwen3-Embedding-0.6B")   # 向量模型名（向量空间契约的一部分）
    embedding_dimensions: Mapped[int] = mapped_column(Integer, default=1024)                         # 向量维度（与 Milvus collection 必须一致，否则写入报错）
    retrieval_method: Mapped[str] = mapped_column(String(40), default="hybrid")                      # 检索方式：hybrid / vector / keyword
    rerank_enabled: Mapped[bool] = mapped_column(default=True)                                       # 是否开启重排（开=更准但更慢更贵）
    rerank_provider: Mapped[str] = mapped_column(String(80), default="qwen_api")                     # 重排服务：qwen_api / heuristic
    rerank_model: Mapped[str] = mapped_column(String(240), default="qwen3-rerank")                   # 重排模型名
    top_k: Mapped[int] = mapped_column(Integer, default=3)                                           # 最终给模型几条证据
    score_threshold_enabled: Mapped[bool] = mapped_column(default=True)                              # 是否启用分数阈值过滤
    score_threshold: Mapped[float] = mapped_column(Float, default=0.8)                               # 阈值：低于它直接丢（太高漏召回，太低引噪声）
    chunk_size: Mapped[int] = mapped_column(Integer, default=800)                                    # 每块目标字符数
    chunk_overlap: Mapped[int] = mapped_column(Integer, default=100)                                 # 相邻块重叠字符数（防止语义被切断）
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())  # 创建时间
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())  # 最后更新时间


class KnowledgeSourceDocument(Base):
    """上传的原始文档（含处理状态机）。

    【业务含义】用户上传的文件记录：存在哪、处理到哪一步、切了多少块。
    【谁写入】api/routes/rag.py（创建，status=queued）→ services/rag/document_processing.py（更新状态）
    【谁读取】前端轮询进度、重试失败文档

    状态机：queued → parsing → splitting → embedding → completed / failed
    为什么要存状态：一份 100 页 PDF 处理要 30 秒以上，必须异步 + 可观测。
    """

    __tablename__ = "knowledge_source_documents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 主键
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)            # 所属仓库
    name: Mapped[str] = mapped_column(String(500))                                                   # 原始文件名（前端展示用）
    source_type: Mapped[str] = mapped_column(String(80), default="upload_file")                      # 来源类型：upload_file（当前只有上传）
    status: Mapped[str] = mapped_column(String(40), default="queued", index=True)                    # 处理状态：queued / parsing / splitting / embedding / completed / failed
    stored_path: Mapped[str] = mapped_column(Text)                                                   # 磁盘路径，如 .data/uploads/xxx.pdf（重试时从这里重读）
    content_type: Mapped[str | None] = mapped_column(String(200), nullable=True)                     # MIME 类型，如 application/pdf
    checksum: Mapped[str] = mapped_column(String(64), index=True)                                    # SHA-256 校验和：同一文件重复上传直接跳过
    file_size: Mapped[int] = mapped_column(Integer, default=0)                                       # 原始字节数
    character_count: Mapped[int] = mapped_column(Integer, default=0)                                 # 解析出的字符数（判断解析是否正常）
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)                                     # 切成了多少块（= 写入 Milvus 的条数）
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)                           # 失败原因（前端展示）
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)  # 上传时间
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())  # 最后更新时间（每次状态流转刷新）
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)    # 处理完成时间


class RetrievalTestRun(Base):
    """检索测试记录。

    【业务含义】用户在"召回测试"页面调参时，每次测试都留一条：用了什么参数、召回了什么。
    【谁写入】api/routes/rag.py 的 retrieval-tests 接口
    【谁读取】前端对比不同参数的召回效果
    """

    __tablename__ = "retrieval_test_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 主键
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)            # 所属仓库
    query: Mapped[str] = mapped_column(Text)                                                         # 测试用的查询词
    retrieval_config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)                     # 本次用的参数（top_k / 阈值 / 方式等）
    results_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)                   # 召回结果（含各阶段分数明细）
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)                                     # 耗时毫秒（用于性能对比）
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)  # 测试时间


# ════════════════════════════════════════════════════════════════
# 第四组：代码图谱（数据流 2 的产物）
# ════════════════════════════════════════════════════════════════

class CodeSymbol(Base):
    """代码符号表（函数 / 类 / 方法）。

    【业务含义】回答"这个函数定义在哪""改这个文件会影响什么"。
    【谁写入】services/code_analysis.py（AST 解析）
    【谁读取】api/routes/code_graph.py、workspace.search_code

    【对应数据流】流 2（仓库代码 → 代码图谱）

    注意：源码【不进】RAG 向量库，但可以建"符号索引"——
    因为符号是结构化的（名字+位置），不是需要语义召回的长文本。
    """

    __tablename__ = "code_symbols"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 主键（code_relations 靠它连边）
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)            # 所属仓库
    path: Mapped[str] = mapped_column(String(500), index=True)                                       # 文件相对路径，如 "app/main.py"
    name: Mapped[str] = mapped_column(String(300), index=True)                                       # 符号名，如 "create_app"
    kind: Mapped[str] = mapped_column(String(80), default="symbol", index=True)                      # 符号类型：function / class / method
    language: Mapped[str] = mapped_column(String(80), default="text", index=True)                    # 语言：python / javascript / java / text ...
    start_line: Mapped[int | None] = mapped_column(Integer, nullable=True)                           # 定义起始行号（1-based）
    end_line: Mapped[int | None] = mapped_column(Integer, nullable=True)                             # 定义结束行号
    branch: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)               # 所属分支（同一文件在不同分支是两套符号）
    commit_sha: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)           # 解析时的 commit（保证符号与代码版本对齐）
    pr_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True, index=True)   # PR 维度符号：回答"这个 PR 改了哪些符号"
    pr_number: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)                # PR 编号（展示用，免 join）
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)                     # 额外信息（参数、装饰器等）
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)  # 入库时间


class CodeRelation(Base):
    """代码关系表（调用 / 继承 / 引用）。

    【业务含义】代码图谱的"边"。source → target，relation_type 说明关系。
    【谁写入】services/code_analysis.py
    【谁读取】api/routes/code_graph.py

    为什么要存 source_name/target_name：
        符号可能还没被解析出来（跨文件、动态调用），先存名字，
        等两边都解析出来再用 ID 关联。
    """

    __tablename__ = "code_relations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 主键
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)            # 所属仓库
    source_symbol_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("code_symbols.id"), nullable=True, index=True)  # 起点符号 ID（未解析出来时为空）
    target_symbol_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("code_symbols.id"), nullable=True, index=True)  # 终点符号 ID（跨文件未解析时为空）
    source_name: Mapped[str | None] = mapped_column(String(300), nullable=True, index=True)          # 冗余存起点名字（ID 还没连上时靠名字兜底）
    target_name: Mapped[str] = mapped_column(String(300), index=True)                                # 终点名字（必须存：调用目标可能还没被解析）
    relation_type: Mapped[str] = mapped_column(String(80), default="calls", index=True)              # 关系类型：calls / imports / inherits
    path: Mapped[str | None] = mapped_column(String(500), nullable=True, index=True)                 # 关系发生的文件路径
    branch: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)               # 所属分支
    pr_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True, index=True)   # PR 维度关系（回答"这个 PR 影响了哪些调用"）
    pr_number: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)                # PR 编号（展示用）
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)                     # 额外信息（调用位置行号等）
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)  # 入库时间


# ════════════════════════════════════════════════════════════════
# 第五组：对话与运行记录（数据流 4 的过程留痕）
# ════════════════════════════════════════════════════════════════

class ChatMessage(Base):
    """对话消息表。

    【业务含义】每一句用户提问和 Agent 回答。
    【谁写入】api/routes/chat.py 的 MessageStore.append()
    【谁读取】ContextAssembler（组装上下文）、前端历史列表

    【对应数据流】流 4（用户问题 → 回答）的过程记录

    为什么 tool_calls 用 JSON：
        不同工具参数结构不同，用 JSON 更灵活。
        代价：没法用 SQL 直接查"哪个工具被调最多"，需要应用层统计。
    """

    __tablename__ = "chat_messages"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 主键（chat_feedback 靠它关联）
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True, index=True)  # 所属仓库（可为空：未选仓库的通用对话）
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)  # 所属会话（可为空：会话创建前的临时消息）
    role: Mapped[str] = mapped_column(String(40), index=True)                                        # 角色：user / assistant
    content: Mapped[str] = mapped_column(Text)                                                       # 消息正文
    route: Mapped[str | None] = mapped_column(String(100), nullable=True)                            # 走了哪条处理路径：chat / issue_analysis / pr_review / ci_debug / workflow ...
    tool_calls: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)                     # 这一轮调了哪些工具（名字+参数+结果摘要）
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)                     # 额外信息（引用、耗时、token 等）
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)  # 消息时间（历史列表按它排序）


class ChatSession(Base):
    """会话封存记录（一轮完整问答的"归档包"）。

    【业务含义】一轮对话结束后，把用户消息 + 回答 + 工具轨迹打包封存，
    供后续提炼记忆、复盘、评测。
    【谁写入】services/chat_memory.py 的 SessionSealer.seal()
    【谁读取】memory_hub（提炼记忆候选）、evals

    【对应数据流】流 5（对话 → 长期记忆）的中间产物
    """

    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 主键（memory_candidates 靠它回溯来源）
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True, index=True)  # 所属仓库
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)  # 所属会话
    user_message_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True, index=True)  # 封存的用户消息 ID（不加外键：消息可能被清理）
    assistant_message_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True, index=True)  # 封存的回答消息 ID
    route: Mapped[str] = mapped_column(String(100), default="unknown")                               # 这轮走的处理路径
    transcript_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)                      # 完整对话记录（含工具轨迹）
    digest: Mapped[str] = mapped_column(Text, default="")                                            # 压缩摘要（比 transcript 短，喂给提炼环节）
    status: Mapped[str] = mapped_column(String(80), default="sealed", index=True)                    # 状态：sealed（已封存）/ 其他扩展态
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)  # 创建时间
    sealed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)       # 封存时间


class ThreadMemory(Base):
    """仓库级长期记忆（每仓库一行）。

    【业务含义】跨会话的"项目常识"：这个项目的约定、决策、待办、用户偏好。
    【谁写入】services/chat_memory.py 的 SessionSealer
    【谁读取】ContextAssembler（每轮对话都注入）、MCP 记忆工具

    【对应数据流】流 5（对话 → 长期记忆）

    设计要点：不是一整段文本，而是拆成 6 个结构化字段。
        因为不同类型的记忆用途不同 —— facts 用于回答事实问题，
        tasks 用于提醒待办，user_preferences 用于调整回答风格。
    """

    __tablename__ = "thread_memories"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 主键
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True, unique=True, index=True)  # 所属仓库（unique：一仓库一条，不断累积更新）
    summary: Mapped[str] = mapped_column(Text, default="")                                           # 整体摘要（一句话说清这个项目）
    facts: Mapped[list[str]] = mapped_column(JSON, default=list)                                     # 客观事实（技术栈、架构约定等）
    decisions: Mapped[list[str]] = mapped_column(JSON, default=list)                                 # 已做的决策（"我们决定用 X 而不用 Y"）
    open_questions: Mapped[list[str]] = mapped_column(JSON, default=list)                            # 未解决的问题
    tasks: Mapped[list[str]] = mapped_column(JSON, default=list)                                     # 待办事项
    user_preferences: Mapped[list[str]] = mapped_column(JSON, default=list)                          # 用户偏好（回答风格、习惯）
    repo_context: Mapped[list[str]] = mapped_column(JSON, default=list)                              # 项目背景（业务是什么、谁在用）
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)                      # 每条记忆的来源引用（可追溯）
    sessions_incorporated: Mapped[int] = mapped_column(Integer, default=0)                           # 已合并了几轮会话（增量更新用）
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())  # 最后更新时间


class ConversationMemory(Base):
    """会话级记忆（每会话一行）。

    【业务含义】只在这个会话内有效的上下文（比如"刚才讨论的那个 PR"）。
    【谁写入】services/chat_memory.py
    【谁读取】ContextAssembler（只在本会话内注入）

    与 ThreadMemory 的区别：
        ThreadMemory = 整个项目的长期常识（跨会话）
        ConversationMemory = 当前这条对话线的上下文（会话结束就不再需要）
    字段结构完全一样，只是作用范围不同。
    """

    __tablename__ = "conversation_memories"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 主键
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)            # 所属仓库
    conversation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("conversations.id"), unique=True, index=True)  # 所属会话（unique：一会话一条）
    summary: Mapped[str] = mapped_column(Text, default="")                                           # 本会话的整体摘要
    facts: Mapped[list[str]] = mapped_column(JSON, default=list)                                     # 本会话确认的事实
    decisions: Mapped[list[str]] = mapped_column(JSON, default=list)                                 # 本会话做出的决策
    open_questions: Mapped[list[str]] = mapped_column(JSON, default=list)                            # 本会话悬而未决的问题
    tasks: Mapped[list[str]] = mapped_column(JSON, default=list)                                     # 本会话产生的待办
    user_preferences: Mapped[list[str]] = mapped_column(JSON, default=list)                          # 本会话中表达的偏好
    repo_context: Mapped[list[str]] = mapped_column(JSON, default=list)                              # 本会话涉及的项目背景
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)                      # 来源引用
    sessions_incorporated: Mapped[int] = mapped_column(Integer, default=0)                           # 已合并几轮会话
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())  # 最后更新时间


class EvidenceItem(Base):
    """证据库条目。

    【业务含义】把 Issue/PR/CI 的摘要建成"可检索的证据"，供 Agent 快速取证。
    【谁写入】services/chat_memory.py 的 EvidenceStore.rebuild_repo()
    【谁读取】EvidenceStore.search()、MCP 的 devflow_search_evidence 工具

    与 Document 的区别：
        Document = 进向量库做语义检索（完整正文）
        EvidenceItem = 轻量摘要，用于关键词快速匹配（不走向量）
    为什么要两套：向量检索有延迟和成本，短查询用关键词更快。
    """

    __tablename__ = "evidence_items"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 主键
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True, index=True)  # 所属仓库
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)  # 关联会话（会话级证据时填）
    source_type: Mapped[str] = mapped_column(String(80), index=True)                                 # 来源类型：issue / pull_request / workflow_run ...
    source_id: Mapped[str] = mapped_column(String(120), index=True)                                  # 源对象 ID（用字符串：兼容 UUID 与数字 ID）
    title: Mapped[str] = mapped_column(String(500))                                                  # 证据标题
    content: Mapped[str] = mapped_column(Text)                                                       # 证据摘要正文（关键词匹配的对象）
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)                     # 额外信息（状态、作者、时间等）
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)  # 重建时间


class KnowledgeGraphEdge(Base):
    """知识图谱的边（实体之间的关系）。

    【业务含义】把 Issue / PR / 文档 / 团队成员连成一张图，
    回答"这个 PR 关联哪些 Issue""谁负责这块代码"。
    【谁写入】services/knowledge_graph.py
    【谁读取】api/routes/knowledge.py 的 graph 接口

    from_type/from_id → to_type/to_id 是通用结构（不限定实体类型），
    所以不能用外键约束，只能靠应用层保证一致性。
    """

    __tablename__ = "knowledge_graph_edges"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 主键
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)            # 所属仓库（图按仓库隔离）
    from_type: Mapped[str] = mapped_column(String(80), index=True)                                   # 起点实体类型：issue / pull_request / document / team_member
    from_id: Mapped[str] = mapped_column(String(180), index=True)                                    # 起点实体 ID（字符串，兼容多种主键）
    to_type: Mapped[str] = mapped_column(String(80), index=True)                                     # 终点实体类型
    to_id: Mapped[str] = mapped_column(String(180), index=True)                                      # 终点实体 ID
    relation: Mapped[str] = mapped_column(String(80), index=True)                                    # 关系：references / authored_by / relates_to
    confidence: Mapped[float] = mapped_column(Float, default=1.0)                                    # 置信度（1.0=显式声明，<1.0=自动推断）
    source: Mapped[str] = mapped_column(String(120), default="explicit")                             # 边的来源：explicit（显式）/ inferred（推断）
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)                     # 额外信息（推断依据等）
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)  # 建边时间


class RecallEvent(Base):
    """召回事件日志。

    【业务含义】Agent 每次调记忆/检索工具都记一条：查了什么、召回几条、耗时多久。
    【谁写入】services/memory_hub.py 的 record_recall_event()
    【谁读取】记忆面板（展示"Agent 最近查了什么"）、调试

    为什么要记：排查"Agent 为什么没找到答案"时，第一件事就是看召回日志。
    """

    __tablename__ = "recall_events"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 主键
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True, index=True)  # 所属仓库
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)  # 所属会话
    tool_name: Mapped[str] = mapped_column(String(120), index=True)                                  # 调的是哪个工具，如 "devflow_search_evidence"
    query: Mapped[str] = mapped_column(Text, default="")                                             # 查询词（用户原话或改写后的）
    scope: Mapped[str] = mapped_column(String(120), default="project")                               # 检索范围：project / conversation / session
    mode: Mapped[str] = mapped_column(String(80), default="hybrid")                                  # 检索方式：hybrid / vector / keyword
    status: Mapped[str] = mapped_column(String(80), default="success", index=True)                   # 结果状态：success / empty / error
    result_count: Mapped[int] = mapped_column(Integer, default=0)                                    # 召回几条（0 是最重要的排查信号）
    results_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)                   # 召回内容快照
    citations_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)                 # 引用信息快照
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)                     # 额外信息（耗时、参数等）
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)  # 调用时间


class MemoryCandidate(Base):
    """记忆候选表 —— 【需要人工批准才能生效】。

    【业务含义】Agent 在对话中发现了"可能值得记住的事实"，先放这里等人审核。
    【谁写入】services/memory_hub.py 的 capture_memory_candidates()
    【谁读取】api/routes/knowledge.py 的 candidates 接口（审核页面）

    【对应数据流】流 5（对话 → 长期记忆）的审核关卡

    ⚠️ 核心业务规则：记忆不是自动生效的。
        Agent 提炼出的"事实"可能是错的（幻觉）。必须人工点"批准"，
        才会晋升为 Document 进 RAG。这是防止错误知识污染长期记忆的关键设计。
    """

    __tablename__ = "memory_candidates"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 主键
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True, index=True)  # 所属仓库
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)  # 来源会话
    session_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("chat_sessions.id"), nullable=True, index=True)  # 来源封存包（回溯"这句话从哪轮对话提炼的"）
    kind: Mapped[str] = mapped_column(String(80), index=True)                                        # 类型：fact / decision / preference
    title: Mapped[str] = mapped_column(String(500))                                                  # 候选记忆标题
    content: Mapped[str] = mapped_column(Text)                                                       # 候选记忆正文
    status: Mapped[str] = mapped_column(String(80), default="pending", index=True)                   # 审核状态：pending / approved / rejected
    source: Mapped[str] = mapped_column(String(120), default="session_sealer")                       # 从哪提炼出来的（session_sealer / 手动）
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)                     # 额外信息（置信度、来源引用）
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)  # 创建时间
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)      # 审核时间（未审核为 NULL）


class AgentRun(Base):
    """Agent 单次运行记录。

    【业务含义】"这次提问，Agent 走了哪条路径、调了哪些工具、最后答了什么"。
    【谁写入】api/routes/chat.py 收到 final 事件后
    【谁读取】trace 查询、评测、反馈关联

    【对应数据流】流 4 的"沉淀"环节

    与 ChatMessage 的区别：
        ChatMessage 面向"对话展示"（用户看历史）
        AgentRun 面向"调试与评测"（开发者看轨迹）
    """

    __tablename__ = "agent_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 主键（chat_feedback.run_id 指向它）
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True)  # 所属仓库
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)  # 所属会话
    user_message: Mapped[str] = mapped_column(Text)                                                  # 用户原始提问
    route: Mapped[str] = mapped_column(String(100))                                                  # 走了哪条分支（chat / workflow / ...）
    tool_calls: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)                     # 完整调用链（含每步参数与结果）
    final_answer: Mapped[str] = mapped_column(Text)                                                  # 最终回答
    status: Mapped[str] = mapped_column(String(80), default="success")                               # 结束状态：success / error / budget_exhausted
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())  # 运行时间


# ════════════════════════════════════════════════════════════════
# 第六组：反馈、工作流与评测（质量闭环）
# ════════════════════════════════════════════════════════════════

class ChatFeedback(Base):
    """回答反馈（点赞 / 点踩）。

    【业务含义】用户对某条回答的评价，是"质量闭环"的人工信号。
    【谁写入】api/routes/feedback.py
    【谁读取】质量指标统计、点踩通知

    设计要点：
        - assistant_message_id 有 unique 约束 → 同一条消息只能评一次（重复提交走 upsert）
        - 点踩会触发飞书通知（notification_status 记录推送状态）
    """

    __tablename__ = "chat_feedback"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 主键
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id", ondelete="CASCADE"), index=True)  # 所属仓库（删仓库时反馈一起删，不留孤儿）
    conversation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)  # 所属会话（级联删除）
    assistant_message_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("chat_messages.id", ondelete="CASCADE"), unique=True, index=True)  # 被评价的回答（unique：一条只能评一次）
    run_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("agent_runs.id", ondelete="SET NULL"), nullable=True, index=True)  # 关联运行记录（删运行时只置空，反馈本身有价值）
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)  # 评价人
    rating: Mapped[str] = mapped_column(String(40), index=True)                                      # 评价：up（点赞）/ down（点踩）
    reason: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)                # 点踩原因分类（如 inaccurate / irrelevant / too_verbose）
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)                                 # 文字补充说明
    review_status: Mapped[str] = mapped_column(String(40), default="open", index=True)               # 人工处理状态：open / resolved / ignored
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)                             # 处理备注（怎么修的）
    notification_status: Mapped[str] = mapped_column(String(40), default="pending", index=True)      # 飞书通知状态：pending / sent / failed / skipped
    notification_error: Mapped[str | None] = mapped_column(Text, nullable=True)                      # 通知失败原因
    notified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)      # 通知发出时间
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)      # 人工处理时间
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)  # 反馈时间
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), index=True)  # 最后更新时间


class AgentWorkflowRun(Base):
    """多 Agent 工作流的整体运行记录（主表）。

    【业务含义】一次"复杂工程判断"（如"这个 PR 能不能合"）的整体执行记录。
    【谁写入】services/agents/workflow_orchestrator.py
    【谁读取】api/routes/chat.py 的 workflow-runs 接口、前端 Runs 页面

    与 AgentTaskRun 是"主-子"关系：一次工作流包含多个子任务。
    """

    __tablename__ = "agent_workflow_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 主键（agent_task_runs.workflow_id 指向它）
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True, index=True)  # 所属仓库
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("conversations.id"), nullable=True, index=True)  # 所属会话
    goal: Mapped[str] = mapped_column(Text)                                                          # 用户原始目标（如"这个 PR 能不能合"）
    spec_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)                            # PlannerAgent 产出的任务图（任务+依赖）
    observation_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)                     # ObserverAgent 的校验结论（问题清单+置信度）
    final_answer: Mapped[str] = mapped_column(Text, default="")                                       # SynthesisAgent 合成的最终结论
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)                              # 指标：耗时、token、重规划次数
    status: Mapped[str] = mapped_column(String(80), default="success", index=True)                   # 整体状态：success / failed / partial
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)  # 开始时间
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)     # 结束时间（运行中为 NULL）


class AgentTaskRun(Base):
    """工作流的单个子任务记录（子表）。

    【业务含义】"这次工作流里，PR 审查这个子任务做了什么、耗时多久、成功没有"。
    【谁写入】services/agents/workflow_orchestrator.py
    【谁读取】前端 Runs 页面的任务展开详情

    为什么要拆子表：
        一次工作流可能跑 5 个子任务，要能单独看每个任务的耗时和结果，
        放在一个 JSON 里无法按状态筛选（比如"找出所有 failed 的子任务"）。
    """

    __tablename__ = "agent_task_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 主键
    workflow_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("agent_workflow_runs.id"), index=True)  # 所属工作流（主-子关系）
    task_id: Mapped[str] = mapped_column(String(120), index=True)                                    # spec 里的任务 ID，如 "t1"（不是 UUID）
    agent_name: Mapped[str] = mapped_column(String(120), index=True)                                 # 执行该任务的 Agent：pr_review / ci_debug / issue_analysis ...
    task_type: Mapped[str] = mapped_column(String(120), index=True)                                  # 任务类型（决定走哪套执行逻辑）
    claim_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)                           # 这个任务要验证的断言（供 Observer 校验）
    input_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)                           # 实际传给 Agent 的输入
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)                          # Agent 输出的结构化结果
    status: Mapped[str] = mapped_column(String(80), default="success", index=True)                   # 状态：success / failed / skipped / error
    error: Mapped[str | None] = mapped_column(Text, nullable=True)                                   # 失败原因（skipped 时写"依赖失败"）
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)       # 开始时间
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)     # 结束时间（用于算单任务耗时）


class EvalRun(Base):
    """评测运行记录。

    【业务含义】一次 RAGAS / 基础评测的结果，用于回归对比（改 prompt 后有没有变差）。
    【谁写入】services/evals/runner.py、ragas_runner.py
    【谁读取】api/routes/evaluations.py、前端 Evals 页面

    为什么只存 result_json 一个大字段：
        评测结果结构复杂且会演进，拆成列会导致频繁改表。
        代价：无法按指标值直接 SQL 筛选。
    """

    __tablename__ = "eval_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 主键
    name: Mapped[str] = mapped_column(String(200))                                                   # 评测名称（区分不同批次/数据集）
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)                          # 全部指标与明细（结构会演进，所以不拆列）
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())  # 评测时间


# ════════════════════════════════════════════════════════════════
# 第七组：安全与审计（数据流 7）
# ════════════════════════════════════════════════════════════════

class ActionDraft(Base):
    """动作草稿表 —— 【AI 写 GitHub 的唯一合法入口】。

    【业务含义】AI 想改 GitHub（关 Issue / 加标签 / 评论）时，不能直接改，
    必须先产出一条草稿，等人工确认。
    【谁写入】services/agents/safety_agent.py（生成草稿）
    【谁读取】api/routes/action_drafts.py（确认/取消）、前端 Drafts 页面

    【对应数据流】流 7（AI 写操作 → 草稿 → 确认 → GitHub）

    ⚠️ 核心业务规则：
        status 必须是 pending_confirmation → confirmed → executed 才能真的改 GitHub。
        没有人工点"确认"，这条草稿永远不会执行。

    status 流转：
        pending_confirmation（等确认）
        → confirmed（已确认，执行中）
        → executed（成功）/ failed（失败）
        → cancelled（用户取消）
    """

    __tablename__ = "action_drafts"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 主键
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repositories.id"), index=True)            # 所属仓库
    draft_type: Mapped[str] = mapped_column(String(100), index=True)                                 # 动作类型：create_issue / comment / close_issue / add_labels
    target_type: Mapped[str] = mapped_column(String(100), index=True)                                # 作用对象类型：issue / pull_request
    target_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True, index=True)  # 作用对象 ID（新建类动作为空）
    title: Mapped[str] = mapped_column(String(300), default="")                                       # 草稿标题（新建 Issue 时用）
    content: Mapped[str] = mapped_column(Text, default="")                                            # 要写入 GitHub 的正文
    risk_level: Mapped[str] = mapped_column(String(80), default="medium", index=True)                # 风险等级：low / medium / high（high 前端显示红色警告）
    status: Mapped[str] = mapped_column(String(80), default="pending_confirmation", index=True)      # 状态：pending_confirmation / confirmed / executed / failed / cancelled
    created_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)  # 哪个 AI/用户生成的
    confirmed_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)  # 哪个人确认的（审计关键字段）
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)     # 实际执行时间
    execution_result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)             # GitHub 的返回结果（新对象 ID、URL 等）
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)                           # 执行失败原因
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)                     # 额外信息（生成依据、证据引用）
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)  # 草稿生成时间
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())  # 最后状态变更时间


class AuditLog(Base):
    """审计日志。

    【业务含义】所有敏感操作的"谁在什么时候干了什么"。
    【谁写入】services/permissions.py 的 write_audit_log()
    【谁读取】api/routes/action_drafts.py 的 audit-logs 接口

    为什么必须有：
        出了问题要能追溯。特别是"谁确认了那条关 Issue 的草稿"这种问题，
        没有审计日志就永远查不清责任。
    """

    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)  # 主键
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)  # 操作人（系统自动操作时为空）
    repo_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("repositories.id"), nullable=True, index=True)  # 所属仓库
    action: Mapped[str] = mapped_column(String(160), index=True)                                     # 动作名，如 "draft.confirm"
    target_type: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)          # 操作对象类型
    target_id: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)            # 操作对象 ID（字符串，兼容多种主键）
    status: Mapped[str] = mapped_column(String(80), default="success", index=True)                   # 结果：success / denied / error
    request_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)                         # 请求快照（当时的参数）
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)                          # 结果快照
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)  # 操作时间
