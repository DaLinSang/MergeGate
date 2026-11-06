"""
对话接口契约
============

【业务定位】"用户提问 → Agent 回答"这条主线的输入输出格式。业务四步中的"定位"靠它完成。
【数据流向】frontend/app/page.tsx（发请求） → api/routes/chat.py（处理） → 前端（渲染轨迹）
【核心规则】必须带 repo_id（否则上下文无意义）；每条回答都要带工具轨迹（让用户看到查了什么）。
"""

from uuid import UUID

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """一次对话请求。"""

    repo_id: UUID                                    # 站在哪个仓库
    conversation_id: UUID | None = None              # 哪条会话，空=新建
    message: str                                     # 用户说了什么
    context: dict = Field(default_factory=dict)      # 额外上下文（如选中的 Issue）


class ChatPlanExecuteRequest(ChatRequest):
    """计划模式的执行请求（用户确认计划后再跑）。"""

    workflow_spec: dict                              # 已确认的计划


class ToolCallTrace(BaseModel):
    """单次工具调用的轨迹。"""

    tool_name: str                                   # 工具名，如 workspace.read_file
    arguments: dict                                  # 调用参数
    status: str                                      # success / error / skipped
    error: str | None = None                         # 错误信息
    error_kind: str | None = None                    # 错误分类（决定是否可重试）
    retryable: bool | None = None                    # 是否可重试
    attempts: int | None = None                      # 重试次数
    tool_use_id: str | None = None                   # 与 LLM 的 tool_call id 对应
    skill_name: str | None = None                    # 激活的技能名
    skill_title: str | None = None                   # 技能标题
    skill_version: str | None = None                 # 技能版本
    skill_steps: list[str] = Field(default_factory=list)  # 技能规定的工作流步骤
    skill_activation_mode: str | None = None         # explicit（用户指定）/ trigger（关键词）
    skill_activation_reason: str | None = None       # 为什么激活
    skill_instruction_digest: str | None = None      # 指令摘要（复现用）


class ChatResponse(BaseModel):
    """一次对话的完整响应。"""

    run_id: UUID | str                               # 运行 ID（查 trace 用）
    message_id: UUID | str                           # 消息 ID（提交反馈用）
    route: str                                       # 走了哪条分支
    tool_calls: list[ToolCallTrace]                  # 工具调用轨迹
    agent_events: list[dict] = []                    # 细粒度事件（时间线）
    answer: str                                      # 最终答案
    citations: list[dict] = []                       # 引用来源
    completion: dict = {}                            # 生成元信息
    compression_stats: dict = {}                     # 压缩统计
    progressive_compaction: dict = {}                # 渐进压缩阶段
    model_usage: list[dict] = []                     # token 用量
    context_diagnostics: dict = {}                   # 上下文压力诊断


class ChatHistoryMessage(BaseModel):
    """历史消息（带过程字段，前端渲染历史时用）。"""

    id: UUID | str                                   # 消息 ID
    conversation_id: UUID | str | None = None        # 所属会话
    role: str                                        # user / assistant
    content: str                                     # 正文
    route: str | None = None                         # 当时走的处理分支
    tool_calls: list[ToolCallTrace] = []             # 当时的工具轨迹
    agent_events: list[dict] = []                    # 当时的事件
    compression_stats: dict = {}                     # 当时的压缩统计
    progressive_compaction: dict = {}                # 当时的压缩阶段
    model_usage: list[dict] = []                     # 当时的 token 用量
    context_diagnostics: dict = {}                   # 当时的上下文诊断
    created_at: str | None = None                    # 创建时间


class PromptRecommendationRequest(BaseModel):
    """推荐提问的请求。"""

    repo_id: UUID                                        # 针对哪个仓库
    exclude: list[str] = Field(default_factory=list)     # 排除已推荐过的
    limit: int = Field(default=5, ge=4, le=5)            # 推荐条数（4~5）


class PromptRecommendationResponse(BaseModel):
    """推荐提问的响应。"""

    agent: str                                       # 由哪个 Agent 生成
    suggestions: list[str]                           # 推荐的问题列表
