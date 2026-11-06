"""
Skill 与 MCP 契约
===============

【业务定位】Skill（技能说明书）和 MCP（外部工具协议）的展示格式。
【数据流向】app/skills/*/SKILL.md（技能文件） → registry.py → api/routes/skills.py → 前端
【核心规则】技能是数据不是代码（Markdown 写的），领域专家也能改；safety_level 标明能否写数据。
"""

from typing import Any

from pydantic import BaseModel, Field


class SkillManifestResponse(BaseModel):
    """技能的清单信息（对应 SKILL.md 的 YAML 头）。"""

    name: str                                               # 技能标识，如 pr-review
    title: str                                              # 中文标题
    version: str                                            # 版本号
    description: str                                        # 一句话说明
    category: str                                           # 分类
    entrypoint: str                                         # 入口工具名
    tools: list[str] = Field(default_factory=list)          # 工具白名单（显式指定时只能用这些）
    input_modes: list[str] = Field(default_factory=list)    # 支持的输入模式
    triggers: list[str] = Field(default_factory=list)       # 触发词（命中则自动激活）
    workflow_steps: list[str] = Field(default_factory=list) # 规定的工作流步骤
    output_contract: str                                    # 输出契约名
    safety_level: str                                       # read_only / write


class SkillResponse(BaseModel):
    """技能详情（含完整指令正文）。"""

    manifest: SkillManifestResponse     # 清单信息
    path: str                           # 技能文件路径
    instructions: str | None = None     # 完整指令正文（会被拼进 prompt）


class SkillListResponse(BaseModel):
    """技能列表。"""

    skills: list[SkillResponse]         # 所有已注册技能


class McpToolResponse(BaseModel):
    """MCP 服务提供的一个工具。"""

    name: str                                               # 工具名
    description: str                                        # 说明（决定 LLM 何时调用）
    input_schema: dict[str, Any] = Field(default_factory=dict)  # 参数 schema


class McpRuntimeResponse(BaseModel):
    """MCP 服务的运行时状态。"""

    status: str                                             # ready / unavailable
    server_name: str                                        # 服务名
    server_version: str                                     # 服务版本
    protocol_version: str                                   # 协议版本
    transport: str                                          # 传输方式：stdio
    tools: list[McpToolResponse] = Field(default_factory=list)  # 工具列表
