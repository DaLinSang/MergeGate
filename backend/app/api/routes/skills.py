"""
Skill 与 MCP 状态路由
=====================

【路由清单】
    GET /skills                 列出所有已注册的 Skill（不含 instructions）
    GET /skills/mcp-status      探测 MCP Server 是否在线，并列出它提供的工具
    GET /skills/{skill_name}    查单个 Skill 详情（含 instructions 全文）

【数据流向】
    services/skills/registry.py::get_skill_registry()  → 本文件
    services/mcp_client.py::StdioMcpClient.inspect_server() → 本文件

【要点】
    1. 列表接口刻意不返回 instructions（可能几万字），详情接口才给 —— 省带宽。
    2. MCP 状态是"每次请求都真连一次"（stdio 起进程），所以慢；前端只在设置页调用。
"""

from fastapi import APIRouter, HTTPException

from app.schemas.skills import McpRuntimeResponse, McpToolResponse, SkillListResponse, SkillManifestResponse, SkillResponse
from app.services.mcp_client import McpClientError, StdioMcpClient
from app.services.skills import get_skill_registry

router = APIRouter()


def _skill_response(skill, include_instructions: bool = False) -> SkillResponse:
    """把内部 Skill 对象转成响应模型。

    include_instructions=False 时 instructions 置 None —— 列表接口用它，
    避免一次返回几十万字的提示词。
    """
    return SkillResponse(
        manifest=SkillManifestResponse.model_validate(skill.manifest.model_dump()),
        path=skill.path,
        instructions=skill.instructions if include_instructions else None,
    )


@router.get("", response_model=SkillListResponse)
async def list_skills() -> SkillListResponse:
    """列出全部 Skill 的元信息（名字 / 描述 / 触发词 / 工具白名单）。"""
    registry = get_skill_registry()
    return SkillListResponse(skills=[_skill_response(skill) for skill in registry.list_skills()])


@router.get("/mcp-status", response_model=McpRuntimeResponse)
async def get_mcp_status() -> McpRuntimeResponse:
    """实时探测 MCP Server：起一个 stdio 子进程，握手 + 列工具。"""
    try:
        inspection = await StdioMcpClient().inspect_server()
    except (McpClientError, OSError) as exc:
        # 503 而不是 500：这是"依赖服务不可用"，不是本服务代码出错
        raise HTTPException(status_code=503, detail=f"MCP Server 验证失败：{exc}") from exc

    # 下面全是在啃 MCP 协议的嵌套结构，逐层兜底成 "unknown"，
    # 避免某个字段缺失导致整个响应 500。
    initialize = inspection.get("initialize") or {}
    server_info = initialize.get("serverInfo") or {}
    tools = [
        McpToolResponse(
            name=str(item.get("name") or ""),
            description=str(item.get("description") or ""),
            input_schema=item.get("inputSchema") or {},
        )
        for item in inspection.get("tools") or []
    ]
    return McpRuntimeResponse(
        status="online",
        server_name=str(server_info.get("name") or "unknown"),
        server_version=str(server_info.get("version") or "unknown"),
        protocol_version=str(initialize.get("protocolVersion") or "unknown"),
        transport="stdio",
        tools=tools,
    )


@router.get("/{skill_name}", response_model=SkillResponse)
async def get_skill(skill_name: str) -> SkillResponse:
    """查单个 Skill 详情，含 instructions 全文（前端"技能详情"抽屉用）。"""
    registry = get_skill_registry()
    skill = registry.get_skill(skill_name)
    if skill is None:
        raise HTTPException(status_code=404, detail="未找到 Skill")
    return _skill_response(skill, include_instructions=True)
