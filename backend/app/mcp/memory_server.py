#!/usr/bin/env python
"""
DevFlow 记忆 MCP Server（stdio）
==============================

【模块职责】
    把 DevFlow 的"记忆能力"暴露成标准 MCP 工具，供 ChatAgent 调用。
    这是一个**独立进程**，通过 stdin/stdout 收发 JSON-RPC 消息（MCP 的 stdio 传输）。

【在系统中的位置】
    ChatAgent（services/agents/chat_agent.py）
      → 调用工具 devflow_search_evidence / devflow_get_thread_context /
        devflow_read_session_events
      → services/mcp_client.py::StdioMcpClient  起子进程、发 JSON-RPC
      → ★ 本文件（子进程）接收请求
      → services/chat_memory.py::MemoryTools   真正干活
      → PostgreSQL（chat_messages / chat_sessions / evidence_items / documents）

    另一条调用路径（不经过本文件）：
      api/routes/chat.py::_build_conversation_tools() 里也有同名工具的**直接实现**
      （call_memory_mcp），它同样走 StdioMcpClient → 本文件。
      也就是说本文件是"记忆工具"的唯一实现，两条路都汇到这里。

【三个工具的分工】
    devflow_search_evidence      语义检索：回忆"历史上有什么相关的东西"
    devflow_get_thread_context   逐字读原文：回忆"当时到底怎么说的"（可跨压缩边界）
    devflow_read_session_events  读封存会话：回忆"那次会话的整体过程和工具调用"

【要点】
    1. 【stdio 协议的铁律】stdout 只能输出 JSON-RPC 消息，一个消息一行。
       任何调试打印都必须走 stderr（`print(..., file=sys.stderr)`），
       否则会把 JSON 流污染，客户端解析直接崩。
    2. 【每个请求建一次 Session】MCP 是长驻进程，但请求之间不该共享 Session ——
       避免一个请求的异常/未提交事务影响下一个。
    3. 【必须先 create_all】这个进程可能比 web 服务先启动（由客户端拉起），
       表还不存在时直接查会报错，所以每次调用前都 ensure 一次（幂等，开销很小）。
    4. 【错误码用 JSON-RPC 标准】-32700 解析错误 / -32601 方法不存在 /
       -32602 参数无效 / -32603 内部错误。客户端（mcp_client.py）按码区分处理。
    5. 【必须 reconfigure 编码】Windows 默认不是 UTF-8，
       不显式设成 utf-8 会让中文在管道里变成乱码。
"""

import asyncio
import json
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any


# 让本文件能被 `python app/mcp/memory_server.py` 直接跑起来（不依赖 PYTHONPATH）。
# 因为它是被当脚本启动的，不在 app 包内，所以要先手动把 backend/ 加进 sys.path。
BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from sqlalchemy.orm import Session  # noqa: E402

from app.db.models import Conversation, Repository  # noqa: E402
from app.db.session import Base, SessionLocal, engine  # noqa: E402
from app.services.chat_memory import MemoryTools  # noqa: E402
from app.services.conversations import ensure_conversation  # noqa: E402


SERVER_NAME = "devflow-memory-mcp"
SERVER_VERSION = "0.1.0"

# 【必须】三个流都设成 UTF-8。stdio 管道在部分平台默认用本地编码，中文会乱码。
for stream in (sys.stdin, sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def _json_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    """拼一个 MCP 工具参数 schema。

    additionalProperties=False：禁止模型传未声明的参数 ——
    否则模型会自创参数（比如自己传 limit=1000），导致行为不可预期。
    """
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


# ── 工具定义（MCP 的 tools/list 就返回这个数组）──
# description 是写给模型看的，所以要明确"什么时候用我、什么时候别用我"，
# 否则模型会在该用结构化工具时误用记忆检索。
TOOLS: list[dict[str, Any]] = [
    {
        "name": "devflow_search_evidence",
        "description": (
            "搜索 DevFlow 仓库知识和持久记忆。"
            "当需要回忆历史 Issue/PR、失败 CI、上传知识、项目文档、会话摘要、Agent 运行记录或已批准决策时，优先使用这个工具。"
            "当前代码、团队成员和实时状态应使用工作区或结构化工具。"
        ),
        "inputSchema": _json_schema(
            {
                "query": {"type": "string", "description": "搜索查询"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "description": "最大结果数"},
                "repo_id": {"type": "string", "description": "必填仓库 UUID。"},
                "conversation_id": {"type": "string", "description": "可选 DevFlow 会话 UUID。"},
            },
            ["query", "repo_id"],
        ),
    },
    {
        "name": "devflow_get_thread_context",
        "description": (
            "读取 DevFlow 仓库会话的完整原始聊天转录，可跨越 Compact Boundary。"
            "支持关键词、消息 UUID、前后范围和字符偏移分页；当用户询问之前说过什么或需要逐字核实时使用。"
        ),
        "inputSchema": _json_schema(
            {
                "keyword": {"type": "string", "description": "可选关键词过滤"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "description": "最大消息数"},
                "message_id": {"type": "string", "description": "可选消息 UUID；用于分页读取一条精确原文。"},
                "before_message_id": {"type": "string", "description": "可选消息 UUID；只读取它之前的记录。"},
                "after_message_id": {"type": "string", "description": "可选消息 UUID；只读取它之后的记录。"},
                "offset": {"type": "integer", "minimum": 0, "description": "读取单条消息时的字符偏移量。"},
                "max_chars": {"type": "integer", "minimum": 200, "maximum": 16000, "description": "本次最多返回的正文字符数。"},
                "repo_id": {"type": "string", "description": "必填仓库 UUID。"},
                "conversation_id": {"type": "string", "description": "可选 DevFlow 会话 UUID。"},
            },
            ["repo_id"],
        ),
    },
    {
        "name": "devflow_read_session_events",
        "description": (
            "读取已封存的 DevFlow 聊天会话、摘要、原始消息和工具事件。"
            "在 search_evidence 之后，或需要会话级回忆时使用。"
        ),
        "inputSchema": _json_schema(
            {
                "session_id": {"type": "string", "description": "可选会话片段 UUID，默认使用最近会话。"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "description": "最大会话数"},
                "include_transcript": {"type": "boolean", "description": "是否返回封存的原始消息和工具事件，默认 true。"},
                "max_chars": {"type": "integer", "minimum": 400, "maximum": 20000, "description": "本次最多返回的正文字符数。"},
                "repo_id": {"type": "string", "description": "必填仓库 UUID。"},
                "conversation_id": {"type": "string", "description": "可选 DevFlow 会话 UUID。"},
            },
            ["repo_id"],
        ),
    },
]


class McpError(Exception):
    """带 JSON-RPC 错误码的业务异常。

    和普通 Exception 的区别：这个会被转成规范的 error 响应，
    普通异常统一兜成 -32603（内部错误）。
    """

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _content(text: str) -> dict[str, Any]:
    """包成 MCP 标准的 content 数组格式（客户端只认 type=text 的项）。"""
    return {"content": [{"type": "text", "text": text}]}


def _resolve_repo(db: Session, args: dict[str, Any]) -> Repository:
    """解析并校验 repo_id。

    【安全】必须校验仓库存在 —— 否则模型传一个不存在的 UUID，
    后面所有查询都会静默返回空，模型会误判"没有相关记忆"。
    """
    repo_id = args.get("repo_id")
    if not repo_id:
        raise McpError(-32602, "repo_id 为必填项")
    try:
        repo = db.get(Repository, uuid.UUID(str(repo_id)))
    except ValueError as exc:
        raise McpError(-32602, f"repo_id 无效：{repo_id}") from exc
    if repo is None:
        raise McpError(-32602, f"未找到仓库：{repo_id}")
    return repo


def _resolve_conversation(db: Session, repo: Repository, args: dict[str, Any]) -> Conversation:
    """解析会话。没传或无效就兜底到该仓库的默认会话（不报错）。

    为什么兜底而不是报错：记忆检索在"全局"视角下也有意义，
    模型经常不传 conversation_id，直接拒绝会浪费一次工具调用。
    """
    conversation_id = args.get("conversation_id")
    if conversation_id:
        try:
            return ensure_conversation(db, repo.id, uuid.UUID(str(conversation_id)))
        except ValueError as exc:
            raise McpError(-32602, f"conversation_id 无效：{conversation_id}") from exc
    return ensure_conversation(db, repo.id)


def _repo_label(repo: Repository) -> str:
    """返回给模型看的仓库标识（owner/repo）。"""
    return repo.full_name


async def _call_memory_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """工具分发的核心：建 Session → 校验参数 → 调 MemoryTools → 包装返回。"""
    # 幂等建表：本进程可能先于 web 服务启动
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        repo = _resolve_repo(db, args)
        conversation = _resolve_conversation(db, repo, args)
        # MemoryTools 是真正的实现（在 services/chat_memory.py 里）
        tools = MemoryTools(db, repo, conversation)

        # 根据name确定走哪个
        if name == "devflow_search_evidence":
            query = str(args.get("query") or "").strip()
            if not query:
                raise McpError(-32602, "query 为必填项")
            result = await tools.search_evidence({"query": query, "limit": int(args.get("limit") or 5)}, {})
            # 记一条召回事件（db.commit 落库），让"记忆被谁用过"可观测
            db.commit()
            # 返回里带"仓库 / 会话"抬头，模型才知道这段证据属于哪个上下文
            return _content(f"仓库：{_repo_label(repo)}\n会话：{conversation.title if conversation else '全局'}\n\n{result.get('answer') or ''}")

        if name == "devflow_get_thread_context":
            result = await tools.get_thread_context(
                {
                    "keyword": str(args.get("keyword") or ""),
                    "limit": int(args.get("limit") or 30),
                    "message_id": args.get("message_id"),
                    "before_message_id": args.get("before_message_id"),
                    "after_message_id": args.get("after_message_id"),
                    "offset": int(args.get("offset") or 0),
                    "max_chars": int(args.get("max_chars") or 6000),
                },
                {},
            )
            db.commit()
            return _content(f"仓库：{_repo_label(repo)}\n会话：{conversation.title if conversation else '全局'}\n\n{result.get('answer') or ''}")

        if name == "devflow_read_session_events":
            result = await tools.read_session_events(
                {
                    "session_id": args.get("session_id"),
                    "limit": int(args.get("limit") or 5),
                    # 默认返回原文：不返回的话模型只能看到摘要，没法逐字核对
                    "include_transcript": bool(args.get("include_transcript", True)),
                    "max_chars": int(args.get("max_chars") or 8000),
                },
                {},
            )
            db.commit()
            return _content(f"仓库：{_repo_label(repo)}\n会话：{conversation.title if conversation else '全局'}\n\n{result.get('answer') or ''}")

        # 走到这里说明 tools/list 声明了但 tools/call 没实现 —— 属于内部 bug
        raise McpError(-32601, f"未知工具：{name}")
    finally:
        # 每个请求都关 Session：避免长驻进程里事务/连接泄漏
        db.close()


def _initialize() -> dict[str, Any]:
    """initialize 响应：告诉客户端协议版本、服务名、支持的能力。

    listChanged=False：工具列表固定（TOOLS 是常量），不需要通知客户端刷新。
    """
    return {
        "protocolVersion": "2024-11-05",
        "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        "capabilities": {"tools": {"listChanged": False}},
    }


async def handle_request(request: dict[str, Any]) -> dict[str, Any] | None:
    """处理一条 JSON-RPC 请求。返回 None 表示"不需要响应"（通知类消息）。

    MCP 握手流程：
      客户端 → initialize        （本函数回 _initialize）
      客户端 → notifications/initialized  （通知，无响应）
      客户端 → tools/list        （回工具清单）
      客户端 → tools/call        （回工具结果）
    """
    request_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}

    # 通知消息没有 id，按 JSON-RPC 规范不能回复
    if method == "notifications/initialized":
        return None

    try:
        if method == "initialize":
            result = _initialize()
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            name = str(params.get("name") or "")
            arguments = params.get("arguments") or {}
            # 参数必须是对象：模型有时会传字符串化的 JSON，这里直接拒绝更清晰
            if not isinstance(arguments, dict):
                raise McpError(-32602, "arguments 必须是对象")
            result = await _call_memory_tool(name, arguments)
        else:
            raise McpError(-32601, f"未找到方法：{method}")
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except McpError as exc:
        # 业务错误：用自己带的码
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": exc.code, "message": exc.message}}
    except Exception as exc:
        # 未预期异常：堆栈打到 stderr（stdout 不能污染），只把消息回给客户端
        print(traceback.format_exc(), file=sys.stderr)
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32603, "message": str(exc)}}


async def main() -> None:
    """主循环：逐行读 stdin → 解析 JSON → 处理 → 写 stdout。

    为什么一行一条消息：stdio 传输没有消息边界，
    JSON-RPC over stdio 的约定就是"一行一个 JSON 对象"。
    """
    # 启动日志走 stderr（stdout 是协议通道）
    print(f"[{SERVER_NAME}] 正在 stdio 上运行", file=sys.stderr)
    while True:
        # readline 是阻塞 IO，用 to_thread 包一下，避免卡住事件循环
        line = await asyncio.to_thread(sys.stdin.readline)
        if not line:
            # EOF：客户端关闭了管道，正常退出
            break
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            # -32700 解析错误。注意 id 为 None —— 连请求都解析不出来，不知道是哪个请求
            response = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}}
        else:
            response = await handle_request(request)
        if response is not None:
            # flush=True 必须加：否则响应会卡在缓冲区，客户端一直等
            print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
