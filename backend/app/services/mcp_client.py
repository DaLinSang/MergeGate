"""
MCP 客户端（stdio 传输）
=======================

【业务定位】
    以子进程方式启动 MCP server，通过 stdin/stdout 说 JSON-RPC 2.0。
    用途：把记忆工具（search_evidence / get_thread_context / read_session_events）
    暴露成标准 MCP 工具，让外部 MCP 客户端（Claude Desktop、Cursor 等）也能调用。

【数据流向】
    上游（谁调它）：
        api/routes/mcp.py（诊断页展示"有哪些工具"）
        services/agents/chat_agent.py（按需调工具）
    下游（它调谁）：
        子进程 app/mcp/memory_server.py
            └ 内部会调 services/chat_memory.py 的 MemoryTools

【核心规则】
    1. 每次调用都是"起进程 → 握手 → 请求 → 杀进程"（无状态）。
    2. 所有异常统一转成 McpClientError（上层只需捕一种）。
    3. 单次请求超时 30 秒，超时立即报错（不能无限等子进程）。

【本文件函数地图】
    StdioMcpClient
        ├ call_tool()         ★ 调一个工具（完整生命周期）
        ├ inspect_server()    握手 + 列工具（诊断用）
        ├ _initialize()       发 initialize 请求
        ├ _start()            起子进程
        ├ _notify()           发通知（不需要响应）
        ├ _request()          ★ 发请求并等响应（含超时与错误处理）
        ├ _write()            写一行 JSON
        └ _stop()             杀进程
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from app.core.config import settings


class McpClientError(RuntimeError):
    """MCP 通信失败（进程起不来 / 超时 / 返回非法 JSON / 服务端报错）。

    【为什么统一成一种异常】
        上层（route / agent）只关心"调用失败了"，不需要区分是哪一步失败。
        细节都在异常消息里，日志能看到。
    """
    pass


class StdioMcpClient:
    """stdio 传输的 MCP 客户端。

    【为什么用 stdio 而不是 HTTP】
        MCP 最轻量的传输方式：不用端口、不用认证、进程退出即清理。
        适合"本地工具服务"这种场景。
    """

    def __init__(self, command: str | None = None, args: list[str] | None = None, cwd: str | None = None) -> None:
        # parents[2]：从 app/services/mcp_client.py 往上三层 → backend/
        backend_root = Path(__file__).resolve().parents[2]
        # 默认用当前 Python 解释器（保证依赖一致）
        self.command = command or sys.executable
        # 默认启动项目自带的 memory_server.py
        self.args = args or [str(backend_root / "app" / "mcp" / "memory_server.py")]
        self.cwd = cwd or str(backend_root)
        # JSON-RPC 请求 ID 自增（用于匹配请求和响应）
        self._next_id = 1

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """调用一个 MCP 工具。

        【调用链】_start() → _initialize() → _notify() → _request() → _stop() → 本函数
        【被谁调用】services/agents/chat_agent.py（工具分发）

        【数据流向】
            入：name（工具名）、arguments（参数）
            出：result dict（MCP 协议里的 result 字段）

        【为什么每次都要起进程】
            无状态设计：用完就杀，不会残留进程、不会串会话。
            代价是每次多几百毫秒启动开销 —— 对低频工具调用可接受。

        【finally 里必须 _stop()】
            即使请求失败也要杀进程，否则子进程会泄漏（越积越多）。
        """
        process = await self._start()
        try:
            # ① 握手
            await self._initialize(process)
            # ② 通知服务端"客户端已就绪"（MCP 协议要求，不需要响应）
            await self._notify(process, "notifications/initialized", {})
            # ③ 真正调工具
            return await self._request(process, "tools/call", {"name": name, "arguments": arguments})
        finally:
            await self._stop(process)

    async def inspect_server(self) -> dict[str, Any]:
        """握手并列出服务端提供的工具（诊断用）。

        【调用链】_start() → _initialize() → _notify() → _request("tools/list") → _stop()
        【被谁调用】api/routes/mcp.py（前端"MCP 状态"页面）

        Perform a real MCP handshake and return the server's advertised tools.

        【和 call_tool 的区别】
            只列工具，不执行。用于验证"服务端能不能起来、有哪些能力"。
        """
        process = await self._start()
        try:
            initialize = await self._initialize(process)
            await self._notify(process, "notifications/initialized", {})
            tools = await self._request(process, "tools/list", {})
            return {"initialize": initialize, "tools": tools.get("tools") or []}
        finally:
            await self._stop(process)

    async def _initialize(self, process: asyncio.subprocess.Process) -> dict[str, Any]:
        """发 initialize 请求（MCP 协议的第一步）。

        【调用链】_request() → 本函数

        【三个字段】
            protocolVersion  协议版本（"2024-11-05" 是当前稳定版）
            capabilities     客户端能力（本项目不声明任何能力，所以是 {}）
            clientInfo       客户端标识（服务端日志里能看到是谁连的）
        """
        return await self._request(
            process,
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "devflow-web-agent", "version": "0.1.0"},
            },
        )

    async def _start(self) -> asyncio.subprocess.Process:
        """启动 MCP server 子进程。

        【调用链】asyncio.create_subprocess_exec() → 本函数
        【被谁调用】call_tool() / inspect_server()

        【三个环境变量，都是必要的】
            PYTHONIOENCODING=utf-8   子进程的 stdin/stdout 用 UTF-8
                                     （否则 Windows 上中文会乱码，JSON 直接解析失败）
            DATABASE_URL             子进程要连同一个数据库（读消息、记忆）
            AUTO_SYNC_ENABLED=false  ★ 子进程绝不能开定时同步
                                     （否则每起一个 MCP 进程就多一个同步任务）

        【为什么用 setdefault 而不是直接赋值】
            AUTO_SYNC_ENABLED 允许外部显式覆盖（比如调试时想开）。
            另外两个是硬要求，所以直接赋值。

        【三个 PIPE】
            stdin  → 我们写请求
            stdout → 我们读响应（JSON-RPC 是"一行一个消息"）
            stderr → 只在出错时读（拿错误信息）
        """
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["DATABASE_URL"] = settings.database_url
        env.setdefault("AUTO_SYNC_ENABLED", "false")
        process = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            cwd=self.cwd,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        return process

    async def _notify(self, process: asyncio.subprocess.Process, method: str, params: dict[str, Any]) -> None:
        """发一条通知（没有 id，服务端不回响应）。

        【调用链】_write() → 本函数

        JSON-RPC 里"有 id = 请求（要响应）"，"没 id = 通知（不要响应）"。
        """
        await self._write(process, {"jsonrpc": "2.0", "method": method, "params": params})

    async def _request(self, process: asyncio.subprocess.Process, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """发请求并等响应（★ 本文件的核心）。

        【调用链】_write() → readline() → json.loads() → 本函数
        【被谁调用】call_tool() / inspect_server() / _initialize()

        【数据流向】
            入：process、method、params
            出：响应里的 result（dict）

        【五道错误处理，缺一不可】
            ① 没有 stdout       → 进程起错了
            ② 读超时（30s）      → 服务端卡死
            ③ 读到空行          → 服务端崩了（顺手把 stderr 读出来，这是最有用的排查信息）
            ④ JSON 解析失败     → 服务端输出了非协议内容（比如 print 调试信息混进 stdout）
            ⑤ 响应里有 error    → 服务端业务报错

        【为什么超时是 30 秒】
            MCP 工具可能查数据库、跑检索。30 秒足够，太长会让前端一直转圈。
        """
        request_id = self._next_id
        self._next_id += 1
        await self._write(process, {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})

        # ① 没有 stdout
        if process.stdout is None:
            raise McpClientError("MCP process has no stdout")

        # ② 读一行（带超时）
        try:
            line = await asyncio.wait_for(process.stdout.readline(), timeout=30)
        except TimeoutError as exc:
            raise McpClientError(f"MCP request timed out: {method}") from exc

        # ③ 空行 = 进程已退出
        if not line:
            stderr = ""
            if process.stderr is not None:
                stderr = (await process.stderr.read()).decode("utf-8", errors="replace")
            raise McpClientError(f"MCP server closed stdout. {stderr}")

        # ④ 解析 JSON
        try:
            response = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise McpClientError(f"MCP server returned an invalid JSON response for {method}") from exc

        # ⑤ 服务端报错
        if "error" in response:
            error = response["error"]
            raise McpClientError(str(error.get("message") or error))

        return response.get("result") or {}

    async def _write(self, process: asyncio.subprocess.Process, payload: dict[str, Any]) -> None:
        """写一行 JSON 到子进程 stdin。

        【调用链】被 _notify() / _request() 调用

        【为什么必须加 "\\n"】
            JSON-RPC over stdio 是"一行一条消息"。
            不加换行，服务端的 readline() 会一直等，永远收不到完整消息。

        【为什么要 drain()】
            写入是缓冲的，不 drain 可能导致数据一直留在缓冲区没发出去。
        """
        if process.stdin is None:
            raise McpClientError("MCP process has no stdin")
        process.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        await process.stdin.drain()

    async def _stop(self, process: asyncio.subprocess.Process) -> None:
        """结束子进程（kill + 等待回收）。

        【调用链】被 call_tool() / inspect_server() 的 finally 调用

        【为什么要判 returncode is None】
            进程可能已经自己退出了。对已退出的进程调 kill() 会抛 ProcessLookupError。

        【为什么要 await wait()】
            kill 只是发信号，不 wait 会留下僵尸进程（zombie）。
        """
        if process.returncode is None:
            process.kill()
        await process.wait()
