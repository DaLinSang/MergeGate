"""
MCP 子包
========

【模块职责】
    存放以 MCP（Model Context Protocol）协议对外暴露能力的服务端实现。
    目前只有一个：memory_server —— 把 DevFlow 的记忆能力暴露成 3 个工具。

【在系统中的位置】
    services/mcp_client.py（客户端，起子进程）
      ⇄ stdio JSON-RPC ⇄
    mcp/memory_server.py（服务端，本包）

【为什么单独成包而不是放 services 里】
    它是一个**独立可执行进程**（`python -m app.mcp.memory_server`），
    有自己的入口和生命周期，不属于"被 import 的业务逻辑"。
    放 services/ 会让人误以为它是个普通模块。
"""
