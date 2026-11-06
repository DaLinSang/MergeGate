"""
LLM 调用客户端
=============

【业务定位】
    对 OpenAI 兼容 /chat/completions 接口的薄封装。全项目所有"调模型"的地方都走这里。

【数据流向】
    上游（谁调它）：
        services/agents/*          → chat_text / chat_json（Agent 决策）
        services/chat_memory.py    → chat_text（压缩摘要）、chat_json（记忆提炼）
        services/rag/qa.py         → chat_text（生成答案）
        services/evals/*           → chat_json（评测打分）
    下游（它调谁）：
        settings.llm_base_url / llm_api_key / llm_model
        context_compression.prompt_cache_model_kwargs()   prompt cache 参数

【核心规则】
    1. **没配 key 就静默返回空**（"" 或 {}），绝不抛异常。
       让上层走自己的降级路径（抽取式答案 / 规则兜底），而不是整条链路崩掉。
    2. **HTTP 错误也返回空**，理由同上。
    3. 三个方法对应三种用途：结构化 JSON / 纯文本 / 流式文本。

【本文件函数地图】
    LLMClient
        ├ chat_json()         ★ 要结构化输出（带 response_format）
        ├ chat_text()         要一段文本（摘要、答案）
        └ chat_text_stream()  流式（SSE，逐 token 吐）
"""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.core.config import settings
from app.services.context_compression import prompt_cache_model_kwargs


class LLMClient:
    """LLM 调用客户端（无状态，每次调用新建一个 httpx 客户端）。

    【为什么不做连接池复用】
        本项目调用频率不高（一轮对话 1~3 次），每次新建连接的开销可接受。
        换成长连接会引入"连接失效要重连"的复杂度，不划算。
    """

    def __init__(self) -> None:
        self.base_url = settings.llm_base_url.rstrip("/")
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model

    async def chat_json(self, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        """调模型并解析成 JSON 对象。

        【调用链】prompt_cache_model_kwargs() → httpx.post → json.loads → 本函数
        【被谁调用】
            services/chat_memory.py::SessionSealer._build_memory_update()  记忆提炼
            services/agents/*                                              Agent 结构化决策
            services/evals/*                                               评测

        【数据流向】
            入：system_prompt（指令）、user_payload（要序列化成 JSON 的数据）
            出：dict（模型返回的 JSON）；失败时返回 {}

        【关键设计：response_format={"type": "json_object"}】
            告诉服务端"必须返回合法 JSON"。
            这是让模型输出结构化数据的标准做法 —— 比在 prompt 里写
            "请返回 JSON" 可靠得多（后者经常被包在 ```json 代码块里）。

        【为什么 user 内容要 json.dumps】
            把结构化输入（如"已有记忆 + 本轮消息"）整体序列化，
            模型更容易理解字段边界，也避免 prompt 注入（数据不会被当指令解析）。

        【失败时为什么返回 {} 而不是抛异常】
            调用方全部按"空 = 走兜底"设计（如 fallback_memory_update）。
            抛异常会让每个调用点都要写 try/except，而且容易漏。
        """
        # 没配 key → 静默返回空（调用方会走兜底）
        if not self.api_key:
            return {}

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            # ★ 强制 JSON 输出
            "response_format": {"type": "json_object"},
        }
        # 合并 prompt cache 参数（启用后 system prompt 会被缓存，省 90% 成本）
        payload.update(prompt_cache_model_kwargs())

        async with httpx.AsyncClient(timeout=60) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return json.loads(content)
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
                # 网络错 / 响应结构不对 / JSON 解析失败 → 一律返回空
                return {}

    async def chat_text(self, system_prompt: str, user_message: str) -> str:
        """调模型并返回纯文本。

        【调用链】prompt_cache_model_kwargs() → httpx.post → 本函数
        【被谁调用】
            services/chat_memory.py::_summary_for()   压缩摘要
            services/rag/qa.py::answer_question()     生成答案
            services/agents/*                         各 Agent 的最终输出

        【数据流向】
            入：system_prompt、user_message（普通字符串，不是 JSON）
            出：模型返回的文本；失败时返回 ""

        【和 chat_json 的区别】
            不加 response_format（允许自由文本），user 内容直接是字符串。
            注意这里的异常处理是"只捕 HTTPStatusError" ——
            因为 response.json() 解析失败会抛别的异常，那样会往上冒泡。
            这是原实现的取舍：非 HTTP 错误说明是代码问题，应该暴露出来。
        """
        if not self.api_key:
            return ""

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        payload.update(prompt_cache_model_kwargs())

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                return ""
            return str(response.json()["choices"][0]["message"].get("content") or "")

    async def chat_text_stream(self, system_prompt: str, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """流式调模型，逐块 yield 文本。

        【调用链】prompt_cache_model_kwargs() → httpx.stream → 逐行解析 SSE → 本函数
        【被谁调用】api/routes/chat.py（SSE 推给前端）

        【数据流向】
            入：system_prompt、messages（完整对话历史，不是单条）
            出：异步生成器，每次 yield 一个文本增量

        【SSE 协议解析（三行核心逻辑）】
            ① 只处理 "data: " 开头的行（其余是空行/注释/event 行）
            ② data 为 "[DONE]" 表示结束
            ③ 每块是 JSON，取 choices[0].delta.content（注意是 delta，不是 message）

        【为什么用 messages 而不是单条 user_message】
            流式场景通常是多轮对话，要把完整历史都传进去。

        【注意：没配 key 时直接 return】
            生成器里 return 就是"不 yield 任何东西"，调用方的 async for 自然结束。
        """
        if not self.api_key:
            return

        # system prompt 放在最前面，后面跟完整历史
        payload_messages = [{"role": "system", "content": system_prompt}, *messages]
        payload: dict[str, Any] = {"model": self.model, "messages": payload_messages, "stream": True}
        payload.update(prompt_cache_model_kwargs())

        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            ) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError:
                    return

                async for line in response.aiter_lines():
                    # ① 只要 data 行
                    if not line.startswith("data: "):
                        continue
                    data = line.removeprefix("data: ").strip()
                    # ② 结束标记
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        # 单块解析失败不影响后续（跳过即可）
                        continue
                    # ③ 取增量内容
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield str(content)
