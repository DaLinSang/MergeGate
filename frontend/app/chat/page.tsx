"use client";

import { useEffect, useState } from "react";

import { Card, PageTitle, PrimaryButton } from "@/components/ui/card";
import { apiGet, apiPostStream } from "@/lib/api";
import type { Repository } from "@/lib/types";

/**
 * Agent 对话页（调试版）
 * =====================
 *
 * 【业务定位】数据流 4 的最简入口：一个输入框 + 一条 SSE 流，用来看 Agent 的原始输出。
 *            正式交互在首页（app/page.tsx），这个页面偏调试：结果直接打 JSON。
 *
 * 【数据流向】
 *    POST /api/chat/stream
 *      → api/routes/chat.py::stream_chat
 *      → 四步准备：ensure_headroom（压缩）→ append 用户消息 → rebuild EvidenceStore → ContextAssembler
 *      → ChatAgent 推理循环（LangGraph，最多 N 轮工具调用）
 *      → SSE 事件流：delta（增量文本）/ tool（工具调用）/ final（最终结果）
 *    → 结果只在本页 state 里，不落库（要落库得带 conversation_id）
 *
 * 【要点】
 *    1. context 里的 messages 传空数组：历史由后端按 conversation_id 取，前端不自己拼
 *    2. time_range: "7d" 是给 Agent 的时间窗提示，影响"最新 CI"这类相对时间的解析
 *    3. final 事件会用完整 JSON 覆盖已流式追加的文本 —— 调试用，方便看结构化字段
 */

// 示例问题，点一下填进输入框。覆盖五类工具：CI / PR / Issue / 报告 / 读代码
const samples = [
  "Why did the latest CI fail?",
  "Summarize the risk in the latest PR",
  "Who should own the latest Issue?",
  "Generate this week's engineering report",
  "Read backend/app/services/agents/chat_agent.py and explain tool calling",
];

export default function ChatPage() {
  const [repos, setRepos] = useState<Repository[]>([]);
  const [repoId, setRepoId] = useState("");
  const [message, setMessage] = useState("Why did the latest CI fail?");
  const [result, setResult] = useState("");          // 流式累加的文本 / 最终 JSON
  const [loading, setLoading] = useState(false);

  // 拉仓库列表；用函数式 set 保留用户已选的仓库（避免覆盖手动选择）
  useEffect(() => {
    apiGet<Repository[]>("/api/repos").then((items) => {
      setRepos(items);
      setRepoId((current) => current || items[0]?.id || "");
    });
  }, []);

  /** 发起对话并消费 SSE 流。 */
  async function send() {
    setLoading(true);
    setResult("");   // 每次提问先清空，避免上一轮的答案残留
    try {
      await apiPostStream("/api/chat/stream", { repo_id: repoId, message, context: { time_range: "7d", messages: [] } }, (event, rawData) => {
        // delta：模型逐字输出的增量，直接往后拼
        if (event === "delta") {
          const data = rawData as { content?: string };
          if (data.content) setResult((current) => current + data.content);
        }
        // final：整轮结束的结构化结果，覆盖上面的文本（调试用，能看到 answer / citations / tool_calls）
        if (event === "final") {
          setResult(JSON.stringify(rawData, null, 2));
        }
      });
    } finally {
      // 无论成功失败都解除 loading，否则按钮会一直转
      setLoading(false);
    }
  }

  return (
    <>
      <PageTitle
        title="Agent Chat"
        description="Use native tool calling to inspect code, search memory, run specialized analysis, and draft safe actions."
      />
      <Card>
        {/* 仓库选择：决定 Agent 能访问哪个仓库的数据（工具闭包里锁死 repo_id） */}
        <select
          className="mb-3 w-full rounded-md border border-line bg-white px-3 py-2 text-sm"
          value={repoId}
          onChange={(event) => setRepoId(event.target.value)}
        >
          {repos.map((repo) => (
            <option key={repo.id} value={repo.id}>{repo.full_name}</option>
          ))}
        </select>
        <textarea
          className="h-32 w-full rounded-md border border-line bg-white px-3 py-3 text-sm leading-6 outline-none transition focus:border-teal-500 focus:ring-2 focus:ring-teal-100"
          value={message}
          onChange={(event) => setMessage(event.target.value)}
        />

        <div className="mt-3 flex flex-wrap gap-2 text-xs text-slate-500">
          {samples.map((sample) => (
            <button
              key={sample}
              onClick={() => setMessage(sample)}
              className="rounded-full border border-line bg-panel px-3 py-1.5 transition hover:border-teal-200 hover:bg-teal-50"
            >
              {sample}
            </button>
          ))}
        </div>

        {/* 三个条件都满足才能发：没在加载、选了仓库、问题非空 */}
        <PrimaryButton onClick={send} disabled={loading || !repoId || !message.trim()} className="mt-4">
          {loading ? "Analyzing..." : "Ask Agent"}
        </PrimaryButton>

        {result && <pre className="mt-4 max-h-[560px] overflow-auto rounded-md border border-slate-800 bg-slate-950 p-4 text-xs text-slate-100">{result}</pre>}
      </Card>
    </>
  );
}
