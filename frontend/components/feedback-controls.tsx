"use client";
// 有 useState / useEffect / 点击事件，必须是客户端组件

import { Check, LoaderCircle, ThumbsDown, ThumbsUp, X } from "lucide-react";
import { useEffect, useState } from "react";

import { apiPost } from "@/lib/api";

/**
 * 回答反馈控件（👍 / 👎）
 * =====================
 *
 * 【业务定位】挂在每条 AI 回答下方，收集"这条答得好不好"，是评测之外的**线上真实信号**。
 *
 * 【数据流向】
 *    好评：点 👍 → 直接提交 → POST /api/chat/feedback
 *    差评：点 👎 → 先开面板选原因 + 填备注 → 提交
 *    → chat_feedback 表
 *    → 若为首次差评 → services/feedback_notifications.py 发飞书通知
 *    → 面板上显示 "trace xxxxxxxx"，可跳 /feedback/trace/{run_id} 回放整条执行链
 *
 * 【要点】
 *    1. reason / comment 只在 unhelpful 时提交，好评传 null —— 避免脏数据混进差评原因分布
 *    2. comment 空串要转 null，否则后端统计里会多出一堆空备注
 *    3. 提交期间 pending 为真：两个按钮都禁用，防止重复提交出两条记录
 *    4. useEffect 依赖 initialFeedback：父组件重新拉数据时覆盖本地 state，否则显示的还是旧状态
 */

/** 一条反馈记录（对应后端 schemas/feedback.py::ChatFeedbackResponse）。 */
export type ChatFeedbackRecord = {
  id: string;                                   // 反馈 ID
  repo_id: string;                              // 所属仓库（隔离维度）
  conversation_id: string;                      // 所属会话
  assistant_message_id: string;                 // 被评价的那条 AI 消息
  run_id?: string | null;                       // 该轮执行的 trace ID，用于回放排障
  rating: "helpful" | "unhelpful";              // 好 / 差
  reason?: FeedbackReason | null;               // 差评原因（好评为空）
  comment?: string | null;                      // 补充说明（可选）
  review_status: "open" | "in_review" | "resolved" | "dismissed";  // 人工处理状态
  notification_status: string;                  // 飞书通知状态（sent / failed / skipped）
  notification_error?: string | null;           // 通知失败原因（不影响反馈本身已落库）
};

/** 差评原因。前端只做展示，真正的枚举校验在后端。 */
type FeedbackReason =
  | "inaccurate"            // 内容不准确
  | "not_relevant"          // 没有解决问题
  | "missing_context"       // 缺少关键上下文
  | "unreliable_citation"   // 引用不可靠
  | "tool_error"            // 工具执行异常
  | "other";                // 其他

type FeedbackControlsProps = {
  repoId: string;                               // 仓库 ID（必传，后端按它做隔离）
  conversationId: string;                       // 会话 ID
  messageId: string;                            // 消息 ID（一条消息只有一条反馈，重复提交是更新）
  initialFeedback?: ChatFeedbackRecord;         // 历史反馈（刷新页面后回显已选状态）
  onSaved: (feedback: ChatFeedbackRecord) => void;  // 保存成功回调，父组件据此更新列表
};

// 差评原因选项。顺序 = 产品希望用户优先选的顺序（越靠前越常见）
const REASONS: Array<{ value: FeedbackReason; label: string }> = [
  { value: "inaccurate", label: "内容不准确" },
  { value: "not_relevant", label: "没有解决问题" },
  { value: "missing_context", label: "缺少关键上下文" },
  { value: "unreliable_citation", label: "引用不可靠" },
  { value: "tool_error", label: "工具执行异常" },
  { value: "other", label: "其他" }
];

/** 反馈按钮组 + 差评面板。 */
export function FeedbackControls({
  repoId,
  conversationId,
  messageId,
  initialFeedback,
  onSaved
}: FeedbackControlsProps) {
  const [feedback, setFeedback] = useState(initialFeedback);            // 已保存的反馈
  const [panelOpen, setPanelOpen] = useState(false);                    // 差评面板是否展开
  const [reason, setReason] = useState<FeedbackReason | undefined>(initialFeedback?.reason ?? undefined);
  const [comment, setComment] = useState(initialFeedback?.comment ?? "");  // 受控 textarea
  const [pending, setPending] = useState(false);                        // 请求进行中
  const [error, setError] = useState("");                               // 行内错误提示

  // 父组件换了消息（列表重新渲染 / 切换会话）时，把本地状态同步回服务端的值
  useEffect(() => {
    setFeedback(initialFeedback);
    setReason(initialFeedback?.reason ?? undefined);
    setComment(initialFeedback?.comment ?? "");
  }, [initialFeedback]);

  /** 提交反馈。好评直接调，差评带上原因和备注。 */
  async function submit(rating: "helpful" | "unhelpful") {
    setPending(true);
    setError("");
    try {
      const result = await apiPost<ChatFeedbackRecord>("/api/chat/feedback", {
        repo_id: repoId,
        conversation_id: conversationId,
        assistant_message_id: messageId,
        rating,
        // 好评不带原因：后端按 unhelpful 才统计原因分布
        reason: rating === "unhelpful" ? reason : null,
        // 空白备注转 null，避免存一堆空字符串
        comment: rating === "unhelpful" ? comment.trim() || null : null
      });
      // 以服务端返回为准（可能带上了 run_id / 通知状态），不做乐观更新
      setFeedback(result);
      setPanelOpen(false);
      onSaved(result);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "反馈提交失败");
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="mt-3 border-t border-line pt-2">
      <div className="flex min-h-8 items-center gap-1">
        {/* 👍 一键好评，不弹面板 */}
        <button
          type="button"
          title="回答有帮助"
          aria-label="回答有帮助"
          aria-pressed={feedback?.rating === "helpful"}
          disabled={pending}
          onClick={() => void submit("helpful")}
          className={`flex h-8 w-8 items-center justify-center rounded-md transition disabled:cursor-not-allowed disabled:opacity-50 ${
            feedback?.rating === "helpful" ? "bg-emerald-50 text-emerald-700" : "text-slate-400 hover:bg-slate-100 hover:text-slate-700"
          }`}
        >
          {/* 面板没开时才在按钮上转圈；面板开着的加载态由提交按钮自己显示 */}
          {pending && !panelOpen ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <ThumbsUp className="h-4 w-4" />}
        </button>
        {/* 👎 先开面板收集原因，不立即提交 */}
        <button
          type="button"
          title="回答没帮助"
          aria-label="回答没帮助"
          aria-pressed={feedback?.rating === "unhelpful"}
          disabled={pending}
          onClick={() => {
            setPanelOpen(true);
            setError("");
          }}
          className={`flex h-8 w-8 items-center justify-center rounded-md transition disabled:cursor-not-allowed disabled:opacity-50 ${
            feedback?.rating === "unhelpful" ? "bg-rose-50 text-rose-700" : "text-slate-400 hover:bg-slate-100 hover:text-slate-700"
          }`}
        >
          <ThumbsDown className="h-4 w-4" />
        </button>
        {/* 已记录提示：带 trace 短码，方便按 run_id 回放这一轮的执行轨迹 */}
        {feedback && !panelOpen && (
          <span className="ml-1 flex min-w-0 items-center gap-1 text-xs text-slate-500">
            <Check className="h-3.5 w-3.5 shrink-0 text-emerald-600" />
            已记录
            {feedback.run_id && <span className="truncate font-mono text-[11px]">trace {feedback.run_id.slice(0, 8)}</span>}
          </span>
        )}
      </div>

      {/* ── 差评面板：选原因 + 填备注 + 提交 ── */}
      {panelOpen && (
        <div className="mt-2 max-w-xl rounded-md border border-rose-100 bg-rose-50/60 p-3">
          <div className="flex items-center justify-between gap-3">
            <span className="text-xs font-medium text-slate-700">这条回答哪里需要改进？</span>
            <button
              type="button"
              title="关闭"
              aria-label="关闭反馈"
              onClick={() => setPanelOpen(false)}
              className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-slate-400 hover:bg-white hover:text-slate-700"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            {/* 原因单选：再点一次不会取消（保持"至少选一个"的语义） */}
            {REASONS.map((item) => (
              <button
                key={item.value}
                type="button"
                aria-pressed={reason === item.value}
                onClick={() => setReason(item.value)}
                className={`rounded-md border px-2.5 py-1 text-xs transition ${
                  reason === item.value
                    ? "border-rose-300 bg-white text-rose-700"
                    : "border-slate-200 bg-white text-slate-600 hover:border-slate-300"
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>
          {/* maxLength 与后端字段长度一致，超长会在提交时被 422 拒掉，这里先挡住 */}
          <textarea
            value={comment}
            maxLength={1000}
            rows={2}
            onChange={(event) => setComment(event.target.value)}
            placeholder="补充说明（可选）"
            className="mt-2 w-full resize-y rounded-md border border-slate-200 bg-white px-3 py-2 text-xs leading-5 text-slate-700 outline-none focus:border-rose-300 focus:ring-2 focus:ring-rose-100"
          />
          <div className="mt-2 flex items-center justify-between gap-3">
            {/* 错误就地显示，不用 toast —— 面板开着，用户视线就在这儿 */}
            <span className="text-xs text-rose-600">{error}</span>
            <button
              type="button"
              disabled={pending}
              onClick={() => void submit("unhelpful")}
              className="flex h-8 shrink-0 items-center gap-1.5 rounded-md bg-slate-800 px-3 text-xs font-medium text-white hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {pending && <LoaderCircle className="h-3.5 w-3.5 animate-spin" />}
              提交反馈
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
