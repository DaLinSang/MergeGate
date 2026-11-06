"use client";

import { useEffect, useMemo, useState } from "react";
import { Check, CheckCircle2, Clock3, FileText, Send, ShieldAlert, X } from "lucide-react";

import { apiGet, apiPost } from "@/lib/api";
import type { Repository } from "@/lib/types";
import { PageTitle, PrimaryButton } from "@/components/ui/card";

/**
 * 安全草稿与人工确认页（数据流 7 的前端）
 * ======================================
 *
 * 【业务定位】项目最重要的安全设计落地页：AI 不能直接改 GitHub，
 *            所有外部写操作先变成"草稿"，必须有人点确认才执行。
 *
 * 【数据流向】
 *    创建：POST /api/action-drafts → action_drafts 表（status=pending_confirmation）
 *    确认：POST /api/action-drafts/{id}/confirm
 *          → api/routes/action_drafts.py → services/action_drafts.py::_execute_github_draft
 *          → 真正调 GitHub API（建 Issue / 评论 / 关单 / 打标签）
 *          → status 改 executed + 写 audit_logs
 *    取消：POST /api/action-drafts/{id}/cancel → status=cancelled
 *    审计：GET /api/action-drafts/audit-logs → 右下角时间线
 *
 * 【要点】
 *    1. 状态机是单向的：pending_confirmation → executed / cancelled / failed，没有回退接口
 *    2. 失败不抛异常：后端落 status=failed + error_message，所以前端要单独渲染错误块
 *    3. 确认按钮只在 pending_confirmation 时出现 —— 状态判断在前端，权限校验在后端
 */

type ActionDraft = {
  id: string;
  repo_id: string;
  draft_type: string;                       // 动作类型：create_issue / comment / close_issue / add_labels / send_report
  target_type: string;                      // 作用对象：repository / issue / pull_request
  title: string;
  content: string;                          // 待审查的正文（确认后原样发给 GitHub）
  risk_level: string;                       // low / medium / high
  status: string;                           // pending_confirmation / executed / cancelled / failed
  execution_result?: Record<string, unknown> | null;   // 执行成功后的返回（如新 Issue 的 html_url）
  error_message?: string | null;            // 执行失败原因
  created_at: string;
  executed_at?: string | null;
};

type AuditLog = { id: string; action: string; status: string; target_type?: string | null; target_id?: string | null; created_at: string };

// 状态 → 文案 + 配色。查不到的状态原样显示（便于发现后端新增状态）
const statusMeta: Record<string, { label: string; className: string }> = {
  pending_confirmation: { label: "等待人工确认", className: "bg-amber-50 text-amber-800 border-amber-200" },
  executed: { label: "已确认并执行", className: "bg-emerald-50 text-emerald-700 border-emerald-200" },
  cancelled: { label: "已取消", className: "bg-slate-100 text-slate-600 border-slate-200" },
  failed: { label: "执行失败", className: "bg-rose-50 text-rose-700 border-rose-200" }
};

export default function ActionDraftsPage() {
  const [repos, setRepos] = useState<Repository[]>([]);
  const [repoId, setRepoId] = useState("");
  const [drafts, setDrafts] = useState<ActionDraft[]>([]);
  const [audits, setAudits] = useState<AuditLog[]>([]);
  const [title, setTitle] = useState("发布检查报告");
  const [content, setContent] = useState("");
  const [loading, setLoading] = useState(false);

  /** 拉草稿 + 审计两路。审计失败不影响草稿展示（catch 成空数组）。 */
  async function load(repo = repoId) {
    const suffix = repo ? `?repo_id=${repo}` : "";
    const [draftItems, auditItems] = await Promise.all([
      apiGet<ActionDraft[]>(`/api/action-drafts${suffix}`),
      apiGet<AuditLog[]>(`/api/action-drafts/audit-logs${suffix}`).catch(() => [])
    ]);
    setDrafts(draftItems);
    setAudits(auditItems);
  }

  // 首屏：先拿仓库，再按第一个仓库拉数据
  useEffect(() => {
    apiGet<Repository[]>("/api/repos")
      .then((items) => {
        setRepos(items);
        setRepoId(items[0]?.id || "");
        return load(items[0]?.id || "");
      })
      .catch(() => setRepos([]));
  }, []);

  /** 创建草稿。这里写死 draft_type=send_report / risk_level=medium，因为页面只做"发报告"这一种。 */
  async function createDraft() {
    if (!repoId || !content.trim()) return;
    setLoading(true);
    try {
      await apiPost<ActionDraft>("/api/action-drafts", {
        repo_id: repoId,
        draft_type: "send_report",
        target_type: "repository",
        title: title.trim() || "发布检查报告",
        content,
        risk_level: "medium",
        metadata: { source: "manual_review" }   // 标记来源：手工创建，区别于 Agent 创建
      });
      setContent("");
      await load();   // 重新拉列表，拿到后端生成的 id / 状态
    } finally {
      setLoading(false);
    }
  }

  /** 确认或取消。两个动作共用一个函数，靠 URL 末段区分。 */
  async function changeDraft(id: string, action: "confirm" | "cancel") {
    setLoading(true);
    try {
      await apiPost(`/api/action-drafts/${id}/${action}`, {});
      await load();   // 成功后重拉，状态与审计都会变
    } finally {
      setLoading(false);
    }
  }

  // 顶部三个统计。用 useMemo 避免每次渲染都遍历两遍
  const pendingCount = useMemo(() => drafts.filter((draft) => draft.status === "pending_confirmation").length, [drafts]);
  const executedCount = useMemo(() => drafts.filter((draft) => draft.status === "executed").length, [drafts]);

  return (
    <>
      <PageTitle title="安全草稿与人工确认" description="Agent 的外部写操作先落成可审查草稿；只有具备批准权限的人确认后，系统才执行并写入审计记录。" />

      {/* 顶部统计：全部 / 待确认 / 已执行 */}
      <section className="grid gap-3 border-y border-line bg-white px-4 py-4 sm:grid-cols-3">
        <Summary icon={FileText} label="全部草稿" value={drafts.length} />
        <Summary icon={Clock3} label="待确认" value={pendingCount} tone="amber" />
        <Summary icon={CheckCircle2} label="已执行" value={executedCount} tone="green" />
      </section>

      <div className="mt-4 grid gap-5 xl:grid-cols-[360px_minmax(0,1fr)]">
        {/* 左栏：手工创建草稿（Agent 也会创建，走同一个表） */}
        <section className="self-start rounded-md border border-line bg-white p-5">
          <div className="flex items-center gap-2 text-sm font-semibold text-ink"><ShieldAlert className="h-4 w-4 text-amber-600" />创建安全草稿</div>
          <p className="mt-2 text-xs leading-5 text-slate-500">这里只保存建议内容，不会直接写入外部系统。</p>
          <label className="mt-4 block text-sm font-medium text-slate-700">仓库
            {/* 切仓库时立即重拉，不等 useEffect —— 否则会闪一下旧数据 */}
            <select className="mt-2 w-full rounded-md border border-line bg-white px-3 py-2 outline-none focus:border-teal-500" value={repoId} onChange={(event) => { setRepoId(event.target.value); void load(event.target.value); }}>
              {repos.map((repo) => <option key={repo.id} value={repo.id}>{repo.full_name}</option>)}
            </select>
          </label>
          <label className="mt-4 block text-sm font-medium text-slate-700">标题
            <input className="mt-2 w-full rounded-md border border-line px-3 py-2 outline-none focus:border-teal-500" value={title} onChange={(event) => setTitle(event.target.value)} />
          </label>
          <label className="mt-4 block text-sm font-medium text-slate-700">待审查内容
            <textarea className="mt-2 min-h-40 w-full resize-none rounded-md border border-line px-3 py-2 text-sm leading-6 outline-none focus:border-teal-500" value={content} onChange={(event) => setContent(event.target.value)} placeholder="输入准备发送的报告或评论内容" />
          </label>
          <PrimaryButton className="mt-4 w-full gap-2" disabled={!repoId || !content.trim() || loading} onClick={createDraft}><FileText className="h-4 w-4" />保存为待确认草稿</PrimaryButton>
        </section>

        {/* 右栏：草稿列表 + 审计时间线 */}
        <div className="space-y-4">
          {drafts.map((draft) => {
            const meta = statusMeta[draft.status] || { label: draft.status, className: "bg-panel text-slate-600 border-line" };
            return (
              <article key={draft.id} className="rounded-md border border-line bg-white p-5">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <h2 className="font-semibold text-ink">{draft.title || draft.draft_type}</h2>
                      <span className={`rounded-full border px-2.5 py-1 text-xs font-medium ${meta.className}`}>{meta.label}</span>
                      {/* 风险徽章只有 high 会标红，其余都按"中低"处理 */}
                      <span className={`rounded-full px-2.5 py-1 text-xs ${draft.risk_level === "high" ? "bg-rose-50 text-rose-700" : "bg-slate-100 text-slate-600"}`}>{draft.risk_level === "high" ? "高风险" : "中低风险"}</span>
                    </div>
                    <div className="mt-2 text-xs text-slate-500">{draft.draft_type} · {draft.target_type} · {new Date(draft.created_at).toLocaleString("zh-CN")}</div>
                  </div>
                  {/* 只有待确认状态才给操作按钮：已执行/已取消的不允许再点 */}
                  {draft.status === "pending_confirmation" && (
                    <div className="flex gap-2">
                      <button className="inline-flex items-center gap-1 rounded-md bg-teal-600 px-3 py-2 text-sm font-medium text-white transition hover:bg-teal-700 disabled:opacity-60" disabled={loading} onClick={() => changeDraft(draft.id, "confirm")}><Check className="h-4 w-4" />确认执行</button>
                      <button className="inline-flex items-center gap-1 rounded-md border border-line px-3 py-2 text-sm text-slate-600 transition hover:bg-panel disabled:opacity-60" disabled={loading} onClick={() => changeDraft(draft.id, "cancel")}><X className="h-4 w-4" />取消</button>
                    </div>
                  )}
                </div>
                {/* whitespace-pre-wrap 保留换行，草稿正文的排版和发出后一致 */}
                <div className="mt-4 whitespace-pre-wrap rounded-md border-l-4 border-teal-500 bg-panel px-4 py-3 text-sm leading-7 text-slate-700">{draft.content}</div>
                {draft.status === "executed" && <div className="mt-3 flex items-center gap-2 text-sm text-emerald-700"><CheckCircle2 className="h-4 w-4" />人工确认完成，执行结果和操作者已经写入审计记录。</div>}
                {/* 失败不抛异常，错误落在这里，必须渲染否则用户不知道为什么没成功 */}
                {draft.error_message && <div className="mt-3 rounded-md bg-rose-50 p-3 text-sm text-rose-700">{draft.error_message}</div>}
              </article>
            );
          })}
          {drafts.length === 0 && <div className="rounded-md border border-dashed border-line bg-white p-5 text-sm text-slate-500">暂无安全草稿。</div>}

          {/* 审计时间线：只显示最近 10 条 */}
          <section className="rounded-md border border-line bg-white p-5">
            <div className="flex items-center gap-2 font-semibold text-ink"><Send className="h-4 w-4 text-teal-700" />审计时间线</div>
            <div className="mt-4 space-y-0">
              {audits.slice(0, 10).map((item, index) => (
                <div key={item.id} className="relative flex gap-3 pb-4">
                  {/* 竖线：最后一条不画，避免线头悬空 */}
                  {index < Math.min(audits.length, 10) - 1 && <span className="absolute left-[7px] top-4 h-full w-px bg-line" />}
                  <span className={`relative mt-1 h-4 w-4 shrink-0 rounded-full border-4 border-white ${item.status === "success" ? "bg-emerald-500" : "bg-rose-500"}`} />
                  <div><div className="text-sm font-medium text-ink">{auditLabel(item.action)}</div><div className="mt-1 text-xs text-slate-500">{item.status} · {new Date(item.created_at).toLocaleString("zh-CN")}</div></div>
                </div>
              ))}
              {audits.length === 0 && <div className="text-sm text-slate-500">暂无审计记录。</div>}
            </div>
          </section>
        </div>
      </div>
    </>
  );
}

/** 顶部统计块。tone 决定图标颜色，默认 teal。 */
function Summary({ icon: Icon, label, value, tone = "teal" }: { icon: typeof FileText; label: string; value: number; tone?: "teal" | "amber" | "green" }) {
  const color = tone === "amber" ? "text-amber-600" : tone === "green" ? "text-emerald-600" : "text-teal-700";
  return <div className="flex items-center gap-3"><Icon className={`h-5 w-5 ${color}`} /><div><div className="text-xs text-slate-500">{label}</div><div className="mt-1 text-xl font-semibold text-ink">{value}</div></div></div>;
}

/** 审计动作名 → 中文。用 startsWith 匹配，因为 confirm 后面会拼上 draft_type。 */
function auditLabel(action: string) {
  if (action.startsWith("action_draft.confirm")) return "人工确认并执行草稿";
  if (action === "action_draft.create") return "创建待确认草稿";
  if (action === "action_draft.cancel") return "取消草稿";
  return action;   // 未知动作原样显示，方便发现新增类型
}
