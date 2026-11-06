"use client";

import { useEffect, useMemo, useState } from "react";

import { apiGet, apiPost } from "@/lib/api";
import type { Repository } from "@/lib/types";
import { Card, PageTitle, PrimaryButton } from "@/components/ui/card";

/**
 * 团队工作区（多仓库聚合）
 * =======================
 *
 * 【业务定位】单仓库周报不够用时，把多个仓库合成一个 workspace，出跨仓库周报。
 *
 * 【数据流向】
 *    创建：POST /api/workspaces {name, repo_ids} → workspaces 表（repo_ids 是数组）
 *    报表：POST /api/workspaces/multi-repo-report
 *          → 二选一：传 workspace_id（按已存的组合）或传 repo_ids（按当前勾选）
 *          → 每个仓库跑一遍统计 → 汇总 metrics + repo_summaries + report_markdown
 *
 * 【要点】
 *    1. workspace_id 与 repo_ids 互斥：传了 id 就传空数组，后端只认一个来源
 *    2. 默认勾选前两个仓库，减少一次点击
 *    3. 日期同样用 toISOString（UTC），东八区凌晨会偏一天
 */

type Workspace = {
  id: string;
  name: string;
  description?: string | null;
  repo_ids: string[];               // 成员仓库（多对多，直接存 ID 数组）
};

type MultiRepoReport = {
  report_markdown: string;                              // 汇总后的 Markdown 周报
  metrics: Record<string, number>;                      // 全局指标（总 PR 数、总失败 CI…）
  repo_summaries: Array<Record<string, unknown>>;       // 每个仓库一段小结
};

/** Date → "YYYY-MM-DD"（UTC 日期，非本地时区） */
function toDateInput(date: Date) {
  return date.toISOString().slice(0, 10);
}

export default function WorkspacesPage() {
  const today = useMemo(() => new Date(), []);
  const start = useMemo(() => {
    const value = new Date(today);
    value.setDate(value.getDate() - 6);   // 近 7 天
    return value;
  }, [today]);
  const [repos, setRepos] = useState<Repository[]>([]);
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [name, setName] = useState("研发团队");
  const [selectedRepos, setSelectedRepos] = useState<string[]>([]);
  const [workspaceId, setWorkspaceId] = useState("");
  const [startDate, setStartDate] = useState(toDateInput(start));
  const [endDate, setEndDate] = useState(toDateInput(today));
  const [report, setReport] = useState<MultiRepoReport | null>(null);
  const [loading, setLoading] = useState(false);

  /** 拉仓库 + workspace 两个列表。用 Promise.all 并发，少一次往返。 */
  async function load() {
    const [repoItems, workspaceItems] = await Promise.all([apiGet<Repository[]>("/api/repos"), apiGet<Workspace[]>("/api/workspaces")]);
    setRepos(repoItems);
    setWorkspaces(workspaceItems);
    setSelectedRepos(repoItems.slice(0, 2).map((repo) => repo.id));   // 默认勾选前两个
    setWorkspaceId(workspaceItems[0]?.id || "");
  }

  useEffect(() => {
    load().catch(() => undefined);   // 首屏失败静默，页面仍可手动操作
  }, []);

  /** 创建 workspace。成功后重新 load()，让列表和下拉框都刷新。 */
  async function createWorkspace() {
    if (!name.trim()) return;
    setLoading(true);
    try {
      await apiPost<Workspace>("/api/workspaces", { name, repo_ids: selectedRepos });
      await load();
    } finally {
      setLoading(false);
    }
  }

  /** 生成多仓库周报。workspace_id 与 repo_ids 只能给一个。 */
  async function generateReport() {
    setLoading(true);
    try {
      const data = await apiPost<MultiRepoReport>("/api/workspaces/multi-repo-report", {
        workspace_id: workspaceId || undefined,
        repo_ids: workspaceId ? [] : selectedRepos,   // 选了 workspace 就忽略勾选，避免两个来源打架
        start_date: startDate,
        end_date: endDate
      });
      setReport(data);
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <PageTitle title="团队工作区" description="把多个仓库聚合成 workspace，生成跨仓库研发周报和风险概览。" />
      <div className="grid gap-4 lg:grid-cols-[380px_1fr]">
        {/* 左栏：创建 workspace + 报表范围 */}
        <Card>
          <label className="text-sm font-medium text-slate-700">
            Workspace 名称
            <input className="mt-2 w-full rounded-md border border-line px-3 py-2 outline-none" value={name} onChange={(event) => setName(event.target.value)} />
          </label>
          <div className="mt-4 text-sm font-medium text-slate-700">仓库</div>
          <div className="mt-2 grid gap-2">
            {repos.map((repo) => (
              <label key={repo.id} className="flex items-center gap-2 rounded-md border border-line px-3 py-2 text-sm">
                {/* 勾选即增删 ID：用 filter 做不可变删除，避免直接 splice 改原数组 */}
                <input
                  type="checkbox"
                  checked={selectedRepos.includes(repo.id)}
                  onChange={(event) => {
                    setSelectedRepos((current) => (event.target.checked ? [...current, repo.id] : current.filter((id) => id !== repo.id)));
                  }}
                />
                {repo.full_name}
              </label>
            ))}
          </div>
          <PrimaryButton className="mt-4" disabled={loading || selectedRepos.length === 0} onClick={createWorkspace}>
            创建 Workspace
          </PrimaryButton>

          <div className="mt-6 grid gap-3 border-t border-line pt-4">
            <label className="text-sm font-medium text-slate-700">
              报表范围
              <select className="mt-2 w-full rounded-md border border-line px-3 py-2" value={workspaceId} onChange={(event) => setWorkspaceId(event.target.value)}>
                {/* 空值 = 用上面勾选的仓库；选了某项 = 用已存的 workspace */}
                <option value="">使用上方勾选仓库</option>
                {workspaces.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name}
                  </option>
                ))}
              </select>
            </label>
            <input className="rounded-md border border-line px-3 py-2" type="date" value={startDate} onChange={(event) => setStartDate(event.target.value)} />
            <input className="rounded-md border border-line px-3 py-2" type="date" value={endDate} onChange={(event) => setEndDate(event.target.value)} />
            <PrimaryButton disabled={loading || (!workspaceId && selectedRepos.length === 0)} onClick={generateReport}>
              生成多仓库周报
            </PrimaryButton>
          </div>
        </Card>

        {/* 右栏：workspace 列表 + 周报正文 */}
        <div className="grid gap-4">
          <Card>
            <h2 className="font-semibold text-ink">Workspace 列表</h2>
            <div className="mt-3 grid gap-2">
              {workspaces.map((item) => (
                <div key={item.id} className="rounded-md border border-line bg-panel px-3 py-2 text-sm text-slate-700">
                  {item.name} · {item.repo_ids.length} 个仓库
                </div>
              ))}
              {workspaces.length === 0 && <div className="text-sm text-slate-500">暂无 workspace。</div>}
            </div>
          </Card>
          <Card>
            <h2 className="font-semibold text-ink">多仓库周报</h2>
            {report ? (
              <pre className="mt-4 max-h-[620px] overflow-auto rounded-md border border-slate-800 bg-slate-950 p-4 text-sm leading-7 text-slate-100">{report.report_markdown}</pre>
            ) : (
              <div className="mt-4 rounded-md border border-line bg-panel p-4 text-sm text-slate-600">创建 workspace 或勾选仓库后生成周报。</div>
            )}
          </Card>
        </div>
      </div>
    </>
  );
}
