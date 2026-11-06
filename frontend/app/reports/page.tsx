"use client";

import { useEffect, useMemo, useState } from "react";

import { Card, PageTitle, PrimaryButton } from "@/components/ui/card";
import { apiGet, apiPost } from "@/lib/api";
import type { Repository } from "@/lib/types";

/**
 * 研发周报生成页
 * ==============
 *
 * 【业务定位】四步主线的"沉淀"环节：把一周的 Issue / PR / CI 事实压成一段能直接发群的 Markdown。
 *
 * 【数据流向】
 *    POST /api/reports/weekly {repo_id, start_date, end_date}
 *      → api/routes/reports.py::weekly_report
 *      → 统计 issues / pull_requests / workflow_runs 三张表
 *      → ReportAgent 生成 Markdown（模板填充 + LLM 润色）
 *      → 返回 report_markdown，前端直接当纯文本展示
 *
 * 【要点】
 *    1. 默认时间窗 = 今天往前 6 天（含今天共 7 天）
 *    2. 日期用 toISOString 取，是 UTC 日期 —— 东八区 00:00~08:00 之间会算成前一天（已知偏差）
 *    3. 周报是纯文本 <pre> 渲染，没接 Markdown 渲染器，所以不会出现富文本样式
 */

type WeeklyReportResponse = {
  report_markdown: string;      // 后端生成好的 Markdown 全文
};

/** Date → "YYYY-MM-DD"。注意 toISOString 是 UTC，不是本地时区。 */
function toDateInput(date: Date) {
  return date.toISOString().slice(0, 10);
}

export default function ReportsPage() {
  // useMemo 固定"今天"，避免每次渲染都 new Date() 导致默认日期跳动
  const today = useMemo(() => new Date(), []);
  const weekStart = useMemo(() => {
    const date = new Date(today);
    date.setDate(date.getDate() - 6);   // 近 7 天 = 今天 - 6
    return date;
  }, [today]);

  const [repos, setRepos] = useState<Repository[]>([]);
  const [repoId, setRepoId] = useState("");
  const [startDate, setStartDate] = useState(toDateInput(weekStart));
  const [endDate, setEndDate] = useState(toDateInput(today));
  const [report, setReport] = useState("");
  const [loading, setLoading] = useState(false);

  // 拉仓库列表并默认选第一个；失败时清空（不弹错，用户看到空下拉就知道没仓库）
  useEffect(() => {
    apiGet<Repository[]>("/api/repos")
      .then((items) => {
        setRepos(items);
        setRepoId(items[0]?.id || "");
      })
      .catch(() => setRepos([]));
  }, []);

  /** 生成周报。日期原样透传，后端按 [start, end] 闭区间统计。 */
  async function generateReport() {
    if (!repoId) return;
    setLoading(true);
    try {
      const data = await apiPost<WeeklyReportResponse>("/api/reports/weekly", {
        repo_id: repoId,
        start_date: startDate,
        end_date: endDate
      });
      setReport(data.report_markdown);
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <PageTitle title="研发周报生成" description="按时间范围统计 Issues、PRs、Merged PRs、Open PRs 和失败 CI，生成适合发给团队负责人的 Markdown 周报。" />

      <div className="grid gap-4 lg:grid-cols-[360px_1fr]">
        {/* 左栏：仓库 + 时间范围 */}
        <Card>
          <div className="grid gap-4">
            <label className="text-sm font-medium text-slate-700">
              仓库
              <select
                className="mt-2 w-full rounded-md border border-line bg-white px-3 py-2 outline-none transition focus:border-teal-500 focus:ring-2 focus:ring-teal-100"
                value={repoId}
                onChange={(event) => setRepoId(event.target.value)}
              >
                {repos.map((repo) => (
                  <option key={repo.id} value={repo.id}>
                    {repo.full_name}
                  </option>
                ))}
              </select>
            </label>

            <label className="text-sm font-medium text-slate-700">
              开始日期
              <input
                type="date"
                className="mt-2 w-full rounded-md border border-line bg-white px-3 py-2 outline-none transition focus:border-teal-500 focus:ring-2 focus:ring-teal-100"
                value={startDate}
                onChange={(event) => setStartDate(event.target.value)}
              />
            </label>

            <label className="text-sm font-medium text-slate-700">
              结束日期
              <input
                type="date"
                className="mt-2 w-full rounded-md border border-line bg-white px-3 py-2 outline-none transition focus:border-teal-500 focus:ring-2 focus:ring-teal-100"
                value={endDate}
                onChange={(event) => setEndDate(event.target.value)}
              />
            </label>

            <PrimaryButton onClick={generateReport} disabled={!repoId || loading}>
              {loading ? "生成中..." : "生成周报"}
            </PrimaryButton>
          </div>
        </Card>

        {/* 右栏：Markdown 原文（不做渲染，方便直接复制粘贴） */}
        <Card>
          <h2 className="font-semibold text-ink">Markdown 周报</h2>
          {report ? (
            <pre className="mt-4 max-h-[640px] overflow-auto rounded-md border border-slate-800 bg-slate-950 p-4 text-sm leading-7 text-slate-100">{report}</pre>
          ) : (
            <div className="mt-4 rounded-md border border-line bg-panel p-4 text-sm text-slate-600">选择仓库和时间范围后生成周报。</div>
          )}
        </Card>
      </div>
    </>
  );
}
