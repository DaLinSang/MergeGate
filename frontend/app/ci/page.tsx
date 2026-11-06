"use client";

import { useEffect, useState } from "react";

import { Card, GhostButton, PageTitle, PrimaryButton } from "@/components/ui/card";
import { apiGet, apiPost } from "@/lib/api";
import type { Repository } from "@/lib/types";

/**
 * CI Debug 分析页
 * ================
 *
 * 【业务定位】回答"这个 CI 为什么挂了"：先看失败 Job/Step，再让 Agent 读日志做归因。
 *
 * 【数据流向】
 *    列表：GET /api/repos → repos[0] → GET /api/repos/{id}/workflow-runs → workflow_runs 表
 *    分析：POST /api/workflow-runs/{id}/analyze
 *          → api/routes/chat.py::analyze_ci → CIAgent
 *          → 日志先 sanitize_ci_log 脱敏，再切"首个错误块"窗口（前 2 后 5 行）送检索
 *    展示：jobs/steps 结构化展示 + 原始日志 + Agent 结论 JSON
 *
 * 【要点】
 *    1. 默认选中第一个 failure 的 run —— 进页面就能看到问题，不用手动找
 *    2. 失败定位靠 conclusion === "failure" 两级过滤（先 job 再 step），Step 才是真正的错误点
 */

type WorkflowRun = {
  id: string;
  name: string;
  status: string;                                                          // queued / in_progress / completed
  conclusion?: string | null;                                              // success / failure / cancelled …
  logs_text?: string | null;                                               // 脱敏后的日志正文
  jobs?: Array<{ name?: string; conclusion?: string; steps?: Array<{ name?: string; conclusion?: string }> }>;
};

export default function CIPage() {
  const [runs, setRuns] = useState<WorkflowRun[]>([]);
  const [selected, setSelected] = useState<WorkflowRun | null>(null);
  const [analysis, setAnalysis] = useState("");

  // 先拿仓库再拿 CI 记录；默认选中第一个失败的 run
  useEffect(() => {
    apiGet<Repository[]>("/api/repos").then((repos) => {
      if (repos[0]) {
        apiGet<WorkflowRun[]>(`/api/repos/${repos[0].id}/workflow-runs`).then((items) => {
          setRuns(items);
          setSelected(items.find((item) => item.conclusion === "failure") ?? items[0] ?? null);
        });
      }
    });
  }, []);

  /** 触发失败归因分析。 */
  async function analyze() {
    if (!selected) return;
    const data = await apiPost(`/api/workflow-runs/${selected.id}/analyze`, {});
    setAnalysis(JSON.stringify(data, null, 2));
  }

  return (
    <>
      <PageTitle title="CI Debug 分析" description="聚合 GitHub Actions 运行状态和失败日志，生成失败归因、排查步骤和相关风险提示。" />
      <div className="grid gap-4 lg:grid-cols-[380px_1fr]">
        {/* 左栏：运行记录（失败标琥珀色，其余绿色） */}
        <Card>
          <div className="flex items-center justify-between">
            <h2 className="font-semibold text-ink">Workflow Runs</h2>
            <span className="rounded-full bg-panel px-2.5 py-1 text-xs text-slate-500">{runs.length} 次</span>
          </div>
          <div className="mt-4 space-y-2">
            {runs.map((run) => (
              <GhostButton key={run.id} onClick={() => setSelected(run)} className={selected?.id === run.id ? "border-teal-300 bg-teal-50 text-teal-900" : ""}>
                <div className="flex items-center justify-between gap-3">
                  <span>{run.name}</span>
                  {/* 还在跑的时候 conclusion 是空的，退回显示 status */}
                  <span className={run.conclusion === "failure" ? "text-amber-700" : "text-emerald-700"}>{run.conclusion ?? run.status}</span>
                </div>
              </GhostButton>
            ))}
          </div>
        </Card>

        {/* 右栏：失败 Job/Step 定位 → 原始日志 → Agent 结论 */}
        <Card>
          <h2 className="font-semibold text-ink">排查建议</h2>
          {(selected?.jobs?.length ?? 0) > 0 && (
            <div className="mt-3 rounded-md border border-line bg-panel p-3 text-xs leading-5 text-slate-700">
              <div className="font-medium text-ink">失败 Job</div>
              {/* 两级过滤：只列失败的 job，每个 job 下只列失败的 step —— 成功的会淹没真正的错误 */}
              {(selected?.jobs ?? [])
                .filter((job) => job.conclusion === "failure")
                .map((job) => (
                  <div key={job.name} className="mt-2">
                    <span className="font-medium">{job.name}</span>
                    {job.steps?.filter((step) => step.conclusion === "failure").map((step) => (
                      <span key={step.name} className="ml-2 rounded-full bg-amber-100 px-2 py-0.5 text-amber-800">
                        {step.name}
                      </span>
                    ))}
                  </div>
                ))}
            </div>
          )}
          {/* 日志只展示 max-h-44，完整日志走分析结果 —— 页面上放全文会卡 */}
          {selected?.logs_text && <pre className="mt-3 max-h-44 overflow-auto rounded-md border border-line bg-panel p-3 text-xs text-slate-700">{selected.logs_text}</pre>}
          <PrimaryButton onClick={analyze} disabled={!selected} className="mt-4">分析失败原因</PrimaryButton>
          {analysis && <pre className="mt-4 max-h-[520px] overflow-auto rounded-md border border-slate-800 bg-slate-950 p-4 text-xs text-slate-100">{analysis}</pre>}
        </Card>
      </div>
    </>
  );
}
