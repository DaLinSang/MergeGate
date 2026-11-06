"use client";

import { useEffect, useState } from "react";

import { Card, GhostButton, PageTitle, PrimaryButton } from "@/components/ui/card";
import { apiGet, apiPost } from "@/lib/api";
import type { Repository } from "@/lib/types";

/**
 * Issue 智能分析页
 * ================
 *
 * 【业务定位】四步主线的"判断"环节：选中一个 Issue，让专用 Agent 给出分类 / 优先级 / 负责人 / 行动项。
 *
 * 【数据流向】
 *    列表：GET /api/repos → 取 repos[0] → GET /api/repos/{id}/issues → issues 表
 *    分析：POST /api/issues/{id}/analyze
 *          → api/routes/chat.py::analyze_issue → IssueAgent
 *          → 工具：相似 Issue 检索（RAG）+ 代码定位（工作区）
 *    展示：结果直接 JSON.stringify 打进 <pre> —— 这个页面是调试用，没做结构化渲染
 *
 * 【要点】
 *    1. 只取 repos[0]：多仓库时永远只看第一个，这是已知简化（首页才有仓库切换）
 *    2. 分析结果不落库，刷新即丢；要留存走 /chat 的会话链路
 */

type Issue = { id: string; number: number; title: string; state: string; labels: string[]; body?: string };

export default function IssuesPage() {
  const [issues, setIssues] = useState<Issue[]>([]);
  const [selected, setSelected] = useState<Issue | null>(null);
  const [analysis, setAnalysis] = useState("");

  // 两级串联请求：先拿仓库，再用仓库 id 拿 Issue。没有仓库时什么都不做（不报错）
  useEffect(() => {
    apiGet<Repository[]>("/api/repos").then((repos) => {
      if (repos[0]) {
        apiGet<Issue[]>(`/api/repos/${repos[0].id}/issues`).then((items) => {
          setIssues(items);
          // 默认选中第一条，避免右侧空白
          setSelected(items[0] ?? null);
        });
      }
    });
  }, []);

  /** 触发后端分析。返回的是结构化 JSON，这里原样展示。 */
  async function analyze() {
    if (!selected) return;
    const data = await apiPost(`/api/issues/${selected.id}/analyze`, {});
    setAnalysis(JSON.stringify(data, null, 2));
  }

  return (
    <>
      <PageTitle title="Issue 智能分析" description="自动生成 Issue 分类、优先级、复杂度、负责人建议、相似问题和下一步行动项。" />
      <div className="grid gap-4 lg:grid-cols-[380px_1fr]">
        {/* 左栏：Issue 列表（点击切换选中项） */}
        <Card>
          <div className="flex items-center justify-between">
            <h2 className="font-semibold text-ink">Issue 列表</h2>
            <span className="rounded-full bg-panel px-2.5 py-1 text-xs text-slate-500">{issues.length} 条</span>
          </div>
          <div className="mt-4 space-y-2">
            {issues.map((issue) => (
              <GhostButton key={issue.id} onClick={() => setSelected(issue)} className={selected?.id === issue.id ? "border-teal-300 bg-teal-50 text-teal-900" : ""}>
                <div className="flex items-center justify-between gap-3">
                  <span>#{issue.number} {issue.title}</span>
                  <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs">{issue.state}</span>
                </div>
              </GhostButton>
            ))}
          </div>
        </Card>

        {/* 右栏：正文 + 分析按钮 + 原始 JSON 结果 */}
        <Card>
          <h2 className="font-semibold text-ink">分析结果</h2>
          {selected && (
            <div className="mt-3 rounded-md bg-panel p-3 text-sm leading-6 text-slate-600">
              <div className="font-medium text-ink">#{selected.number} {selected.title}</div>
              <div className="mt-1">{selected.body || "暂无正文"}</div>
            </div>
          )}
          <PrimaryButton onClick={analyze} disabled={!selected} className="mt-4">分析当前 Issue</PrimaryButton>
          {analysis && <pre className="mt-4 max-h-[520px] overflow-auto rounded-md border border-slate-800 bg-slate-950 p-4 text-xs text-slate-100">{analysis}</pre>}
        </Card>
      </div>
    </>
  );
}
