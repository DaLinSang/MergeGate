"use client";

import { useEffect, useState } from "react";

import { apiGet, apiPost } from "@/lib/api";
import type { Repository } from "@/lib/types";
import { Card, PageTitle, PrimaryButton } from "@/components/ui/card";

/**
 * 代码图谱页（数据流 2 的查询端）
 * ==============================
 *
 * 【业务定位】回答"这个符号在哪、被谁调用、这个 PR 改了什么"。
 *            代码走结构化检索（符号 / 关系），不走向量检索 —— 精确结构用 grep 比语义更准。
 *
 * 【数据流向】
 *    建图（不在这里，在 /repos 页触发）：
 *        code_analysis.py 解析 → code_search.py 建索引 → knowledge_graph.py 写关系
 *        → code_symbols / code_relations 两张表
 *    查询：GET /api/repos/{id}/code-graph/search?q=...&pr_id=...
 *          → 返回 symbols（符号）+ relations（调用/导入）+ documents（代码片段）
 *    快照：POST /api/pull-requests/{id}/snapshot
 *          → 把 PR 分支 checkout 到独立目录并建图 → 之后带 pr_id 查询就查 PR 分支的代码
 *
 * 【要点】
 *    1. 带 pr_id 查询 = 查 PR 分支；不带 = 查主分支。切 PR 时必须清空旧结果，否则看着像同一份代码
 *    2. 快照失败不阻塞：status 不是 ready 时退回主分支查询，message 里写原因
 *    3. relations 只显示前 30 条（一次调用链可能上千条边，全渲染会卡）
 */

type PullRequest = {
  id: string;
  number: number;
  title: string;
  head_branch?: string | null;   // PR 源分支，建快照时 checkout 它
};

type CodeGraph = {
  query: string;                                                            // 原始查询词
  symbols: Array<{ id: string; path: string; name: string; kind: string; start_line?: number | null; pr_number?: number | null }>;
  relations: Array<{ id: string; source_name?: string | null; target_name: string; relation_type: string; path?: string | null; pr_number?: number | null }>;
  documents: Array<{ id: string; title: string; path?: string; snippet: string; pr_number?: number | null }>;
};

type PrSnapshot = {
  status: string;                        // ready / failed / pending
  message?: string | null;               // 失败原因或提示
  snapshot_name?: string | null;         // 快照目录名
  branch?: string | null;                // 实际 checkout 的分支
  commit_sha?: string | null;            // 快照对应的 commit
  pr_id?: string | null;
  pr_number?: number | null;
  indexed_symbols?: number | null;       // 这次快照建了多少符号
};

export default function CodeGraphPage() {
  const [repos, setRepos] = useState<Repository[]>([]);
  const [repoId, setRepoId] = useState("");
  const [prs, setPrs] = useState<PullRequest[]>([]);
  const [prId, setPrId] = useState("");
  const [query, setQuery] = useState("");
  const [graph, setGraph] = useState<CodeGraph | null>(null);
  const [snapshot, setSnapshot] = useState<PrSnapshot | null>(null);
  const [loading, setLoading] = useState(false);

  /** 切仓库时重拉 PR 列表。失败静默成空数组（没 PR 也能搜主分支代码）。 */
  async function loadPrs(id: string) {
    if (!id) return;
    const items = await apiGet<PullRequest[]>(`/api/repos/${id}/pull-requests`).catch(() => []);
    setPrs(items);
    setPrId(items[0]?.id || "");
    setSnapshot(null);   // 换仓库了，旧快照信息作废
  }

  // 首屏：仓库 → PR 列表
  useEffect(() => {
    apiGet<Repository[]>("/api/repos")
      .then((items) => {
        setRepos(items);
        setRepoId(items[0]?.id || "");
        return loadPrs(items[0]?.id || "");
      })
      .catch(() => setRepos([]));
  }, []);

  /**
   * 搜索图谱。
   * 传了 snapshotPrId（刚建好快照）就用它；否则看当前快照是否 ready —— ready 才带 pr_id。
   */
  async function searchGraph(snapshotPrId?: string) {
    if (!repoId) return;
    setLoading(true);
    try {
      const params = new URLSearchParams({ q: query });
      const activeSnapshotPrId = snapshotPrId ?? (snapshot?.status === "ready" ? prId : "");
      if (activeSnapshotPrId) params.set("pr_id", activeSnapshotPrId);
      setGraph(await apiGet<CodeGraph>(`/api/repos/${repoId}/code-graph/search?${params.toString()}`));
    } finally {
      setLoading(false);
    }
  }

  /** 建 PR 快照并立刻用它查一次。快照没 ready 就退回主分支。 */
  async function prepareSnapshot() {
    if (!prId) return;
    setLoading(true);
    try {
      const nextSnapshot = await apiPost<PrSnapshot>(`/api/pull-requests/${prId}/snapshot`, {});
      setSnapshot(nextSnapshot);
      // 快照失败不阻塞查询：传 undefined 让 searchGraph 退回主分支
      await searchGraph(nextSnapshot.status === "ready" ? prId : undefined);
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <PageTitle title="代码图谱" description="检索代码 symbol、调用/导入关系，并为 PR 准备独立分支快照。" />
      <div className="grid gap-4 lg:grid-cols-[380px_1fr]">
        {/* 左栏：仓库 / 查询词 / PR 快照 */}
        <Card>
          <label className="text-sm font-medium text-slate-700">
            仓库
            <select
              className="mt-2 w-full rounded-md border border-line px-3 py-2"
              value={repoId}
              onChange={(event) => {
                setRepoId(event.target.value);
                setGraph(null);                    // 清空结果：否则会拿旧仓库的符号当成新仓库的
                loadPrs(event.target.value);
              }}
            >
              {repos.map((repo) => (
                <option key={repo.id} value={repo.id}>
                  {repo.full_name}
                </option>
              ))}
            </select>
          </label>
          <label className="mt-4 block text-sm font-medium text-slate-700">
            搜索 symbol / path
            <input className="mt-2 w-full rounded-md border border-line px-3 py-2" value={query} onChange={(event) => setQuery(event.target.value)} />
          </label>
          <PrimaryButton className="mt-3" disabled={!repoId || loading} onClick={() => searchGraph()}>
            搜索图谱
          </PrimaryButton>
          <div className="mt-6 border-t border-line pt-4">
            <label className="text-sm font-medium text-slate-700">
              PR 分支快照
              <select
                className="mt-2 w-full rounded-md border border-line px-3 py-2"
                value={prId}
                onChange={(event) => {
                  setPrId(event.target.value);
                  // 换 PR 必须同时清快照和结果，否则会误以为结果来自新 PR
                  setSnapshot(null);
                  setGraph(null);
                }}
              >
                {prs.map((pr) => (
                  <option key={pr.id} value={pr.id}>
                    #{pr.number} {pr.title}
                  </option>
                ))}
              </select>
            </label>
            <PrimaryButton className="mt-3" disabled={!prId || loading} onClick={prepareSnapshot}>
              准备 PR 快照
            </PrimaryButton>
            {/* 快照结果：branch / commit 用来核对"我查的到底是哪版代码" */}
            {snapshot && (
              <div className="mt-3 rounded-md border border-line bg-panel p-3 text-sm text-slate-700">
                <div className="font-medium text-ink">{snapshot.snapshot_name || "PR 分支快照"}</div>
                <div className="mt-1">状态：{snapshot.status}</div>
                {snapshot.branch && <div>分支：{snapshot.branch}</div>}
                {/* commit 只显示前 12 位，够辨识又不占宽度 */}
                {snapshot.commit_sha && <div>Commit：{snapshot.commit_sha.slice(0, 12)}</div>}
                {typeof snapshot.indexed_symbols === "number" && <div>代码图符号：{snapshot.indexed_symbols}</div>}
                {snapshot.message && <div className="mt-1 text-slate-500">{snapshot.message}</div>}
              </div>
            )}
          </div>
        </Card>

        {/* 右栏：符号 / 关系 / 片段三段 */}
        <div className="grid gap-4">
          <Card>
            <h2 className="font-semibold text-ink">Symbols</h2>
            <div className="mt-3 grid gap-2">
              {(graph?.symbols || []).map((item) => (
                <div key={item.id} className="rounded-md border border-line bg-panel px-3 py-2 text-sm">
                  <span className="font-medium text-ink">{item.name}</span>
                  <span className="ml-2 text-slate-500">
                    {/* start_line 可能是 0，用 || 1 兜底成第一行 */}
                    {item.kind} · {item.path}:{item.start_line || 1}
                    {item.pr_number ? ` · PR #${item.pr_number}` : ""}
                  </span>
                </div>
              ))}
              {!graph && <div className="text-sm text-slate-500">索引代码后可搜索 symbol。</div>}
            </div>
          </Card>
          <Card>
            <h2 className="font-semibold text-ink">Relations</h2>
            <div className="mt-3 grid gap-2">
              {/* 只渲染前 30 条：调用关系可能上千条，全画会拖垮浏览器 */}
              {(graph?.relations || []).slice(0, 30).map((item) => (
                <div key={item.id} className="rounded-md border border-line bg-panel px-3 py-2 text-sm text-slate-700">
                  {/* 没有 source_name 时退回 path，再没有就写 module —— 保证关系两端都看得见 */}
                  {item.source_name || item.path || "module"} - {item.relation_type} - {item.target_name}
                  {item.pr_number ? <span className="ml-2 text-slate-500">PR #{item.pr_number}</span> : null}
                </div>
              ))}
            </div>
          </Card>
          <Card>
            <h2 className="font-semibold text-ink">相关代码片段</h2>
            {(graph?.documents || []).map((item) => (
              <pre key={item.id} className="mt-3 max-h-48 overflow-auto rounded-md border border-slate-800 bg-slate-950 p-3 text-xs text-slate-100">
                {item.title}
                {"\n\n"}
                {item.snippet}
              </pre>
            ))}
          </Card>
        </div>
      </div>
    </>
  );
}
