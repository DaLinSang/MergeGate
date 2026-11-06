"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import {
  ArrowLeft,
  BookOpen,
  Check,
  ChevronRight,
  FileSearch,
  FileText,
  History,
  LoaderCircle,
  Play,
  Plus,
  RotateCcw,
  Save,
  Settings,
  TestTube2,
  X
} from "lucide-react";

import { RetrievalSettings } from "@/components/rag/retrieval-settings";
import { apiGet, apiPatch, apiPost, apiUpload } from "@/lib/api";
import {
  formatBytes,
  formatDate,
  formatNumber,
  statusLabel,
  type KnowledgeBase,
  type KnowledgeChunk,
  type KnowledgeConfig,
  type KnowledgeDocument,
  type ModelCatalog,
  type RetrievalHistory,
  type RetrievalResponse
} from "@/lib/rag";

/**
 * 知识库详情页（RAG 主控台）
 * ========================
 *
 * 【业务定位】RAG 三个环节都能在这里看到：入库（文档 tab）、召回（召回测试 tab）、参数（设置 tab）。
 *            这是理解 RAG 数据流 3 + 4 的最好入口。
 *
 * 【数据流向】
 *    文档 tab：上传 → POST /api/rag/repositories/{id}/documents（multipart）
 *              → 后台流水线 parsing → splitting → embedding → Milvus
 *              → 前端每秒轮询状态，直到全部终态
 *    召回 tab：POST /api/rag/repositories/{id}/retrieval-tests
 *              → 六阶段流水线（向量+BM25 召回 → 融合 → 去重 → 重排 → 压缩 → 准入）
 *              → 返回 results + pipeline.stages（每阶段耗时与数量）
 *    设置 tab：PATCH /api/rag/repositories/{id}/config
 *              换了 embedding 模型 → 额外 POST /api/rag/{id}/reindex（旧向量维度不兼容）
 *
 * 【要点】
 *    1. 轮询只在"有文档没跑完"时启动，全部终态就停 —— 否则会一直打后端
 *    2. 换 embedding 模型必须 reindex：不同模型的向量不在同一个空间里，混着查等于乱答
 *    3. 保存按钮的 disabled 条件含 chunk_overlap >= chunk_size：重叠不小于切片长度会无限切
 */

type Tab = "documents" | "retrieval" | "settings";

// 三个页签。icon 直接存组件引用
const tabs: { value: Tab; label: string; icon: typeof FileText }[] = [
  { value: "documents", label: "文档", icon: FileText },
  { value: "retrieval", label: "召回测试", icon: TestTube2 },
  { value: "settings", label: "设置", icon: Settings }
];

// 流水线阶段 → 中文。后端返回的 name 是英文 key，这里做展示映射
const pipelineStageLabel: Record<string, string> = {
  vector_recall: "向量召回",
  keyword_recall: "BM25 召回",
  weighted_fusion: "加权融合",
  single_retriever_selection: "单路候选",
  deduplication_and_parent_aggregation: "去重聚合",
  rerank: "重排",
  parent_expansion: "父级扩展",
  score_threshold: "分数门槛",
  context_compression: "证据压缩",
  answer_gate: "回答判断"
};

// 准入闸门的四种结论 → 中文
const answerGateLabel: Record<string, string> = {
  answer: "证据充足",
  ask_clarification: "需要补充范围",
  insufficient_evidence: "证据不足",
  conflict: "证据冲突"
};

export default function KnowledgeBaseDetailPage() {
  const id = String(useParams<{ id: string }>().id);   // 路由参数 = repo_id（知识库跟着仓库走）
  const fileInput = useRef<HTMLInputElement>(null);   // 隐藏的文件输入，用按钮代理点击
  const [tab, setTab] = useState<Tab>("documents");
  const [knowledge, setKnowledge] = useState<KnowledgeBase | null>(null);   // 服务端的知识库快照
  const [documents, setDocuments] = useState<KnowledgeDocument[]>([]);
  const [catalog, setCatalog] = useState<ModelCatalog | null>(null);
  const [config, setConfig] = useState<KnowledgeConfig | null>(null);       // 本地可编辑的配置副本
  const [selectedDocument, setSelectedDocument] = useState<KnowledgeDocument | null>(null);  // 打开抽屉的那篇
  const [chunks, setChunks] = useState<KnowledgeChunk[]>([]);
  const [query, setQuery] = useState("");
  const [retrieval, setRetrieval] = useState<RetrievalResponse | null>(null);
  const [history, setHistory] = useState<RetrievalHistory[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [testing, setTesting] = useState(false);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");   // 成功提示（绿色条）
  const [error, setError] = useState("");       // 失败提示（红色条）

  /** 首屏三路并发：知识库概要 / 文档列表 / 可选模型清单。 */
  const load = useCallback(async () => {
    const [base, docs, models] = await Promise.all([
      apiGet<KnowledgeBase>(`/api/rag/repositories/${id}`),
      apiGet<KnowledgeDocument[]>(`/api/rag/repositories/${id}/documents`),
      apiGet<ModelCatalog>("/api/rag/models")
    ]);
    setKnowledge(base);
    setDocuments(docs);
    setCatalog(models);
    setConfig(base.config);   // 注意：轮询也会重置 config，用户未保存的改动会被覆盖
  }, [id]);

  /** 召回测试的历史记录。失败静默（不影响主流程）。 */
  const loadHistory = useCallback(() => {
    apiGet<RetrievalHistory[]>(`/api/rag/repositories/${id}/retrieval-tests`).then(setHistory).catch(() => null);
  }, [id]);

  // 首屏：从 URL 的 ?tab= 恢复页签，再拉数据
  useEffect(() => {
    const initial = new URLSearchParams(window.location.search).get("tab");
    if (initial === "retrieval" || initial === "settings") setTab(initial);
    load()
      .catch((reason: Error) => setError(reason.message))
      .finally(() => setLoading(false));
    loadHistory();
  }, [load, loadHistory]);

  /**
   * 入库进度轮询：只要还有文档处于非终态（queued/parsing/splitting/embedding）就每秒重拉。
   * 依赖 documents —— 状态一变就重建 interval；全部完成时直接 return，不建定时器。
   */
  useEffect(() => {
    if (!documents.some((item) => !["completed", "failed"].includes(item.status))) return;
    const timer = window.setInterval(() => load().catch(() => null), 1000);
    return () => window.clearInterval(timer);
  }, [documents, load]);

  /** 切页签并把 tab 写进 URL —— 刷新后还停在同一个 tab。用 replaceState 不留历史记录。 */
  function changeTab(next: Tab) {
    setTab(next);
    window.history.replaceState(null, "", `${window.location.pathname}?tab=${next}`);
  }

  /** 上传文件。逐个串行上传，避免一次把后端的解析队列打满。 */
  async function uploadFiles(list: FileList | null) {
    if (!list?.length) return;
    setUploading(true);
    setError("");
    try {
      for (const file of Array.from(list)) {
        const form = new FormData();
        form.append("file", file);
        await apiUpload<KnowledgeDocument>(`/api/rag/repositories/${id}/documents`, form);
      }
      await load();   // 上传完立刻刷新，触发轮询
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "上传失败");
    } finally {
      setUploading(false);
      // 必须清空 input.value：否则再次选同一个文件不会触发 onChange
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  /** 打开右侧抽屉看真实切片内容。只有 completed 的文档能点。 */
  async function openChunks(document: KnowledgeDocument) {
    setSelectedDocument(document);
    setChunks([]);   // 先清空，避免看到上一篇的切片
    try {
      setChunks(await apiGet<KnowledgeChunk[]>(`/api/rag/repositories/${id}/documents/${document.id}/chunks`));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "读取分段失败");
    }
  }

  /** 失败文档重试：重新走一遍解析 → 切片 → 向量化。 */
  async function retry(document: KnowledgeDocument) {
    await apiPost(`/api/rag/repositories/${id}/documents/${document.id}/retry`, {});
    await load();
  }

  /** 跑一次召回测试。参数取自当前 config（即"按这套设置会召回什么"）。 */
  async function runRetrieval() {
    if (!query.trim() || !config) return;
    setTesting(true);
    setError("");
    try {
      const result = await apiPost<RetrievalResponse>(`/api/rag/repositories/${id}/retrieval-tests`, {
        query: query.trim(),
        top_k: config.top_k,
        retrieval_method: config.retrieval_method,
        rerank_enabled: config.rerank_enabled,
        score_threshold_enabled: config.score_threshold_enabled,
        score_threshold: config.score_threshold
      });
      setRetrieval(result);
      loadHistory();   // 新测试会出现在历史里
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "召回测试失败");
    } finally {
      setTesting(false);
    }
  }

  /**
   * 保存设置。
   * 【关键】换 embedding 模型/供应商 → 必须 reindex：旧向量是用别的模型算的，维度/空间都不兼容，
   * 不重建就会出现"新问题检索不到旧文档"或维度报错。
   * 只改分段长度则不重建：那只影响以后新上传的文档。
   */
  async function saveSettings() {
    if (!config || !knowledge) return;
    setSaving(true);
    setMessage("");
    setError("");
    const reindexNeeded = config.embedding_model !== knowledge.config.embedding_model || config.embedding_provider !== knowledge.config.embedding_provider;
    try {
      const updated = await apiPatch<KnowledgeBase>(`/api/rag/repositories/${id}/config`, config);
      setKnowledge(updated);
      setConfig(updated.config);
      // chunk_count 为 0 表示还没有向量，没必要 reindex
      if (reindexNeeded && updated.chunk_count) {
        await apiPost(`/api/rag/${id}/reindex`, {});
        setMessage("设置已保存，现有分段已重新向量化。平时修改分段长度只会影响新上传的文档。");
      } else {
        setMessage("设置已保存。");
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "保存失败");
    } finally {
      setSaving(false);
    }
  }

  // 三种渲染态：加载中 / 加载失败 / 正常
  if (loading) return <div className="grid min-h-[70vh] place-items-center text-sm text-slate-500">正在加载项目知识...</div>;
  if (!knowledge || !config) return <div className="m-6 border border-red-200 bg-red-50 p-4 text-sm text-red-700">{error || "未找到仓库"}</div>;

  return (
    <div className="min-h-[calc(100vh-74px)] bg-[#f7f8fa]">
      {/* 顶部：返回 + 仓库名 + 三个总量 */}
      <header className="border-b border-line bg-white px-4 py-3">
        <div className="flex min-w-0 items-center gap-3">
          <Link href="/knowledge" title="返回项目知识" className="grid h-9 w-9 shrink-0 place-items-center rounded-md border border-line text-slate-600 hover:text-ink"><ArrowLeft className="h-4 w-4" /></Link>
          <span className="grid h-9 w-9 shrink-0 place-items-center rounded-md bg-teal-50 text-teal-700"><BookOpen className="h-4 w-4" /></span>
          <div className="min-w-0">
            <h1 className="truncate text-sm font-semibold text-ink">{knowledge.name}</h1>
            <p className="truncate text-xs text-slate-400">{knowledge.full_name}</p>
          </div>
          <div className="ml-auto hidden items-center gap-5 text-xs text-slate-500 sm:flex">
            <span>{formatNumber(knowledge.document_count)} 文档</span><span>{formatNumber(knowledge.chunk_count)} 分段</span><span>{formatNumber(knowledge.character_count)} 字符</span>
          </div>
        </div>
      </header>

      <div className="grid min-h-[calc(100vh-135px)] md:grid-cols-[190px_minmax(0,1fr)]">
        {/* 左侧栏：页签 + 模型运行状态（这一栏是"为什么检索不准"的第一排查点） */}
        <aside className="border-b border-line bg-white p-3 md:border-b-0 md:border-r">
          <nav className="flex gap-1 overflow-x-auto md:block md:space-y-1">
            {tabs.map((item) => {
              const Icon = item.icon;
              return (
                <button key={item.value} type="button" onClick={() => changeTab(item.value)} className={`flex h-10 shrink-0 items-center gap-2 rounded-md px-3 text-sm transition md:w-full ${tab === item.value ? "bg-teal-50 font-medium text-teal-800" : "text-slate-600 hover:bg-slate-50"}`}>
                  <Icon className="h-4 w-4" /> {item.label}
                </button>
              );
            })}
          </nav>
          <div className="mt-5 hidden border-t border-line pt-4 text-xs leading-5 text-slate-400 md:block">
            <div className="font-medium text-slate-600">运行状态</div>
            <div className="mt-2">Embedding</div>
            <div className={knowledge.runtime.embedding.available ? "text-emerald-700" : "text-amber-700"}>
              {knowledge.runtime.embedding.available
                ? `${knowledge.runtime.embedding.provider} 就绪`
                : `${knowledge.runtime.embedding.provider} 不可用`}
            </div>
            <div className="mt-2">Rerank</div>
            {/* rerank 有三级状态：正常 / 降级到启发式 / 完全不可用 */}
            <div className={knowledge.runtime.rerank.available ? "text-emerald-700" : "text-amber-700"}>
              {knowledge.runtime.rerank.available
                ? `${knowledge.runtime.rerank.provider} 就绪`
                : knowledge.runtime.rerank.fallback === "heuristic"
                  ? "启发式重排"
                  : `${knowledge.runtime.rerank.provider} 不可用`}
            </div>
          </div>
        </aside>

        <main className="min-w-0 p-4 sm:p-6">
          {error ? <div className="mb-4 flex items-center justify-between bg-red-50 px-4 py-3 text-sm text-red-700"><span>{error}</span><button onClick={() => setError("")} title="关闭"><X className="h-4 w-4" /></button></div> : null}

          {/* ── 页签 1：文档（入库进度 + 切片查看） ── */}
          {tab === "documents" ? (
            <div className="mx-auto max-w-6xl">
              <div className="flex items-end justify-between gap-4">
                <div><h2 className="text-lg font-semibold text-ink">文档</h2><p className="mt-1 text-sm text-slate-500">查看解析状态和真实分段内容。</p></div>
                {/* 隐藏的 file input：accept 限定可解析的格式，multiple 支持一次传多份 */}
                <input ref={fileInput} type="file" multiple accept=".txt,.md,.markdown,.pdf,.docx,.json,.csv,.log" className="hidden" onChange={(event) => uploadFiles(event.target.files)} />
                <button type="button" onClick={() => fileInput.current?.click()} disabled={uploading} className="inline-flex h-10 items-center gap-2 rounded-md bg-teal-700 px-4 text-sm font-medium text-white disabled:opacity-50">
                  {uploading ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />} 添加文件
                </button>
              </div>

              <div className="mt-5 overflow-hidden border border-line bg-white shadow-sm">
                {/* 表头：移动端隐藏，靠每行内联的"分段/字符"小标签兜底 */}
                <div className="hidden grid-cols-[minmax(220px,1fr)_120px_100px_100px_140px_34px] gap-3 border-b border-line bg-slate-50 px-4 py-2.5 text-xs font-medium text-slate-500 md:grid">
                  <span>名称</span><span>状态</span><span>分段</span><span>字符数</span><span>上传时间</span><span />
                </div>
                {documents.map((document) => (
                  // 整行是按钮，但只有 completed 才能点开切片（其他状态没内容可看）
                  <button key={document.id} type="button" onClick={() => document.status === "completed" && openChunks(document)} className="grid w-full gap-2 border-b border-slate-100 px-4 py-3 text-left last:border-b-0 hover:bg-slate-50 md:grid-cols-[minmax(220px,1fr)_120px_100px_100px_140px_34px] md:items-center md:gap-3">
                    <span className="flex min-w-0 items-center gap-2"><FileText className="h-4 w-4 shrink-0 text-teal-700" /><span className="truncate text-sm font-medium text-ink">{document.name}</span></span>
                    <span className={`w-fit rounded px-2 py-1 text-xs ${document.status === "completed" ? "bg-emerald-50 text-emerald-700" : document.status === "failed" ? "bg-red-50 text-red-700" : "bg-amber-50 text-amber-700"}`}>
                      {statusLabel[document.status]}
                    </span>
                    <span className="text-xs text-slate-500"><span className="md:hidden">分段 </span>{formatNumber(document.chunk_count)}</span>
                    <span className="text-xs text-slate-500"><span className="md:hidden">字符 </span>{formatNumber(document.character_count)}</span>
                    <span className="text-xs text-slate-400">{formatDate(document.created_at)}</span>
                    {/* 失败行把右侧箭头换成重试图标；stopPropagation 防止同时触发行点击 */}
                    {document.status === "failed" ? <span onClick={(event) => { event.stopPropagation(); retry(document); }} title="重试" className="grid h-7 w-7 place-items-center text-red-600"><RotateCcw className="h-4 w-4" /></span> : <ChevronRight className="hidden h-4 w-4 text-slate-300 md:block" />}
                  </button>
                ))}
                {!documents.length ? <div className="py-20 text-center text-sm text-slate-500">还没有文档，添加一份资料开始构建。</div> : null}
              </div>
            </div>
          ) : null}

          {/* ── 页签 2：召回测试（六阶段流水线的可视化） ── */}
          {tab === "retrieval" ? (
            <div className="mx-auto max-w-7xl">
              <div><h2 className="text-lg font-semibold text-ink">召回测试</h2><p className="mt-1 text-sm text-slate-500">输入真实问题，观察被召回的分段、分数和排序。</p></div>
              <div className="mt-5 grid gap-5 xl:grid-cols-[minmax(360px,.72fr)_minmax(520px,1.28fr)]">
                <div className="space-y-5">
                  <section className="border border-line bg-white p-4 shadow-sm">
                    <label className="text-xs font-medium text-slate-600">查询文本</label>
                    {/* Ctrl/Cmd + Enter 直接提交；maxLength 与后端字段限制一致 */}
                    <textarea value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") runRetrieval(); }} maxLength={250} placeholder="例如：新员工可以请几天年假？" className="mt-2 min-h-36 w-full resize-y rounded-md border border-line p-3 text-sm leading-6 outline-none focus:border-teal-600" />
                    <div className="mt-3 flex items-center justify-between"><span className="text-xs text-slate-400">{query.length} / 250</span><button type="button" disabled={!query.trim() || testing} onClick={runRetrieval} className="inline-flex h-9 items-center gap-2 rounded-md bg-teal-700 px-4 text-sm font-medium text-white disabled:opacity-40">{testing ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />} 测试</button></div>
                  </section>
                  <section className="border border-line bg-white p-4 shadow-sm">
                    <div className="flex items-center justify-between"><h3 className="flex items-center gap-2 text-sm font-semibold text-ink"><History className="h-4 w-4" /> 测试记录</h3><span className="text-xs text-slate-400">最近 {history.length} 条</span></div>
                    <div className="mt-3 divide-y divide-slate-100">
                      {history.map((item) => <div key={item.id} className="py-3"><div className="truncate text-sm text-slate-700">{item.query}</div><div className="mt-1 flex gap-3 text-xs text-slate-400"><span>{item.result_count} 条结果</span><span>{item.took_ms} ms</span><span>{formatDate(item.created_at)}</span></div></div>)}
                      {!history.length ? <div className="py-8 text-center text-xs text-slate-400">暂无测试记录</div> : null}
                    </div>
                  </section>
                </div>

                <section className="min-w-0">
                  <div className="flex items-center justify-between"><h3 className="text-sm font-semibold text-ink">召回结果</h3>{retrieval ? <span className="text-xs text-slate-500">{retrieval.result_count} 条 · {retrieval.took_ms} ms</span> : null}</div>
                  {/* Retrieval Trace：把每个阶段的"进来多少、出去多少"按顺序排开 —— 一眼看出是哪一步把候选筛没了 */}
                  {retrieval?.retrieval_config.pipeline?.stages?.length ? (
                    <div className="mt-3 border border-line bg-white p-4 shadow-sm">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="text-xs font-medium text-slate-600">Retrieval Trace</div>
                        <span className={`rounded px-2 py-1 text-xs font-medium ${retrieval.retrieval_config.answer_gate?.decision === "answer" ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700"}`}>
                          {answerGateLabel[String(retrieval.retrieval_config.answer_gate?.decision ?? "")] ?? "等待判断"}
                        </span>
                      </div>
                      <div className="mt-3 flex flex-wrap items-center gap-2">
                        {retrieval.retrieval_config.pipeline.stages.map((stage, index) => {
                          const name = String(stage.name ?? "unknown");
                          // 不同阶段的数量字段名不统一，两个都试一下
                          const outputCount = stage.output_count ?? stage.candidate_count;
                          return (
                            <div key={`${name}-${index}`} className="flex items-center gap-2">
                              {index ? <span className="text-slate-300">→</span> : null}
                              <span className="rounded border border-slate-200 bg-slate-50 px-2 py-1 text-xs text-slate-600">
                                {/* 没映射到中文的原样显示英文 key，方便发现新阶段 */}
                                {pipelineStageLabel[name] ?? name}{typeof outputCount === "number" ? ` · ${outputCount}` : ""}
                              </span>
                            </div>
                          );
                        })}
                      </div>
                      {retrieval.retrieval_config.answer_gate?.reason ? <p className="mt-3 text-xs leading-5 text-slate-500">{retrieval.retrieval_config.answer_gate.reason}</p> : null}
                    </div>
                  ) : null}
                  <div className="mt-3 space-y-3">
                    {retrieval?.results.map((result, index) => (
                      <article key={`${result.document_id}-${index}`} className="border border-line bg-white p-4 shadow-sm">
                        <div className="flex items-center gap-3 border-b border-slate-100 pb-3">
                          {/* 序号是前端按返回顺序排的，和回答里的 [1][2] 不是同一套编号（那套用 result.citation） */}
                          <span className="grid h-7 w-7 shrink-0 place-items-center rounded bg-teal-50 text-xs font-semibold text-teal-700">{index + 1}</span>
                          <div className="min-w-0 flex-1"><h4 className="truncate text-sm font-medium text-ink">{result.title}</h4><p className="mt-0.5 truncate text-xs text-slate-400">{String(result.metadata.filename ?? result.source_type)}</p></div>
                          {/* 分数保留 3 位：融合分差异往往在小数点后第 2~3 位 */}
                          <span className="rounded bg-emerald-50 px-2 py-1 text-xs font-semibold text-emerald-700">{result.score.toFixed(3)}</span>
                        </div>
                        <p className="mt-3 whitespace-pre-wrap text-sm leading-6 text-slate-700">{result.snippet}</p>
                        {/* 底部证据行：切片号 / 字数 / 实际走的后端 / 融合方式 / 重排模式 */}
                        <div className="mt-3 flex flex-wrap gap-3 text-xs text-slate-400"><span>分段 #{String(result.metadata.chunk_index ?? index + 1)}</span><span>{result.snippet.length} 字符</span><span>{String(result.retrieval.keyword_backend ?? result.retrieval.vector_backend ?? "retrieval")}</span><span>{String(result.retrieval.fusion_method ?? retrieval.retrieval_config.fusion_method ?? config.retrieval_method)}</span><span>{String(result.retrieval.rerank_mode ?? result.retrieval.mode ?? "retrieval")}</span></div>
                      </article>
                    ))}
                    {!retrieval ? <div className="grid min-h-80 place-items-center border border-dashed border-slate-300 bg-white text-center"><div><FileSearch className="mx-auto h-8 w-8 text-slate-300" /><p className="mt-3 text-sm text-slate-500">输入问题后查看真实召回分段</p></div></div> : null}
                    {/* 查到了但被阈值/闸门全部筛掉：这个空态和"还没查"必须区分开 */}
                    {retrieval && !retrieval.results.length ? <div className="grid min-h-64 place-items-center border border-line bg-white text-sm text-slate-500">没有分段通过当前检索条件。</div> : null}
                  </div>
                </section>
              </div>
            </div>
          ) : null}

          {/* ── 页签 3：设置（复用 RetrievalSettings 受控组件） ── */}
          {tab === "settings" ? (
            <div className="mx-auto max-w-4xl">
              <div><h2 className="text-lg font-semibold text-ink">知识库设置</h2><p className="mt-1 text-sm text-slate-500">控制后续入库和召回测试使用的模型与参数。</p></div>
              <div className="mt-5 border border-line bg-white p-5 shadow-sm sm:p-7">
                <RetrievalSettings value={config} catalog={catalog} onChange={setConfig} />
                {message ? <div className="mt-5 flex items-center gap-2 bg-emerald-50 px-3 py-2.5 text-sm text-emerald-700"><Check className="h-4 w-4" /> {message}</div> : null}
                {/* 重叠 >= 切片长度会让切片原地打转，直接禁用保存 */}
                <div className="mt-6 flex justify-end border-t border-line pt-5"><button type="button" onClick={saveSettings} disabled={saving || config.chunk_overlap >= config.chunk_size} className="inline-flex h-10 items-center gap-2 rounded-md bg-teal-700 px-5 text-sm font-medium text-white disabled:opacity-40">{saving ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />} 保存设置</button></div>
              </div>
            </div>
          ) : null}
        </main>
      </div>

      {/* 右侧抽屉：文档的真实切片。点遮罩关闭，点内容不关（stopPropagation） */}
      {selectedDocument ? (
        <div className="fixed inset-0 z-50 flex justify-end bg-black/30" onMouseDown={() => setSelectedDocument(null)}>
          <div className="h-full w-full max-w-2xl overflow-y-auto border-l border-line bg-[#f7f8fa] shadow-2xl" onMouseDown={(event) => event.stopPropagation()}>
            <header className="sticky top-0 z-10 flex items-center gap-3 border-b border-line bg-white px-5 py-4"><FileText className="h-5 w-5 text-teal-700" /><div className="min-w-0 flex-1"><h2 className="truncate text-sm font-semibold text-ink">{selectedDocument.name}</h2><p className="text-xs text-slate-400">{selectedDocument.chunk_count} 个分段 · {formatBytes(selectedDocument.file_size)}</p></div><button title="关闭" onClick={() => setSelectedDocument(null)} className="grid h-8 w-8 place-items-center text-slate-500"><X className="h-4 w-4" /></button></header>
            <div className="space-y-3 p-4 sm:p-5">
              {/* 切片按 position 展示；父块信息在 content 里已拼好（父子块检索） */}
              {chunks.map((chunk) => <article key={chunk.id} className="border border-line bg-white p-4 shadow-sm"><div className="mb-3 flex items-center justify-between"><span className="text-xs font-semibold text-teal-700">分段 #{chunk.position}</span><span className="text-xs text-slate-400">{chunk.character_count} 字符</span></div><p className="whitespace-pre-wrap text-sm leading-6 text-slate-700">{chunk.content}</p></article>)}
              {!chunks.length ? <div className="py-20 text-center text-sm text-slate-500">正在读取分段...</div> : null}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
