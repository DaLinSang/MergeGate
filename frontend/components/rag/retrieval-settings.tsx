"use client";
// ↑ 这个组件有 onChange / 受控输入，必须跑在浏览器端。App Router 默认是服务端组件，少了这行会报错

import { Check, FileSearch, Layers3, Search } from "lucide-react";

import type { KnowledgeConfig, ModelCatalog, RetrievalMethod } from "@/lib/rag";

/**
 * RAG 检索参数面板
 * ================
 *
 * 【业务定位】app/rag 与 app/knowledge 共用的"调参台"。用户在这里改 top_k / 阈值 / 模型，
 *            改完由父组件 PATCH 回后端，下一次检索立即生效（不重建索引）。
 *
 * 【数据流向】
 *    入：value（父组件持有的配置）、catalog（可选模型清单，进页面时拉一次）
 *    出：onChange(整份新配置) —— 受控组件，自己不存 state
 *    落库：父组件 → apiPatch("/rag/repositories/{id}/config") → knowledge_configs 表
 *
 * 【要点】
 *    1. 全程不可变更新（{...value, [key]: next}），直接改 value 会导致 React 不重渲染
 *    2. 切 Embedding 模型必须同时改 dimensions —— 维度不同不能写进同一个 Milvus collection
 *    3. catalog 为 null 时下拉退化成"只有当前值"一项，避免白屏
 */

type Props = {
  value: KnowledgeConfig;                          // 当前配置（父组件持有，本组件只读）
  catalog: ModelCatalog | null;                    // 可选模型清单；null = 还没拉到
  onChange: (value: KnowledgeConfig) => void;      // 任一字段变化都回传整份配置
  showChunking?: boolean;                          // 是否显示分段设置（知识库详情页才显示）
};

// 三种检索方式。icon 直接用 lucide 组件，省一个映射表
const methods: { value: RetrievalMethod; label: string; note: string; icon: typeof Search }[] = [
  { value: "vector", label: "向量检索", note: "擅长理解语义相近的表达", icon: Layers3 },
  { value: "fulltext", label: "全文检索", note: "适合专有名词与精确关键词", icon: FileSearch },
  { value: "hybrid", label: "混合检索", note: "融合语义与关键词结果", icon: Search }
];

/** 检索参数面板。所有输入都是受控的，值只从 props.value 来。 */
export function RetrievalSettings({ value, catalog, onChange, showChunking = true }: Props) {
  // 泛型 K 保证 key 和 next 类型匹配：update("top_k", "abc") 会在编译期报错
  const update = <K extends keyof KnowledgeConfig,>(key: K, next: KnowledgeConfig[K]) => onChange({ ...value, [key]: next });

  return (
    <div className="space-y-7">
      {/* ── 分段设置：只影响"下次入库"，改完已有文档不会重新切片 ── */}
      {showChunking ? (
        <section>
          <h3 className="text-sm font-semibold text-ink">分段设置</h3>
          <p className="mt-1 text-xs text-slate-500">按自然段落切分，过长段落会继续按长度拆开。</p>
          <div className="mt-3 grid gap-3 sm:grid-cols-2">
            <label className="text-xs font-medium text-slate-600">
              最大分段长度
              <input
                type="number"
                min={200}
                max={4000}
                value={value.chunk_size}
                onChange={(event) => update("chunk_size", Number(event.target.value))}
                className="mt-1.5 h-10 w-full rounded-md border border-line bg-white px-3 text-sm text-ink outline-none focus:border-teal-600"
              />
            </label>
            <label className="text-xs font-medium text-slate-600">
              分段重叠长度
              <input
                type="number"
                min={0}
                max={1000}
                value={value.chunk_overlap}
                onChange={(event) => update("chunk_overlap", Number(event.target.value))}
                className="mt-1.5 h-10 w-full rounded-md border border-line bg-white px-3 text-sm text-ink outline-none focus:border-teal-600"
              />
            </label>
          </div>
        </section>
      ) : null}

      {/* ── Embedding 模型：改模型必须同步改维度 ── */}
      <section>
        <h3 className="text-sm font-semibold text-ink">索引方式</h3>
        <p className="mt-1 text-xs text-slate-500">DeepSeek 官方 API 暂无 Embedding，默认使用 Qwen。</p>
        <label className="mt-3 block text-xs font-medium text-slate-600">
          Embedding 模型
          {/* HTML select 的 value 只能是字符串，所以把 provider 和 model 用 "::" 拼成一个值 */}
          <select
            value={`${value.embedding_provider}::${value.embedding_model}`}
            onChange={(event) => {
              const [provider, model] = event.target.value.split("::");
              const selected = catalog?.embedding_models.find((item) => item.provider === provider && item.model === model);
              // dimensions 一起换：不同维度的向量不能共存于同一个 collection，换了模型却不换维度会导致写入失败
              onChange({ ...value, embedding_provider: provider, embedding_model: model, embedding_dimensions: selected?.dimensions ?? value.embedding_dimensions });
            }}
            className="mt-1.5 h-10 w-full rounded-md border border-line bg-white px-3 text-sm text-ink outline-none focus:border-teal-600"
          >
            {(catalog?.embedding_models ?? []).map((model) => (
              <option key={`${model.provider}-${model.model}`} value={`${model.provider}::${model.model}`}>
                {/* 不可用的模型不隐藏，而是标出来 —— 后端会自动降级，用户需要知道 */}
                {model.label}{model.available ? "" : " · 未配置，自动降级"}
              </option>
            ))}
            {/* catalog 还没拉到（首次渲染）：塞一个当前值兜底，否则下拉框是空的 */}
            {!catalog ? <option value={`${value.embedding_provider}::${value.embedding_model}`}>{value.embedding_model}</option> : null}
          </select>
        </label>
      </section>

      {/* ── 检索方式：三选一卡片 ── */}
      <section>
        <h3 className="text-sm font-semibold text-ink">检索设置</h3>
        <div className="mt-3 grid gap-2 sm:grid-cols-3">
          {methods.map((method) => {
            const Icon = method.icon;
            const selected = value.retrieval_method === method.value;
            return (
              <button
                key={method.value}
                type="button"
                onClick={() => update("retrieval_method", method.value)}
                className={`relative min-h-28 rounded-md border p-3 text-left transition ${selected ? "border-teal-600 bg-teal-50/60" : "border-line bg-white hover:border-slate-400"}`}
              >
                {selected ? <Check className="absolute right-2 top-2 h-4 w-4 text-teal-700" /> : null}
                <Icon className={`h-5 w-5 ${selected ? "text-teal-700" : "text-slate-500"}`} />
                <span className="mt-3 block text-sm font-medium text-ink">{method.label}</span>
                <span className="mt-1 block text-xs leading-5 text-slate-500">{method.note}</span>
              </button>
            );
          })}
        </div>
      </section>

      {/* ── Rerank 开关：关了就不显示模型下拉，省一次模型调用 ── */}
      <section className="border-t border-line pt-5">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h3 className="text-sm font-semibold text-ink">Rerank 模型</h3>
            <p className="mt-1 text-xs text-slate-500">对召回候选重新排序，API 不可用时自动使用本地重排。</p>
          </div>
          {/* 自绘开关：用 role="switch" + aria-checked 保证读屏器能正确播报状态 */}
          <button
            type="button"
            role="switch"
            aria-checked={value.rerank_enabled}
            onClick={() => update("rerank_enabled", !value.rerank_enabled)}
            className={`relative h-6 w-11 shrink-0 rounded-full transition ${value.rerank_enabled ? "bg-teal-700" : "bg-slate-300"}`}
          >
            <span className={`absolute top-1 h-4 w-4 rounded-full bg-white transition ${value.rerank_enabled ? "left-6" : "left-1"}`} />
          </button>
        </div>
        {value.rerank_enabled ? (
          <select
            value={`${value.rerank_provider}::${value.rerank_model}`}
            onChange={(event) => {
              const [provider, model] = event.target.value.split("::");
              onChange({ ...value, rerank_provider: provider, rerank_model: model });
            }}
            className="mt-3 h-10 w-full rounded-md border border-line bg-white px-3 text-sm text-ink outline-none focus:border-teal-600"
          >
            {(catalog?.rerank_models ?? []).map((model) => (
              <option key={`${model.provider}-${model.model}`} value={`${model.provider}::${model.model}`}>
                {model.label}{model.available ? "" : " · 未配置，自动降级"}
              </option>
            ))}
            {!catalog ? <option value={`${value.rerank_provider}::${value.rerank_model}`}>{value.rerank_model}</option> : null}
          </select>
        ) : null}
      </section>

      {/* ── Top K + 分数阈值 ── */}
      <section className="grid gap-4 border-t border-line pt-5 sm:grid-cols-2">
        <label className="text-xs font-medium text-slate-600">
          Top K
          <div className="mt-2 flex items-center gap-3">
            {/* 滑块和数字框共用同一个值；注意滑块上限 10、数字框上限 20，两者范围不一致 */}
            <input type="range" min={1} max={10} value={value.top_k} onChange={(event) => update("top_k", Number(event.target.value))} className="min-w-0 flex-1 accent-teal-700" />
            <input type="number" min={1} max={20} value={value.top_k} onChange={(event) => update("top_k", Number(event.target.value))} className="h-9 w-16 rounded-md border border-line text-center text-sm outline-none focus:border-teal-600" />
          </div>
        </label>
        <div>
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-600">Score 阈值</span>
            {/* 阈值开关：不勾选时滑块置灰但值保留，重新勾选能恢复上次的值 */}
            <input type="checkbox" checked={value.score_threshold_enabled} onChange={(event) => update("score_threshold_enabled", event.target.checked)} className="h-4 w-4 accent-teal-700" />
          </div>
          <div className="mt-2 flex items-center gap-3">
            <input type="range" min={0} max={1} step={0.01} disabled={!value.score_threshold_enabled} value={value.score_threshold} onChange={(event) => update("score_threshold", Number(event.target.value))} className="min-w-0 flex-1 accent-teal-700 disabled:opacity-40" />
            <span className="w-12 text-right text-sm text-slate-600">{value.score_threshold.toFixed(2)}</span>
          </div>
        </div>
      </section>
    </div>
  );
}
