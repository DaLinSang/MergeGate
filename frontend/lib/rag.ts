/**
 * RAG 知识库的前端契约
 * ====================
 *
 * 【业务定位】app/rag/page.tsx 与 components/rag/retrieval-settings.tsx 共用的全部类型。
 *            对应后端 app/schemas/rag.py，字段名必须逐一对应。
 *
 * 【数据流向】
 *    入库：文件 → apiUpload → document_processing → chunking → embeddings → Milvus
 *          前端只能看到 status 变化（queued → parsing → splitting → embedding → completed）
 *    检索：query → apiPost("/rag/.../retrieve") → 六阶段流水线 → RetrievalResponse
 *          → 渲染结果卡片 + 引用编号
 *
 * 【要点】
 *    1. status 是"入库流水线走到哪一步"，不是 HTTP 状态；前端靠它显示进度条
 *    2. RetrievalResult.citation 是回答里的 [1][2] 编号，必须由后端统一发号，前端不能自己排
 *    3. retrieval_config 里塞了 answer_gate / pipeline，是"这次检索为什么给这个答案"的证据
 */

/** 检索方式：vector（纯语义）/ fulltext（纯 BM25）/ hybrid（双路加权融合，默认） */
export type RetrievalMethod = "vector" | "fulltext" | "hybrid";

/** 知识库的可调参数。改完 PATCH 回后端，下次检索立即生效（不重建索引）。 */
export type KnowledgeConfig = {
  embedding_provider: string;             // 向量化服务商（openai / deepseek / local …）
  embedding_model: string;                // 向量模型名
  embedding_dimensions: number;           // 向量维度，必须和 Milvus collection 建表时一致
  retrieval_method: RetrievalMethod;      // 检索方式
  rerank_enabled: boolean;                // 是否重排（开了更准但更慢，多一次模型调用）
  rerank_provider: string;                // 重排服务商
  rerank_model: string;                   // 重排模型名
  top_k: number;                          // 最终返回条数（重排后的截断位置）
  score_threshold_enabled: boolean;       // 是否启用分数门槛（过滤低相关结果）
  score_threshold: number;                // 门槛值，低于它的候选直接丢弃
  chunk_size: number;                     // 切片大小（字符数）
  chunk_overlap: number;                  // 相邻切片重叠（防止答案被切断在边界上）
};

/** 某个模型在当前环境的可用状态（不是配置，是探测结果）。 */
export type RuntimeModel = {
  provider: string;                       // 服务商
  model: string;                          // 模型名
  available: boolean;                     // 当前是否可用（没配 key 或探测失败为 false）
  fallback?: string | null;               // 不可用时降级到哪个模型
  enabled?: boolean;                      // 用户是否勾选启用
  dimensions?: number;                    // 向量维度（仅 embedding 模型有）
};

/** 知识库 = 一个仓库的 RAG 视图（不是独立实体，跟着仓库走）。 */
export type KnowledgeBase = {
  id: string;                             // 知识库 ID（= 仓库 ID）
  name: string;                           // 仓库名
  full_name: string;                      // owner/name
  description: string;                    // 描述
  provider: string;                       // 平台类型
  document_count: number;                 // 文档数（已入库的）
  chunk_count: number;                    // 切片总数（决定 Milvus 里的记录数）
  character_count: number;                // 总字符数（估算 token 成本用）
  config: KnowledgeConfig;                // 当前检索配置
  runtime: { embedding: RuntimeModel; rerank: RuntimeModel };  // 两个模型的实时可用性
  created_at?: string;                    // 创建时间
  updated_at?: string;                    // 更新时间
};

/** 一篇待入库 / 已入库的文档。 */
export type KnowledgeDocument = {
  id: string;                             // 文档 ID
  repo_id: string;                        // 所属仓库（所有查询都必须带，否则会跨库串数据）
  name: string;                           // 文件名
  source_type: string;                    // 来源类型：issue / pr / doc_file / code_file …
  status: "queued" | "parsing" | "splitting" | "embedding" | "completed" | "failed";  // 入库流水线阶段
  content_type?: string;                  // MIME 类型
  file_size: number;                      // 原始字节数
  character_count: number;                // 抽取后的字符数（≠ file_size，PDF 会差很多）
  chunk_count: number;                    // 切成了几片
  error_message?: string;                 // 失败原因（status=failed 时才有）
  created_at?: string;                    // 入库开始时间
  completed_at?: string;                  // 入库完成时间
};

/** 单个切片。前端只用来看"这篇文档被切成了什么"，不参与检索。 */
export type KnowledgeChunk = {
  id: string;                             // 切片 ID
  position: number;                       // 在文档内的序号（父块拼回时按它排序）
  title: string;                          // 所属标题（切片时按标题分层）
  content: string;                        // 切片正文
  character_count: number;                // 字符数
  enabled: boolean;                       // 是否参与检索（关掉 = 软删除，向量还在）
};

/** 模型商店里的一项（可选项，带推荐标记）。 */
export type CatalogModel = {
  provider: string;                       // 服务商
  model: string;                          // 模型名
  label: string;                          // 展示名
  available: boolean;                     // 是否可用
  recommended: boolean;                   // 是否官方推荐（前端打"推荐"标）
  fallback?: string | null;               // 降级模型
  dimensions?: number;                    // 向量维度
};

/** 可选模型清单，进 RAG 页面时拉一次。 */
export type ModelCatalog = {
  embedding_models: CatalogModel[];       // 可选向量模型
  rerank_models: CatalogModel[];          // 可选重排模型
  deepseek: { embedding_available: boolean; rerank_available: boolean; note: string };  // 特殊说明（DeepSeek 不提供向量/重排）
};

/** 一条检索结果（= 一张结果卡片）。 */
export type RetrievalResult = {
  citation: number;                       // 引用编号，对应回答里的 [1][2]，后端统一发号
  document_id: string;                    // 命中的文档
  source_id: string;                      // 命中的具体来源（issue 号 / 文件路径 / PR 号）
  source_type: string;                    // 来源类型，决定卡片图标
  title: string;                          // 标题
  snippet: string;                        // 摘录（压缩后的片段，不是全文）
  score: number;                          // 融合后的最终分（重排分优先）
  metadata: Record<string, unknown>;      // 附加信息（行号、作者、时间…）
  retrieval: Record<string, unknown>;     // 检索过程证据（向量分 / BM25 分 / 各阶段耗时）
};

/** 一次检索的完整响应。 */
export type RetrievalResponse = {
  id: string;                             // 检索记录 ID（可回看历史）
  query: string;                          // 原始问题
  results: RetrievalResult[];             // 结果列表（已按 score 降序）
  result_count: number;                   // 结果条数
  took_ms: number;                        // 总耗时（毫秒）
  retrieval_config: KnowledgeConfig & {   // 本次实际生效的配置（可能是默认值，不一定是用户当前设置）
    runtime?: unknown;                    // 运行时信息
    fusion_method?: string;               // 融合方式（加权 / RRF）
    answer_gate?: {                       // 准入闸门：决定"给答案"还是"拒答"
      decision?: "answer" | "ask_clarification" | "insufficient_evidence" | "conflict";  // 四种结论
      reason?: string;                    // 判定理由（前端展示给用户看）
      evidence_count?: number;            // 证据条数
      top_score?: number;                 // 最高分（不够门槛就拒答）
    };
    pipeline?: {                          // 六阶段流水线的执行记录
      retrieval_method?: string;          // 实际用的检索方式（可能因 Milvus 挂了降级）
      fusion_method?: string;             // 实际用的融合方式
      stages?: Array<Record<string, unknown> & { name?: string }>;  // 每个阶段的耗时与数量
    };
  };
  created_at?: string;                    // 检索时间
};

/** 检索历史列表项（只给列表用，不含结果正文）。 */
export type RetrievalHistory = {
  id: string;                             // 检索记录 ID
  query: string;                          // 问题
  result_count: number;                   // 结果条数
  took_ms: number;                        // 耗时
  created_at?: string;                    // 时间
};

/** 入库状态 → 中文文案。集中在这里，避免每个页面各写一套 switch。 */
export const statusLabel: Record<KnowledgeDocument["status"], string> = {
  queued: "排队中",                        // 已入库任务，等 worker 取
  parsing: "解析中",                       // pypdf / python-docx 抽文本
  splitting: "分段中",                     // chunking，父块 + 子块
  embedding: "向量化",                     // 调 embedding 模型 + 写 Milvus
  completed: "可用",                       // 可以检索了
  failed: "失败"                           // 看 error_message
};

/** 数字千分位。用于文档数 / 切片数这类可能上万的量。 */
export function formatNumber(value: number) {
  return new Intl.NumberFormat("zh-CN").format(value);
}

/** 字节数转可读大小。只到 MB —— 单篇文档不会到 GB。 */
export function formatBytes(value: number) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

/** 时间戳转 "月-日 时:分"。空值统一显示 "-"，避免页面出现 Invalid Date。 */
export function formatDate(value?: string) {
  if (!value) return "-";
  return new Intl.DateTimeFormat("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }).format(
    new Date(value)
  );
}
