# RAG 代码学习路线（DevFlow AI）

> 目标：把 `backend/app/services/rag/` 这 12 个文件，按**数据真实流动的顺序**串起来读，
> 每一步都知道「在哪个文件、哪个函数、为什么这么设计」。
>
> 配套：代码里的注释已按 v5 标准写（文件头有函数地图，长函数有调用链）。

---

## 0. 一句话理解 RAG

> **RAG = 先去资料里找到相关片段，再把片段塞进 prompt 让模型基于它回答。**

不用 RAG 的模型只能靠训练记忆回答，会瞎编；用了 RAG，答案有出处、可核对、可更新。

本项目把 RAG 拆成**两条独立的链路**，理解这一点是读懂代码的前提：

```
【入库链路】离线跑，慢一点没关系，重点是"存得对"
   GitHub Issue / PR / CI 日志 / 用户上传文档
        ↓  解析 → 切片 → 向量化
   PostgreSQL（完整全文）  +  Milvus（向量 + 截断正文）

【查询链路】在线跑，每次提问都要走，重点是"快且准"
   用户问题
        ↓  向量化 → 召回 → 融合 → 重排 → 扩展 → 过滤 → 压缩 → 准入
   带引用的答案
```

**两条链路唯一的连接点是"向量空间契约"**：入库和查询必须用同一个 embedding 模型、同一维度，
否则向量不在同一个空间，相似度算出来是随机的（而且不报错）。

---

## 1. 入库链路：文件怎么变成可检索的向量

```
用户上传 report.pdf
    │
    │ ① api/routes/rag.py
    │    存文件 → 建 KnowledgeSourceDocument(status=queued) → 返回 202
    │    把任务丢给 BackgroundTasks
    ↓
    │ ② services/rag/document_processing.py :: process_knowledge_document()
    │    ← 整个入库流程的"总指挥"，读它就能看到全貌
    │
    ├─ ③ document_extraction.py :: extract_document_text()
    │      PDF → 逐页提取 + 插 <!-- page: n -->
    │      DOCX → 先 zip 安全校验，再段落/表格转 Markdown
    │      其他 → 多编码尝试解码
    │
    ├─ ④ chunking.py :: chunk_document()
    │      按文件类型选策略：markdown / plain / json / csv / code
    │      产出：content + section_path + page + parent_id + child_index
    │
    ├─ ⑤ embeddings.py :: embed_knowledge_texts(is_query=False)
    │      向量化的是 "标题\n正文"（不是只有正文！）
    │
    └─ ⑥ vector_store.py :: add_documents()
          PG 写完整全文（事实源）
          Milvus 写向量 + 截断正文（索引副本）
```

### 每一步的关键设计（读代码时重点看这些）

| 步骤 | 文件 | 关键设计 | 为什么重要 |
|---|---|---|---|
| ③ 解析 | `document_extraction.py` | PDF 插页码标记、DOCX 转 Markdown | 给切片器提供结构信息，否则引用无法定位到页/章节 |
| ③ 解析 | `document_extraction.py` | zip 炸弹校验在 `docx.Document()` **之前** | 校验放后面就白做了，恶意文件已经进内存 |
| ④ 切片 | `chunking.py` | **父子块**：小切片检索，大父块回答 | 见下方"核心概念" |
| ④ 切片 | `chunking.py` | 代码块切断要补回 ` ``` ` 围栏 | 否则切片语法不完整，模型读不懂 |
| ⑤ 向量化 | `embeddings.py` | **标题 + 正文**一起向量化 | 最便宜的召回率提升：正文只说"默认 30 秒"，标题是"超时设置"，用户问"超时多久"才能命中 |
| ⑤ 向量化 | `embeddings.py` | `validate_embedding_contract()` 每次必校验 | 防止不同模型的向量混进同一个 collection |
| ⑥ 双写 | `vector_store.py` | PG 存全文，Milvus 存截断版 | Milvus 的 `VARCHAR max_length` 单位是**字节**，8192 字节 ≈ 2700 汉字 |
| ⑥ 双写 | `vector_store.py` | `_clip_utf8()` 按字节截断 | 按字符截断会超长报错 |
| ⑥ 双写 | `vector_store.py` | `delete_document_group()` 先删再写 | 重传同一文件保证幂等，否则重复召回 |

---

## 2. 查询链路：问题怎么变成带引用的答案

```
用户问"登录为什么失败"
    │
    │ ① knowledge_base.py :: retrieve_knowledge()
    │    ← 六阶段流水线的总编排，读它看全局
    │
    ├─ ② embeddings.py :: embed_knowledge_query()
    │      is_query=True → 加查询专用前缀（Qwen3-Embedding 的训练方式）
    │
    ├─ ③ retrieval.py :: search_similar_documents()
    │      ├ _milvus_vector_candidates()  向量召回
    │      ├ _keyword_candidates()        BM25 关键词召回（查 PG 的 documents 表）
    │      ├ _merge_candidates()          ★ 融合：0.62×向量 + 0.30×关键词 + 0.08×双路命中
    │      ├ _deduplicate_candidates()    去重（精确指纹 + Jaccard 0.9）
    │      └ _limit_parent_occupancy()    同一父块最多留 2 个切片
    │
    ├─ ④ rerank.py :: rerank_documents()
    │      粗排后的 top-N 做精排（Qwen rerank 或 11 项加权启发式）
    │
    ├─ ⑤ retrieval.py :: expand_parent_context()
    │      小切片命中 → 用 parent_id 拼回大父块（"小的准，大的全"）
    │
    ├─ ⑥ 分数阈值过滤（config.score_threshold）
    │
    ├─ ⑦ context_compression.py :: compress_retrieval_results()
    │      按 query 抽取最相关句子，控制 token
    │
    ├─ ⑧ answer_gate.py :: evaluate_answer_gate()
    │      四种决策：answer / clarify / refuse / conflict
    │      ★ 不是 answer 就【不调 LLM】—— 省成本，也防幻觉
    │
    └─ ⑨ qa.py :: answer_question()
           <knowledge> 包裹证据 → 调 LLM → 失败则降级 extractive_answer()
```

### 六阶段流水线在 `knowledge_base.py` 里的记录方式

每个阶段都记 `input_count` / `output_count`，这是排查"为什么没结果"的关键：

```
① hybrid_recall      50 → 50
② rerank             50 → 8      ← 如果这里变成 0，是重排模型的问题
③ parent_expansion   8  → 8
④ score_threshold    8  → 0      ← 如果这里变成 0，是阈值设太高
⑤ context_compression 0 → 0
⑥ answer_gate        0  → refuse
```

**一句话**：哪一步 `output_count` 归零，问题就在那一步。

---

## 3. 核心概念速查（这些是 RAG 的通用知识，不只本项目）

| 概念 | 一句话 | 本项目在哪 |
|---|---|---|
| **Chunk（切片）** | 把长文档切成小块，每块单独向量化 | `chunking.py::chunk_document()` |
| **Overlap（重叠）** | 相邻块重叠一段，防止语义被切断 | `config.chunk_overlap`（默认 100） |
| **父子块** | 用小切片做检索（准），命中后返回它的大父块（全） | `parent_id` + `expand_parent_context()` |
| **Hybrid（混合检索）** | 向量（语义）+ 关键词（精确）一起召回，再融合 | `retrieval.py::_merge_candidates()` |
| **Rerank（重排）** | 粗排取 50 条，用更强的模型精排出前 8 条 | `rerank.py::rerank_documents()` |
| **Score Threshold** | 分数低于阈值直接丢，宁缺毋滥 | `config.score_threshold`（默认 0.8） |
| **Context Compression** | 把证据压缩到 token 预算内，只留相关句子 | `context_compression.py` |
| **Answer Gate（准入）** | 判断"证据够不够回答"，不够就拒答/反问 | `answer_gate.py::evaluate_answer_gate()` |
| **Citation（引用）** | 答案里标 [1][2]，能点回原始证据 | `qa.py::build_sources()` |
| **向量空间契约** | 同一 collection 必须同模型同维度 | `embeddings.py::validate_embedding_contract()` |
| **数据分流** | 什么进向量库、什么直查数据库 | `source_policy.py::retrieval_mode()` |

---

## 4. 推荐阅读顺序（分 4 轮，每轮 1~2 小时）

### 第 1 轮：搞清"什么进 RAG"（30 分钟）

```
source_policy.py（74 行，最短，先建立边界感）
```

**读的时候回答三个问题**：
1. 哪些类型进 RAG？哪些不进？
2. 为什么 `code_file`（源码）两边都不在？
3. `retrieval_mode()` 为什么默认返回 `direct` 而不是 `rag`？

### 第 2 轮：入库链路（2 小时）

```
document_extraction.py   （169 行）解析：字节 → 文本
    ↓
chunking.py              （1203 行，最难，但最重要）
    ↓
embeddings.py            （282 行）向量化
    ↓
vector_store.py          （640 行）双写
    ↓
document_processing.py   （187 行）把上面四步串起来 —— 最后读它，会豁然开朗
```

**`chunking.py` 的读法**（1203 行，别从头读）：
1. 先看 `chunk_document()`（约 886 行处）—— 它是分发器
2. 再按后缀看具体策略：`_markdown_blocks()` / `_plain_blocks()` / `_json_blocks()` / `_csv_chunks()`
3. 最后看 `_pack_blocks()` 和 `_emit_chunk()` —— 父子块就是在这里生成的

### 第 3 轮：查询链路（2 小时）

```
retrieval.py             （824 行）混合检索 + 融合
    ↓
rerank.py                （176 行）重排
    ↓
knowledge_base.py        （449 行）六阶段编排 —— 相当于查询侧的 main
    ↓
answer_gate.py           （176 行）准入判定
    ↓
qa.py                    （371 行）生成答案
```

**`retrieval.py` 的读法**：
1. 先看文件顶部的 5 个常量（权重 0.62 / 0.30 / 0.08）
2. 再看 `search_similar_documents()`（约 674 行）—— 主流程
3. 然后逐个看它调用的私有函数

### 第 4 轮：补齐（1 小时）

```
indexing.py（245 行）—— Issue/PR/CI 怎么进 RAG（和上传文件走的是同一套底层）
```

重点看 `sanitize_ci_log()` —— CI 日志脱敏的 6 类规则（私钥/token/AWS/JWT/Bearer/add-mask）。

---

## 5. 六个容易踩的坑（面试也常问）

### ① Milvus 的 `VARCHAR max_length` 单位是**字节**，不是字符

```python
max_length=8192   # 不是 8192 个字符！一个汉字 3 字节 → 约 2700 汉字
```

按字符截断，中英文混合内容会超长报错。所以 `vector_store._clip_utf8()` 按字节切。

### ② 向量空间契约不一致 = 静默错误

不同模型的向量不在同一个空间。混进同一个 collection，余弦相似度算出来是**随机的**，
检索结果完全错乱，**而且不报错**。这是最难排查的一类 bug。

### ③ `is_query` 用错会降低召回率

Qwen3-Embedding 这类模型对"文档"和"查询"用不同前缀。
入库用 `is_query=False`，查询用 `is_query=True`，混用不会报错，只会变差。

### ④ 孤儿向量

删了 PG 的记录但没删 Milvus → 向量还在被召回，回 PG 查不到正文 → 引用里出现空内容。
所以 `delete_milvus_documents(strict=True)` 默认"删不掉就报错"。

### ⑤ 整组替换的代价

`replace_documents()` 是"先删光再写入"。如果这次拉取失败（payloads 为空），
原有数据会被删光。调用方必须先确认"拿到了完整数据"。

### ⑥ 父块占用不限制会浪费预算

同一个父块的 5 个切片都被召回时，5 条证据其实是同一段内容。
所以 `_limit_parent_occupancy()` 限制每个父块最多 2 个切片。

---

## 6. 注释进度

**全部 12 个文件已完成 v5 注释**（4,796 行 → 7,880 行，+64%）。

| 文件 | 行数 | 状态 |
|---|---|---|
| `source_policy.py` | 74 → 188 | ✅ v5 完成 |
| `document_extraction.py` | 169 → 344 | ✅ v5 完成 |
| `embeddings.py` | 282 → 542 | ✅ v5 完成 |
| `vector_store.py` | 640 → 1,046 | ✅ v5 完成 |
| `document_processing.py` | 187 → 317 | ✅ v5 完成 |
| `chunking.py` | 1,203 → 1,833 | ✅ v5 完成 |
| `retrieval.py` | 824 → 1,288 | ✅ v5 完成 |
| `rerank.py` | 176 → 317 | ✅ v5 完成 |
| `knowledge_base.py` | 449 → 667 | ✅ v5 完成 |
| `answer_gate.py` | 176 → 338 | ✅ v5 完成 |
| `qa.py` | 371 → 594 | ✅ v5 完成 |
| `indexing.py` | 245 → 406 | ✅ v5 完成 |

> 「v5」= 文件头有【在 RAG 流水线中的位置】+【本文件函数地图】；
> 长函数有【调用链】【被谁调用】【数据流向】；函数体内按「# ── 步骤 N ──」分块。

**验证**：12 个文件 AST 逐部件比对全部一致（零逻辑改动）；全量测试 209 passed。

---

## 7. 一张图记住全流程

```
┌───────────────────── 入库（离线）─────────────────────┐
│  文件 ──解析──> 文本 ──切片──> chunks ──向量化──> 向量  │
│                                    │                  │
│                          ┌─────────┴─────────┐        │
│                          ↓                   ↓        │
│                   PG: 完整全文        Milvus: 向量     │
│                   （事实源）          （索引副本）      │
└──────────────────────────┬───────────────────────────┘
                           │ 向量空间契约（同模型同维度）
┌──────────────────────────┴───────────────────────────┐
│  问题 ──向量化──> 向量 ──召回──> 候选 ──融合──> 排序    │
│         ──重排──> 精排 ──父块扩展──> 完整上下文         │
│         ──压缩──> 预算内 ──准入──> 够不够？             │
│         ──生成──> 带引用的答案                          │
└──────────────────────────────────────────────────────┘
```
