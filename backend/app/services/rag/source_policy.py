"""
数据分流策略（RAG 的第一道门）
============================

【在 RAG 流水线中的位置】
    整条入库链路的第一步，也是"要不要入库"的判定器：

        业务数据产生
             ↓
        ★ 本文件：retrieval_mode() 判断走 rag 还是 direct
             ↓ rag                    ↓ direct
        vector_store 入库         留在原表，用 SQL 直查
             ↓
        retrieval 检索

    所以读 RAG 代码，从这里开始最合理 —— 先搞清"什么该进向量库"。

【业务定位】
    决定"什么数据进 RAG 向量库、什么数据直查数据库"。
    这不是技术选择，是架构边界的显式声明：写在这里，就是为了让人能一眼看到边界在哪。

【数据流向】
    上游（谁调它）：
        services/rag/indexing.py      → should_vectorize() 判断要不要建索引
        services/rag/vector_store.py  → should_vectorize() 写入前再拦一道
        services/rag/retrieval.py     → is_searchable_source() 过滤可检索类型
    下游：无 —— 纯常量 + 纯函数，不碰数据库、不发请求

【核心规则】
    只有"需要跨来源语义召回 + 会反复使用"的非结构化知识才进 RAG。
    源码（code_file）被【刻意排除】—— 理由见文件末尾的说明。

【本文件函数地图】（5 个函数，30 秒能读完）
    ① retrieval_mode()        → 核心判定：rag / direct
    ② should_vectorize()      → 语义包装：要不要写向量
    ③ is_searchable_source()  → 语义包装：能不能被检索
    ④ normalize_source_types()→ 工具：清洗类型列表（去空去重）
    ⑤ policy_metadata()       → 把判定结果转成可入库的元数据
"""

from collections.abc import Iterable
from typing import Literal


# 检索方式的两种取值。
# 用 Literal 而不是 str：IDE 和类型检查器能提示，写错值会立刻被发现。
RetrievalMode = Literal["rag", "direct"]


# ── 白名单：进 RAG 的（非结构化、值得语义召回、会反复用）──
# RAG is reserved for reusable, unstructured project knowledge where semantic
# recall is useful. Structured live facts remain in their source tables/APIs.
#
# 判断某个类型该不该进来的三个自问：
#   ① 它是长文本吗？（短字段用 SQL 精确查更准更快）
#   ② 用户会换着说法问它吗？（"登录失败" vs "认证报错" → 需要语义）
#   ③ 它会反复被引用吗？（一次性数据不值得建索引）
RAG_SOURCE_TYPES = frozenset(
    {
        "issue",            # Issue 正文（长文本，需要语义召回）
        "pull_request",     # PR 说明与 Review 讨论
        "workflow_run",     # 失败 CI 日志
        "project_doc",      # 项目文档（README/设计文档）
        "knowledge_file",   # 用户上传的知识文件
        "memory_note",      # 已人工批准的长期记忆
    }
)

# ── 另一侧：直查数据库的（结构化、实时、有明确字段）──
# 注意：这个集合【不参与判定】。retrieval_mode() 只查 RAG 白名单，
# 不在白名单里的一律 direct。这里列出来是为了"文档价值"——
# 让人看到"哪些类型是故意不进 RAG 的"，而不是漏掉了。
DIRECT_SOURCE_TYPES = frozenset(
    {
        "project_overview",     # 项目概览（实时统计）
        "project_manifest",     # 依赖清单
        "team_member",          # 团队成员（要精确名单，不能靠语义猜）
        "weekly_report",        # 周报（派生数据）
        "chat_session",         # 会话（按会话范围读，不做全局检索）
        "conversation_memory",  # 会话记忆
        "thread_memory",        # 仓库长期记忆
    }
)


def retrieval_mode(source_type: str) -> RetrievalMode:
    """【核心判定】某类数据走 RAG 还是直查。

    【调用链】无 —— 只查一个 frozenset，没有内部调用。
    【被谁调用】should_vectorize() / is_searchable_source() / policy_metadata()
               以及 rag/indexing.py、rag/vector_store.py

    【默认 direct 而不是 rag】
        保守策略：不确定就不向量化。
        如果默认 rag，新增一种数据类型时会悄悄进库，产生一堆没人用的向量，
        而且检索时可能被召回成噪声。默认 direct 则是"显式加入白名单才生效"。
    """
    if source_type in RAG_SOURCE_TYPES:
        return "rag"
    return "direct"


def normalize_source_types(source_types: Iterable[str | None] | None) -> list[str]:
    """清洗来源类型列表：去掉空值、去重（保持原顺序）。

    【调用链】无
    【被谁调用】api/routes/rag.py、services/rag/retrieval.py（处理前端传来的过滤条件）

    【为什么用 dict.fromkeys 而不是 set】
        set 会打乱顺序。前端传 ["issue", "pull_request"] 时，
        返回顺序最好保持一致，日志和调试都更好读。
        dict.fromkeys 既能去重又保序（Python 3.7+ 字典有序）。

    【为什么容忍 None / 空串】
        前端多选组件取消全选时可能传 [null]，不能因此报错。
    """
    normalized = [str(source_type).strip() for source_type in (source_types or []) if source_type and str(source_type).strip()]
    return list(dict.fromkeys(normalized))


def should_vectorize(source_type: str) -> bool:
    """要不要把这类数据写入向量库。

    【调用链】retrieval_mode() → 本函数
    【被谁调用】services/rag/indexing.py（建索引前）、vector_store.add_document()（写入前）

    和 is_searchable_source() 逻辑完全一样，但语义不同：
        这个问"要不要【写】"，那个问"能不能【读】"。
    之所以分开写两个名字，是因为调用点的意图不同 —— 读代码时一眼能懂。
    """
    return retrieval_mode(source_type) == "rag"


def is_searchable_source(source_type: str) -> bool:
    """这类数据能不能被语义检索到。

    【调用链】retrieval_mode() → 本函数
    【被谁调用】services/rag/retrieval.py（检索时过滤 source_type）

    理论上 should_vectorize 为 True 的类型才可检索，
    但保留独立函数是为了将来可能出现"入库了但暂不检索"的场景（比如灰度）。
    """
    return retrieval_mode(source_type) == "rag"


def policy_metadata(source_type: str) -> dict[str, object]:
    """把判定结果转成可入库的元数据。

    【调用链】retrieval_mode() → 本函数
    【被谁调用】services/rag/vector_store.py（写 Milvus 时塞进 metadata）

    【为什么要把策略写进元数据】
        检索时可以直接按 metadata["rag_enabled"] 过滤，
        不必在 Python 里再算一遍；也让"这条向量为什么在这里"可追溯。

    返回两个字段：
        retrieval_mode → "rag" / "direct"（原始判定）
        rag_enabled    → 布尔值（方便 Milvus 表达式过滤）
    """
    mode = retrieval_mode(source_type)
    return {
        "retrieval_mode": mode,
        "rag_enabled": mode == "rag",
    }


# ════════════════════════════════════════════════════════════════
# 【延伸理解】为什么源码（code_file）不进 RAG？
# ════════════════════════════════════════════════════════════════
#
# 这是本项目一个刻意的架构决策，也是面试常问的点：
#
# 1. 源码有精确结构，不需要"语义猜"
#    想找 create_app 的定义，用 AST 索引直接定位到文件+行号（services/code_analysis.py），
#    比向量检索准得多，也不会"猜错"。
#
# 2. 代码会被频繁修改，向量化成本高且容易过期
#    每次 commit 都要重新切片+向量化，索引和真实代码很容易不一致。
#    而"按文件路径实时读"永远是最新的（services/code_search.py / worktree_manager.py）。
#
# 3. 代码的"相似"不等于"相关"
#    向量检索会把所有长得很像的函数都召回（比如十几个 get_xxx），噪声极大。
#
# 所以本项目的做法是：
#    源码 → 代码图谱（code_symbols / code_relations）+ 实时文件读取
#    文档/Issue/PR/日志 → RAG 向量库
# 这两条路在 Agent 里是两套不同的工具，不要混。
