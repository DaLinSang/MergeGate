"""
文档处理流水线（RAG 入库的调度器）
=================================

【在 RAG 流水线中的位置】
    这是"用户上传文件"这条链路的【总指挥】，把前面几个零件串起来：

        api/routes/rag.py 存文件 → 建 KnowledgeSourceDocument(status=queued)
                              → 返回 202，把本文件挂到 BackgroundTasks
             ↓
        ★ 本文件：process_knowledge_document()
             ① parsing    → document_extraction.py 解析成文本
             ② splitting  → chunking.py 切片（读 knowledge_base_configs 的参数）
             ③ embedding  → embeddings.py 向量化（标题 + 正文）
             ④ 双写       → vector_store.py 写 PostgreSQL + Milvus
             ↓
        status=completed（或 failed + error_message）

【业务定位】
    数据流 3（文档入库）的调度器：解析 → 切片 → 向量化 → 双写。

【数据流向】
    上游（谁调它）：
        api/routes/rag.py → BackgroundTasks.add_task(process_knowledge_document, id)
    下游（它调谁）：
        document_extraction.extract_document_text()  解析
        chunking.chunk_document()                    切片
        embeddings.embed_knowledge_texts()           向量化
        knowledge_base.get_or_create_config()        取该仓库的切片参数
        vector_store.delete_document_group()         删旧数据（保证幂等）
        vector_store.add_documents()                 双写

【核心规则】
    1. 状态机：queued → parsing → splitting → embedding → completed / failed
       每进一个阶段就 commit 一次 —— 前端才能看到进度条在动。
    2. 任何一步失败：回滚 + 标 failed + 记录原因（前端能看到"为什么失败"）。
    3. 后台任务不在请求生命周期内，必须自己开 Session（不能复用请求的 db）。

【本文件函数地图】
    process_knowledge_document()  ← 主流程（4 个阶段）
        ├─ _set_status()          改状态（独立 Session）
        ├─ extract_text()         → document_extraction
        ├─ get_or_create_config() → knowledge_base（取切片参数）
        ├─ chunk_document()       → chunking
        ├─ _embed_in_batches()    → embeddings（分批）
        ├─ delete_document_group()+ add_documents() → vector_store（双写）
        └─ _embed_in_batches() 内部：分批循环

【一个容易被忽略的点：为什么"标题 + 正文"一起向量化】
    见阶段 ③ 的注释 —— 这是提升召回率最便宜的技巧。
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.db.models import KnowledgeSourceDocument
from app.db.session import SessionLocal
from app.services.rag.chunking import chunk_document
from app.services.rag.document_extraction import MAX_DOCUMENT_BYTES, extract_document_text
from app.services.rag.embeddings import embed_knowledge_texts
from app.services.rag.knowledge_base import get_or_create_config
from app.services.rag.vector_store import DocumentPayload, add_documents, delete_document_group

# 支持的文件后缀（与前端上传限制保持一致）
# 注意：这里只做"白名单校验"，真正的解析分发在 document_extraction.extract_document_text()
SUPPORTED_KNOWLEDGE_SUFFIXES = {".txt", ".md", ".markdown", ".mdx", ".pdf", ".docx", ".json", ".csv", ".log"}

# 向量化批大小：64 条一批，平衡显存/内存与吞吐
# 太小 → HTTP 往返次数多、模型利用率低；太大 → 一次编码几千条会 OOM
EMBEDDING_BATCH_SIZE = 64


def extract_text(filename: str, data: bytes) -> str:
    """解析入口（薄封装，便于测试替换）。

    【调用链】document_extraction.extract_document_text() → 本函数
    【被谁调用】process_knowledge_document()

    【为什么要包一层】
        测试时可以用 monkeypatch 替换 extract_text，不必真的造一个 PDF。
        这种"为测试留缝"的薄封装在小项目里看着多余，但很实用。
    """
    return extract_document_text(filename, data)


def _embed_in_batches(
    # * 前面的参数：可以用位置传参，也可以关键字传参。这里就是 texts
    # * 后面的所有参数（provider / model / dimensions）：强制只能用关键字传递，不能按位置传参！
    texts: list[str],
    *,
    provider: str,
    model: str,
    dimensions: int,
) -> list[list[float]]:
    """分批向量化。为什么要分批：一次编码几千条会 OOM，分批能控制内存峰值。

    【调用链】embeddings.embed_knowledge_texts() → 本函数（循环调用）
    【被谁调用】process_knowledge_document()（阶段 ③）

    【数据流向】
        入：texts（所有切片的"标题+正文"）、provider/model/dimensions（契约）
        出：向量列表，顺序与 texts 严格一致（靠 extend 的顺序保证）

    【为什么用 extend 而不是 append】
        每批返回的是"一批向量"（list[list[float]]），要摊平成一个长列表。
        append 会得到嵌套两层的列表，顺序也就乱了。

    【注意：这里没有 is_query】
        入库场景一律 is_query=False（用文档前缀）。
        查询才需要 is_query=True，见 embeddings.embed_knowledge_query()。
    """
    vectors: list[list[float]] = []
    # range 步长为批大小，start 是每批的起点
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        vectors.extend(
            embed_knowledge_texts(
                texts[start : start + EMBEDDING_BATCH_SIZE],   # 切片取这一批
                provider=provider,
                model=model,
                dimensions=dimensions,
            )
        )
    return vectors


def _set_status(document_id: uuid.UUID, status: str, *, error: str | None = None) -> None:
    """更新文档状态（独立开 Session，因为后台任务不能复用请求的 Session）。

    【调用链】无 —— 只操作 KnowledgeSourceDocument 一张表
    【被谁调用】process_knowledge_document()（进入 parsing 阶段前）

    【为什么要独立开 Session】
        本函数在后台任务里跑，请求的 Session 在响应返回时就已关闭。
        复用会拿到"已关闭的 Session"，抛 InvalidRequestError。

    【为什么这里要 commit 而主流程用 flush】
        状态是给前端轮询看的，必须真的落盘。
        如果只在事务里改，前端查到的还是旧状态，进度条永远不动。

    【为什么查不到文档就 return 而不是报错】
        可能文档在处理期间被用户删了。这不是错误，直接结束即可。
    """
    db = SessionLocal()
    try:
        document = db.get(KnowledgeSourceDocument, document_id)
        if document is None:
            return
        document.status = status
        document.error_message = error
        document.updated_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        # 无论成功失败都要关，否则连接池会被耗光
        db.close()


def process_knowledge_document(document_id: uuid.UUID) -> None:
    """【主流程】处理一篇上传文档：解析 → 切片 → 向量化 → 双写 → 完成。

    【调用链】追这个函数会进入（按执行顺序）：
        1. _set_status()                    → 本文件，标 parsing
        2. extract_text()                   → document_extraction.extract_document_text()
                                               └ 内部：_extract_pdf / _extract_docx / decode_text
        3. get_or_create_config()           → knowledge_base（读该仓库的切片参数）
        4. chunk_document()                 → chunking（五种策略 + 父子块）
        5. _embed_in_batches()              → embeddings.embed_knowledge_texts()
                                               └ 内部：validate_embedding_contract → _embed_local / _embed_openai_compatible
        6. delete_document_group()          → vector_store（删旧数据，保证幂等）
        7. add_documents()                  → vector_store（写 PG + Milvus）

    【被谁调用】
        api/routes/rag.py → BackgroundTasks.add_task()（上传接口返回 202 后异步执行）

    【数据流向】
        入：document_id（KnowledgeSourceDocument 主键）
        出：无返回值 —— 结果全部体现在数据库里：
            status / character_count / chunk_count / completed_at / error_message
            + documents 表新增 N 行 + Milvus 新增 N 条向量

    【为什么整个过程不在一个事务里】
        解析 + 切片 + 向量化可能要几十秒。如果包在一个事务里，
        数据库会长时间持锁，其他请求全被阻塞。
        所以：状态更新各自 commit（短事务），只有最后"删旧 + 写新"放在一个事务里。
    """
    # ── 阶段 0：标记开始（前端进度条从这里开始动）──
    _set_status(document_id, "parsing")

    db = SessionLocal()
    try:
        source = db.get(KnowledgeSourceDocument, document_id)
        if source is None:
            return

        # ── 阶段 1：读文件 + 解析 ──
        # 多读 1 字节用于判断"是否超过上限"（读完 12MB+1 才知道超没超）
        with Path(source.stored_path).open("rb") as handle:
            data = handle.read(MAX_DOCUMENT_BYTES + 1)
        if len(data) > MAX_DOCUMENT_BYTES:
            raise ValueError(f"knowledge file exceeds the {MAX_DOCUMENT_BYTES} byte limit")

        text = extract_text(source.name, data)
        if not text.strip():
            # 常见原因：扫描版 PDF（无文字层）、空文件、纯图片 DOCX
            raise ValueError("文档中没有可索引的文本")

        # ── 阶段 2：切片 ──
        # 先改状态 + commit，前端才能看到"正在切片"
        source.status = "splitting"
        source.character_count = len(text)
        db.commit()

        config = get_or_create_config(db, source.repo_id)      # 取该仓库的切片参数
        chunks = chunk_document(
            text,
            source.name,
            max_chars=config.chunk_size,        # 每块目标字符数（默认 800）
            overlap=config.chunk_overlap,       # 相邻块重叠（默认 100，防语义被切断）
        )
        if not chunks:
            raise ValueError("文档切分后没有有效片段")

        # ── 阶段 3：向量化 ──
        source.status = "embedding"
        source.chunk_count = len(chunks)
        db.commit()

        # 给每个切片生成人类可读的标题（用于引用展示）
        titles = []
        for index, chunk in enumerate(chunks):
            label = chunk.get("section_title")          # 文档自带章节标题 section_title（首选）
            if not label and chunk.get("page"):         # 没有章节标题 → 使用页码 Page N
                label = f"Page {chunk['page']}"
            label = str(label or f"Chunk {index + 1}")  # 都没有 → 兜底 Chunk N
            # 父块拆出的子块带序号，避免标题重复 ---->【有些长的父 chunk，内容太长，会二次拆分成多个更小的子 chunk】
            # 如果存在 child_index，就拼接后缀 (1)、(2)
            child_suffix = f" ({int(chunk.get('child_index') or 0) + 1})" if chunk.get("child_index") else ""
            titles.append(f"{source.name} · {label}{child_suffix}")     # 拼接完整标题，加入 titles 列表

        contents = [str(chunk["content"]) for chunk in chunks]  # 把所有 chunk 的文本内容，单独提取出来，放到 contents 数组

        # 【关键】向量化的是"标题 + 正文"，不是只有正文。
        #   标题里含文件名和章节名，能显著提升检索命中率。
        #   举例：正文里只写"该参数默认 30 秒"，但标题是"配置文件 · 超时设置"，
        #        用户问"超时多久"时，带标题的向量才能命中。
        vectors = _embed_in_batches(
            # 把两个列表，按顺序，一一配对，打包成一组一组的 (title, content)
            [f"{title}\n{content}" for title, content in zip(titles, contents)],
            provider=config.embedding_provider,
            model=config.embedding_model,
            dimensions=config.embedding_dimensions,
        )

        # ── 阶段 4：双写（PostgreSQL 存全文 + Milvus 存向量）──
        payloads = [
            DocumentPayload(
                source_type="knowledge_file",
                source_id=source.id,          # 同一个文档的所有切片共享 source_id
                title=title,
                content=content,              # 完整正文（Milvus 里会截断，这里不截）
                metadata={
                    # 把切片时算出的所有元数据带进来（section_path / page / parent_id 等）
                    **{key: value for key, value in chunk.items() if key != "content" and value is not None},
                    "display_title": source.name,
                    "filename": source.name,
                    "knowledge_document_id": str(source.id),
                    "checksum": source.checksum,
                    "file_size": source.file_size,
                    "content_type": source.content_type,
                    "chunk_index": index,          # 第几块（用于按顺序展示）
                    "chunk_chars": len(content),   # 这块多长
                    "total_chunks": len(chunks),   # 共几块
                    "chunk_strategy": "structure_aware_v1",
                },
            )
            for index, (title, content, chunk) in enumerate(zip(titles, contents, chunks))
        ]

        # 先删旧数据（重传同一个文件时保证幂等），再写入
        # 为什么要删：同一文件重复上传时，旧的切片/向量必须清掉，
        #            否则会有两份内容，检索时重复召回。
        delete_document_group(db, source.repo_id, source.id, source_type="knowledge_file", commit=False)
        add_documents(
            db,
            source.repo_id,
            payloads,
            commit=False,           # 最后统一 commit，保证"要么全成功要么全失败"
            vectors=vectors,
            embedding_info={
                "embedding_provider": config.embedding_provider,
                "embedding_model": config.embedding_model,
                "embedding_dimensions": config.embedding_dimensions,
            },
        )

        # ── 阶段 5：完成 ──
        source.status = "completed"
        source.error_message = None
        source.completed_at = datetime.now(timezone.utc)
        source.updated_at = source.completed_at
        db.commit()

    except Exception as exc:
        # 任何一步失败 → 回滚 + 标 failed + 记录原因（前端能看到"为什么失败"）
        # 为什么要 rollback 再改状态：不 rollback 的话，
        #   事务里已写入的 documents 会和"failed"状态一起被 commit，留下脏数据。
        db.rollback()
        source = db.get(KnowledgeSourceDocument, document_id)
        if source is not None:
            source.status = "failed"
            # 截断到 1000 字符：异常信息可能极长（比如整段堆栈），塞满字段没意义
            source.error_message = str(exc)[:1000]
            source.updated_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        db.close()
