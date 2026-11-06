"""
检索路由
========

【路由清单】
    POST /search    语义检索（向量 + 关键词混合召回）

【数据流向】
    前端搜索框
      → 本文件 semantic_search()
      → services/rag/retrieval.py::search_similar_documents()
      → Milvus（向量）+ PostgreSQL（关键词 / 正文）
      → 返回 SearchResponse

【要点】
    这里是"只检索不生成"的裸接口，给前端搜索页用。
    需要"带引用回答"的走 /rag/answer，那条链路还会过 rerank + answer_gate。
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.models import Repository
from app.db.session import get_db
from app.schemas.search import SearchRequest, SearchResponse
from app.services.rag.retrieval import search_similar_documents

router = APIRouter()


@router.post("", response_model=SearchResponse)
async def semantic_search(payload: SearchRequest, db: Session = Depends(get_db)) -> SearchResponse:
    """混合检索：返回匹配的文档片段（不生成答案）。"""
    # 先校验仓库存在，避免拿一个不存在的 repo_id 去查向量库（会返回空，掩盖真正的问题）
    repo = db.get(Repository, payload.repo_id)
    if repo is None:
        raise HTTPException(status_code=404, detail="未找到仓库")

    # 六个可选过滤条件。值为 None 的会在下面被剔除，
    # 所以这里可以放心全列出来，不用写一堆 if。
    filters = {
        "label": payload.label,
        "path": payload.path,
        "module": payload.module,
        "state": payload.state,
        "start_date": payload.start_date,
        "end_date": payload.end_date,
    }
    results = search_similar_documents(
        db,
        payload.repo_id,
        payload.query,
        source_type=payload.source_type,
        limit=payload.limit,
        # 只传有值的过滤条件：传 None 会被当成"要求该字段等于 null"，导致查不到任何结果
        metadata_filters={key: value for key, value in filters.items() if value},
    )
    return SearchResponse(results=results)
