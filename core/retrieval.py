"""检索：混合检索（向量 + 全文）+ 可选重排。

设计要点（面试会被问）：
1. 纯向量对精确标识符（错误码、型号、API 名）不敏感，纯全文对语义改写不敏感，
   所以要混合。权重由 recall@k 曲线扫出来（docs/02）。
2. 重排只对 top-k_candidate（默认 20）做，用更贵的模型换更准的排序，成本可控。
3. 元数据过滤下推到 SQL（metadata @> jsonb），不要先全量召回再在 Python 里过滤。
"""

from __future__ import annotations

from typing import Any

from core.config import settings
from core.db import get_pool, to_vector_literal

Vector = list[float]


async def hybrid_search(
    query_embedding: Vector,
    query_text: str,
    top_k: int | None = None,
    metadata_filter: dict | None = None,
) -> list[dict[str, Any]]:
    """混合检索：α·向量相似度 + (1-α)·全文相关度。

    用 SQL 加权融合而不是 Python 侧拼接，因为：候选集在 DB 内排序，
    避免把整个候选集拉回应用再算。对百万级 chunk，这一点决定性能。
    """
    top_k = top_k or settings.top_k
    pool = await get_pool()
    meta = metadata_filter or {}
    vec = to_vector_literal(query_embedding)

    sql = """
        SELECT id, doc_id, chunk_index, content, metadata,
               ({alpha}::float * (1 - (embedding <=> $1::vector))
              + {beta}::float * ts_rank_cd(content_tsv, plainto_tsquery('simple', $2))) AS score
        FROM chunks
        WHERE ($3::jsonb = '{{}}'::jsonb OR metadata @> $3::jsonb)
        ORDER BY score DESC
        LIMIT $4
    """.format(alpha=settings.hybrid_vector_weight, beta=1 - settings.hybrid_vector_weight)

    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, vec, query_text, meta, top_k)
    return [dict(r) for r in rows]


async def vector_search(
    query_embedding: Vector,
    top_k: int | None = None,
    metadata_filter: dict | None = None,
) -> list[dict[str, Any]]:
    """纯向量检索（作为对照组，用于评测里证明混合检索确实更好）。"""
    top_k = top_k or settings.top_k
    pool = await get_pool()
    meta = metadata_filter or {}
    vec = to_vector_literal(query_embedding)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, doc_id, chunk_index, content, metadata,
                   (1 - (embedding <=> $1::vector)) AS score
            FROM chunks
            WHERE ($2::jsonb = '{}'::jsonb OR metadata @> $2::jsonb)
            ORDER BY embedding <=> $1::vector
            LIMIT $3
            """,
            vec, meta, top_k,
        )
    return [dict(r) for r in rows]


async def rerank(query: str, candidates: list[dict[str, Any]], top_k: int | None = None) -> list[dict[str, Any]]:
    """重排：用 LLM 对候选做逐条相关性打分。

    默认用 mock 打分（离线可跑）；生产换 cross-encoder 或更强的 LLM。
    关键设计：只重排 candidates（<=20），不是整个索引。
    """
    top_k = top_k or settings.top_k
    if not candidates:
        return []
    # 轻量 LLM-as-reranker：让模型给每个候选打 1-5 分。
    # 这里用核心 llm 层，保持 provider 可插拔。
    from core.llm import get_llm

    llm = get_llm()
    scored: list[tuple[float, dict]] = []
    for c in candidates:
        prompt = [
            {"role": "system", "content": "你是相关性打分器，只输出 1-5 的整数。"},
            {"role": "user", "content": f"问题：{query}\n资料：{c['content'][:300]}\n相关性(1-5)："},
        ]
        res = await llm.generate(prompt, max_tokens=4)
        try:
            score = float(res.text.strip()[:1])
        except ValueError:
            score = 0.0
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]
