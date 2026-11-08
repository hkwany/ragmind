"""数据库访问：asyncpg 连接池 + pgvector 的写入与检索。

注意：向量检索的 SQL 直接在这里写死，而不是用 ORM。
理由：pgvector 的查询性能和索引选择高度依赖原始 SQL，
ORM 会把这些细节藏起来，反而更难调优和讲清楚。
"""

from __future__ import annotations

import json

import asyncpg

from core.config import settings

_pool: asyncpg.Pool | None = None


def to_vector_literal(vec: list[float] | str) -> str:
    """把 Python 向量转成 pgvector 接受的文本格式 '[x,y,...]'。

    asyncpg 不识别 vector 扩展类型，传 list 会被当成 text 编码而报
    "expected str, got list"；`$n::vector` 的入参必须显式转成字符串。
    """
    if isinstance(vec, str):
        return vec
    return "[" + ",".join(f"{x:.9g}" for x in vec) + "]"


async def _init_connection(conn: asyncpg.Connection) -> None:
    """连接级 codec：jsonb 自动在 Python dict 与 JSON 之间转换。

    不注册的话 asyncpg 会把 jsonb 读成 str，调用方拿 `metadata["title"]` 会炸；
    写入 dict 也会报 "expected str, got dict"。
    """
    await conn.set_type_codec(
        "jsonb",
        encoder=lambda v: json.dumps(v, ensure_ascii=False),
        decoder=json.loads,
        schema="pg_catalog",
    )


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            settings.pg_dsn, min_size=1, max_size=int(__import__("os").getenv("PG_POOL_MAX", "8")),
            command_timeout=30, init=_init_connection,
        )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def init_schema(pool: asyncpg.Pool | None = None) -> None:
    pool = pool or await get_pool()
    import pathlib

    sql = pathlib.Path(__file__).resolve().parents[1] / "sql" / "001_schema.sql"
    async with pool.acquire() as conn:
        await conn.execute(sql.read_text(encoding="utf-8"))


async def upsert_chunks(records: list[dict], embedding_vectors: list[list[float]]) -> int:
    """批量写入 chunk + 向量。records 与 vectors 一一对应。

    返回写入条数。嵌入在 ingest 侧先算好（可并发、可重试），
    DB 只负责存 —— 职责分离让「换嵌入模型」不需要动 DB 逻辑。
    """
    pool = await get_pool()
    rows = [
        (r["doc_id"], r["chunk_index"], r["content"], to_vector_literal(vec),
         r.get("metadata", {}))
        for r, vec in zip(records, embedding_vectors)
    ]
    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO chunks (doc_id, chunk_index, content, embedding, metadata)
            VALUES ($1, $2, $3, $4::vector, $5::jsonb)
            ON CONFLICT (doc_id, chunk_index)
            DO UPDATE SET content = EXCLUDED.content,
                          embedding = EXCLUDED.embedding,
                          metadata = EXCLUDED.metadata
            """,
            rows,
        )
    return len(rows)


async def upsert_documents(docs: list[dict]) -> int:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO documents (id, title, source, metadata)
            VALUES ($1, $2, $3, $4::jsonb)
            ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title, updated_at = now()
            """,
            [(d["id"], d["title"], d.get("source"),
              d.get("metadata", {})) for d in docs],
        )
    return len(docs)


async def count_chunks() -> int:
    pool = await get_pool()
    async with pool.acquire() as conn:
        return int(await conn.fetchval("SELECT count(*) FROM chunks"))
