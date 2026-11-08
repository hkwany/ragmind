"""RagMind · 文档摄入（ingest）

离线流水线：解析 → 分块 → 嵌入 → 入库。
和在线问答（api）分开，因为两者对延迟/吞吐的要求完全不同：
ingest 可以慢、可以批处理、可以失败重试；api 必须快。

用法：
  python -m services.ingest.worker --dir docs_sample --strategy fixed
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from core.chunking import chunk_text
from core.db import close_pool, get_pool, init_schema, upsert_chunks, upsert_documents
from core.embeddings import get_embedder

SUPPORTED = {".md", ".txt"}


def read_docs(directory: Path) -> list[dict]:
    """读目录下所有 md/txt，返回 [{id, title, source, content}]。"""
    docs = []
    for p in sorted(directory.rglob("*")):
        if p.suffix.lower() not in SUPPORTED or not p.is_file():
            continue
        content = p.read_text(encoding="utf-8", errors="replace")
        if not content.strip():
            continue
        doc_id = hashlib.sha1(str(p).encode()).hexdigest()[:12]
        docs.append({"id": doc_id, "title": p.stem, "source": str(p), "content": content})
    return docs


async def run(directory: Path, strategy: str, batch: int) -> None:
    pool = await get_pool()
    await init_schema(pool)
    embedder = get_embedder()

    docs = read_docs(directory)
    if not docs:
        raise SystemExit(f"no documents found under {directory}")

    await upsert_documents([{k: d[k] for k in ("id", "title", "source")} for d in docs])
    print(f"documents: {len(docs)}")

    total_chunks = 0
    for d in docs:
        chunks = chunk_text(d["content"], strategy)
        # 嵌入分批算（省内存、可断点重试）
        embeddings = []
        for i in range(0, len(chunks), batch):
            embeddings.extend(embedder.embed_batch(chunks[i : i + batch]))

        records = [
            {"doc_id": d["id"], "chunk_index": i, "content": c,
             "metadata": {"title": d["title"]}}
            for i, c in enumerate(chunks)
        ]
        n = await upsert_chunks(records, embeddings)
        total_chunks += n
        print(f"  {d['title']}: {len(chunks)} chunks, {n} upserted")

    print(f"done. total chunks: {total_chunks}")
    await close_pool()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("docs_sample"))
    ap.add_argument("--strategy", choices=["fixed", "semantic"], default="fixed")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    asyncio.run(run(a.dir, a.strategy, a.batch))
