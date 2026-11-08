"""检索评测（离线、不依赖 LLM）。

只测检索层，快速、便宜、可复现 —— 每次改分块/权重后先跑这个。
生成层指标（faithfulness 等）需要 LLM-as-judge，见 docs/03，此处留接口。

用法：
  python -m scripts.eval_ragas            # 默认混合检索
  python -m scripts.eval_ragas --mode vector   # 对照组：纯向量
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from core.db import close_pool, get_pool
from core.embeddings import get_embedder
from core.retrieval import hybrid_search, vector_search


def load_evalset(path: Path = Path("evalset.json")) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


async def run(mode: str, top_k: int) -> None:
    await get_pool()
    embedder = get_embedder()
    evalset = load_evalset()

    recall_hits = 0
    mrr_sum = 0.0
    total = len(evalset)
    per_item: list[dict] = []

    for item in evalset:
        q_emb = embedder.embed(item["question"])
        if mode == "vector":
            results = await vector_search(q_emb, top_k=top_k)
        else:
            results = await hybrid_search(q_emb, item["question"], top_k=top_k)

        ref_title = item["reference_doc"].removesuffix(".md")
        # 用 chunk 的 metadata.title 匹配（doc_id 是 hash，title 稳定可读）
        hit = any(r.get("metadata", {}).get("title") == ref_title for r in results)

        # MRR：第一个命中的排名（0-based → 1/rank）
        rank = next((i + 1 for i, r in enumerate(results)
                     if r.get("metadata", {}).get("title") == ref_title), None)
        if rank:
            mrr_sum += 1.0 / rank

        if hit:
            recall_hits += 1
        per_item.append({"question": item["question"], "hit": hit, "rank": rank})

    recall_at_k = recall_hits / total if total else 0.0
    mrr = mrr_sum / total if total else 0.0
    print(f"mode={mode} top_k={top_k} n={total}")
    print(f"recall@{top_k} = {recall_at_k:.3f}")
    print(f"MRR         = {mrr:.3f}")
    for it in per_item:
        if not it["hit"]:
            print(f"  MISS: {it['question']}")
    await close_pool()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hybrid", "vector"], default="hybrid")
    ap.add_argument("--top-k", type=int, default=5)
    a = ap.parse_args()
    asyncio.run(run(a.mode, a.top_k))
