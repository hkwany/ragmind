"""RagMind · 问答 API（FastAPI）

生产化关注点都在这一个文件里体现，面试会逐条问：
1. 流式 SSE：首 token 优先，用户感知比完整延迟重要
2. 语义缓存：相似问题命中，跳过检索+生成（成本与延迟的最大杠杆）
3. 限流：按租户令牌桶
4. 守卫：注入检测 + 溯源约束 + 生成后 grounding 校验
5. token 核算：每次问答打点 prompt/completion token，成本账可算
6. 可观测：Prometheus 指标 + 结构化日志
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import redis.asyncio as aioredis
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from core.config import settings
from core.db import close_pool, get_pool
from core.embeddings import get_embedder
from core.guardrails import build_grounded_prompt, detect_injection, grounding_score
from core.llm import get_llm
from core.retrieval import hybrid_search, rerank

REQ_COUNT = Counter("rag_requests_total", "问答请求数", ["status"])
LATENCY = Histogram("rag_seconds", "端到端耗时", ["stage", "cache_hit"],
                    buckets=(0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10))
TOKENS = Counter("rag_tokens_total", "token 用量", ["kind"])
CACHE_HIT = Counter("rag_cache_hit_total", "缓存命中")
GUARD = Counter("rag_guard_rejected_total", "守卫拦截", ["reason"])
GROUNDED = Counter("rag_grounded_total", "溯源校验", ["result"])


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    metadata_filter: dict[str, Any] = Field(default_factory=dict)  # 如 {"product": "pay"}
    stream: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = await get_pool()
    app.state.redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    try:
        await app.state.redis.ping()
        app.state.redis_ok = True
    except Exception:
        app.state.redis_ok = False
    yield
    await app.state.redis.aclose()
    await close_pool()


app = FastAPI(title="RagMind", version="0.1.0", lifespan=lifespan)


async def _rate_allow(redis: aioredis.Redis, tenant: str) -> bool:
    key = f"rl:{tenant}:{int(time.time())}"
    n = await redis.incr(key)
    await redis.expire(key, 2)
    return n <= int(os.getenv("RATE_PER_SEC", "10"))


async def _cache_get(redis: aioredis.Redis, query_emb: list[float]) -> dict | None:
    # 简化语义缓存：用向量相似度。生产应维护「嵌入 → 答案」的向量索引，
    # 这里退化为「按归一化问题字符串」的近似，说明语义缓存思想即可。
    # 用 query embedding 的前 8 维做粗粒度 key（近似，不精确）。
    key = "cache:" + "".join(f"{int(v * 1000) & 0xff:02x}" for v in query_emb[:8])
    raw = await redis.get(key)
    return json.loads(raw) if raw else None


async def _cache_set(redis: aioredis.Redis, query_emb: list[float], answer: dict) -> None:
    key = "cache:" + "".join(f"{int(v * 1000) & 0xff:02x}" for v in query_emb[:8])
    await redis.set(key, json.dumps(answer, ensure_ascii=False), ex=settings.cache_ttl_seconds)


def _answer_payload(question: str, answer: str, sources: list[dict], cache_hit: bool, tokens: dict) -> dict:
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "cache_hit": cache_hit,
        "tokens": tokens,
    }


@app.post("/v1/chat")
async def chat(
    body: ChatRequest,
    request: Request,
    x_tenant_id: str = Header("default", alias="X-Tenant-Id"),
) -> Response:
    start = time.perf_counter()
    redis: aioredis.Redis = request.app.state.redis

    # 0) 限流
    if request.app.state.redis_ok and not await _rate_allow(redis, x_tenant_id):
        REQ_COUNT.labels(status="429").inc()
        raise HTTPException(429, detail={"error": "rate_limited"})

    # 1) 守卫：注入检测（输入侧，启发式）
    inj = detect_injection(body.question)
    if inj["blocked"]:
        GUARD.labels(reason="injection").inc()
        REQ_COUNT.labels(status="400").inc()
        raise HTTPException(400, detail={"error": "blocked", "reasons": inj["reasons"]})

    # 2) 嵌入 + 语义缓存
    embedder = get_embedder()
    q_emb = embedder.embed(body.question)
    cache_hit = False
    if request.app.state.redis_ok:
        cached = await _cache_get(redis, q_emb)
        if cached:
            cache_hit = True
            CACHE_HIT.inc()
            LATENCY.labels(stage="total", cache_hit="1").observe(time.perf_counter() - start)
            REQ_COUNT.labels(status="200").inc()
            return JSONResponse(cached)

    # 3) 检索（+ 可选重排）
    t_retr = time.perf_counter()
    candidates = await hybrid_search(q_emb, body.question, top_k=settings.top_k_candidate,
                                     metadata_filter=body.metadata_filter)
    if settings.rerank_enabled and candidates:
        candidates = await rerank(body.question, candidates, top_k=settings.top_k)
    context_chunks = candidates[: settings.top_k]
    LATENCY.labels(stage="retrieve", cache_hit="0").observe(time.perf_counter() - t_retr)

    # 4) 生成
    llm = get_llm()
    messages = build_grounded_prompt(body.question, context_chunks)

    if body.stream:
        async def gen():
            first = True
            async for piece in llm.stream(messages):
                if first:
                    LATENCY.labels(stage="first_token", cache_hit="0").observe(time.perf_counter() - start)
                    first = False
                yield f"data: {json.dumps({'delta': piece})}\n\n"
            yield "data: [DONE]\n\n"

        REQ_COUNT.labels(status="200").inc()
        return StreamingResponse(gen(), media_type="text/event-stream")

    t_gen = time.perf_counter()
    result = await llm.generate(messages)
    LATENCY.labels(stage="generate", cache_hit="0").observe(time.perf_counter() - t_gen)
    TOKENS.labels(kind="prompt").inc(result.prompt_tokens)
    TOKENS.labels(kind="completion").inc(result.completion_tokens)

    # 5) 溯源校验：答案与上下文重叠过低 → 降级
    gs = grounding_score(result.text, context_chunks)
    if gs < 0.1:
        GROUNDED.labels(result="degraded").inc()
        answer = "根据现有资料，我无法回答这个问题。"
    else:
        GROUNDED.labels(result="ok").inc()
        answer = result.text

    sources = [{"doc_id": c["doc_id"], "chunk_index": c["chunk_index"],
                "score": round(c["score"], 4) if c.get("score") is not None else None}
               for c in context_chunks]
    payload = _answer_payload(body.question, answer, sources, cache_hit=False, tokens={
        "prompt": result.prompt_tokens, "completion": result.completion_tokens})

    if request.app.state.redis_ok:
        await _cache_set(redis, q_emb, payload)

    LATENCY.labels(stage="total", cache_hit="0").observe(time.perf_counter() - start)
    REQ_COUNT.labels(status="200").inc()
    return JSONResponse(payload)


@app.get("/healthz", response_class=PlainTextResponse)
async def healthz() -> str:
    return "ok"


@app.get("/readyz")
async def readyz(request: Request) -> JSONResponse:
    try:
        async with request.app.state.pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return JSONResponse({"ready": True})
    except Exception:
        return JSONResponse({"ready": False, "reason": "postgres"}, status_code=503)


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
