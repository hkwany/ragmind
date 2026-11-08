"""集中配置：所有可调参数从环境变量读，便于复现和对比实验。

面试要点：参数不进代码，才能做「同一评测集、不同参数」的对照实验。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass
class Settings:
    # 数据库
    pg_dsn: str = field(default_factory=lambda: os.getenv(
        "PG_DSN", "postgresql://ragmind:ragmind@localhost:5432/ragmind"))

    # 嵌入：mock（离线可跑）/ openai / local
    embedding_provider: str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "mock"))
    embedding_dim: int = field(default_factory=lambda: _int("EMBEDDING_DIM", 384))
    embedding_model: str = field(default_factory=lambda: os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))

    # LLM：mock / openai / vllm（OpenAI 兼容）
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "mock"))
    llm_base_url: str = field(default_factory=lambda: os.getenv(
        "LLM_BASE_URL", "http://localhost:8000/v1"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "mock-model"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "sk-mock"))

    # 检索
    top_k: int = field(default_factory=lambda: _int("TOP_K", 5))
    top_k_candidate: int = field(default_factory=lambda: _int("TOP_K_CANDIDATE", 20))
    hybrid_vector_weight: float = field(default_factory=lambda: _float("HYBRID_VECTOR_WEIGHT", 0.6))
    rerank_enabled: bool = field(default_factory=lambda: os.getenv("RERANK_ENABLED", "false") == "true")

    # 缓存
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    cache_ttl_seconds: int = field(default_factory=lambda: _int("CACHE_TTL_SECONDS", 300))
    cache_similarity: float = field(default_factory=lambda: _float("CACHE_SIMILARITY", 0.95))

    # 生成
    max_new_tokens: int = field(default_factory=lambda: _int("MAX_NEW_TOKENS", 512))
    generation_temperature: float = field(default_factory=lambda: _float("TEMPERATURE", 0.1))

    # 分块
    chunk_size: int = field(default_factory=lambda: _int("CHUNK_SIZE", 512))
    chunk_overlap: int = field(default_factory=lambda: _int("CHUNK_OVERLAP", 64))


settings = Settings()
