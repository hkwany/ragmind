-- ============================================================
-- RagMind 01 · pgvector 表结构 + 混合检索索引
-- 设计要点见 docs/02-rag-design-decisions.md
-- ============================================================

CREATE EXTENSION IF NOT EXISTS vector;

-- 文档表：一个源文件一行，chunks 通过 doc_id 关联
CREATE TABLE IF NOT EXISTS documents (
    id          text PRIMARY KEY,               -- 源文件的稳定 id（路径 hash 或 URL）
    title       text NOT NULL,
    source      text,                           -- 原始来源（文件路径 / URL）
    metadata    jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);

-- 分块表：检索的基本单位
-- 注意：embedding 维度必须和嵌入模型一致。
-- 默认 384（all-MiniLM-L6-v2）；切 bge-small-zh-v1.5 是 512，改这里 + 重建索引。
CREATE TABLE IF NOT EXISTS chunks (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id      text NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index int  NOT NULL,                  -- 文档内顺序，用于重组上下文
    content     text NOT NULL,
    embedding   vector(384),
    metadata    jsonb NOT NULL DEFAULT '{}'::jsonb,  -- 章节、标签、产品线等过滤条件
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (doc_id, chunk_index)
);

-- 向量索引：HNSW，余弦距离
-- 说明：为什么 cosine 不是 L2 —— 文本嵌入的方向比长度更有语义意义，这是默认选择。
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- 全文检索列：用于混合检索的文本侧
-- 中文场景建议换 zhparser 或 pg_jieba；这里用 simple（英文按空格/标点切）。
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED;
CREATE INDEX IF NOT EXISTS idx_chunks_tsv
    ON chunks USING gin (content_tsv);

-- 元数据过滤：常见过滤维度（产品线、文档类型）建索引，避免过滤时全表扫
CREATE INDEX IF NOT EXISTS idx_chunks_metadata
    ON chunks USING gin (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id
    ON chunks (doc_id);

-- ------------------------------------------------------------
-- 混合检索参考 SQL（核心实现见 core/retrieval.py）
-- ------------------------------------------------------------
-- 纯向量：
--   SELECT id, doc_id, chunk_index, content,
--          1 - (embedding <=> $1) AS score
--   FROM chunks ORDER BY embedding <=> $1 LIMIT 20;
--
-- 混合（向量 + 全文），权重在应用层算或这里加权：
--   SELECT id, doc_id, chunk_index, content,
--          (0.6 * (1 - (embedding <=> $1))
--         + 0.4 * ts_rank_cd(content_tsv, plainto_tsquery('simple', $2))) AS score
--   FROM chunks
--   WHERE metadata @> $3::jsonb            -- 元数据过滤
--     AND (embedding <=> $1) < 0.5          -- 向量粗过滤
--   ORDER BY score DESC
--   LIMIT 20;
