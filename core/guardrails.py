"""守卫：prompt 注入检测 + 答案溯源校验。

设计要点：
1. 输入侧检测是启发式，能被绕过——所以它只是第一道低成本过滤，不是最终防线。
2. 真正的兜底是「溯源约束」：prompt 里强制『只能根据资料回答』，生成后再校验
   答案与检索上下文的重叠度，重叠过低就降级为「我不知道」。
   这是从架构上让幻觉「说出口」之前就被拦下，而不是事后靠规则堵。
"""

from __future__ import annotations

import re
from typing import Any

# 典型注入特征：要求模型忽略系统指令 / 扮演 / 泄露 system prompt
INJECTION_PATTERNS = [
    re.compile(r"忽略.{0,6}(指令|限制|规则|上文|之前)", re.I),
    re.compile(r"(override|ignore|disregard)\s+(the\s+)?(instructions|prompt|system)", re.I),
    re.compile(r"(你现在是|你是我的).{0,20}(开发者|助手之外)", re.I),
    re.compile(r"(system\s*prompt|系统提示词|隐藏指令)", re.I),
    re.compile(r"(泄露|输出|打印).{0,10}(提示词|指令|prompt)", re.I),
]

# 指示模型「无条件服从」的危险信号
UNSAFE_KEYWORDS = ["越狱", "jailbreak", "不需要遵守任何", "无视所有"]


def detect_injection(text: str) -> dict[str, Any]:
    """返回 {blocked: bool, reasons: [...]}。

    注意措辞：这是启发式，只拦明显攻击，不声称能拦住精心构造的越狱。
    """
    reasons: list[str] = []
    for pat in INJECTION_PATTERNS:
        if pat.search(text):
            reasons.append(pat.pattern[:30])
    for kw in UNSAFE_KEYWORDS:
        if kw.lower() in text.lower():
            reasons.append(kw)
    return {"blocked": bool(reasons), "reasons": reasons}


def build_grounded_prompt(question: str, context_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """组装带溯源约束的 prompt。

    关键：把检索到的上下文作为 system 的 _context 字段传入，
    并用明确的指令约束模型只能基于资料回答。
    """
    ctx = "\n\n---\n\n".join(
        f"[{i + 1}] (来源: {c['doc_id']} chunk {c['chunk_index']})\n{c['content']}"
        for i, c in enumerate(context_chunks)
    )
    system = (
        "你是企业内部知识库助手。只能根据下面【资料】回答，不得使用外部知识。\n"
        "如果资料中没有答案，直接回答『根据现有资料，我无法回答这个问题』，不要编造。\n"
        "回答时在句尾用 [1][2] 标注引用来源。\n\n"
        f"【资料】\n{ctx}"
    )
    messages = [
        {"role": "system", "content": system, "_context": ctx},
        {"role": "user", "content": question},
    ]
    return messages


def grounding_score(answer: str, context_chunks: list[dict[str, Any]]) -> float:
    """粗略的溯源校验：答案与上下文的字符 n-gram 重叠率。

    这是 cheap 的代理指标，不是真正的忠实度（那个要用 LLM-as-judge）。
    用途：重叠过低时触发降级，而不是直接拿一个明显幻觉的答案返回。
    """
    if not context_chunks:
        return 0.0
    ctx = " ".join(c["content"] for c in context_chunks)

    def _ngrams(s: str, n: int = 3) -> set[str]:
        s = "".join(ch for ch in s if ch.isalnum())
        return {s[i : i + n] for i in range(len(s) - n + 1)}

    a = _ngrams(answer)
    b = _ngrams(ctx)
    if not a:
        return 0.0
    return len(a & b) / len(a)
