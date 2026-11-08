"""生成样例知识库文档 + 评测集。

造一批「内部技术文档 / FAQ」性质的 md 文件，覆盖真实 RAG 的几个难点：
- 精确标识符（错误码、API 名）—— 测混合检索的全文侧
- 语义相近但答案不同的文档 —— 测检索区分度（干扰文档）
- 口语化问题 —— 测查询改写价值

用法：
  python -m scripts.prep_sample_docs
"""

from __future__ import annotations

import json
from pathlib import Path

DOCS = {
    "redis-ops.md": """# Redis 生产运维手册

## 内存告警处理
当 Redis 内存使用率超过 80% 时，第一步查看 maxmemory 配置和当前内存策略。
如果 maxmemory-policy 是 noeviction，写操作会直接报错 OOM。
建议先确认是否有大 key，再考虑扩容或调整淘汰策略 allkeys-lru。

## 连接数过高
客户端连接数突增时，检查是否短连接风暴。设置合理的 timeout 和连接池上限。
生产环境建议使用连接池，避免每个请求都新建连接。

## 常见错误码
E1001：连接被拒绝，通常是 maxclients 达到上限。
E1002：OOM，写命令被 noeviction 策略拒绝。
""",

    "mysql-ops.md": """# MySQL 运维手册

## 连接数过高
当 Too many connections 报错时，检查 max_connections 和当前活跃连接。
定位是哪个应用占满了连接池，必要时限流或增加连接数。

## 慢查询处理
开启 slow_query_log，用 EXPLAIN 分析执行计划，检查是否缺索引或索引失效。

## 常见错误码
E2001：Too many connections。
E2002：锁等待超时（lock wait timeout）。
""",

    "auth-faq.md": """# 登录认证 FAQ

## 登录不上怎么办
登录不上通常有两个原因：密码错误，或账号被锁定。
先检查是否连续输错密码导致账号锁定，锁定后需联系管理员解锁或等待 30 分钟自动解锁。

## 如何重置密码
在登录页点击「忘记密码」，通过邮箱验证码重置。

## 单点登录
SSO 登录失败时，检查浏览器时间是否与服务器时间偏差过大（token 时间戳校验失败）。
""",

    "deploy-checklist.md": """# 生产部署检查清单

## 上线前检查
1. 确认数据库迁移脚本已执行
2. 确认配置中心参数已更新
3. 确认监控告警已配置
4. 准备回滚方案

## 灰度发布
新版本先 5% 流量灰度，观察 30 分钟核心指标，再逐步放量。

## 回滚
任何变更都要能在 5 分钟内回退。保留上一个版本的镜像和配置。
""",
}

# 评测集：问题 + 标准答案来源 + 干扰文档
EVALSET = [
    {
        "question": "Redis 内存使用率 90% 了，第一步该做什么？",
        "reference_doc": "redis-ops.md",
        "answer": "查看 maxmemory 配置和当前内存策略",
        "distractors": ["mysql-ops.md"],
    },
    {
        "question": "报错 E1002 是什么意思？",
        "reference_doc": "redis-ops.md",
        "answer": "Redis OOM，写命令被 noeviction 策略拒绝",
        "distractors": ["mysql-ops.md"],
    },
    {
        "question": "登录不上怎么办",
        "reference_doc": "auth-faq.md",
        "answer": "先检查密码错误还是账号被锁定",
        "distractors": ["redis-ops.md", "mysql-ops.md"],
    },
    {
        "question": "Too many connections 怎么处理",
        "reference_doc": "mysql-ops.md",
        "answer": "检查 max_connections 和活跃连接，定位占用连接池的应用",
        "distractors": ["redis-ops.md"],
    },
    {
        "question": "新版本上线怎么灰度",
        "reference_doc": "deploy-checklist.md",
        "answer": "5% 流量灰度，观察 30 分钟再放量",
        "distractors": ["auth-faq.md"],
    },
]


def main() -> None:
    base = Path("docs_sample")
    base.mkdir(exist_ok=True)
    for name, content in DOCS.items():
        (base / name).write_text(content, encoding="utf-8")

    eval_path = Path("evalset.json")
    eval_path.write_text(json.dumps(EVALSET, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(DOCS)} docs to docs_sample/ and {len(EVALSET)} QA to evalset.json")


if __name__ == "__main__":
    main()
