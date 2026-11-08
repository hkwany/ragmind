SHELL := /bin/bash
COMPOSE := docker compose -f deploy/docker-compose.yml
PY := python3

.PHONY: help up down logs ps seed smoke eval eval-vector check clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

up: ## 启动核心栈（pgvector/redis/api）
	$(COMPOSE) up -d --build
	@for i in $$(seq 1 60); do curl -sf localhost:8000/readyz >/dev/null && break || sleep 2; done
	@curl -s localhost:8000/readyz; echo

down: ## 停止保留数据卷
	$(COMPOSE) down

clean: ## 停止并删除数据卷
	$(COMPOSE) down -v

logs: ## 跟踪日志
	$(COMPOSE) logs -f --tail=100

ps: ## 容器状态
	$(COMPOSE) ps

seed: ## 造样例文档 + 嵌入入库
	$(PY) -m scripts.prep_sample_docs
	@test -n "$$(find docs_sample -type f \( -name '*.md' -o -name '*.txt' \) 2>/dev/null)" \
		|| { echo "docs_sample 为空，先检查 scripts/prep_sample_docs.py"; exit 1; }
	# api 把宿主机 docs_sample 只读挂载到 /app/docs_sample；--build 确保容器代码与本地一致
	$(COMPOSE) up -d --build api
	$(COMPOSE) exec -T api \
		python -m services.ingest.worker --dir /app/docs_sample --strategy fixed

smoke: ## 冒烟：问一句 + 验证注入拦截
	@$(PY) scripts/smoke.py

eval: ## 检索评测（混合检索）
	@test -f evalset.json || $(PY) -m scripts.prep_sample_docs
	$(COMPOSE) up -d --build api
	$(COMPOSE) cp evalset.json api:/app/evalset.json
	@$(COMPOSE) exec -T api python -m scripts.eval_ragas --mode hybrid

eval-vector: ## 检索评测（纯向量对照组）
	@test -f evalset.json || $(PY) -m scripts.prep_sample_docs
	$(COMPOSE) up -d --build api
	$(COMPOSE) cp evalset.json api:/app/evalset.json
	@$(COMPOSE) exec -T api python -m scripts.eval_ragas --mode vector

check: ## 静态检查（不依赖 Docker）
	@$(PY) -m compileall -q core services scripts && echo "compile OK"
	@$(PY) -c 'import ast, pathlib; files = list(pathlib.Path("core").rglob("*.py")) + list(pathlib.Path("services").rglob("*.py")) + list(pathlib.Path("scripts").rglob("*.py")); [ast.parse(f.read_text(encoding="utf-8"), filename=str(f)) for f in files]; print("ast OK")'
