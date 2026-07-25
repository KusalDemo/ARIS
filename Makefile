# Makefile shortcuts for ARIS (run from this directory).
# make lint  — run Ruff checker + format check on Python trees.
# make test  — run pytest (coverage gate turns on once the test suite exists).
COMPOSE := docker compose -f infra/docker/docker-compose.yml
PYTHON3 ?= python3

.PHONY: lint fmt typecheck test test-cov up down compose-up-full compose-down compose-up-dsb compose-up-kafka compose-down-dsb demo helm-template gen-proto benchmark-config-check benchmark-run benchmark-config-check-dsb dsb-submodule-init

gen-proto:
	chmod +x scripts/gen_proto.sh
	./scripts/gen_proto.sh

lint:
	ruff check aris-rl services/policy-service services/telemetry-bridge benchmarks scripts
	ruff format --check aris-rl services/policy-service services/telemetry-bridge benchmarks scripts

fmt:
	ruff format aris-rl services/policy-service services/telemetry-bridge benchmarks scripts

typecheck:
	mypy aris-rl/src/aris_rl

test:
	pytest aris-rl/tests tests/integration -q

test-cov:
	pytest aris-rl/tests tests/integration --cov=aris_rl --cov-report=term-missing --cov-fail-under=85

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

compose-up-full:
	$(COMPOSE) up -d --build

compose-down:
	$(COMPOSE) down

policy-static:
	ARIS_CONFIG=/config/benchmark-static.yaml $(COMPOSE) up -d --force-recreate policy-service
	@echo "ARIS_CONFIG=/config/benchmark-static.yaml (inference off)"

policy-adaptive:
	ARIS_CONFIG=/config/benchmark-adaptive.yaml $(COMPOSE) up -d --force-recreate policy-service
	@echo "ARIS_CONFIG=/config/benchmark-adaptive.yaml (needs data/models/policy.ts)"

dsb-submodule-init:
	git submodule update --init --recursive

compose-up-dsb:
	@echo "stub: DeathStarBench overlay (optional, Week 5+)"
	@exit 1

compose-down-dsb:
	@echo "stub: compose-down-dsb (optional)"
	@exit 1

compose-up-kafka:
	@echo "stub: Kafka overlay (optional)"
	@exit 1

demo: compose-up-full
	@echo ""
	@echo "ARIS demo stack is up. Try:"
	@echo "  Echo health:     curl -s http://127.0.0.1:18001/health/live"
	@echo "  Through Envoy:   curl -s 'http://127.0.0.1:10000/echo?message=hi' -H 'x-aris-route: /echo' -H 'x-aris-error-rate: 0.01'"
	@echo "  Policy health:   curl -s http://127.0.0.1:18080/health/live"
	@echo "  Prometheus:      http://127.0.0.1:9090"
	@echo "  Grafana:         http://127.0.0.1:3000  (admin / admin)"
	@echo ""

helm-template:
	@if command -v helm >/dev/null 2>&1; then \
		helm template aris-test infra/helm/aris >/dev/null && echo "helm template OK"; \
	else \
		echo "helm not installed; skipping template render"; \
	fi

benchmark-config-check:
	@echo "stub: benchmarks.harness after Week 4"
	@exit 1

benchmark-config-check-dsb:
	@echo "stub: dsb benchmark config after Week 4"
	@exit 1

benchmark-run:
	@echo "stub: benchmarks.run_experiment after Week 4"
	@exit 1
