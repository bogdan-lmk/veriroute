IMAGE ?= ghcr.io/bogdan-lmk/veriroute
TAG   ?= dev

.PHONY: test build push smoke run-practice

test:
	uv run --group dev pytest

build:
	docker buildx build --platform linux/amd64 -t $(IMAGE):$(TAG) --load .

push:
	docker buildx build --platform linux/amd64 -t $(IMAGE):$(TAG) --push .

# Smoke test under the grader's exact resource limits.
# On Apple Silicon this runs under Rosetta/QEMU: functionally valid,
# but timings are meaningless — measure real timings on x86 (task 0.5).
smoke: build
	rm -rf /tmp/veriroute-smoke && mkdir -p /tmp/veriroute-smoke/input /tmp/veriroute-smoke/output
	cp eval/practice_tasks.json /tmp/veriroute-smoke/input/tasks.json
	docker run --rm --platform linux/amd64 --memory=4g --cpus=2 \
		-v /tmp/veriroute-smoke/input:/input:ro \
		-v /tmp/veriroute-smoke/output:/output \
		$(IMAGE):$(TAG)
	@echo "--- results ---" && cat /tmp/veriroute-smoke/output/results.json && echo

# Run the agent locally against the practice tasks with your own
# Fireworks credentials (development only — the grader injects its own).
run-practice:
	mkdir -p /tmp/veriroute-local
	AGENT_INPUT_PATH=eval/practice_tasks.json \
	AGENT_OUTPUT_PATH=/tmp/veriroute-local/results.json \
	uv run python -m agent.main
	@echo "--- results ---" && cat /tmp/veriroute-local/results.json && echo
