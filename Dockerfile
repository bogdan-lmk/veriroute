# veriroute — AMD Hackathon Track 1 submission container.
# linux/amd64 only; graded on 4 GB RAM / 2 vCPU / no GPU / 10 min.
#
# llama.cpp is NOT compiled here (QEMU cross-builds are slow and flaky):
# we ship the official release binaries, which are built with runtime CPU
# dispatch (GGML_CPU_ALL_VARIANTS) and run on any x86-64 grader CPU.
# Phase 0 ships the binaries without a model; the GGUF lands in Phase 2.

FROM python:3.12-slim AS dl
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ARG LLAMA_TAG=b9930
RUN curl -fsSL -o /tmp/llama.tar.gz \
      "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/llama-${LLAMA_TAG}-bin-ubuntu-x64.tar.gz" \
    && mkdir /llama \
    && tar -xzf /tmp/llama.tar.gz -C /llama --strip-components=1 \
    && rm /tmp/llama.tar.gz

FROM python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/bogdan-lmk/veriroute"
# ffmpeg: frame extraction for the Track 2 captioning mode.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# The release tarball is flat: llama-server and ALL libggml-cpu-*.so CPU
# variants already sit in one directory, which is required — runtime
# CPU-variant discovery ignores LD_LIBRARY_PATH (llama.cpp #17491).
COPY --from=dl /llama/ /app/llama/

# Fail the build if the prebuilt binaries need a newer glibc than this base.
RUN if ldd /app/llama/llama-server | grep -q "not found"; then \
      echo "unresolved shared libraries:" && ldd /app/llama/llama-server; exit 1; \
    fi

COPY agent/ /app/agent/
# Track 1 router model: verified answers cost zero tokens by the rules.
COPY models/qwen2.5-1.5b-instruct-q4_k_m.gguf /app/model.gguf
# Track 2 captioner models: SmolVLM2 eye + Gemma 3 stylist (both local).
COPY models/smolvlm2-500m-q8.gguf /app/smolvlm.gguf
COPY models/smolvlm2-mmproj.gguf /app/smolvlm-mmproj.gguf
COPY models/gemma-3-4b-it-Q4_K_M.gguf /app/gemma.gguf
# Track 1 router uses Gemma-3-4B (measured accurate on text); qwen stays as a
# fast fallback. The grader sets no env, so bake the hybrid defaults here.
ENV AGENT_LLAMA_MODEL=/app/gemma.gguf
ENV AGENT_LOCAL_TIMEOUT_S=32
# Track 2 escalation goes through our relay: a URL is baked, the API key is
# NOT — it lives on our box and the relay dies after the event. The Track 1
# path never reads this; it uses harness-injected env only.
WORKDIR /app

# Exec form: python is PID 1 and receives SIGTERM directly.
ENTRYPOINT ["python", "-m", "agent.main"]
