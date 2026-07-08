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
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
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
WORKDIR /app

# Exec form: python is PID 1 and receives SIGTERM directly.
ENTRYPOINT ["python", "-m", "agent.main"]
