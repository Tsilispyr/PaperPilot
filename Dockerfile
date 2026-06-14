# syntax=docker/dockerfile:1
# =============================================================================
# Dockerfile  -  FAST DEVELOPMENT BUILD  (default)
# =============================================================================
#
# Cache strategy:
#   +--------------------------------------------------------------------------+
#   | Layer               | Invalidated when…        | Approx time             |
#   +--------------------------------------------------------------------------+
#   | apt packages        | base image changes        | ~60s  (then cached)    |
#   | uv install          | never (pip tool only)     | ~5s   (then cached)    |
#   | DEP INSTALL (*)     | pyproject.toml changes    | ~8min (then cached)    |
#   | COPY src            | any source file changes   | instant                |
#   | editable install    | any source file changes   | ~3s   (no network)     |
#   +--------------------------------------------------------------------------+
#
# (*) --mount=type=cache persists the uv package download cache in a Docker
#     BuildKit cache volume that survives even `docker build --no-cache`.
#     Packages are re-extracted from the local cache (~2s) instead of
#     re-downloaded from PyPI (~8 min). This is the key improvement over the
#     previous version - a clean forced rebuild still uses cached .whl files.
#
# EVERYDAY USE:
#     docker compose up --build          # fast, uses layer + download cache
#
# FORCED CLEAN (layers rebuilt, packages NOT re-downloaded):
#     docker build --no-cache -t paperpilot .
#
# NUCLEAR OPTION (everything from scratch):
#     docker build -f Dockerfile.clean --no-cache -t paperpilot .
# =============================================================================

FROM python:3.11-slim

#  System dependencies 
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv (fast pip replacement) - cached pip download
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install uv

# Copy manifests ONLY - source is NOT copied yet
COPY pyproject.toml ./
COPY README.md ./

# Pre-install CPU-only torch (must come BEFORE the full dep install)
# torch from PyPI defaults to the CUDA variant (~2 GB of nvidia-* packages).
# Installing the CPU wheel first satisfies the torch>=2.2.0 constraint so the
# next step never pulls NVIDIA CUDA libraries. Saves ~1.5 GB and ~10 min.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install torch --index-url https://download.pytorch.org/whl/cpu

#  SLOW LAYER (cached until pyproject.toml changes) 
# torch is already installed above; uv resolves it as satisfied and skips it.
# The uv download cache is mounted so re-installing after --no-cache still
# reads .whl files from the BuildKit cache volume, not from the internet.
RUN --mount=type=cache,target=/root/.cache/uv \
    . /opt/venv/bin/activate && \
    mkdir -p src/paperpilot && \
    touch src/paperpilot/__init__.py && \
    uv pip install -e . && \
    rm -rf src/paperpilot

ENV PATH="/opt/venv/bin:$PATH"

#  Copy real source (invalidates cache below, NOT the dep layer above) 
COPY src ./src
COPY public ./public
COPY chainlit.md ./
COPY .chainlit ./.chainlit

#  FAST LAYER (~3s, no internet) - just re-registers your package in the venv 
RUN --mount=type=cache,target=/root/.cache/uv \
    . /opt/venv/bin/activate && uv pip install --no-deps -e .

EXPOSE 8000
CMD ["python", "-m", "paperpilot"]
