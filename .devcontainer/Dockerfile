# Multi-stage build for LoveBug
# Stage 1: Dependencies only - Install all dependencies but no source code
FROM python:3.12-slim AS deps

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies (fish shell, build tools, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    fish \
    && rm -rf /var/lib/apt/lists/*

# Install pipx and uv
RUN pip install --no-cache-dir pipx && \
    pipx install uv && \
    pipx ensurepath

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock* README.md ./

# Sync dependencies and create virtual environment (dev dependencies included)
RUN uv sync --frozen

# Stage 2: Builder - Build wheel (for production image)
FROM deps AS builder

# Copy source code and build wheel
COPY src/ ./src/
COPY LICENSE ./
RUN uv build --wheel

# Stage 3: Runtime - Minimal production image
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create non-root user
RUN useradd --create-home --uid 1000 app
USER app
WORKDIR /app

# Copy built wheel from builder stage
COPY --from=builder --chown=app:app /app/dist/*.whl ./

# Install only the wheel for minimal footprint
RUN pip install --no-cache-dir *.whl

# Default command (override as needed)
CMD ["python", "-m", "lovebug"]

# Stage 4: Development - For local development with all dev tools
FROM python:3.12-slim AS development

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    fish \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy uv and virtual environment from deps stage
COPY --from=deps /root/.local /root/.local
COPY --from=deps /app/.venv /app/.venv

# Set PATH to include uv and the project's virtual environment
ENV PATH="/app/.venv/bin:/root/.local/bin:${PATH}"

# Create dev user with fish shell
RUN useradd --create-home --shell /usr/bin/fish --uid 1000 dev
USER dev
WORKDIR /workspace

# Note: Source code is mounted at runtime by VS Code, not copied into the image
# This prevents layer invalidation when code changes

# Set fish as default shell
CMD ["/usr/bin/fish", "-l"]
