# Multi-stage build for LoveBug
# Stage 1: Builder - Install dependencies and build wheel
FROM python:3.12-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install pipx and uv
RUN pip install --no-cache-dir pipx && \
    pipx install uv && \
    pipx ensurepath

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock* README.md ./

# Sync dependencies and create virtual environment
RUN uv sync --frozen

# Copy source code and build wheel
COPY src/ ./src/
COPY LICENSE ./
RUN uv build --wheel

# Stage 2: Runtime - Minimal production image
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

# Development stage - For local development with all dev tools
FROM builder AS development

# Install fish shell and dev tools
USER root
RUN apt-get update && apt-get install -y \
    fish \
    && rm -rf /var/lib/apt/lists/*

# Copy uv to a standard location accessible to all users
COPY --from=builder /root/.local/bin/uv /usr/local/bin/uv
RUN chmod a+x /usr/local/bin/uv

# Create dev user with fish shell
RUN useradd --create-home --shell /usr/bin/fish --uid 1000 dev

# Copy the virtual environment from builder stage
COPY --from=builder --chown=dev:dev /app/.venv /app/.venv

# Switch to dev user
USER dev
WORKDIR /workspace

# Set PATH to include uv and the project's virtual environment
ENV PATH="/app/.venv/bin:/usr/local/bin:/root/.local/bin:${PATH}"

# Copy entire project for development
COPY --chown=dev:dev . ./

# Debug: Verify uv is available
RUN echo "USER: $(whoami)" && \
    echo "PATH: $PATH" && \
    which uv && \
    uv --version

# Install pre-commit hooks
RUN uv run pre-commit install

# Set fish as default shell and start interactive session
CMD ["/usr/bin/fish", "-l"]
