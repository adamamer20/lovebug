# Multi-stage build for LoveBug
# Stage 1: Dependencies only - Install all dependencies but no source code
FROM python:3.12-slim AS deps

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies (build tools, curl, git, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install latest Quarto CLI dynamically
RUN QUARTO_TAG=$(curl -s https://api.github.com/repos/quarto-dev/quarto-cli/releases/latest \
                   | grep -Po '"tag_name": "\K.*?(?=")') \
 && VERSION=${QUARTO_TAG#v} \
 && curl -LsSf https://github.com/quarto-dev/quarto-cli/releases/download/${QUARTO_TAG}/quarto-${VERSION}-linux-amd64.tar.gz \
      -o quarto.tar.gz \
 && mkdir -p /usr/local/quarto \
 && tar -xzf quarto.tar.gz -C /usr/local/quarto --strip-components=1 \
 && ln -s /usr/local/quarto/bin/quarto /usr/local/bin/quarto \
 && rm quarto.tar.gz \
 && quarto --version

# Install TinyTeX via Quarto (cleaner integration)
RUN quarto install tinytex --update-path

# Install uv to /usr/local/bin (system-wide accessible)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && mv /root/.local/bin/uv /usr/local/bin/uv

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
RUN python -m pip install --no-cache-dir --user *.whl && \
    rm *.whl

# Set PATH to include user's local bin for packages installed with --user
ENV PATH="/home/app/.local/bin:${PATH}"

# Default command (override as needed)
CMD ["python", "-m", "lovebug"]

# Stage 4: Development - For local development with all dev tools
FROM deps AS development

# ENV variables are inherited from 'deps' stage
# Quarto, uv, git, curl, build-essential are all inherited from 'deps'

# Install additional development-specific system tools
RUN apt-get update && apt-get install -y \
    fish \
    && rm -rf /var/lib/apt/lists/*

# Set PATH to include the project's virtual environment
# (uv is system-wide in /usr/local/bin, venv takes priority)
ENV PATH="/app/.venv/bin:${PATH}"

# Create dev user with fish shell
RUN useradd --create-home --shell /usr/bin/fish --uid 1000 dev
USER dev
WORKDIR /workspace

# Note: Source code is mounted at runtime by VS Code, not copied into the image
# This prevents layer invalidation when code changes

# Set fish as default shell
CMD ["/usr/bin/fish", "-l"]
