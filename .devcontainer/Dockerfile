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

# Install uv from source
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

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
RUN python -m pip install --no-cache-dir --user *.whl && \
    rm *.whl

# Set PATH to include user's local bin for packages installed with --user
ENV PATH="/home/app/.local/bin:${PATH}"

# Default command (override as needed)
CMD ["python", "-m", "lovebug"]

# Stage 4: Development - For local development with all dev tools
FROM python:3.12-slim AS development

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies, including sudo and Docker CLI
RUN apt-get update && apt-get install -y \
    fish \
    git \
    curl \
    sudo \
    ca-certificates \
    gnupg \
    lsb-release \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg \
         | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && echo \
     "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
     https://download.docker.com/linux/debian \
     $(lsb_release -cs) stable" \
      > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install Node.js (using NodeSource) and then globally install Claude Code CLI
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
 && apt-get install -y nodejs \
 && npm install -g @anthropic-ai/claude-code \
 && rm -rf /var/lib/apt/lists/*

# Copy uv from deps stage
COPY --from=deps /root/.local/bin/uv /usr/local/bin/uv

# Prepare /app directory for venv and its project files
RUN mkdir -p /app
COPY --from=deps /app/.venv /app/.venv
COPY --from=deps /app/pyproject.toml /app/pyproject.toml
COPY --from=deps /app/uv.lock /app/uv.lock

# Set PATH to include the project's virtual environment
ENV PATH="/app/.venv/bin:${PATH}"

# Create docker group and dev user, configure passwordless sudo, and set ownership
RUN groupadd -g 999 docker \
    && echo "dev ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/dev-nopasswd \
    && chmod 0440 /etc/sudoers.d/dev-nopasswd \
    && useradd --create-home --shell /usr/bin/fish --uid 1000 --groups docker dev \
    && mkdir -p /workspace \
    && chown -R dev:dev /home/dev /workspace /app

USER dev
WORKDIR /workspace

# Note: Source code is mounted at runtime by VS Code, not copied into the image
# This prevents layer invalidation when code changes

# Set fish as default shell
CMD ["/usr/bin/fish", "-l"]
