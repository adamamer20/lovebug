#!/bin/bash
# Test script to verify devcontainer setup

set -e

echo "🧪 Testing devcontainer setup..."

# Test 1: Verify we're running as dev user
echo "👤 Current user: $(whoami)"
if [ "$(whoami)" = "dev" ]; then
    echo "✅ Running as dev user"
else
    echo "❌ Not running as dev user"
    exit 1
fi

# Test 2: Verify sudo access
echo "🔐 Testing sudo access..."
if sudo -n true 2>/dev/null; then
    echo "✅ Passwordless sudo access confirmed"
else
    echo "❌ Sudo access failed"
    exit 1
fi

# Test 3: Verify fish shell
echo "🐟 Testing fish shell..."
if command -v fish >/dev/null 2>&1; then
    echo "✅ Fish shell available"
else
    echo "❌ Fish shell not found"
    exit 1
fi

# Test 4: Verify uv installation
echo "📦 Testing uv installation..."
if command -v uv >/dev/null 2>&1; then
    echo "✅ uv available: $(uv --version)"
else
    echo "❌ uv not found"
    exit 1
fi

# Test 5: Verify Python venv
echo "🐍 Testing Python virtual environment..."
if [ -x "/app/.venv/bin/python" ]; then
    echo "✅ Python venv available: $(/app/.venv/bin/python --version)"
else
    echo "❌ Python venv not found at /app/.venv"
    exit 1
fi

# Test 6: Verify file permissions
echo "📁 Testing file permissions..."
if [ -w "/app" ] && [ -w "/workspace" ]; then
    echo "✅ Dev user can write to /app and /workspace"
else
    echo "❌ Permission issues with /app or /workspace"
    exit 1
fi

echo "🎉 All tests passed! Devcontainer setup is correct."
