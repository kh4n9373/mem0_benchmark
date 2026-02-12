#!/bin/bash
# =============================================================================
# Pre-flight Check - Verify EVERYTHING before running benchmark
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo ""
}

check_pass() { echo -e "${GREEN}✅ $1${NC}"; }
check_fail() { echo -e "${RED}❌ $1${NC}"; FAILED=true; }
check_warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }
check_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

FAILED=false

clear
print_header "🚀 Mem0 Benchmark - Pre-flight Check"

echo "Verifying all requirements..."
echo ""

# =============================================================================
# System Requirements
# =============================================================================

echo "System Requirements:"

# Python
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 --version | cut -d' ' -f2)
    PY_MAJOR=$(echo $PY_VERSION | cut -d. -f1)
    PY_MINOR=$(echo $PY_VERSION | cut -d. -f2)
    if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 10 ]; then
        check_pass "Python $PY_VERSION"
    else
        check_fail "Python $PY_VERSION (need 3.10+)"
    fi
else
    check_fail "Python 3.10+ not found"
fi

# UV
if command -v uv &> /dev/null; then
    check_pass "UV package manager ($(uv --version))"
else
    check_warn "UV not installed (will auto-install)"
fi

# Docker
if command -v docker &> /dev/null; then
    check_pass "Docker ($(docker --version | cut -d' ' -f3 | tr -d ','))"
else
    check_fail "Docker not found (needed for Neo4j)"
fi

# Git
if command -v git &> /dev/null; then
    check_pass "Git"
else
    check_warn "Git not found (optional)"
fi

echo ""

# =============================================================================
# GPU Check
# =============================================================================

echo "GPU Availability:"

if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    check_pass "Found $GPU_COUNT GPU(s): $GPU_NAME ($GPU_MEM)"
    
    # Check VRAM
    MEM_GB=$(echo $GPU_MEM | grep -oP '\d+' | head -1)
    if [ "$MEM_GB" -lt 16 ]; then
        check_warn "GPU has <16GB VRAM. Recommended: 24GB+"
    fi
else
    check_warn "No NVIDIA GPU detected (CPU mode will be very slow)"
fi

echo ""

# =============================================================================
# Critical Services
# =============================================================================

echo "Service Status:"

# LLM Server
if curl -s -f -m 2 http://localhost:8002/v1/models > /dev/null 2>&1; then
    MODEL=$(curl -s http://localhost:8002/v1/models 2>/dev/null | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
    check_pass "LLM server running (model: ${MODEL:-unknown})"
    
    # Test tool calling
    TOOL_TEST=$(curl -s http://localhost:8002/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "'$MODEL'",
        "messages": [{"role": "user", "content": "test"}],
        "tools": [{"type": "function", "function": {"name": "test", "parameters": {}}}],
        "tool_choice": "auto",
        "max_tokens": 10
      }' 2>&1)
    
    if echo "$TOOL_TEST" | grep -q "tool_calls\|choices"; then
        check_pass "Tool calling is supported ✓"
    else
        check_fail "Tool calling NOT supported (CRITICAL!)"
        echo ""
        echo -e "${YELLOW}Your LLM server lacks tool calling support.${NC}"
        echo -e "${YELLOW}Restart with:${NC}"
        echo "  vllm serve Qwen/Qwen2.5-7B-Instruct --port 8002 --tool-call-parser openai"
        echo ""
    fi
else
    check_fail "LLM server not running on port 8002 (CRITICAL!)"
    echo ""
    echo -e "${YELLOW}Start LLM server with:${NC}"
    echo "  vllm serve Qwen/Qwen2.5-7B-Instruct \\"
    echo "    --port 8002 \\"
    echo "    --gpu-memory-utilization 0.9 \\"
    echo "    --tool-call-parser openai"
    echo ""
fi

# Neo4j
if pgrep -f "neo4j" > /dev/null; then
    check_pass "Neo4j process running"
else
    check_warn "Neo4j not running (will auto-start)"
fi

if curl -s -f -m 2 http://localhost:7474 > /dev/null 2>&1; then
    check_pass "Neo4j web interface accessible"
fi

# Embedding Model
EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-m3}"
if command -v uv &> /dev/null; then
    MODEL_CHECK=$(uv run python -c "
from sentence_transformers import SentenceTransformer
import sys
try:
    # Try to load from cache without downloading
    model = SentenceTransformer('$EMBEDDING_MODEL', cache_folder=None)
    print('cached')
except:
    print('missing')
" 2>/dev/null)
    
    if [ "$MODEL_CHECK" = "cached" ]; then
        check_pass "Embedding model cached ($EMBEDDING_MODEL)"
    else
        check_warn "Embedding model not cached (will download on first run)"
    fi
else
    check_warn "Cannot check embedding model (uv not available)"
fi

echo ""

# =============================================================================
# Project Structure
# =============================================================================

echo "Project Files:"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check scripts
if [ -f "$SCRIPT_DIR/full_benchmark_locomo.sh" ] && [ -x "$SCRIPT_DIR/full_benchmark_locomo.sh" ]; then
    check_pass "Benchmark script (locomo)"
else
    check_fail "Benchmark script missing or not executable"
fi

# Check pyproject.toml
if [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
    check_pass "pyproject.toml"
else
    check_fail "pyproject.toml not found"
fi

# Check dataset
if [ -f "$SCRIPT_DIR/data/locomo/processed_data/locomo_lite.json" ]; then
    LINES=$(wc -l < "$SCRIPT_DIR/data/locomo/processed_data/locomo_lite.json")
    check_pass "LOCOMO dataset ($LINES lines)"
else
    check_warn "LOCOMO dataset not found (will need to download)"
fi

echo ""

# =============================================================================
# Ports Check
# =============================================================================

echo "Port Availability:"

check_port() {
    if lsof -i :$1 > /dev/null 2>&1 || netstat -tuln 2>/dev/null | grep -q ":$1 "; then
        return 0
    else
        return 1
    fi
}

# Port 8002
if check_port 8002; then
    check_pass "Port 8002 (LLM server)"
else
    check_warn "Port 8002 not in use (LLM needs to start)"
fi

# Port 7687
if check_port 7687; then
    check_pass "Port 7687 (Neo4j bolt)"
else
    check_warn "Port 7687 not in use (Neo4j needs to start)"
fi

# Port 7474
if check_port 7474; then
    check_pass "Port 7474 (Neo4j web)"
else
    check_warn "Port 7474 not in use (Neo4j needs to start)"
fi

echo ""

# =============================================================================
# Disk Space
# =============================================================================

echo "Disk Space:"

AVAIL=$(df -BG "$SCRIPT_DIR" | tail -1 | awk '{print $4}' | tr -d 'G')
if [ "$AVAIL" -gt 50 ]; then
    check_pass "Available: ${AVAIL}GB"
elif [ "$AVAIL" -gt 20 ]; then
    check_warn "Available: ${AVAIL}GB (Recommended: 50GB+)"
else
    check_fail "Available: ${AVAIL}GB (Insufficient!)"
fi

echo ""

# =============================================================================
# Summary
# =============================================================================

print_header "Summary"

if [ "$FAILED" = true ]; then
    echo -e "${RED}❌ Pre-flight check FAILED${NC}"
    echo ""
    echo "Please fix the issues above before running the benchmark."
    echo ""
    echo "Common fixes:"
    echo "  • Install Docker: https://docker.com"
    echo "  • Start LLM server: See LLM_SERVER_GUIDE.md"
    echo "  • Free up disk space"
    echo ""
    exit 1
else
    echo -e "${GREEN}✅ All checks passed!${NC}"
    echo ""
    echo "You're ready to run the benchmark:"
    echo ""
    echo "  ./full_benchmark_locomo.sh"
    echo ""
    echo "Or for LongMemEval:"
    echo ""
    echo "  ./full_benchmark_longmemeval.sh"
    echo ""
fi
