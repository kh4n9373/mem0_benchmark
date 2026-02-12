#!/bin/bash
# =============================================================================
# Quick Setup Check - UV-based
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
}

check_item() {
    local name="$1"
    local check_command="$2"
    local help_text="$3"
    
    echo -n "Checking $name... "
    if eval "$check_command" > /dev/null 2>&1; then
        echo -e "${GREEN}✅${NC}"
        return 0
    else
        echo -e "${RED}❌${NC}"
        if [ -n "$help_text" ]; then
            echo -e "${YELLOW}  → $help_text${NC}"
        fi
        return 1
    fi
}

print_header "🔍 Mem0 Benchmark - Setup Check"

all_good=true

# UV
if ! check_item "uv package manager" "command -v uv" "Install: curl -LsSf https://astral.sh/uv/install.sh | sh"; then
    all_good=false
fi

# Docker
if ! check_item "Docker" "command -v docker" "Install from docker.com"; then
    all_good=false
fi

# Neo4j
if ! check_item "Neo4j container" "docker ps | grep -q neo4j" "Run: ./manage_services.sh neo4j-start"; then
    all_good=false
fi

if ! check_item "Neo4j connection" "curl -s -f -m 2 http://localhost:7474 > /dev/null" "Check: ./manage_services.sh neo4j-logs"; then
    all_good=false
fi

# LLM Server
if ! check_item "LLM server (port 8002)" "curl -s -f -m 2 http://localhost:8002/v1/models > /dev/null" "See: ./manage_services.sh llm-help"; then
    all_good=false
fi

# Datasets
echo ""
echo "Checking datasets..."
if ! check_item "LOCOMO dataset" "[ -f data/locomo/processed_data/locomo_lite.json ]" "Will be checked at runtime"; then
    echo -e "${YELLOW}  Note: Dataset path will be validated when running benchmark${NC}"
fi

# GPU
echo ""
echo "Checking GPU..."
if command -v nvidia-smi > /dev/null 2>&1; then
    gpu_count=$(nvidia-smi --list-gpus | wc -l)
    echo -e "${GREEN}✅ Found $gpu_count GPU(s)${NC}"
else
    echo -e "${YELLOW}⚠️  No NVIDIA GPU (CPU mode will be slower)${NC}"
fi

# Summary
echo ""
print_header "📋 Summary"

if $all_good; then
    echo -e "${GREEN}✨ All checks passed! Ready to run benchmarks.${NC}"
    echo ""
    echo "Run:"
    echo "  ./full_benchmark_locomo.sh      # LOCOMO benchmark"
    echo "  ./full_benchmark_longmemeval.sh # LongMemEval benchmark"
else
    echo -e "${YELLOW}⚠️  Some checks failed. Review errors above.${NC}"
    echo ""
    echo "Quick fixes:"
    echo "  ./manage_services.sh start      # Start services"
    echo "  ./manage_services.sh llm-help   # LLM setup guide"
fi

echo ""
