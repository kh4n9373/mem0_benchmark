#!/bin/bash
# =============================================================================
# LongMemEval Full Benchmark - Automated Execution with UV
# =============================================================================

set -e

# Force use GPU 0 (avoid GPU 1 if busy)
export CUDA_VISIBLE_DEVICES=0

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_PATH="$SCRIPT_DIR/data/locomo/processed_data/longmemeval_processed_data.json"  # Auto-relative
MEMORY_DIR="$SCRIPT_DIR/longmemeval_memory_benchmark"  # Use absolute
RESULTS_DIR="$SCRIPT_DIR/longmemeval_results_benchmark"
LOG_DIR="$SCRIPT_DIR/worker_logs/longmemeval"

# Services
LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
LLM_BASE_URL="http://localhost:8002/v1"
NEO4J_URL="bolt://localhost:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="password"

# Models
EMBEDDING_MODEL="BAAI/bge-m3"

# =============================================================================
# Helpers
# =============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
}

print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

exit_with_error() {
    print_error "$1"
    echo ""
    echo -e "${YELLOW}Please fix the above issue and try again.${NC}"
    exit 1
}

# =============================================================================
# Setup
# =============================================================================

setup_environment() {
    print_header "STEP 1: Environment Setup"
    
    cd "$SCRIPT_DIR"
    
    # Check uv
    if ! command -v uv &> /dev/null; then
        print_info "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
    print_success "uv: $(uv --version)"
    
    # Sync dependencies
    print_info "Syncing dependencies with uv..."
    uv sync
    print_success "Dependencies synced"
}

verify_services() {
    print_header "STEP 2: Service Checks"
    
    # Check Neo4j
    print_info "Checking Neo4j..."
    if ! pgrep -f "neo4j" > /dev/null; then
        print_warning "Neo4j not running. Starting automatically..."
        if command -v docker &> /dev/null; then
            docker run -d --name neo4j-mem0 -p 7474:7474 -p 7687:7687 \
                -e NEO4J_AUTH=$NEO4J_USERNAME/$NEO4J_PASSWORD neo4j 2>/dev/null || \
            docker start neo4j-mem0 2>/dev/null
            sleep 10
            print_success "Neo4j started"
        else
            exit_with_error "Docker not found. Install Docker or start Neo4j manually:\n  ./manage_services.sh neo4j-start"
        fi
    else
        print_success "Neo4j already running"
    fi
    
    uv run python - <<EOF
import sys
from neo4j import GraphDatabase
try:
    driver = GraphDatabase.driver("$NEO4J_URL", auth=("$NEO4J_USERNAME", "$NEO4J_PASSWORD"))
    with driver.session() as session:
        session.run("RETURN 1")
    driver.close()
    print("✅ Neo4j connection OK")
except Exception as e:
    print(f"❌ Neo4j connection failed: {e}")
    sys.exit(1)
EOF
    
    [ $? -eq 0 ] || exit_with_error "Neo4j connection failed"
    
    # Check LLM server
    print_info "Checking LLM server..."
    if ! curl -s -f -m 5 "$LLM_BASE_URL/models" > /dev/null 2>&1; then
        print_error "LLM server not running on port 8002"
        echo ""
        echo -e "${YELLOW}╔══════════════════════════════════════════════════════════╗${NC}"
        echo -e "${YELLOW}║  CRITICAL: LLM Server with Tool Calling Required        ║${NC}"
        echo -e "${YELLOW}╚══════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo "Mem0 requires an LLM with tool calling support."
        echo ""
        echo "Recommended: Qwen2.5-7B-Instruct"
        echo ""
        echo -e "${GREEN}Start with:${NC}"
        echo "  vllm serve Qwen/Qwen2.5-7B-Instruct \\"
        echo "    --port 8002 \\"
        echo "    --gpu-memory-utilization 0.9 \\"
        echo "    --tool-call-parser openai"
        echo ""
        echo -e "${BLUE}Alternative models:${NC}"
        echo "  • Qwen/Qwen2.5-14B-Instruct (better quality)"
        echo "  • meta-llama/Llama-3.1-8B-Instruct"
        echo "  • gpt-4o-mini (OpenAI API)"
        echo ""
        echo -e "${YELLOW}For more help:${NC} ./manage_services.sh llm-help"
        echo ""
        exit 1
    fi
    print_success "LLM server responding"
}

check_dataset() {
    print_header "STEP 3: Dataset Check"
    
    if [ ! -f "$DATASET_PATH" ]; then
        print_warning "Dataset not found. Downloading..."
        mkdir -p "$(dirname "$DATASET_PATH")"
        
        uv run python - <<EOF
from huggingface_hub import snapshot_download
try:
    snapshot_download(
        repo_id="KhangPTT373/longmemeval",
        local_dir="data/longmemeval",
        repo_type="dataset"
    )
    print("✅ Dataset downloaded")
except Exception as e:
    print(f"❌ Download failed: {e}")
    exit(1)
EOF
        [ $? -eq 0 ] || exit_with_error "Failed to download dataset"
    fi
    
    local lines=$(wc -l < "$DATASET_PATH")
    print_success "Dataset found ($lines lines)"
}

download_models() {
    print_header "STEP 4: Download Models (if needed)"
    
    # Download embedding model
    print_info "Checking embedding model: $EMBEDDING_MODEL..."
    
    # Export for Python subprocess
    export EMBEDDING_MODEL
    
    uv run python - <<EOF
import sys
import os
from sentence_transformers import SentenceTransformer

try:
    model_name = os.environ.get('EMBEDDING_MODEL', 'BAAI/bge-m3')
    print(f"Loading/downloading {model_name}...")
    model = SentenceTransformer(model_name)
    print("✅ Embedding model ready")
    sys.exit(0)
except Exception as e:
    print(f"❌ Failed to load embedding model: {e}")
    print("")
    print("Possible causes:")
    print("  • No internet connection to HuggingFace")
    print("  • Model name incorrect")
    print("  • Insufficient disk space")
    print("")
    sys.exit(1)
EOF
    
    if [ $? -ne 0 ]; then
        exit_with_error "Failed to download embedding model"
    fi
}

clean_previous() {
    print_header "STEP 5: Cleanup"
    
    rm -rf "$MEMORY_DIR" "$RESULTS_DIR" 2>/dev/null || true
    mkdir -p "$LOG_DIR"
    print_success "Clean workspace ready"
}

run_benchmark() {
    print_header "STEP 6: Running Benchmark"
    
    print_info "Configuration:"
    echo "  Dataset:    $DATASET_PATH"
    echo "  LLM:        $LLM_MODEL"
    echo "  Embedding:  $EMBEDDING_MODEL"
    echo "  Neo4j:      $NEO4J_URL"
    echo "  GPU:        $CUDA_VISIBLE_DEVICES"
    echo ""
    
    uv run python benchmark/mem0_full_pipeline.py \
        "$DATASET_PATH" \
        "$MEMORY_DIR" \
        "$RESULTS_DIR" \
        --max_workers 1 \
        --llm_model "$LLM_MODEL" \
        --api_key "dummy" \
        --base_url "$LLM_BASE_URL" \
        --embedding_model "$EMBEDDING_MODEL" \
        --top_k 100 \
        --context_k 5 \
        --eval_ks "3,5,10" \
        --disable_thinking \
        --log_dir "$LOG_DIR" \
        --stream_logs \
        --graph_provider "neo4j" \
        --graph_url "$NEO4J_URL" \
        --graph_username "$NEO4J_USERNAME" \
        --graph_password "$NEO4J_PASSWORD"
    
    if [ $? -eq 0 ]; then
        print_success "Benchmark completed!"
        echo ""
        print_info "Results: $RESULTS_DIR"
    else
        exit_with_error "Benchmark failed. Check logs: $LOG_DIR"
    fi
}

# =============================================================================
# Main
# =============================================================================

main() {
    clear 2>/dev/null || true
    print_header "🚀 LONGMEMEVAL BENCHMARK"
    
    setup_environment
    verify_services
    check_dataset
    download_models
    clean_previous
    run_benchmark
    
    print_header "✨ DONE!"
}

main
