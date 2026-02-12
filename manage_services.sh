#!/bin/bash
# =============================================================================
# Service Management Script for Mem0 Benchmark
# =============================================================================
# Manages Neo4j and provides helper commands for LLM server
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
NEO4J_CONTAINER="neo4j-mem0"
NEO4J_PASSWORD="password"
LLM_PORT=8002
LLM_MODEL="Qwen/Qwen3-8B"

print_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# =============================================================================
# Neo4j Functions
# =============================================================================

neo4j_status() {
    print_header "Neo4j Status"
    
    if docker ps | grep -q "$NEO4J_CONTAINER"; then
        print_success "Neo4j is running"
        docker ps | grep "$NEO4J_CONTAINER"
        echo ""
        print_info "Web UI: http://localhost:7474"
        print_info "Bolt: bolt://localhost:7687"
        print_info "Auth: neo4j / $NEO4J_PASSWORD"
    else
        print_warning "Neo4j is not running"
    fi
}

neo4j_start() {
    print_header "Starting Neo4j"
    
    if docker ps | grep -q "$NEO4J_CONTAINER"; then
        print_warning "Neo4j is already running"
        return
    fi
    
    # Remove old container if exists
    if docker ps -a | grep -q "$NEO4J_CONTAINER"; then
        print_info "Removing old container..."
        docker rm -f "$NEO4J_CONTAINER" > /dev/null 2>&1
    fi
    
    print_info "Starting Neo4j container..."
    docker run -d \
        --name "$NEO4J_CONTAINER" \
        -p 7474:7474 \
        -p 7687:7687 \
        -e NEO4J_AUTH=neo4j/$NEO4J_PASSWORD \
        -v neo4j-data:/data \
        neo4j:latest
    
    print_info "Waiting for Neo4j to be ready..."
    sleep 10
    
    if docker ps | grep -q "$NEO4J_CONTAINER"; then
        print_success "Neo4j started successfully!"
        echo ""
        print_info "Web UI: http://localhost:7474"
        print_info "Username: neo4j"
        print_info "Password: $NEO4J_PASSWORD"
    else
        print_error "Failed to start Neo4j"
        docker logs "$NEO4J_CONTAINER"
        exit 1
    fi
}

neo4j_stop() {
    print_header "Stopping Neo4j"
    
    if docker ps | grep -q "$NEO4J_CONTAINER"; then
        print_info "Stopping container..."
        docker stop "$NEO4J_CONTAINER"
        print_success "Neo4j stopped"
    else
        print_warning "Neo4j is not running"
    fi
}

neo4j_restart() {
    neo4j_stop
    sleep 2
    neo4j_start
}

neo4j_logs() {
    print_header "Neo4j Logs"
    docker logs -f "$NEO4J_CONTAINER"
}

neo4j_clean() {
    print_header "Cleaning Neo4j Data"
    
    print_warning "This will delete all Neo4j data!"
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        print_info "Cancelled"
        return
    fi
    
    neo4j_stop
    print_info "Removing container and volume..."
    docker rm -f "$NEO4J_CONTAINER" 2>/dev/null || true
    docker volume rm neo4j-data 2>/dev/null || true
    print_success "Neo4j data cleaned"
}

# =============================================================================
# LLM Server Functions
# =============================================================================

llm_status() {
    print_header "LLM Server Status"
    
    if curl -s -f -m 2 "http://localhost:$LLM_PORT/v1/models" > /dev/null 2>&1; then
        print_success "LLM server is running on port $LLM_PORT"
        
        # Get models
        models=$(curl -s "http://localhost:$LLM_PORT/v1/models" | grep -o '"id":"[^"]*"' | cut -d'"' -f4 || echo "")
        if [ -n "$models" ]; then
            echo ""
            print_info "Available models:"
            echo "$models" | while read -r model; do
                echo "  • $model"
            done
        fi
    else
        print_warning "LLM server is not running on port $LLM_PORT"
        echo ""
        print_info "To start LLM server, see: llm-help"
    fi
}

llm_help() {
    print_header "LLM Server Setup Guide"
    
    echo -e "${RED}⚠️  CRITICAL: Tool calling support is REQUIRED!${NC}"
    echo ""
    echo "Mem0 needs an LLM that supports function/tool calling."
    echo ""
    
    echo -e "${BLUE}Recommended: Qwen2.5 with vLLM${NC}"
    echo "  # Install vLLM"
    echo "  pip install vllm"
    echo ""
    echo "  # Start server (RECOMMENDED)"
    echo "  vllm serve Qwen/Qwen2.5-7B-Instruct \\"
    echo "    --port $LLM_PORT \\"
    echo "    --gpu-memory-utilization 0.9 \\"
    echo "    --tool-call-parser openai"
    echo ""
    echo -e "${GREEN}Why Qwen2.5?${NC}"
    echo "  ✅ Native tool calling support"
    echo "  ✅ Fast and efficient"
    echo "  ✅ Good quality"
    echo ""
    
    echo -e "${BLUE}Option 2: SGLang${NC}"
    echo "  # Install"
    echo "  pip install sglang[all]"
    echo ""
    echo "  # Start server"
    echo "  python -m sglang.launch_server \\"
    echo "    --model $LLM_MODEL \\"
    echo "    --port $LLM_PORT \\"
    echo "    --mem-fraction-static 0.9"
    echo ""
    
    echo -e "${BLUE}Option 3: llama.cpp${NC}"
    echo "  # Download model in GGUF format first"
    echo "  ./server \\"
    echo "    -m qwen3-8b.gguf \\"
    echo "    --port $LLM_PORT \\"
    echo "    --ctx-size 8192"
    echo ""
    
    echo -e "${BLUE}Option 4: Ollama${NC}"
    echo "  # Install Ollama and pull model"
    echo "  ollama pull qwen2.5:7b"
    echo ""
    echo "  # Ollama serves on port 11434 by default"
    echo "  # Update LLM_BASE_URL in benchmark scripts to:"
    echo "  # http://localhost:11434/v1"
    echo ""
}

# =============================================================================
# Combined Functions
# =============================================================================

services_status() {
    neo4j_status
    llm_status
}

services_start() {
    neo4j_start
    echo ""
    llm_status
}

services_stop() {
    neo4j_stop
}

# =============================================================================
# Main Menu
# =============================================================================

show_menu() {
    print_header "🛠️  Mem0 Benchmark - Service Manager"
    
    echo "Neo4j Management:"
    echo "  neo4j-start     Start Neo4j database"
    echo "  neo4j-stop      Stop Neo4j database"
    echo "  neo4j-restart   Restart Neo4j database"
    echo "  neo4j-status    Check Neo4j status"
    echo "  neo4j-logs      View Neo4j logs"
    echo "  neo4j-clean     Clean Neo4j data (WARNING: deletes all data)"
    echo ""
    echo "LLM Server Management:"
    echo "  llm-status      Check LLM server status"
    echo "  llm-help        Show LLM server setup guide"
    echo ""
    echo "Combined:"
    echo "  status          Check all services"
    echo "  start           Start all services"
    echo "  stop            Stop all services"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
}

# =============================================================================
# Command Router
# =============================================================================

case "${1:-help}" in
    # Neo4j
    neo4j-start)    neo4j_start ;;
    neo4j-stop)     neo4j_stop ;;
    neo4j-restart)  neo4j_restart ;;
    neo4j-status)   neo4j_status ;;
    neo4j-logs)     neo4j_logs ;;
    neo4j-clean)    neo4j_clean ;;
    
    # LLM
    llm-status)     llm_status ;;
    llm-help)       llm_help ;;
    
    # Combined
    status)         services_status ;;
    start)          services_start ;;
    stop)           services_stop ;;
    
    # Help
    help|--help|-h) show_menu ;;
    
    *)
        print_error "Unknown command: $1"
        echo ""
        show_menu
        exit 1
        ;;
esac

