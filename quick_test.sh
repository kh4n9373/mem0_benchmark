#!/bin/bash
# Quick test pipeline với 20 questions đầu tiên
# Run from: mem0/ directory

cd "$(dirname "$0")"

echo "🧪 Running quick test (20 questions)..."
echo ""

# Use timestamped directories to avoid permission issues
timestamp=$(date +%Y%m%d_%H%M%S)

python3 mem0_full_pipeline.py \
    data/locomo/processed_data/locomo_small.json \
    test_memory_${timestamp} \
    test_results_${timestamp} \
    --max_workers 1 \
    --llm_model Qwen/Qwen3-8B \
    --api_key dummy \
    --base_url http://localhost:8001/v1 \
    --embedding_model facebook/contriever \
    --top_k 100 \
    --context_k 5 \
    --eval_ks "3,5,10" \
    --limit 20 \
    --disable_thinking
