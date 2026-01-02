cd "$(dirname "$0")"

echo "🚀 Running full locomo benchmark (all questions)..."
echo ""

python3 mem0_full_pipeline.py \
    data/locomo/processed_data/locomo_lite.json \
    locomo_memory_benchmark \
    locomo_results_benchmark \
    --max_workers 2 \
    --llm_model Qwen/Qwen3-8B \
    --api_key dummy \
    --base_url http://localhost:8001/v1 \
    --embedding_model facebook/contriever \
    --top_k 100 \
    --context_k 5 \
    --eval_ks "3,5,10" \
    --disable_thinking \
    --log_dir "worker_logs/locomo" \
    --stream_logs
