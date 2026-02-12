#!/bin/bash
# Run GPT4SCORE evaluation on retrieval results
# Usage: ./run_gpt4score_eval.sh <retrieval_results.json> [context_k] [judge_model] [llm_model]

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <retrieval_results.json> [context_k] [judge_model] [llm_model]"
    echo ""
    echo "Arguments:"
    echo "  retrieval_results.json  - Input retrieval results file (required)"
    echo "  context_k              - Number of chunks for generation (default: 5)"
    echo "  judge_model            - Model for LLM judge (default: Qwen/Qwen3-8B)"
    echo "  llm_model              - Model for answer generation (default: Qwen/Qwen3-8B)"
    echo ""
    echo "Example:"
    echo "  $0 locomo_results_contriever/retrieval_results.json 5 Qwen/Qwen3-8B Qwen/Qwen3-8B"
    exit 1
fi

RETRIEVAL_FILE="$1"
CONTEXT_K=5
JUDGE_MODEL="Qwen/Qwen3-8B"
LLM_MODEL="Qwen/Qwen3-8B"

# Generate output filename
BASENAME=$(basename "$RETRIEVAL_FILE" .json)
DIRNAME=$(dirname "$RETRIEVAL_FILE")
OUTPUT_FILE="${DIRNAME}/${BASENAME}_gpt4score.json"

echo "=================================="
echo "🚀 GPT4SCORE EVALUATION"
echo "=================================="
echo "Input: $RETRIEVAL_FILE"
echo "Output: $OUTPUT_FILE"
echo "Context-K: $CONTEXT_K"
echo "Judge Model: $JUDGE_MODEL"
echo "LLM Model: $LLM_MODEL"
echo "=================================="
echo ""

# Check if input file exists
if [ ! -f "$RETRIEVAL_FILE" ]; then
    echo "❌ Error: Input file not found: $RETRIEVAL_FILE"
    exit 1
fi

# Run GPT4SCORE evaluation
START_TIME=$(date +%s)

python3 gpt4score_evaluator.py \
    --input "$RETRIEVAL_FILE" \
    --output "$OUTPUT_FILE" \
    --context_k "$CONTEXT_K" \
    --judge_model "$JUDGE_MODEL" \
    --llm_model "$LLM_MODEL" \
    --base_url "http://localhost:8001/v1" \
    --api_key "dummy" \
    --max_concurrent 20 \
    --disable_thinking

STATUS=$?

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
if [ $STATUS -eq 0 ]; then
    echo "✅ GPT4SCORE evaluation completed successfully!"
    echo "Time taken: ${DURATION}s ($((DURATION/60))min)"
    echo ""
    echo "📊 View results:"
    echo "   cat $OUTPUT_FILE | jq '.overall'"
    echo "   cat $OUTPUT_FILE | jq '.by_category'"
else
    echo "❌ GPT4SCORE evaluation failed!"
    exit 1
fi


"""
bash run_gpt4score_eval.sh locomo_results_benchmark/retrieval_results_20260105_025735.json
"""