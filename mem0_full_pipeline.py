import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, Any


def run_command(cmd: list, description: str, log_file: str = None):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"▶ {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    if log_file:
        print(f"Logging to: {log_file}")
        with open(log_file, 'w') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    else:
        result = subprocess.run(cmd)
    
    duration = time.time() - start_time
    
    if result.returncode != 0:
        print(f"❌ {description} FAILED (exit code {result.returncode}, took {duration:.2f}s)")
        sys.exit(1)
    else:
        print(f"✅ {description} completed successfully ({duration:.2f}s)")
    
    return duration


def print_results_summary(
    retrieval_results_path: str,
    generation_results_path: str,
    total_time: float
):
    """Print a nice summary of all results."""
    print("\n" + "="*60)
    print("📊 BENCHMARK RESULTS SUMMARY")
    print("="*60)
    
    # Load retrieval results
    if os.path.exists(retrieval_results_path):
        with open(retrieval_results_path, 'r') as f:
            ret_data = json.load(f)
        
        print("\n--- RETRIEVAL EVALUATION ---")
        macro_avgs = ret_data.get("macro_avgs", {})
        micro_avgs = ret_data.get("micro_avgs", {})
        
        for k in sorted(macro_avgs.keys()):
            ma = macro_avgs[k]
            mi = micro_avgs[k]
            print(f"\n@ k={k}")
            print(f"  Macro: P={ma['precision']:.4f}  R={ma['recall']:.4f}  F1={ma['f1']:.4f}  nDCG={ma['ndcg']:.4f}")
            print(f"  Micro: P={mi['precision']:.4f}  R={mi['recall']:.4f}  F1={mi['f1']:.4f}  nDCG={mi['ndcg']:.4f}")
        
        # Category breakdown
        category_avgs = ret_data.get("category_avgs", {})
        if category_avgs:
            print("\n  By Category:")
            for cat in sorted(category_avgs.keys()):
                cat_data = category_avgs[cat]
                k = "10"  # Show k=10 for categories
                if k in cat_data.get("macro_avgs", {}):
                    ma = cat_data["macro_avgs"][k]
                    print(f"    Cat {cat}: P={ma['precision']:.4f}  R={ma['recall']:.4f}  F1={ma['f1']:.4f}  nDCG={ma['ndcg']:.4f}")
    
    # Load generation results
    if os.path.exists(generation_results_path):
        with open(generation_results_path, 'r') as f:
            gen_data = json.load(f)
        
        print("\n--- GENERATION EVALUATION ---")
        overall = gen_data.get("overall", {})
        print(f"\nOverall (n={overall.get('count', 0)})")
        print(f"  F1:           {overall.get('f1', 0):.4f}")
        print(f"  BLEU:         {overall.get('bleu', 0):.4f}")
        print(f"  ROUGE-1:      {overall.get('rouge1', 0):.4f}")
        print(f"  ROUGE-2:      {overall.get('rouge2', 0):.4f}")
        print(f"  ROUGE-L:      {overall.get('rougeL', 0):.4f}")
        print(f"  BERTScore-F1: {overall.get('bertscore_f1', 0):.4f}")
        
        # Category breakdown
        by_category = gen_data.get("by_category", {})
        if by_category:
            print("\n  By Category:")
            for cat in sorted(by_category.keys()):
                metrics = by_category[cat]
                print(f"    Cat {cat} (n={metrics.get('count', 0)}): F1={metrics.get('f1', 0):.4f}  BLEU={metrics.get('bleu', 0):.4f}  BERTScore={metrics.get('bertscore_f1', 0):.4f}")
    
    print("\n" + "="*60)
    print(f"⏱️  Total pipeline time: {total_time:.2f}s ({total_time/60:.2f}min)")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Mem0 Full Benchmark Pipeline: Index -> Retrieve -> Evaluate"
    )
    
    # Required args
    parser.add_argument("dataset_file", help="Path to input dataset JSON (e.g., locomo_small.json)")
    parser.add_argument("memory_dir", help="Path to store Mem0 memory database")
    parser.add_argument("output_dir", help="Path to store all output results")
    
    # Parallel processing
    parser.add_argument("--max_workers", type=int, default=2, help="Number of parallel indexing workers")
    parser.add_argument("--log_dir", default="worker_logs", help="Directory to store worker logs")
    parser.add_argument("--stream_logs", action="store_true", help="Stream worker logs to console")
    
    # LLM config
    parser.add_argument("--llm_model", default="Qwen/Qwen3-8B", help="LLM model for Mem0")
    parser.add_argument("--api_key", default="dummy", help="API key for LLM")
    parser.add_argument("--base_url", default="http://localhost:8001/v1", help="LLM API base URL")
    parser.add_argument("--disable_thinking", action="store_true", default=True, help="Disable thinking for Qwen")
    
    # Embedding config
    parser.add_argument("--embedding_model", default="facebook/contriever", help="Embedding model")
    
    # Retrieval config
    parser.add_argument("--top_k", type=int, default=100, help="Number of chunks to retrieve")
    parser.add_argument("--eval_ks", default="3,5,10", help="K values for retrieval evaluation")
    
    # Generation config
    parser.add_argument("--context_k", type=int, default=5, help="Number of chunks to use as context for generation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions (for testing)")
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute paths
    args.dataset_file = os.path.abspath(args.dataset_file)
    args.memory_dir = os.path.abspath(args.memory_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define output file paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    retrieval_output = os.path.join(args.output_dir, f"retrieval_results_{timestamp}.json")
    retrieval_eval_output = os.path.join(args.output_dir, f"retrieval_eval_{timestamp}.json")
    generation_eval_output = os.path.join(args.output_dir, f"generation_eval_{timestamp}.json")
    pipeline_log = os.path.join(args.output_dir, f"pipeline_{timestamp}.log")
    
    # Track timing
    pipeline_start = time.time()
    timings = {}
    
    print("\n" + "="*60)
    print("🚀 MEM0 FULL BENCHMARK PIPELINE")
    print("="*60)
    print(f"Dataset:       {args.dataset_file}")
    print(f"Memory Dir:    {args.memory_dir}")
    print(f"Output Dir:    {args.output_dir}")
    print(f"Workers:       {args.max_workers}")
    print(f"LLM Model:     {args.llm_model}")
    print(f"Embedding:     {args.embedding_model}")
    print(f"Top-K:         {args.top_k}")
    print(f"Context-K:     {args.context_k}")
    print("="*60)
    
    # Get script paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parallel_index_script = os.path.join(script_dir, "mem0_parallel_index.py")
    retrieve_script = os.path.join(script_dir, "mem0_process_retrieve.py")
    retrieval_eval_script = os.path.join(os.path.dirname(script_dir), "retrieval_evaluator.py")
    generation_eval_script = os.path.join(os.path.dirname(script_dir), "generation_evaluator.py")
    
    python_exe = sys.executable
    
    # Step 1: Parallel Indexing
    index_cmd = [
        python_exe,
        parallel_index_script,
        args.dataset_file,
        args.memory_dir,
        "--max_workers", str(args.max_workers),
        "--log_dir", args.log_dir,
        "--model_name", args.embedding_model,
        "--llm_backend", "openai",
        "--llm_model", args.llm_model,
        "--api_key", args.api_key,
        "--base_url", args.base_url,
    ]
    if args.disable_thinking:
        index_cmd.append("--disable_thinking")
    if args.stream_logs:
        index_cmd.append("--stream_logs")
    
    timings['indexing'] = run_command(
        index_cmd,
        "STEP 1/4: Parallel Indexing (with LLM summarization)"
    )
    
    # Step 2: Retrieval
    retrieve_cmd = [
        python_exe,
        retrieve_script,
        args.dataset_file,
        args.memory_dir,
        retrieval_output,
        "--model_name", args.embedding_model,
        "--top_k", str(args.top_k)
    ]
    
    timings['retrieval'] = run_command(
        retrieve_cmd,
        "STEP 2/4: Retrieval"
    )
    
    # Step 3: Retrieval Evaluation
    retrieval_eval_cmd = [
        python_exe,
        retrieval_eval_script,
        "--input", retrieval_output,
        "--ks", args.eval_ks,
        "--out", retrieval_eval_output
    ]
    
    timings['retrieval_eval'] = run_command(
        retrieval_eval_cmd,
        "STEP 3/4: Retrieval Evaluation"
    )
    
    # Step 4: Generation Evaluation
    generation_eval_cmd = [
        python_exe,
        generation_eval_script,
        retrieval_output,
        "--ground-truth", args.dataset_file,
        "--output", generation_eval_output,
        "--context-k", str(args.context_k),
        "--llm_model", args.llm_model,
        "--base_url", args.base_url,
        "--api_key", args.api_key,
    ]
    if args.disable_thinking:
        generation_eval_cmd.append("--disable_thinking")
    if args.limit:
        generation_eval_cmd.extend(["--limit", str(args.limit)])
    
    timings['generation_eval'] = run_command(
        generation_eval_cmd,
        "STEP 4/4: Generation Evaluation"
    )
    
    total_time = time.time() - pipeline_start
    
    # Print summary
    print_results_summary(retrieval_eval_output, generation_eval_output, total_time)
    
    # Save pipeline metadata
    pipeline_metadata = {
        "timestamp": timestamp,
        "config": {
            "dataset_file": args.dataset_file,
            "memory_dir": args.memory_dir,
            "output_dir": args.output_dir,
            "max_workers": args.max_workers,
            "llm_model": args.llm_model,
            "embedding_model": args.embedding_model,
            "top_k": args.top_k,
            "context_k": args.context_k,
            "eval_ks": args.eval_ks,
            "disable_thinking": args.disable_thinking,
        },
        "timings": timings,
        "total_time": total_time,
        "output_files": {
            "retrieval_results": retrieval_output,
            "retrieval_evaluation": retrieval_eval_output,
            "generation_evaluation": generation_eval_output,
        }
    }
    
    metadata_file = os.path.join(args.output_dir, f"pipeline_metadata_{timestamp}.json")
    with open(metadata_file, 'w') as f:
        json.dump(pipeline_metadata, f, indent=2)
    
    print("\n📁 Output Files:")
    print(f"  - Retrieval Results:     {retrieval_output}")
    print(f"  - Retrieval Evaluation:  {retrieval_eval_output}")
    print(f"  - Generation Evaluation: {generation_eval_output}")
    print(f"  - Pipeline Metadata:     {metadata_file}")
    print("\n✅ Pipeline completed successfully!\n")


if __name__ == "__main__":
    main()


"""
Example usage:

python3 /home/vinhpq/mem_baseline/mem0/mem0_full_pipeline.py \
    /home/vinhpq/mem_baseline/locomo_dataset/processed_data/locomo_small.json \
    /home/vinhpq/mem_baseline/mem0/locomo_memory_benchmark \
    /home/vinhpq/mem_baseline/mem0/benchmark_results \
    --max_workers 2 \
    --llm_model Qwen/Qwen3-8B \
    --api_key dummy \
    --base_url http://localhost:8001/v1 \
    --embedding_model facebook/contriever \
    --top_k 100 \
    --context_k 5 \
    --eval_ks "3,5,10" \
    --disable_thinking

# For quick testing (limit to first N questions):
python3 /home/vinhpq/mem_baseline/mem0/mem0_full_pipeline.py \
    /home/vinhpq/mem_baseline/locomo_dataset/processed_data/locomo_small.json \
    /home/vinhpq/mem_baseline/mem0/locomo_memory_test \
    /home/vinhpq/mem_baseline/mem0/test_results \
    --max_workers 1 \
    --limit 10
"""

