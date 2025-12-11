
import argparse
import subprocess
import sys
import time
from typing import List
import os

def run_parallel_indexing():
    parser = argparse.ArgumentParser(description="Run parallel indexing for Mem0")
    
    # Parallel processing args
    parser.add_argument("--max_workers", type=int, default=1, help="Number of parallel workers")
    
    # Pass-through args
    args, unknown_args = parser.parse_known_args()
    
    workers: List[subprocess.Popen] = []
    
    print(f"Starting {args.max_workers} parallel workers...")
    start_time = time.time()
    
    # Path to the actual indexing script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "mem0_process_index.py")
    
    # Create logs directory in script directory
    logs_dir = os.path.join(script_dir, "worker_logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    try:
        for i in range(args.max_workers):
            # Construct command for this worker
            python_executable = sys.executable
                
            cmd = [
                python_executable,
                script_path,
            ] + unknown_args + [
                "--num_shards", str(args.max_workers),
                "--shard_id", str(i)
            ]
            
            log_file = os.path.join(logs_dir, f"worker_{i}.log")
            print(f"Starting worker {i+1}/{args.max_workers} -> logging to {log_file}")
            
            # Redirect stdout and stderr to file
            with open(log_file, "w") as f:
                process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            workers.append(process)
            
        print("\nAll workers started.")
        print(f"To view progress, run: tail -f {logs_dir}/worker_*.log")
        print("Waiting for completion...")
        
        # Wait for all workers to complete
        exit_codes = []
        for p in workers:
            exit_codes.append(p.wait())
            
        duration = time.time() - start_time
        print(f"\nAll workers completed in {duration:.2f}s")
        
        # Check if any worker failed
        if any(code != 0 for code in exit_codes):
            print("WARNING: Some workers failed!")
            sys.exit(1)
        else:
            print("SUCCESS: All workers completed successfully.")
            
    except KeyboardInterrupt:
        print("\nStopping all workers...")
        for p in workers:
            p.terminate()
        sys.exit(1)

if __name__ == "__main__":
    run_parallel_indexing()

"""
python3 /home/vinhpq/mem_baseline/mem0/mem0_parallel_index.py \
    /home/vinhpq/mem_baseline/locomo_dataset/processed_data/locomo_small.json \
    /home/vinhpq/mem_baseline/mem0/locomo_memory_small_llm \
    --max_workers 2 \
    --model_name facebook/contriever \
    --llm_backend openai \
    --llm_model Qwen/Qwen3-8B \
    --api_key dummy \
    --base_url http://localhost:8001/v1 \
    --disable_thinking
"""