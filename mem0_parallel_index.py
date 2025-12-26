
import argparse
import subprocess
import sys
import time
from typing import List
import os

import threading

def run_parallel_indexing():
    parser = argparse.ArgumentParser(description="Run parallel indexing for Mem0")
    
    # Parallel processing args
    parser.add_argument("--max_workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--log_dir", default=None, help="Directory to store worker logs")
    parser.add_argument("--stream_logs", action="store_true", help="Stream worker logs to console")
    
    # Pass-through args
    args, unknown_args = parser.parse_known_args()
    
    workers: List[subprocess.Popen] = []
    
    print(f"Starting {args.max_workers} parallel workers...")
    start_time = time.time()
    
    # Path to the actual indexing script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "mem0_process_index.py")
    
    # Create logs directory
    if args.log_dir:
        logs_dir = os.path.abspath(args.log_dir)
    else:
        logs_dir = os.path.join(script_dir, "worker_logs")
    
    os.makedirs(logs_dir, exist_ok=True)
    
    def log_streamer(proc, log_path, worker_id):
        """Reads stdout/stderr from process and writes to both file and console."""
        with open(log_path, "w") as f:
            for line in proc.stdout:
                # Write to file
                f.write(line)
                f.flush()
                # Write to console with worker prefix
                sys.stdout.write(f"[{worker_id}] {line}")
                sys.stdout.flush()

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
            
            if args.stream_logs:
                # Pipe mode for streaming
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1 # Line buffered
                )
                
                # Start streamer thread
                t = threading.Thread(target=log_streamer, args=(process, log_file, i))
                t.daemon = True
                t.start()
                
            else:
                # File redirection mode (original)
                with open(log_file, "w") as f:
                    process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            
            workers.append(process)
            
        print("\nAll workers started.")
        print(f"To view progress, run: tail -f {logs_dir}/worker_*.log")
        print("Waiting for completion...")
        
        # Wait for workers with polling (Fail-Fast)
        while True:
            all_finished = True
            any_failed = False
            
            for p in workers:
                ret = p.poll()
                if ret is None:
                    all_finished = False
                elif ret != 0:
                    any_failed = True
                    break # Break inner loop
            
            if any_failed:
                print("\n❌ A worker failed! Terminating all other workers...")
                for p in workers:
                    if p.poll() is None:
                        p.terminate()
                sys.exit(1)
            
            if all_finished:
                break
                
            time.sleep(1) # Poll every second
            
        duration = time.time() - start_time
        print(f"\nAll workers completed in {duration:.2f}s")
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