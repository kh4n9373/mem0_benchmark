# 🚀 Mem0 Benchmark Setup Guide

Simple step-by-step guide to run Mem0 benchmarks.

---

## Prerequisites

### Hardware
- **GPU**: 1x 40GB VRAM (e.g., A100) or 2x 24GB VRAM (e.g., RTX 3090)
- **RAM**: 32GB+
- **Storage**: 50GB free

### Software
- Python 3.9-3.11
- CUDA 11.8+ (`nvidia-smi`)
- Docker
- Git

---

## Setup & Run (5 Steps)

### Step 1: Install vLLM

```bash
conda create -n vllm python=3.10
conda activate vllm
pip install vllm
```

### Step 2: Clone Repository

```bash
git clone <your-repo-url>
cd mem0
```

### Step 3: Start LLM Server (Terminal 1)

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 8002 \
  --gpu-memory-utilization 0.9 \
  --tool-call-parser hermes \
  --enable-auto-tool-choice
```

**Wait until**: `Uvicorn running on http://0.0.0.0:8002` appears.

### Step 4: Run Benchmark (Terminal 2)

```bash
cd mem0
bash ./full_benchmark_locomo.sh
```

**What it does**:
- Auto-installs all dependencies
- Auto-starts Neo4j
- **Auto-downloads dataset from HuggingFace** (~3GB, first run only)
- Auto-downloads embedding model (~2GB, first run only)
- Runs full benchmark (10-30 min)

### Step 5: View Results

```bash
cat locomo_results_benchmark/retrieval_eval_*.json
```

---

## Alternative: LongMemEval Benchmark

```bash
bash ./full_benchmark_longmemeval.sh
```

---

## Configuration

### Use Different GPU

Edit `full_benchmark_locomo.sh` line 9:

```bash
export CUDA_VISIBLE_DEVICES=1  # Change to 0, 1, etc.
```

### Use Better Model (More Stable)

Restart LLM server with 14B:

```bash
pkill -f "vllm serve"

CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-14B-Instruct \
  --port 8002 \
  --gpu-memory-utilization 0.85 \
  --tool-call-parser hermes \
  --enable-auto-tool-choice
```

---

## Pre-flight Check (Optional)

```bash
./preflight_check.sh
```

Checks: Python, GPU, Docker, LLM server, Neo4j.

---

## Files & Results

### Scripts
- `full_benchmark_locomo.sh` - Run LOCOMO benchmark
- `full_benchmark_longmemeval.sh` - Run LongMemEval benchmark
- `preflight_check.sh` - System checks
- `manage_services.sh` - Start/stop Neo4j

### Results
```
locomo_results_benchmark/
├── retrieval_eval_*.json      # Metrics here!
├── retrieval_results_*.json
├── generation_eval_*.json
└── pipeline_metadata_*.json
```

---

## Quick Reference

### Minimal Commands

```bash
# Terminal 1: Start LLM
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 8002 --tool-call-parser hermes --enable-auto-tool-choice

# Terminal 2: Run Benchmark
cd mem0 && bash ./full_benchmark_locomo.sh

# View Results
cat locomo_results_benchmark/retrieval_eval_*.json
```

---

**Done!** 🎉

For troubleshooting, check:
- `./preflight_check.sh` - System status
- `worker_logs/locomo/worker_0.log` - Detailed logs
