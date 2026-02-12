# Mem0 Benchmark

Run Mem0 benchmarks on LOCOMO and LongMemEval datasets.

---

## Quick Start

### 1. Start LLM Server

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-7B-Instruct \
  --port 8002 \
  --gpu-memory-utilization 0.9 \
  --tool-call-parser openai \
  --enable-auto-tool-choice
```

### 2. Run Benchmark

```bash
bash ./full_benchmark_locomo.sh
```

**First run**: Auto-downloads dataset (~3GB) + embedding model (~2GB).

### 3. View Results

```bash
cat locomo_results_benchmark/retrieval_eval_*.json
```

---

## Prerequisites

- **GPU**: 40GB VRAM (or 2x 24GB)
- **CUDA**: 11.8+
- **Docker**: For Neo4j
- **vLLM**: `pip install vllm`

---

## Files

- `full_benchmark_locomo.sh` - Run LOCOMO benchmark
- `full_benchmark_longmemeval.sh` - Run LongMemEval benchmark
- `preflight_check.sh` - Check system requirements
- `manage_services.sh` - Manage Neo4j service
- `SETUP_GUIDE.md` - Detailed setup instructions

---

## Results

```
locomo_results_benchmark/
├── retrieval_eval_*.json      ← Main metrics here!
├── retrieval_results_*.json
├── generation_eval_*.json
└── pipeline_metadata_*.json
```

---

## Troubleshooting

```bash
# Check system status
./preflight_check.sh

# View logs
tail -f worker_logs/locomo/worker_0.log

# Restart Neo4j
./manage_services.sh neo4j-restart
```

---

For detailed setup instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md).
