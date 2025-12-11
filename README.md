# Mem0 Benchmark

Benchmark Mem0 on LoCoMo dataset: Index → Retrieve → Evaluate (retrieval + generation).

## Quick Start

```bash
# 1. Setup
conda create -n mem0 python=3.10
conda activate mem0
cd mem0/
./setup.sh

# 2. Run
./quick_test.sh              # Quick test (20 questions, ~10 min)
./full_benchmark_locomo.sh          # Full benchmark for locomo
./full_benchmark_longmemeval.sh     # Or benchmark for longmemeval
```

## What It Does

```
Dataset (conversations + questions)
    ↓
[1] Index with LLM summarization
    ↓
[2] Retrieve top-K chunks for each question
    ↓
[3] Evaluate retrieval (Precision, Recall, F1, nDCG)
    ↓
[4] Generate answers with LLM + retrieved context
    ↓
[5] Evaluate generation (F1, BLEU, ROUGE, BERTScore)
    ↓
Results (JSON + terminal summary)
```

## Files

```
mem0/
├── setup.sh                 Setup everything
├── quick_test.sh           Quick test
├── full_benchmark_{dataset_name}.sh       Full benchmark
│
├── mem0_full_pipeline.py   Main orchestrator
├── mem0_parallel_index.py  Parallel indexing
├── mem0_process_index.py   Indexing logic
├── mem0_process_retrieve.py Retrieval logic
│
├── requirements.txt        Dependencies
├── README.md               This file
└── INSTALL.md              Detailed install guide
```

## Output

Results saved to `benchmark_results/` or `test_results_pipeline/`:

- `retrieval_results_*.json` - Raw retrieval data
- `retrieval_eval_*.json` - Retrieval metrics
- `generation_eval_*.json` - Generation metrics (main results)
- `pipeline_metadata_*.json` - Config + timings

## Key Metrics

**Generation (most important):**
- **BERTScore** (0.85+) - Semantic similarity with ground truth
- **F1** (0.1-0.3) - Token overlap (lower with LLM summarization, normal)
- **ROUGE** (0.1-0.2) - N-gram overlap

**Retrieval:**
- **Precision** - % retrieved chunks that are relevant
- **Recall** - % ground truth evidences found
- **nDCG** - Ranking quality

## Configuration

Edit `quick_test.sh` or `full_benchmark_{dataset}.sh` to change:

```bash
--llm_model Qwen/Qwen3-8B           # LLM model name
--base_url http://localhost:8001/v1 # LLM server URL
--max_workers 2                      # Parallel indexing workers
--top_k 100                          # Retrieve top-K chunks
--context_k 5                        # Use top-K for LLM context
```

## Requirements

- Python 3.8+
- ~5 GB disk space
- LLM server (vLLM, OpenAI API, or compatible)

**Optional:** Start LLM server before running:
```bash
vllm serve Qwen/Qwen2.5-3B-Instruct --port 8001
```

## Troubleshooting

**Setup fails?**
```bash
pip install -r requirements.txt  # Manual install
```

**Dataset missing?**
```bash
rm -rf data/locomo && ./setup.sh  # Re-download
```

**Benchmark fails?**
```bash
# Check worker logs
tail -f worker_logs/worker_0.log

# Check LLM server
curl http://localhost:8001/v1/models
```

**Need help?** See `INSTALL.md` for detailed installation guide.

## Dependencies

Installed by `setup.sh` from `requirements.txt`:

- mem0ai - Memory framework
- openai - LLM client
- chromadb - Vector store
- sentence-transformers - Embeddings
- rouge-score, sacrebleu, bert-score - Metrics

**External (must be in parent directory):**
- `../retrieval_evaluator.py`
- `../generation_evaluator.py`

---

**Version**: 1.1 | **Dataset**: LoCoMo (KhangPTT373/locomo)
