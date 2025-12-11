# Installation

## Automatic (Recommended)

```bash
conda create -n mem0 python=3.10
conda activate mem0
cd mem0/
./setup.sh
```

Done! Now run `./quick_test.sh`

## Manual Installation

### 1. Create environment

```bash
conda create -n mem0 python=3.10
conda activate mem0
cd mem0/
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or essential packages only:
```bash
pip install mem0ai openai chromadb sentence-transformers \
            huggingface-hub rouge-score sacrebleu bert-score tqdm
```

### 3. Download dataset

```bash
python3 <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="KhangPTT373/locomo",
    local_dir="data/locomo",
    repo_type="dataset"
)
EOF
```

### 4. Create directories

```bash
mkdir -p worker_logs benchmark_results test_results
```

### 5. Verify

```bash
python3 -c "import mem0; print('✅ Installed')"
ls data/locomo/processed_data/locomo_small.json
```

## System Requirements

- **Python**: 3.8+ (3.10 recommended)
- **Disk**: ~5 GB (dataset + dependencies)
- **RAM**: 8 GB minimum (16 GB recommended)
- **OS**: Linux, macOS, Windows (WSL)

## Troubleshooting

**"conda: command not found"**
```bash
# Install Miniconda first
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

**"pip install fails"**
```bash
# Try updating pip
pip install --upgrade pip
pip install -r requirements.txt
```

**"Dataset download fails"**
```bash
# Check internet, retry
rm -rf data/locomo
./setup.sh
```

**Package conflicts**
```bash
# Fresh environment
conda env remove -n mem0
conda create -n mem0 python=3.10
conda activate mem0
./setup.sh
```

## Next Steps

After installation:

1. **(Optional)** Start LLM server:
   ```bash
   vllm serve Qwen/Qwen2.5-3B-Instruct --port 8001
   ```

2. Run quick test:
   ```bash
   ./quick_test.sh
   ```

3. Check results:
   ```bash
   ls test_results_pipeline/*.json
   ```

---

Need more info? See `README.md`
