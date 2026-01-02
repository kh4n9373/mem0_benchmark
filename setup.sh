#!/bin/bash
# Mem0 Benchmark Setup Script
# Simple setup: Install dependencies and download dataset

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "🔧 Mem0 Benchmark Setup"
echo "============================================================"
echo ""

# 1. Check Python version
echo "▶ Checking Python..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "   Python version: $PYTHON_VERSION"

if [[ $(echo "$PYTHON_VERSION < 3.8" | bc -l) -eq 1 ]]; then
    echo "❌ Python 3.8+ required (found $PYTHON_VERSION)"
    exit 1
fi
echo "✅ Python version OK"
echo ""

# 2. Check conda environment (optional but recommended)
echo "▶ Checking environment..."
if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
    echo "   Conda environment: $CONDA_DEFAULT_ENV"
    if [[ "$CONDA_DEFAULT_ENV" != "mem0" ]]; then
        echo "⚠️  Not in 'mem0' environment. Consider:"
        echo "   conda create -n mem0 python=3.10"
        echo "   conda activate mem0"
        echo ""
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo "✅ In mem0 conda environment"
    fi
else
    echo "   No conda environment detected (using system Python)"
    echo "⚠️  Recommended: Create a conda environment first"
    echo ""
fi
echo ""

# 3. Install Python dependencies
echo "▶ Installing Python packages..."
echo "   This may take a few minutes..."
echo ""

# Set TMPDIR to avoid disk space issues on small system partitions
export TMPDIR=$(pwd)/.tmp
export TEMP=$TMPDIR
export TMP=$TMPDIR
mkdir -p "$TMPDIR"

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "⬇️  Installing uv for faster installation..."
    pip install uv --quiet
fi

if [ -f "requirements.txt" ]; then
    echo "⚡ Using uv for fast installation..."
    uv pip install -r requirements.txt --quiet --system
    if [ $? -eq 0 ]; then
        echo "✅ Python packages installed from requirements.txt"
    else
        echo "❌ Failed to install packages from requirements.txt"
        echo "   Try manually: uv pip install -r requirements.txt --system"
        exit 1
    fi
else
    # Fallback: install essential packages individually
    echo "⚠️  requirements.txt not found. Installing essential packages..."
    uv pip install mem0ai openai chromadb sentence-transformers huggingface-hub \
                rouge-score sacrebleu bert-score tqdm --quiet --system
    echo "✅ Essential packages installed"
fi
echo ""

# 4. Download dataset
echo "▶ Downloading dataset..."
if [ -d "data/locomo/processed_data" ] && [ -f "data/locomo/processed_data/locomo_small.json" ]; then
    echo "✅ Dataset already exists at: data/locomo/processed_data/locomo_small.json"
else
    echo "   Downloading from HuggingFace (KhangPTT373/locomo)..."
    mkdir -p data
    
    python3 <<'EOF'
from huggingface_hub import snapshot_download
import os

try:
    snapshot_download(
        repo_id="KhangPTT373/locomo",
        local_dir="data/locomo",
        repo_type="dataset"
    )
    print("✅ Dataset downloaded successfully!")
except Exception as e:
    print(f"❌ Failed to download dataset: {e}")
    print("   Please check your internet connection and try again")
    exit(1)
EOF
    
    if [ $? -ne 0 ]; then
        exit 1
    fi
    
    # Verify download
    if [ -f "data/locomo/processed_data/locomo_processed_data.json" ]; then
        echo "✅ Dataset verified: data/locomo/processed_data/locomo_processed_data.json"
    else
        echo "❌ Dataset file not found"
        echo "   Expected: data/locomo/processed_data/locomo_processed_data.json"
        exit 1
    fi
fi
echo ""

# 5. Create necessary directories
echo "▶ Creating directories..."
mkdir -p worker_logs
mkdir -p benchmark_results
mkdir -p test_results
echo "✅ Directories created"
echo ""

# 6. Verify installation
echo "▶ Verifying installation..."
python3 -c "import mem0; print('✅ mem0ai')" 2>/dev/null || echo "⚠️  mem0ai not found"
python3 -c "import openai; print('✅ openai')" 2>/dev/null || echo "⚠️  openai not found"
python3 -c "import chromadb; print('✅ chromadb')" 2>/dev/null || echo "⚠️  chromadb not found"
python3 -c "import rouge_score; print('✅ rouge_score')" 2>/dev/null || echo "⚠️  rouge_score not found"
python3 -c "import bert_score; print('✅ bert_score')" 2>/dev/null || echo "⚠️  bert_score not found"
echo ""

# 7. Summary
echo "============================================================"
echo "✅ Setup completed!"
echo "============================================================"
echo ""
echo "📋 What was done:"
echo "  ✓ Python dependencies installed (using uv)"
echo "  ✓ LoCoMo dataset downloaded to data/locomo/"
echo "  ✓ Directories created (worker_logs/, benchmark_results/, etc.)"
echo ""
echo "📝 Next steps:"
echo ""
echo "  1. (Optional) Start LLM server:"
echo "     vllm serve <your-model> --port 8001"
echo ""
echo "  2. Run quick test (20 questions, ~5-10 min):"
echo "     ./quick_test.sh"
echo ""
echo "  3. Or run full benchmark (199 questions, ~45-60 min):"
echo "     ./full_benchmark_{dataset_name}.sh"
echo ""
echo "💡 Tips:"
echo "  - Edit *.sh files to change LLM model/server settings"
echo "  - Check worker_logs/ if indexing fails"
echo "  - Results saved to benchmark_results/ or test_results/"
echo ""
echo "📚 Documentation:"
echo "  - GETTING_STARTED.md - Quick start guide"
echo "  - README.md          - Full documentation"
echo ""
