
import os
from mem0 import Memory

# Configuration mirroring the benchmark
config = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "test_memories",
            "path": "./test_memory_db",
        }
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "BAAI/bge-m3",
            "model_kwargs": {"local_files_only": True}
        }
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "Qwen/Qwen3-8B",
            "api_key": "dummy",
            "openai_base_url": "http://localhost:8002/v1",

        }
    }
}

print("Initializing Memory...")
try:
    memory = Memory.from_config(config)
    print("Memory initialized.")
except Exception as e:
    print(f"Failed to initialize Memory: {e}")
    exit(1)

print("Adding test memory (should trigger LLM)...")
try:
    # Adding a memory triggers LLM to summarize/extract
    result = memory.add("I am working on benchmarking Mem0 with a local LLM.", user_id="test_user")
    print("\n✅ Success! Result:")
    print(result)
except Exception as e:
    print(f"\n❌ Error adding memory: {e}")
