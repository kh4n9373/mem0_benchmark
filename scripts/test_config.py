
from mem0 import Memory
import os

print("Testing Memory initialization...")

# Minimal config based on what we are using
config = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "test_memories",
            "path": "test_db",
        }
    },
    "embedder": {
            "provider": "huggingface",
            "config": {
                "model": "facebook/contriever"
            }
    },
        "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
                "api_key": "dummy"
        }
    }
}

try:
    print("Initializing Memory from config...")
    memory = Memory.from_config(config)
    print("Memory initialized successfully!")
    
    # Test simple add
    print("Adding memory...")
    memory.add("This is a test memory", user_id="test_user", infer=False)
    print("Memory added successfully!")

except Exception as e:
    print(f"FAILED with error: {e}")
    import traceback
    traceback.print_exc()
