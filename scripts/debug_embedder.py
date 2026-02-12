import os
import sys
import mem0
from mem0.embeddings.huggingface import HuggingFaceEmbedding
from mem0.configs.embeddings.base import BaseEmbedderConfig

print(f"mem0 location: {mem0.__file__}")

config = BaseEmbedderConfig(
    model="BAAI/bge-m3", 
    model_kwargs={"local_files_only": True}
)
print(f"Config huggingface_base_url: {config.huggingface_base_url}")
print(f"Config model_kwargs: {config.model_kwargs}")

embedder = HuggingFaceEmbedding(config)
print(f"Embedder client: {getattr(embedder, 'client', 'None')}")
print(f"Embedder model type: {type(embedder.model)}")

# Test embedding
try:
    vec = embedder.embed("test")
    print(f"Embedding success, vector len: {len(vec)}")
except Exception as e:
    print(f"Embedding failed: {e}")
