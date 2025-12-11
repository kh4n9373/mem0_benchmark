
import argparse
import json
import os
import sys
from pathlib import Path
from tqdm.auto import tqdm
from mem0 import Memory


def process_indexing(
    input_file: str,
    base_output_dir: str,
    model_name: str = "facebook/contriever",
    llm_backend: str = "openai",
    llm_model: str = "gpt-4o-mini",
    api_key: str = "dummy",
    base_url: str = None,
    disable_thinking: bool = False, # Kept for API compat, though assumed true by user context
    num_shards: int = 1,
    shard_id: int = 0,
):
    """Create and index memory systems for each conversation using Mem0."""
    
    print(f"Loading dataset from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} conversations")
    
    # Create base output directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Store metadata about indexed conversations
    index_metadata = []
    
    for conv_idx, conversation in enumerate(tqdm(dataset, desc=f"Indexing (shard {shard_id}/{num_shards})")):
        # Sharding check
        if conv_idx % num_shards != shard_id:
            continue
            
        conv_id = conversation.get("conv_id", f"conv_{conv_idx}")
        
        chunks = []
        dialogs = conversation.get('dialogs')
        
        # Extract chunks with separated timestamp
        for session in dialogs:
            timestamp = session['datetime']
            for message in session['messages']:
                chunks.append({
                    "content": message['content'],
                    "timestamp": timestamp
                })
        
        # Create a separate directory for each conversation's memory
        conv_dir = os.path.join(base_output_dir, conv_id)
        os.makedirs(conv_dir, exist_ok=True)
        
        print(f"\nProcessing conversation {conv_idx + 1}/{len(dataset)}: {conv_id}")
        print(f"  Chunks to add: {len(chunks)}")
        
        try:
            # Configure Mem0 Memory
            config = {
                "vector_store": {
                    "provider": "chroma",
                    "config": {
                        "collection_name": "memories",
                        "path": conv_dir,
                    }
                },
                "embedder": {
                     "provider": "huggingface",
                     "config": {
                         "model": model_name
                     }
                },
                 "llm": {
                    "provider": llm_backend,
                    "config": {
                        "model": llm_model,
                        "api_key": api_key, # "dummy" or real key
                    }
                }
            }
            
            if base_url:
                config["llm"]["config"]["openai_base_url"] = base_url

            memory = Memory.from_config(config)
            
            added_count = 0
            for i, chunk_data in enumerate(tqdm(chunks, desc=f"Adding chunks [{conv_id}]", leave=False)):
                try:
                    content = chunk_data["content"]
                    timestamp = chunk_data["timestamp"]
                     # Truncate if too long
                    # chunk_text = content[:2000] if len(content) > 2000 else content
                    chunk_text_with_time = f"[Timestamp: {timestamp}]\n{content}"
                    chunk_text_with_time = chunk_text_with_time[:2000] if len(chunk_text_with_time) > 2000 else chunk_text_with_time

                    memory.add(
                        chunk_text_with_time, 
                        user_id=conv_id, 
                        metadata={
                            "timestamp": timestamp,
                            "original_content": content
                        },
                        infer=True 
                    )
                    added_count += 1
                except Exception as e:
                    print(f"  Error adding chunk {i} for {conv_id}: {e}")
                    continue
            
            # Save metadata
            metadata = {
                "conv_id": conv_id,
                "num_chunks": len(chunks),
                "num_indexed": added_count,
                "persist_directory": conv_dir,
                "model_name": model_name,
            }
            index_metadata.append(metadata)
            
            print(f"  Successfully indexed {added_count}/{len(chunks)} chunks")
            
        except Exception as e:
            print(f"  Failed to index conversation {conv_id}: {e}")
            continue
    
    # Save index metadata
    metadata_file = os.path.join(base_output_dir, f"index_metadata_{shard_id}.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(index_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Indexing complete (Shard {shard_id})!")
    print(f"   Indexed {len(index_metadata)} conversations")
    print(f"   Metadata saved to: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description="Index locomo dataset with Mem0")
    parser.add_argument("input_file", help="Path to input JSON dataset")
    parser.add_argument("output_dir", help="Base directory to save memory systems")
    parser.add_argument("--model_name", default="facebook/contriever", help="Embedding model")
    
    # LLM Args
    parser.add_argument("--llm_backend", default="openai", help="LLM backend")
    parser.add_argument("--llm_model", default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--api_key", default="dummy", help="API key")
    parser.add_argument("--base_url", default=None, help="LLM API base URL")
    parser.add_argument("--disable_thinking", action="store_true", help="Disable thinking (not used but kept for compat)")
    
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--shard_id", type=int, default=0, help="Shard ID")

    args = parser.parse_args()
    
    try:
        process_indexing(
            input_file=args.input_file,
            base_output_dir=args.output_dir,
            model_name=args.model_name,
            llm_backend=args.llm_backend,
            llm_model=args.llm_model,
            api_key=args.api_key,
            base_url=args.base_url,
            disable_thinking=args.disable_thinking,
            num_shards=args.num_shards,
            shard_id=args.shard_id,
        )
    except Exception as e:
        print(f"Error during indexing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
"""
python3 /home/vinhpq/mem_baseline/mem0/mem0_parallel_index.py \
    /home/vinhpq/mem_baseline/locomo_dataset/processed_data/locomo_processed_data.json \
    /home/vinhpq/mem_baseline/mem0/locomo_memory \
    --max_workers 5 \
    --model_name facebook/contriever \
    --llm_backend openai \
    --llm_model qwen3-8b \
    --api_key dummy \
    --base_url https://120aa1600a49.ngrok-free.app/v1 \
    --disable_thinking
"""