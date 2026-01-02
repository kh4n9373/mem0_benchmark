
import argparse
import json
import os
import sys
from pathlib import Path
from tqdm.auto import tqdm
from mem0 import Memory
import openai


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
        
        # Check if already completed
        # Note: We rely on completed.flag for total completion, but we also check individual progress below
        completed_flag = os.path.join(conv_dir, "completed.flag")
        if os.path.exists(completed_flag):
            print(f"\nSkipping conversation {conv_idx + 1}/{len(dataset)}: {conv_id} (Already indexed)")
            continue
        
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
                        "max_tokens": 4096,
                        "temperature": 0.0,
                    }
                }
            }
            
            if disable_thinking:
                config["custom_fact_extraction_prompt"] = (
                    "You are a helpful assistant that extracts facts and memories from conversations. "
                    "Your response MUST be a valid JSON object with a key 'facts' containing a list of strings. "
                    "Do NOT include any thinking, reasoning, or markdown code blocks. "
                    "Do NOT output anything other than the JSON object."
                )
            
            if base_url:
                config["llm"]["config"]["openai_base_url"] = base_url

            # Load granular progress
            progress_file = os.path.join(conv_dir, "indexing_progress.json")
            completed_indices = set()
            
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'r') as f:
                        saved_progress = json.load(f)
                        # Support legacy format (int) by converting to range, or new format (list)
                        if "last_indexed_chunk" in saved_progress:
                            # Migrate legacy to new format
                            last_idx = saved_progress.get("last_indexed_chunk", 0)
                            completed_indices = set(range(last_idx))
                        else:
                            completed_indices = set(saved_progress.get("completed_indices", []))
                            
                    if completed_indices:
                        print(f"  Resuming: {len(completed_indices)}/{len(chunks)} chunks already completed")
                except Exception as e:
                    print(f"  Warning: Could not read progress file: {e}")

            memory = Memory.from_config(config)
            
            failed_chunks_count = 0
            
            for i, chunk_data in enumerate(tqdm(chunks, desc=f"Adding chunks [{conv_id}]", leave=False)):
                if i in completed_indices:
                    continue
                    
                try:
                    content = chunk_data["content"]
                    timestamp = chunk_data["timestamp"]
                    
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
                    
                    # Mark as success
                    completed_indices.add(i)
                    
                    # Save progress immediately
                    with open(progress_file, 'w') as f:
                        json.dump({"completed_indices": list(completed_indices), "timestamp": timestamp}, f)
                    
                except (openai.APIConnectionError, openai.InternalServerError, openai.APITimeoutError, openai.RateLimitError) as e:
                    # TRANSIENT ERROR: Do NOT mark as completed. Will retry next run.
                    print(f"\n  ⚠️  Transient API Error processing chunk {i} for {conv_id}: {e}")
                    print("     -> Marking as FAILED (will retry next time)")
                    failed_chunks_count += 1
                    continue
                    
                except openai.AuthenticationError as e:
                    # FATAL ERROR: Stop everything
                    print(f"\n❌ FATAL Authentication Error: {e}")
                    sys.exit(1)
                    
                except Exception as e:
                    # CONTENT/LOGIC ERROR: Mark as completed (skip next time) per user request
                    print(f"\n  ⚠️  Content/Logic Error processing chunk {i} for {conv_id}: {e}")
                    print("     -> Marking as COMPLETED/SKIPPED (will NOT retry)")
                    
                    completed_indices.add(i)
                    with open(progress_file, 'w') as f:
                        json.dump({"completed_indices": list(completed_indices), "timestamp": timestamp}, f)
                    
                    # Optional: Could add to a separate "error_log" list in the file if desired, 
                    # but broadly treats as "done/skipped".
            
            # Report status for this conversation
            if failed_chunks_count > 0:
                print(f"  ⚠️  Finished with {failed_chunks_count} FAILED chunks (Transient errors).")
                print("      Run script again to retry these chunks.")
            else:
                # Only write completed flag if ALL chunks are accounted for (completed or skipped-as-completed)
                # AND we didn't have transient failures that caused us to skip the loop
                if len(completed_indices) == len(chunks):
                    with open(completed_flag, "w") as f:
                        f.write("completed")
                    print(f"  ✅ Conversation completed successfully ({len(chunks)} chunks)")
            
            # Save metadata
            metadata = {
                "conv_id": conv_id,
                "num_chunks": len(chunks),
                "num_indexed": len(completed_indices),
                "persist_directory": conv_dir,
                "model_name": model_name,
                "status": "partial" if failed_chunks_count > 0 else "complete"
            }
            index_metadata.append(metadata)
            
        except Exception as e:
            print(f"  Failed to initialize/process conversation {conv_id}: {e}")
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