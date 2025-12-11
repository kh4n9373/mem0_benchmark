
import argparse
import json
import os
import sys
from tqdm.auto import tqdm
from mem0 import Memory

def extract_content(msg):
    if isinstance(msg, dict):
        return msg['content']
    return str(msg)

def process_retrieval(
    input_file: str,
    memory_dir_base: str,
    output_file: str,
    model_name: str = "facebook/contriever",
    top_k: int = 100
):
    print(f"Loading queries from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    print(f"Loaded {len(dataset)} records")
    
    results = []
    
    # Process each user/session
    # We assume 'dataset' is list of records, maybe same format as input or the 'longmemeval' format?
    # The user used 'locomo_processed_data.json' for indexing.
    # For retrieval, user likely uses 'locomo_processed_data.json' (if it contains questions) 
    # OR a different query file.
    # Wait, the user command for retrieve in A-mem was:
    # process_retrieve.py .../locomo_processed_data.json ...
    # So it uses the same file which contains both dialogs (history) and questions?
    # Let's double check locomo_processed_data format if possible or assume standard behavior.
    
    # The A-mem process_retrieve loops over "dataset" and checks for 'questions' in each item.
    
    processed_count = 0
    
    for item in tqdm(dataset, desc="Retrieving"):
        conv_id = item.get("conv_id")
        questions = item.get("qas", [])
        
        if not questions:
            continue
            
        # Initialize Memory for this conversation
        conv_dir = os.path.join(memory_dir_base, conv_id)
        if not os.path.exists(conv_dir):
            if processed_count < 5: 
                print(f"Warning: Memory directory not found for {conv_id} at {conv_dir}")
            continue

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
                    "provider": "openai",
                    "config": {
                        "model": "gpt-4o-mini",
                         "api_key": "dummy"
                    }
                }
            }
            memory = Memory.from_config(config)
            
            for idx, q in enumerate(questions):
                query_text = q.get("question", "")
                qid = q.get("question_id", idx)
                evidences = q.get("evidences", [])
                answer = q.get("answer")
                
                # Retrieve
                search_results = memory.search(query_text, user_id=conv_id, limit=top_k)
                
                # Mem0.search returns dict {"results": [...]}
                actual_results = search_results.get("results", [])
                
                # Format results
                retrieved_chunks = []
                for res in actual_results:
                     retrieved_chunks.append({
                        "id": res.get("id"),
                        "content": metadata.get("original_content", res.get("memory")),  # 👈 ƯU TIÊN LẤY CONTENT GỐC
                        "score": res.get("score"),
                        "timestamp": metadata.get("timestamp")
                        # Metadata might be in 'metadata' key or top level depending on get_all vs search?
                        # In search, it returns OutputData -> parsed.
                        # OutputData has payload.
                        # Wait, main.py search returns formatted dicts.
                        # example: {'id': '...', 'memory': '...', 'score': 0.8, ...}
                        # Metadata is merged into the dict if it's promoted keys, else in 'metadata'?
                        # Let's assume 'metadata' key or check if timestamp is promoted. 
                        # promoted keys: user_id, agent_id, run_id, actor_id, role. 
                        # timestamp is likely in 'metadata'.
                    })


                results.append({
                    "question_id": qid,
                    "question": query_text,
                    "answer": answer,
                    "chunks": retrieved_chunks,
                    "evidences": evidences,
                    "category": q.get("category"), # Keep category for eval
                    "conv_id": conv_id
                })
                
                processed_count += 1
                
        except Exception as e:
            print(f"Error processing conv {conv_id}: {e}")
            continue

    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"Saved {len(results)} results to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("memory_dir")
    parser.add_argument("output_file")
    parser.add_argument("--model_name", default="facebook/contriever")
    parser.add_argument("--top_k", type=int, default=100)
    
    args = parser.parse_args()
    
    process_retrieval(
        args.input_file,
        args.memory_dir,
        args.output_file,
        model_name=args.model_name,
        top_k=args.top_k
    )

if __name__ == "__main__":
    main()
"""
python3 /home/vinhpq/mem_baseline/mem0/mem0_process_retrieve.py \
    /home/vinhpq/mem_baseline/locomo_dataset/processed_data/locomo_small.json \
    /home/vinhpq/mem_baseline/mem0/locomo_memory_small_llm \
    /home/vinhpq/mem_baseline/mem0/locomo_retrieve_result_llm.json 
"""