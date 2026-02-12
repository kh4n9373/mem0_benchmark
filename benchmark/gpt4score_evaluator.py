#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT4SCORE Evaluator for Retrieval Results

Takes retrieval results, generates answers using top-k chunks,
and evaluates with LLM-as-judge to compute GPT4SCORE accuracy.
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any
from collections import defaultdict
from datetime import datetime
from tqdm.auto import tqdm

# Import LLM judge components
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from llm_judge_memtree import AsyncLLMJudge, parse_judge_response

# Judge prompt template (same as llm_judge_memtree.py)
JUDGE_PROMPT = """[User Question]
{question}

[The Start of Reference Answer]
{answer}
[The End of Reference Answer]

[The Start of Model's Response]
{response}
[The End of Model's Response]

Is the model response correct? Answer [[yes]] or [[no]] only."""


def load_retrieval_results(input_file: str) -> List[Dict]:
    """Load retrieval results from JSON file."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def format_context_from_chunks(chunks: List[Dict], context_k: int) -> str:
    """Format context from top-k chunks."""
    top_chunks = chunks[:context_k]
    
    context_parts = []
    for i, chunk in enumerate(top_chunks, 1):
        content = chunk.get('content', '')
        context_parts.append(f"[Chunk {i}]\n{content}")
    
    return "\n\n".join(context_parts)


def generate_answer_from_chunks(
    question: str,
    chunks: List[Dict],
    context_k: int,
    llm_client: Any,
    llm_model: str,
    disable_thinking: bool = True,
) -> str:
    """
    Generate answer using top-k chunks via LLM.
    
    Args:
        question: User question
        chunks: Retrieved chunks with scores
        context_k: Number of top chunks to use
        llm_client: OpenAI-compatible client
        llm_model: Model name to use
        disable_thinking: Whether to disable thinking in LLM
        
    Returns:
        Generated answer string
    """
    # Format context
    context = format_context_from_chunks(chunks, context_k)
    
    # Create prompt
    system_prompt = """You are a helpful assistant. Answer the user's question based on the provided context chunks.
If the answer is not in the context, say "I don't have enough information to answer this question."
Be concise and accurate."""
    
    user_prompt = f"""Context:
{context}

Question: {question}

Please provide a concise answer based on the context above."""
    
    # Call LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    extra_body = {}
    if disable_thinking:
        extra_body["enable_thinking"] = False
    
    try:
        response = llm_client.chat.completions.create(
            model=llm_model,  # Use actual model name from parameter
            messages=messages,
            temperature=0.0,
            max_tokens=512,
            extra_body=extra_body if extra_body else None,
        )
        
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"Error generating answer: {e}")
        return ""



def prepare_judge_tasks_from_retrieval(
    retrieval_results: List[Dict],
    generated_answers: Dict[str, str],
) -> List[Dict]:
    """
    Prepare judge tasks from retrieval results and generated answers.
    
    Args:
        retrieval_results: List of retrieval result dicts
        generated_answers: Dict mapping question_id to generated answer
        
    Returns:
        List of judge task dicts
    """
    tasks = []
    
    for result in retrieval_results:
        question_id = result.get("question_id", "")
        question = result.get("question", "")
        ground_truth = result.get("answer")
        category = str(result.get("category", "unknown"))
        
        predicted_answer = generated_answers.get(question_id, "")
        
        # Skip if ground_truth is null or empty
        if ground_truth is None or not predicted_answer:
            continue
        
        ground_truth = str(ground_truth).strip()
        predicted_answer = str(predicted_answer).strip()
        
        if not ground_truth or not predicted_answer:
            continue
        
        # Format judge prompt
        prompt = JUDGE_PROMPT.format(
            question=question,
            answer=ground_truth,
            response=predicted_answer,
        )
        
        tasks.append({
            "question_id": question_id,
            "category": category,
            "question": question,
            "ground_truth": ground_truth,
            "predicted_answer": predicted_answer,
            "prompt": prompt,
        })
    
    return tasks


def calculate_gpt4score(results: List[Dict]) -> Dict[str, Any]:
    """
    Calculate GPT4SCORE accuracy metrics from judge results.
    
    Args:
        results: List of judge result dicts
        
    Returns:
        Dict with overall and by-category accuracy
    """
    # Overall
    all_scores = [r["judge_score"] for r in results]
    overall_acc = sum(all_scores) / len(all_scores) if all_scores else 0.0
    
    # By category
    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r["judge_score"])
    
    acc_by_category = {
        cat: sum(scores) / len(scores) if scores else 0.0
        for cat, scores in by_category.items()
    }
    
    return {
        "overall": {"accuracy": overall_acc, "count": len(all_scores)},
        "by_category": {
            cat: {"accuracy": acc, "count": len(by_category[cat])}
            for cat, acc in acc_by_category.items()
        },
    }


async def evaluate_with_gpt4score_async(
    input_file: str,
    output_file: str,
    context_k: int,
    judge_model: str,
    llm_model: str,
    api_key: str,
    base_url: str,
    judge_api_key: str,
    judge_base_url: str,
    max_concurrent: int,
    disable_thinking: bool,
):
    """
    Main async function to evaluate retrieval results with GPT4SCORE.
    """
    from openai import OpenAI
    
    print(f"\n{'='*60}")
    print(f"GPT4SCORE EVALUATION")
    print(f"{'='*60}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Context-K: {context_k}")
    print(f"LLM Model: {llm_model}")
    print(f"Judge Model: {judge_model}")
    print(f"Max Concurrent: {max_concurrent}")
    print(f"{'='*60}\n")
    
    # Load retrieval results
    print("📥 Loading retrieval results...")
    retrieval_results = load_retrieval_results(input_file)
    print(f"   Loaded {len(retrieval_results)} questions\n")
    
    # Initialize async LLM client for generation
    print("🤖 Initializing LLM client for answer generation...")
    from openai import AsyncOpenAI
    async_llm_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    
    # Generate answers using top-k chunks (async batch)
    print(f"✍️  Generating answers using top-{context_k} chunks...")
    
    async def generate_one(result):
        """Generate answer for one question."""
        question_id = result.get("question_id", "")
        question = result.get("question", "")
        chunks = result.get("chunks", [])
        
        if not chunks:
            return question_id, ""
        
        # Format context
        context = format_context_from_chunks(chunks, context_k)
        
        # Create prompt
        system_prompt = """You are a helpful assistant. Answer the user's question based on the provided context chunks.
If the answer is not in the context, say "I don't have enough information to answer this question."
Be concise and accurate."""
        
        user_prompt = f"""Context:
{context}

Question: {question}

Please provide a concise answer based on the context above."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        extra_body = {}
        if disable_thinking:
            extra_body["enable_thinking"] = False
        
        try:
            response = await async_llm_client.chat.completions.create(
                model=llm_model,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
                extra_body=extra_body if extra_body else None,
            )
            answer = response.choices[0].message.content.strip()
            return question_id, answer
        except Exception as e:
            print(f"Error generating answer for {question_id}: {e}")
            return question_id, ""
    
    # Process in batches with progress bar
    import asyncio
    batch_size = max_concurrent
    generated_answers = {}
    
    for i in tqdm(range(0, len(retrieval_results), batch_size), desc="Generating answers (batched)"):
        batch = retrieval_results[i:i+batch_size]
        tasks = [generate_one(result) for result in batch]
        results = await asyncio.gather(*tasks)
        
        for question_id, answer in results:
            if answer:
                generated_answers[question_id] = answer
    
    print(f"   Generated {len(generated_answers)} answers\n")

    
    # Prepare judge tasks
    print("📋 Preparing judge tasks...")
    judge_tasks = prepare_judge_tasks_from_retrieval(
        retrieval_results=retrieval_results,
        generated_answers=generated_answers,
    )
    print(f"   Prepared {len(judge_tasks)} judge tasks\n")
    
    # Initialize LLM judge
    print(f"⚖️  Initializing LLM judge ({judge_model})...")
    judge = AsyncLLMJudge(
        api_key=judge_api_key,
        model=judge_model,
        base_url=judge_base_url,
        max_concurrent=max_concurrent,
    )
    
    # Run judge evaluation
    print(f"🔍 Running LLM judge evaluation...")
    prompts = [task["prompt"] for task in judge_tasks]
    judge_responses = await judge.judge_batch(prompts)
    
    # Parse judge responses and create results
    results = []
    for task, response in zip(judge_tasks, judge_responses):
        judge_score = parse_judge_response(response)
        
        results.append({
            "question_id": task["question_id"],
            "category": task["category"],
            "question": task["question"],
            "ground_truth": task["ground_truth"],
            "predicted_answer": task["predicted_answer"],
            "judge_score": judge_score,
            "judge_response": response,
        })
    
    print(f"   Judged {len(results)} responses\n")
    
    # Calculate GPT4SCORE
    print("📊 Calculating GPT4SCORE metrics...")
    metrics = calculate_gpt4score(results)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"GPT4SCORE RESULTS")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {metrics['overall']['accuracy']:.4f} (n={metrics['overall']['count']})")
    print(f"\nBy Category:")
    for cat, cat_metrics in sorted(metrics['by_category'].items()):
        print(f"  {cat}: {cat_metrics['accuracy']:.4f} (n={cat_metrics['count']})")
    print(f"{'='*60}\n")
    
    # Save results
    output_data = {
        "overall": metrics["overall"],
        "by_category": metrics["by_category"],
        "results": results,
        "metadata": {
            "input_file": input_file,
            "context_k": context_k,
            "llm_model": llm_model,
            "judge_model": judge_model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    }
    
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to: {output_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval results with GPT4SCORE"
    )
    
    # Required args
    parser.add_argument("--input", required=True, help="Input retrieval results JSON file")
    parser.add_argument("--output", required=True, help="Output GPT4SCORE results JSON file")
    
    # Generation config
    parser.add_argument("--context_k", type=int, default=5,
                        help="Number of top chunks to use for generation")
    parser.add_argument("--llm_model", default="Qwen/Qwen3-8B",
                        help="LLM model for answer generation")
    parser.add_argument("--api_key", default="dummy",
                        help="API key for LLM")
    parser.add_argument("--base_url", default="http://localhost:8001/v1",
                        help="LLM API base URL")
    parser.add_argument("--disable_thinking", action="store_true",
                        help="Disable thinking in LLM")
    
    # Judge config
    parser.add_argument("--judge_model", default="gpt-4o",
                        help="Model for LLM judge")
    parser.add_argument("--judge_api_key", default=None,
                        help="API key for judge (defaults to --api_key)")
    parser.add_argument("--judge_base_url", default=None,
                        help="Judge API base URL (defaults to --base_url)")
    parser.add_argument("--max_concurrent", type=int, default=10,
                        help="Max concurrent judge API calls")
    
    args = parser.parse_args()
    
    # Default judge API settings to LLM API settings
    judge_api_key = args.judge_api_key or args.api_key
    judge_base_url = args.judge_base_url or args.base_url
    
    # Run async evaluation
    import asyncio
    asyncio.run(evaluate_with_gpt4score_async(
        input_file=args.input,
        output_file=args.output,
        context_k=args.context_k,
        judge_model=args.judge_model,
        llm_model=args.llm_model,
        api_key=args.api_key,
        base_url=args.base_url,
        judge_api_key=judge_api_key,
        judge_base_url=judge_base_url,
        max_concurrent=args.max_concurrent,
        disable_thinking=args.disable_thinking,
    ))


if __name__ == "__main__":
    main()

"""

"""