#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generation Evaluator - Complete RAG Evaluation Pipeline

This script:
1. Loads retrieved chunks from retrieval results (from process_retrieve.py)
2. Uses LLM to generate answers based on question + retrieved context
3. Compares generated answers with ground truth
4. Calculates comprehensive metrics: F1, BLEU, ROUGE-1/2/L, BERTScore
5. Groups results by category and provides detailed analysis
"""

import json
import argparse
import os
import re
from typing import Dict, List, Any
from collections import defaultdict
from tqdm import tqdm

# Metrics imports
try:
    from rouge_score import rouge_scorer
except ImportError:
    print("Installing rouge_score...")
    os.system("pip install rouge_score")
    from rouge_score import rouge_scorer

try:
    import sacrebleu
except ImportError:
    print("Installing sacrebleu...")
    os.system("pip install sacrebleu")
    import sacrebleu

try:
    from bert_score import score as bert_score
except ImportError:
    print("Installing bert_score...")
    os.system("pip install bert_score")
    from bert_score import score as bert_score

# LLM imports
from openai import OpenAI


def load_ground_truth_from_dataset(dataset_file: str) -> Dict[str, str]:
    """
    Load ground truth answers from the original locomo dataset.
    
    Expected format:
    [
      {
        "conv_id": "conv-1",
        "qas": [
          {"question": "...", "answer": "...", "category": "1"},
          ...
        ]
      },
      ...
    ]
    
    Returns dict with multiple lookup keys:
    - "{conv_id}_q{i}" format
    - string index "0", "1", "2" etc
    - question text as key
    """
    with open(dataset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    gt_map = {}
    for conv in data:
        conv_id = conv.get("conv_id", "")
        qas = conv.get("qas", [])
        for i, qa in enumerate(qas):
            question_id = f"{conv_id}_q{i}"
            answer = qa.get("answer", "")
            question_text = qa.get("question", "")
            # Convert answer to string (some may be integers)
            answer_str = str(answer)
            
            # Multiple keys for flexible lookup
            gt_map[question_id] = answer_str
            gt_map[str(i)] = answer_str
            if question_text:
                gt_map[question_text] = answer_str
    
    return gt_map


def generate_answer_with_llm(
    client: OpenAI,
    model: str,
    question: str,
    context_chunks: List,
    max_context_chunks: int = 5,
    disable_thinking: bool = True
) -> str:
    """Generate answer using LLM with retrieved context chunks."""
    # Handle both List[str] and List[dict] formats
    chunks_text = []
    for chunk in context_chunks[:max_context_chunks]:
        if isinstance(chunk, dict):
            # Extract content from dict (from mem0 retrieval)
            chunks_text.append(chunk.get("content", ""))
        else:
            # Already a string
            chunks_text.append(str(chunk))
    
    # Take top-k chunks as context
    context = "\n\n".join(chunks_text)
    
    # Clean context of timestamps if present
    # Format: "2023-01-15 14:30: message content"
    context_lines = []
    for line in context.split("\n"):
        # Remove timestamp prefix if it matches the pattern
        cleaned = re.sub(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\s*', '', line)
        context_lines.append(cleaned.strip())
    context = "\n".join([l for l in context_lines if l])
    
    prompt = f"""Based on the following conversation memories, answer the question concisely.

Context from memory:
{context}

Question: {question}

Answer (be concise and direct, just give the answer without explanation):"""
    
    extra_body = {}
    if disable_thinking:
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. Give concise, direct answers only. Do not explain your reasoning."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.1,
            extra_body=extra_body
        )
        answer = response.choices[0].message.content.strip()
        
        # If response still contains thinking tags, extract the answer part
        if "<think>" in answer:
            if "</think>" in answer:
                answer = answer.split("</think>")[-1].strip()
            else:
                answer = answer.split("<think>")[0].strip()
        
        # Clean up any remaining tags
        answer = re.sub(r'<[^>]+>', '', answer).strip()
        
        return answer if answer else "N/A"
    except Exception as e:
        print(f"Error generating answer: {e}")
        return ""


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()
    
    if not pred_tokens or not gt_tokens:
        return 0.0
    
    common = set(pred_tokens) & set(gt_tokens)
    
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        'rouge1': sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
        'rouge2': sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0,
        'rougeL': sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0
    }


def compute_bleu(predictions: List[str], references: List[str]) -> float:
    """Compute BLEU score."""
    if not predictions or not references:
        return 0.0
    
    # sacrebleu expects references as list of lists
    refs = [[ref] for ref in references]
    
    try:
        bleu = sacrebleu.corpus_bleu(predictions, list(zip(*refs)))
        return bleu.score / 100.0  # Normalize to 0-1
    except Exception as e:
        print(f"BLEU computation error: {e}")
        return 0.0


def compute_bertscore(predictions: List[str], references: List[str]) -> float:
    """Compute BERTScore F1."""
    if not predictions or not references:
        return 0.0
    
    try:
        P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
        return F1.mean().item()
    except Exception as e:
        print(f"BERTScore computation error: {e}")
        return 0.0


def evaluate_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute all metrics for a set of predictions and references."""
    
    # F1 scores
    f1_scores = [compute_f1(p, r) for p, r in zip(predictions, references)]
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    # ROUGE scores
    rouge_scores = compute_rouge(predictions, references)
    
    # BLEU score
    bleu = compute_bleu(predictions, references)
    
    # BERTScore
    bertscore = compute_bertscore(predictions, references)
    
    return {
        'f1': avg_f1,
        'bleu': bleu,
        'rouge1': rouge_scores['rouge1'],
        'rouge2': rouge_scores['rouge2'],
        'rougeL': rouge_scores['rougeL'],
        'bertscore_f1': bertscore,
        'count': len(predictions)
    }


def main():
    parser = argparse.ArgumentParser(description="Complete RAG Evaluation: Generation + Metrics")
    parser.add_argument("retrieval_file", help="Path to retrieval results JSON (from process_retrieve.py)")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth dataset JSON (e.g., locomo_processed_data.json)")
    parser.add_argument("--output", default="generation_eval_results.json", help="Output file path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to process (for testing)")
    parser.add_argument("--context-k", type=int, default=5, help="Number of top retrieved chunks to use as context")
    parser.add_argument("--llm_model", default="Qwen/Qwen3-8B", help="LLM model name")
    parser.add_argument("--base_url", default="http://localhost:8001/v1", help="LLM API base URL")
    parser.add_argument("--api_key", default="local-anything", help="API key for LLM")
    parser.add_argument("--disable_thinking", action="store_true", help="Disable thinking for Qwen models")
    
    args = parser.parse_args()
    
    # Load retrieval results
    print(f"Loading retrieval results from {args.retrieval_file}...")
    with open(args.retrieval_file, 'r', encoding='utf-8') as f:
        retrieval_data = json.load(f)
    
    # Load ground truth
    print(f"Loading ground truth from {args.ground_truth}...")
    gt_map = load_ground_truth_from_dataset(args.ground_truth)
    
    print(f"Loaded {len(retrieval_data)} retrieval records")
    print(f"Loaded {len(gt_map)} ground truth answers")
    
    # Initialize LLM client
    print(f"Initializing LLM client (model: {args.llm_model})...")
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    
    # Apply limit if specified
    if args.limit:
        retrieval_data = retrieval_data[:args.limit]
        print(f"Processing first {args.limit} questions only (--limit)")
    
    # Process each question
    results = []
    predictions_by_category = defaultdict(list)
    references_by_category = defaultdict(list)
    all_predictions = []
    all_references = []
    
    print(f"\nGenerating answers with LLM (context-k={args.context_k})...")
    for record in tqdm(retrieval_data, desc="Processing"):
        question_id = record.get("question_id", "")
        question = record.get("question", "")
        category = record.get("category", "unknown")
        
        # Get ground truth (try multiple keys)
        ground_truth = gt_map.get(str(question_id), "") or gt_map.get(question, "")
        if not ground_truth:
            print(f"Warning: No ground truth found for question_id={question_id}, question='{question}'")
            continue
        
        # Get retrieved chunks
        chunks = record.get("chunks", [])
        if not chunks:
            print(f"Warning: No chunks retrieved for {question_id}")
            chunks = []
        
        # Generate answer
        generated_answer = generate_answer_with_llm(
            client, args.llm_model, question, chunks, args.context_k, args.disable_thinking
        )
        
        # Compute per-question F1
        f1 = compute_f1(generated_answer, ground_truth)
        
        # Store result
        result = {
            "question_id": question_id,
            "question": question,
            "category": str(category),
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "f1": f1,
            "num_chunks_retrieved": len(chunks)
        }
        results.append(result)
        
        # Collect for batch metrics
        all_predictions.append(generated_answer)
        all_references.append(ground_truth)
        predictions_by_category[str(category)].append(generated_answer)
        references_by_category[str(category)].append(ground_truth)
    
    print(f"\nComputing overall metrics...")
    overall_metrics = evaluate_metrics(all_predictions, all_references)
    
    print(f"\nComputing per-category metrics...")
    category_metrics = {}
    for cat in sorted(predictions_by_category.keys()):
        preds = predictions_by_category[cat]
        refs = references_by_category[cat]
        category_metrics[cat] = evaluate_metrics(preds, refs)
    
    # Build output
    output = {
        "config": {
            "retrieval_file": args.retrieval_file,
            "ground_truth": args.ground_truth,
            "llm_model": args.llm_model,
            "context_k": args.context_k,
            "base_url": args.base_url,
            "disable_thinking": args.disable_thinking
        },
        "overall": overall_metrics,
        "by_category": category_metrics,
        "per_question": results
    }
    
    # Save results
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("GENERATION EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\n=== Overall Metrics (n={overall_metrics['count']}) ===")
    print(f"F1:           {overall_metrics['f1']:.4f}")
    print(f"BLEU:         {overall_metrics['bleu']:.4f}")
    print(f"ROUGE-1:      {overall_metrics['rouge1']:.4f}")
    print(f"ROUGE-2:      {overall_metrics['rouge2']:.4f}")
    print(f"ROUGE-L:      {overall_metrics['rougeL']:.4f}")
    print(f"BERTScore-F1: {overall_metrics['bertscore_f1']:.4f}")
    
    print(f"\n=== By Category ===")
    for cat, metrics in sorted(category_metrics.items()):
        print(f"\nCategory {cat} (n={metrics['count']}):") 
        print(f"  F1:           {metrics['f1']:.4f}")
        print(f"  BLEU:         {metrics['bleu']:.4f}")
        print(f"  ROUGE-1:      {metrics['rouge1']:.4f}")
        print(f"  ROUGE-2:      {metrics['rouge2']:.4f}")
        print(f"  ROUGE-L:      {metrics['rougeL']:.4f}")
        print(f"  BERTScore-F1: {metrics['bertscore_f1']:.4f}")
    
    print(f"\n✅ Results saved to: {args.output}")


if __name__ == "__main__":
    main()


"""
Example usage:

python3 generation_evaluator.py \
    /mnt/hungpv/projects/memory/amem/locomo_results_20260102_161503/retrieval_results_20260102_161503.json \
    --ground-truth /home/vinhpq/mem_baseline/mem0/data/locomo/processed_data/locomo_processed_data.json \
    --output /home/vinhpq/mem_baseline/amem_temp/generation_eval_results.json \
    --context-k 5 \
    --llm_model Qwen/Qwen3-8B \
    --base_url http://localhost:8001/v1 \
    --api_key dummy \
    --disable_thinking
"""
