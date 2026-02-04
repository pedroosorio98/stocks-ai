# -*- coding: utf-8 -*-
"""
Q&A System Evaluation - Tests ai-app.py answer() function
Evaluates quick question-answering using RAG
"""

import sys
from pathlib import Path

# Add parent directory to import root modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from typing import List, Dict, Any
import argparse

from rag import retrieve_internal, retrieve_web_exa, build_context
from openai import OpenAI
from metrics_qa import (
    calculate_answer_relevance,
    check_sources_cited,
    check_has_sources_section,
    check_retrieval_accuracy,
    calculate_response_quality,
    check_must_contain
)

client = OpenAI()


def load_dataset(filepath: str = "evals/dataset_qa.jsonl") -> List[Dict]:
    """Load Q&A test cases"""
    dataset = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    print(f"✓ Loaded {len(dataset)} Q&A test cases")
    return dataset


def generate_answer(query: str, use_web: bool = True) -> Dict[str, Any]:
    """
    Generate answer using YOUR actual ai-app.py logic
    Returns structured data for evaluation
    """
    # Use YOUR actual RAG pipeline
    internal = retrieve_internal(query, k=12)
    web = retrieve_web_exa(query, k=4) if use_web else []
    context = build_context(internal, web)
    
    # Use YOUR actual prompt
    prompt = f"""You are a helpful research assistant.
Answer the user using the provided context.
If something is not in context, say you don't know.
Always include a short 'Sources' section listing the INTERNAL and WEB citations you used.

User question: {query}

Context:
{context}
"""
    
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    
    answer_text = resp.choices[0].message.content
    
    return {
        "text": answer_text,
        "internal_chunks": internal,
        "web_chunks": web,
        "internal_count": len(internal),
        "web_count": len(web)
    }


def run_single_test(test_case: Dict, verbose: bool = True) -> Dict[str, Any]:
    """Run single Q&A test case"""
    
    test_id = test_case["id"]
    query = test_case["query"]
    expected_tickers = test_case.get("expected_sources_contains", [])
    must_cite_internal = test_case.get("must_cite_internal", True)
    must_contain = test_case.get("must_contain", [])  # NEW: Get required terms
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"TEST: {test_id}")
        print(f"{'='*70}")
        print(f"Query: {query}")
        print(f"Expected: {', '.join(expected_tickers)}")
        if must_contain:
            print(f"Must contain: {', '.join(must_contain)}")
    
    try:
        # Generate answer
        if verbose:
            print(f"\n  → Generating answer...")
        
        result = generate_answer(query, use_web=True)
        
        if verbose:
            print(f"  → Retrieved {result['internal_count']} internal + {result['web_count']} web chunks")
        
        # Evaluate
        if verbose:
            print(f"  → Evaluating...")
        
        relevance = calculate_answer_relevance(query, result['text'])
        retrieval_acc = check_retrieval_accuracy(result['internal_chunks'], expected_tickers)
        has_sources = check_has_sources_section(result['text'])
        quality = calculate_response_quality(result['text'])
        content_check = check_must_contain(result['text'], must_contain)  # NEW: Check required terms
        
        # Calculate pass/fail (based on RETRIEVAL, not answer text)
        passed = (
            retrieval_acc["passed"] and  # ← Main criterion: retrieved from right folders
            has_sources and              # ← Has Sources section
            relevance >= 0.3 and         # ← Answer is relevant to query
            content_check["passed"]      # ← NEW: Contains required terms
        )
        
        # Calculate overall score
        overall_score = (
            relevance * 0.25 +
            retrieval_acc["precision"] * 0.40 +  # ← Retrieval is most important (40%)
            (1.0 if has_sources else 0.0) * 0.15 +
            content_check["precision"] * 0.20    # ← NEW: Content check (20%)
        )
        
        # Print results
        if verbose:
            print(f"\n  📊 RESULTS:")
            print(f"  {'─'*66}")
            print(f"  Overall Score:      {overall_score:.1%}")
            print(f"  Status:             {'✓ PASSED' if passed else '✗ FAILED'}")
            print(f"  {'─'*66}")
            print(f"  Answer Relevance:   {relevance:.1%}")
            print(f"  Retrieval Accuracy: {retrieval_acc['precision']:.1%}")
            print(f"  Retrieved From:     {', '.join(retrieval_acc['found']) if retrieval_acc['found'] else 'None'}")
            if retrieval_acc['missing']:
                print(f"  Missing Data From:  {', '.join(retrieval_acc['missing'])}")
            print(f"  Has Sources:        {'✓' if has_sources else '✗'}")
            
            # NEW: Show content check results
            if must_contain:
                print(f"  Content Check:      {content_check['precision']:.1%}")
                if content_check['found']:
                    print(f"  Found Terms:        {', '.join(content_check['found'])}")
                if content_check['missing']:
                    print(f"  Missing Terms:      {', '.join(content_check['missing'])}")
            
            print(f"  Response Quality:   {quality['word_count']} words, {quality['has_numbers']} numbers")
            print(f"  {'─'*66}")
        
        return {
            "test_id": test_id,
            "query": query,
            "answer": result['text'],
            "metrics": {
                "overall_score": overall_score,
                "relevance": relevance,
                "retrieval_accuracy": retrieval_acc,  # ← Main metric
                "has_sources": has_sources,
                "content_check": content_check,  # ← NEW: Content validation
                "quality": quality
            },
            "chunks_retrieved": {
                "internal": result['internal_count'],
                "web": result['web_count']
            },
            "passed": passed,
            "error": None
        }
        
    except Exception as e:
        if verbose:
            print(f"\n  ✗ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return {
            "test_id": test_id,
            "query": query,
            "answer": None,
            "metrics": None,
            "chunks_retrieved": None,
            "passed": False,
            "error": str(e)
        }


def run_full_evaluation(
    dataset_path: str = "evals/dataset_qa.jsonl",
    output_path: str = "evals/results/qa_results.json"
):
    """Run full Q&A evaluation suite"""
    
    print("\n" + "="*70)
    print("Q&A SYSTEM EVALUATION")
    print("="*70)
    
    dataset = load_dataset(dataset_path)
    
    results = []
    passed_count = 0
    
    for i, test_case in enumerate(dataset, 1):
        print(f"\n[{i}/{len(dataset)}]", end=" ")
        
        result = run_single_test(test_case, verbose=True)
        results.append(result)
        
        if result['passed']:
            passed_count += 1
    
    # Calculate summary
    total = len(results)
    accuracy = passed_count / total if total > 0 else 0
    
    valid_results = [r for r in results if r['metrics'] is not None]
    
    avg_overall = sum(r['metrics']['overall_score'] for r in valid_results) / len(valid_results) if valid_results else 0
    avg_relevance = sum(r['metrics']['relevance'] for r in valid_results) / len(valid_results) if valid_results else 0
    avg_retrieval = sum(r['metrics']['retrieval_accuracy']['precision'] for r in valid_results) / len(valid_results) if valid_results else 0
    
    summary = {
        "total_tests": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "pass_rate": accuracy,
        "avg_overall_score": avg_overall,
        "avg_relevance": avg_relevance,
        "avg_retrieval_accuracy": avg_retrieval
    }
    
    # Save results
    output = {
        "summary": summary,
        "results": results
    }
    
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total Tests:       {total}")
    print(f"Passed:            {passed_count} ({accuracy:.1%})")
    print(f"Failed:            {total - passed_count}")
    print(f"─"*70)
    print(f"Avg Overall Score: {avg_overall:.1%}")
    print(f"Avg Relevance:     {avg_relevance:.1%}")
    print(f"Avg Retrieval:     {avg_retrieval:.1%}")  # ← Shows if right data retrieved
    print("="*70)
    print(f"\n✓ Results saved to: {output_path}")
    
    if total - passed_count > 0:
        print(f"\n❌ Failed Tests:")
        for r in results:
            if not r['passed']:
                reason = r['error'] if r['error'] else "Quality thresholds not met"
                print(f"  • {r['test_id']}: {reason}")
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Run Q&A evaluation suite")
    parser.add_argument('--dataset', default='evals/dataset_qa.jsonl')
    parser.add_argument('--output', default='evals/results/qa_results.json')
    parser.add_argument('--test-id', help='Run specific test only')
    
    args = parser.parse_args()
    
    dataset = load_dataset(args.dataset)
    
    if args.test_id:
        dataset = [t for t in dataset if t['id'] == args.test_id]
        if not dataset:
            print(f"Error: Test ID '{args.test_id}' not found")
            return
    
    if len(dataset) == 1:
        result = run_single_test(dataset[0], verbose=True)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Result saved to: {args.output}")
    else:
        run_full_evaluation(args.dataset, args.output)


if __name__ == "__main__":
    main()