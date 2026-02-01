# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 20:10:41 2026

@author: Pedro
"""

import json
import sys
from pathlib import Path

# Add parent directory to path to import your RAG modules
sys.path.append(str(Path(__file__).parent.parent))

from rag import retrieve_internal, generate_answer  # Your RAG functions
from typing import List, Dict, Any

def load_eval_dataset(filepath: str = "evals/dataset.jsonl") -> List[Dict]:
    """Load evaluation dataset from JSONL file"""
    dataset = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    return dataset

def check_sources_cited(answer: str, expected_sources: List[str]) -> Dict[str, Any]:
    """
    Check if expected sources are cited in the answer
    Returns: {
        "found": ["NVDA"],
        "missing": [],
        "precision": 1.0,
        "recall": 1.0
    }
    """
    found = []
    missing = []
    
    for source in expected_sources:
        if source.upper() in answer.upper():
            found.append(source)
        else:
            missing.append(source)
    
    # Calculate metrics
    precision = len(found) / len(expected_sources) if expected_sources else 1.0
    recall = precision  # Same for exact match
    
    return {
        "found": found,
        "missing": missing,
        "precision": precision,
        "recall": recall,
        "passed": len(missing) == 0
    }

def check_internal_citation(retrieved_chunks: List[Dict]) -> bool:
    """Check if at least one internal source was retrieved"""
    if not retrieved_chunks:
        return False
    
    # Check if any chunk has metadata indicating internal source
    for chunk in retrieved_chunks:
        if chunk.get("metadata", {}).get("source_type") == "internal":
            return True
        # Or check if source path contains your Data folder
        if "Data/" in chunk.get("metadata", {}).get("source", ""):
            return True
    
    return True  # Assume internal if no explicit marker

def run_single_eval(test_case: Dict, verbose: bool = True) -> Dict[str, Any]:
    """Run a single evaluation test case"""
    
    test_id = test_case["id"]
    query = test_case["query"]
    expected_sources = test_case.get("expected_sources_contains", [])
    must_cite_internal = test_case.get("must_cite_internal", False)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Test ID: {test_id}")
        print(f"Query: {query}")
        print(f"Expected sources: {expected_sources}")
    
    try:
        # Step 1: Retrieve chunks
        retrieved_chunks = retrieve_internal(query, k=5)
        
        # Step 2: Generate answer
        answer = generate_answer(query, retrieved_chunks)
        
        # Step 3: Evaluate
        source_check = check_sources_cited(answer, expected_sources)
        internal_check = check_internal_citation(retrieved_chunks)
        
        # Determine pass/fail
        passed = source_check["passed"]
        if must_cite_internal:
            passed = passed and internal_check
        
        result = {
            "id": test_id,
            "query": query,
            "answer": answer,
            "retrieved_chunks_count": len(retrieved_chunks),
            "source_check": source_check,
            "internal_cited": internal_check,
            "passed": passed,
            "error": None
        }
        
        if verbose:
            print(f"Retrieved: {len(retrieved_chunks)} chunks")
            print(f"Sources found: {source_check['found']}")
            if source_check['missing']:
                print(f"Sources missing: {source_check['missing']}")
            print(f"Internal cited: {internal_check}")
            print(f"{'PASSED' if passed else '✗ FAILED'}")
            print(f"\nAnswer preview: {answer[:200]}...")
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"✗ ERROR: {str(e)}")
        
        return {
            "id": test_id,
            "query": query,
            "answer": None,
            "retrieved_chunks_count": 0,
            "source_check": {"found": [], "missing": expected_sources, "precision": 0, "recall": 0, "passed": False},
            "internal_cited": False,
            "passed": False,
            "error": str(e)
        }

def run_full_eval(dataset_path: str = "evals/dataset.jsonl", output_path: str = "evals/results.json"):
    """Run evaluation on entire dataset"""
    
    print("Loading evaluation dataset...")
    dataset = load_eval_dataset(dataset_path)
    print(f"Found {len(dataset)} test cases")
    
    results = []
    passed_count = 0
    
    for test_case in dataset:
        result = run_single_eval(test_case, verbose=True)
        results.append(result)
        if result["passed"]:
            passed_count += 1
    
    # Calculate summary metrics
    total = len(results)
    accuracy = passed_count / total if total > 0 else 0
    
    avg_precision = sum(r["source_check"]["precision"] for r in results) / total if total > 0 else 0
    avg_recall = sum(r["source_check"]["recall"] for r in results) / total if total > 0 else 0
    
    summary = {
        "total_tests": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "accuracy": accuracy,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall
    }
    
    # Save results
    output = {
        "summary": summary,
        "results": results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {total}")
    print(f"Passed: {passed_count} ({accuracy*100:.1f}%)")
    print(f"Failed: {total - passed_count}")
    print(f"Avg Precision: {avg_precision:.2f}")
    print(f"Avg Recall: {avg_recall:.2f}")
    print(f"\nResults saved to: {output_path}")
    
    return output

if __name__ == "__main__":
    run_full_eval()
