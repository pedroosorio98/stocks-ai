# -*- coding: utf-8 -*-
"""
Report Generation Evaluation - Tests reports.py
Evaluates comprehensive PDF report generation
"""

import sys
from pathlib import Path

# Add parent directory to import root modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from typing import List, Dict, Any
import argparse

from reports import CompanyReportGenerator
from metrics_reports import (
    check_section_presence,
    calculate_section_completeness,
    verify_facts,
    count_data_backed_claims,
    evaluate_source_quality,
    check_length_appropriateness,
    llm_judge_quality,
    check_must_contain_terms
)


def load_dataset(filepath: str = "evals/dataset_reports.jsonl") -> List[Dict]:
    """Load report test cases"""
    dataset = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    print(f"✓ Loaded {len(dataset)} report test cases")
    return dataset


def generate_report(ticker: str) -> Dict[str, Any]:
    """
    Generate report using YOUR actual reports.py
    Returns report data for evaluation
    """
    print(f"  → Generating full report for {ticker}...")
    
    generator = CompanyReportGenerator(ticker)
    
    # Generate all sections
    report_data = generator.generate_all_sections()
    
    # Combine all sections into full text
    full_text = "\n\n".join([
        f"# {section_name}\n\n{content}"
        for section_name, content in report_data.items()
    ])
    
    # Get sources used
    internal_sources = list(generator.internal_sources)
    web_sources = list(generator.web_sources)
    
    return {
        "ticker": ticker,
        "sections": report_data,
        "full_text": full_text,
        "internal_sources": internal_sources,
        "web_sources": web_sources,
        "total_sources": len(internal_sources) + len(web_sources)
    }


def run_single_test(test_case: Dict, verbose: bool = True) -> Dict[str, Any]:
    """Run single report generation test"""
    
    test_id = test_case["id"]
    ticker = test_case["ticker"]
    expected_sections = test_case.get("expected_sections", [])
    ground_truth_facts = test_case.get("ground_truth_facts", [])
    must_contain_terms = test_case.get("must_contain_terms", [])  # NEW
    min_words = test_case.get("min_word_count", 1500)
    max_words = test_case.get("max_word_count", 3000)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"TEST: {test_id}")
        print(f"{'='*70}")
        print(f"Ticker: {ticker}")
        print(f"Expected sections: {', '.join(expected_sections)}")
        if must_contain_terms:
            print(f"Must contain: {', '.join(must_contain_terms)}")
    
    try:
        # Generate report
        result = generate_report(ticker)
        
        if verbose:
            print(f"  → Generated {len(result['sections'])} sections")
            print(f"  → Used {len(result['internal_sources'])} internal + {len(result['web_sources'])} web sources")
        
        # Evaluate
        if verbose:
            print(f"  → Evaluating...")
        
        sections = check_section_presence(result['full_text'])
        completeness = calculate_section_completeness(result['full_text'])
        
        if ground_truth_facts:
            facts = verify_facts(result['full_text'], ground_truth_facts)
        else:
            facts = {"accuracy": None}
        
        data_backing = count_data_backed_claims(result['full_text'])
        
        # Format sources for evaluation
        sources = []
        for fname, location in result['internal_sources']:
            sources.append({
                "type": "internal",
                "file": fname,
                "location": location
            })
        for title, url in result['web_sources']:
            sources.append({
                "type": "web",
                "title": title,
                "url": url
            })
        
        source_quality = evaluate_source_quality(sources)
        length = check_length_appropriateness(result['full_text'], min_words, max_words)
        llm_judge = llm_judge_quality(result['full_text'], f"Generate comprehensive report for {ticker}")
        terms_check = check_must_contain_terms(result['full_text'], must_contain_terms)  # NEW
        
        # Calculate overall score
        weights = {
            'completeness': 0.25,
            'facts': 0.15,
            'data_backing': 0.20,
            'sources': 0.15,
            'llm_judge': 0.10,
            'length': 0.05,
            'terms': 0.10  # NEW: Must-contain terms
        }
        
        overall = completeness * weights['completeness']
        
        if facts['accuracy'] is not None:
            overall += facts['accuracy'] * weights['facts']
        
        overall += data_backing['ratio'] * weights['data_backing']
        overall += source_quality['score'] * weights['sources']
        overall += (llm_judge['overall'] / 5.0) * weights['llm_judge']
        overall += length['score'] * weights['length']
        overall += terms_check['precision'] * weights['terms']  # NEW
        
        passed = overall >= 0.75  # 75% threshold
        
        # Print results
        if verbose:
            print(f"\n  📊 RESULTS:")
            print(f"  {'─'*66}")
            print(f"  Overall Score:        {overall:.1%}")
            print(f"  Status:               {'✓ PASSED' if passed else '✗ FAILED'}")
            print(f"  {'─'*66}")
            print(f"  Section Completeness: {completeness:.1%}")
            if facts['accuracy'] is not None:
                print(f"  Factual Accuracy:     {facts['accuracy']:.1%}")
            print(f"  Data-Backed Claims:   {data_backing['ratio']:.1%} ({data_backing['backed_claims']}/{data_backing['total_claims']})")
            print(f"  Source Quality:       {source_quality['score']:.1%}")
            print(f"  LLM Judge:            {llm_judge['overall']:.1f}/5.0")
            print(f"  Length:               {length['word_count']} words ({length['status']})")
            
            # NEW: Show terms check
            if must_contain_terms:
                print(f"  Terms Coverage:       {terms_check['precision']:.1%}")
                if terms_check['found']:
                    print(f"  Found Terms:          {', '.join(terms_check['found'])}")
                if terms_check['missing']:
                    print(f"  Missing Terms:        {', '.join(terms_check['missing'])}")
            
            print(f"  {'─'*66}")
            
            print(f"\n  📑 Sections Found:")
            for section, found in sections.items():
                status = "✓" if found else "✗"
                print(f"    {status} {section}")
            
            if llm_judge.get('brief_feedback'):
                print(f"\n  💬 LLM Feedback: {llm_judge['brief_feedback']}")
        
        return {
            "test_id": test_id,
            "ticker": ticker,
            "report": result,
            "metrics": {
                "overall_score": overall,
                "sections": sections,
                "completeness": completeness,
                "facts": facts,
                "data_backing": data_backing,
                "source_quality": source_quality,
                "length": length,
                "llm_judge": llm_judge,
                "terms_check": terms_check  # NEW
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
            "ticker": ticker,
            "report": None,
            "metrics": None,
            "passed": False,
            "error": str(e)
        }


def run_full_evaluation(
    dataset_path: str = "evals/dataset_reports.jsonl",
    output_path: str = "evals/results/reports_results.json"
):
    """Run full report evaluation suite"""
    
    print("\n" + "="*70)
    print("REPORT GENERATION EVALUATION")
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
    avg_completeness = sum(r['metrics']['completeness'] for r in valid_results) / len(valid_results) if valid_results else 0
    avg_data_backing = sum(r['metrics']['data_backing']['ratio'] for r in valid_results) / len(valid_results) if valid_results else 0
    avg_llm_judge = sum(r['metrics']['llm_judge']['overall'] for r in valid_results) / len(valid_results) if valid_results else 0
    
    summary = {
        "total_tests": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "pass_rate": accuracy,
        "avg_overall_score": avg_overall,
        "avg_completeness": avg_completeness,
        "avg_data_backing": avg_data_backing,
        "avg_llm_judge": avg_llm_judge
    }
    
    # Save results
    output = {
        "summary": summary,
        "results": results
    }
    
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total Tests:          {total}")
    print(f"Passed:               {passed_count} ({accuracy:.1%})")
    print(f"Failed:               {total - passed_count}")
    print(f"─"*70)
    print(f"Avg Overall Score:    {avg_overall:.1%}")
    print(f"Avg Completeness:     {avg_completeness:.1%}")
    print(f"Avg Data-Backing:     {avg_data_backing:.1%}")
    print(f"Avg LLM Judge:        {avg_llm_judge:.1f}/5.0")
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
    parser = argparse.ArgumentParser(description="Run report evaluation suite")
    parser.add_argument('--dataset', default='evals/dataset_reports.jsonl')
    parser.add_argument('--output', default='evals/results/reports_results.json')
    parser.add_argument('--test-id', help='Run specific test only')
    parser.add_argument('--ticker', help='Run test for specific ticker')
    
    args = parser.parse_args()
    
    dataset = load_dataset(args.dataset)
    
    if args.test_id:
        dataset = [t for t in dataset if t['id'] == args.test_id]
        if not dataset:
            print(f"Error: Test ID '{args.test_id}' not found")
            return
    
    if args.ticker:
        dataset = [t for t in dataset if t['ticker'].upper() == args.ticker.upper()]
        if not dataset:
            print(f"Error: No tests found for ticker '{args.ticker}'")
            return
    
    if len(dataset) == 1:
        result = run_single_test(dataset[0], verbose=True)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n✓ Result saved to: {args.output}")
    else:
        run_full_evaluation(args.dataset, args.output)


if __name__ == "__main__":
    main()