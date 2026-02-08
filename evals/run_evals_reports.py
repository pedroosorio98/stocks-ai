# -*- coding: utf-8 -*-

import sys
from pathlib import Path
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
    check_must_contain_terms,
    check_probability_scenarios,
    # NEW METRICS
    check_any_margins_cited,
    check_market_share_mentioned,
    check_key_drivers_count,
    check_opportunities_and_risks,
    check_scenario_completeness,
    check_stock_price_implications,
    check_competitor_comparison
)


def load_dataset(filepath: str = "dataset_reports.jsonl") -> List[Dict]:
    """Load report test cases"""
    dataset = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    print(f"✓ Loaded {len(dataset)} report test cases")
    return dataset


def generate_report(ticker: str) -> Dict[str, Any]:
    """Generate report using reports.py"""
    print(f"  → Generating full report for {ticker}...")
    
    generator = CompanyReportGenerator(ticker)
    report_data = generator.generate_all_sections()
    
    full_text = "\n\n".join([
        f"# {section_name}\n\n{content}"
        for section_name, content in report_data.items()
    ])
    
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
    """Run single report generation test with ENHANCED metrics"""
    
    # Extract test parameters
    test_id = test_case["id"]
    ticker = test_case["ticker"]
    expected_sections = test_case.get("expected_sections", [])
    ground_truth_facts = test_case.get("ground_truth_facts", [])
    must_contain_terms = test_case.get("must_contain_terms", [])
    
    # Scenario requirements
    require_scenarios = test_case.get("require_scenarios", False)
    scenario_count = test_case.get("scenario_count", 3)
    
    # NEW: Additional requirements
    require_any_margins = test_case.get("require_any_margins", False)
    require_market_share = test_case.get("require_market_share", False)
    require_key_drivers = test_case.get("require_key_drivers", False)
    min_drivers = test_case.get("min_drivers", 3)
    max_drivers = test_case.get("max_drivers", 5)
    require_opps_risks = test_case.get("require_opportunities_and_risks", False)
    min_opps_risks = test_case.get("min_opportunities_risks", 2)
    require_scenario_types = test_case.get("require_scenario_types", False)
    require_stock_price = test_case.get("require_stock_price_discussion", False)
    require_competitor_comp = test_case.get("require_competitor_comparison", False)
    
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
        if require_scenarios:
            print(f"Require scenarios: {scenario_count} probability scenarios")
        if require_any_margins:
            print(f"Require: All 3 margins (gross, EBITDA, net)")
        if require_key_drivers:
            print(f"Require: {min_drivers}-{max_drivers} key drivers")
    
    try:
        # Generate report
        result = generate_report(ticker)
        
        if verbose:
            print(f"  → Generated {len(result['sections'])} sections")
            print(f"  → Used {len(result['internal_sources'])} internal + {len(result['web_sources'])} web sources")
            print(f"  → Evaluating...")
        
        # Run basic evaluations
        sections = check_section_presence(result['full_text'])
        completeness = calculate_section_completeness(result['full_text'])
        
        if ground_truth_facts:
            facts = verify_facts(result['full_text'], ground_truth_facts)
        else:
            facts = {"accuracy": None}
        
        data_backing = count_data_backed_claims(result['full_text'])
        
        # Format sources
        sources = []
        for fname, location in result['internal_sources']:
            sources.append({"type": "internal", "file": fname, "location": location})
        for title, url in result['web_sources']:
            sources.append({"type": "web", "title": title, "url": url})
        
        source_quality = evaluate_source_quality(sources)
        length = check_length_appropriateness(result['full_text'], min_words, max_words)
        llm_judge = llm_judge_quality(result['full_text'], f"Generate comprehensive report for {ticker}")
        terms_check = check_must_contain_terms(result['full_text'], must_contain_terms)
        
        # Scenario checks
        if require_scenarios:
            scenarios_check = check_probability_scenarios(result['full_text'], scenario_count)
        else:
            scenarios_check = None
        
        # NEW: Run additional checks if required
        margins_check = check_any_margins_cited(result['full_text']) if require_any_margins else None
        market_share_check = check_market_share_mentioned(result['full_text']) if require_market_share else None
        drivers_check = check_key_drivers_count(result['full_text'], min_drivers, max_drivers) if require_key_drivers else None
        opps_risks_check = check_opportunities_and_risks(result['full_text'], min_opps_risks) if require_opps_risks else None
        scenario_types_check = check_scenario_completeness(result['full_text']) if require_scenario_types else None
        stock_price_check = check_stock_price_implications(result['full_text']) if require_stock_price else None
        competitor_comp_check = check_competitor_comparison(result['full_text']) if require_competitor_comp else None
        
        # Calculate overall score with NEW weights
        weights = {
            'completeness': 0.20,
            'facts': 0.10,
            'data_backing': 0.10,
            'sources': 0.08,
            'llm_judge': 0.08,
            'length': 0.04,
            'terms': 0.08,
            'scenarios': 0.08,
            'margins': 0.06,
            'market_share': 0.04,
            'key_drivers': 0.06,
            'opps_risks': 0.04,
            'scenario_types': 0.02,
            'stock_price': 0.02
        }
        
        overall = completeness * weights['completeness']
        
        if facts['accuracy'] is not None:
            overall += facts['accuracy'] * weights['facts']
        
        overall += data_backing['ratio'] * weights['data_backing']
        overall += source_quality['score'] * weights['sources']
        overall += (llm_judge['overall'] / 5.0) * weights['llm_judge']
        overall += length['score'] * weights['length']
        overall += terms_check['precision'] * weights['terms']
        
        # Scenarios
        if scenarios_check:
            scenario_score = 1.0 if scenarios_check['passed'] else 0.5
            overall += scenario_score * weights['scenarios']
        
        # NEW: Add scores for additional metrics
        if margins_check and margins_check['passed']:
            overall += weights['margins']
        if market_share_check and market_share_check['passed']:
            overall += weights['market_share']
        if drivers_check and drivers_check['passed']:
            overall += weights['key_drivers']
        if opps_risks_check and opps_risks_check['passed']:
            overall += weights['opps_risks']
        if scenario_types_check and scenario_types_check['passed']:
            overall += weights['scenario_types']
        if stock_price_check and stock_price_check['passed']:
            overall += weights['stock_price']
        
        passed = overall >= 0.7 # 0.75
        
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
            
            if must_contain_terms:
                print(f"  Terms Coverage:       {terms_check['precision']:.1%}")
                if terms_check['missing']:
                    print(f"  Missing Terms:        {', '.join(terms_check['missing'])}")
            
            # Scenario checks
            if scenarios_check:
                print(f"  Scenarios Check:      {'✓ PASSED' if scenarios_check['passed'] else '✗ FAILED'}")
                print(f"  Probabilities Found:  {', '.join(scenarios_check['probabilities_found'])}")
                print(f"  Sum:                  {scenarios_check['sum']}%")
            
            # NEW: Show additional metric results
            if margins_check:
                print(f"  All Margins Cited:    {'✓ PASSED' if margins_check['passed'] else '✗ FAILED'}")
                if margins_check['missing_margins']:
                    print(f"    Missing:            {', '.join(margins_check['missing_margins'])}")
            
            if market_share_check:
                print(f"  Market Share:         {'✓ PASSED' if market_share_check['passed'] else '✗ FAILED'}")
                if market_share_check['market_share_value']:
                    print(f"    Value:              {market_share_check['market_share_value']}%")
            
            if drivers_check:
                print(f"  Key Drivers:          {'✓ PASSED' if drivers_check['passed'] else '✗ FAILED'} ({drivers_check['driver_count']} found)")
            
            if opps_risks_check:
                print(f"  Opportunities:        {opps_risks_check['opportunities_count']} found")
                print(f"  Risks:                {opps_risks_check['risks_count']} found")
                print(f"  Opps & Risks:         {'✓ PASSED' if opps_risks_check['passed'] else '✗ FAILED'}")
            
            if scenario_types_check:
                print(f"  Scenario Types:       {'✓ PASSED' if scenario_types_check['passed'] else '✗ FAILED'}")
                if scenario_types_check['missing_scenarios']:
                    print(f"    Missing:            {', '.join(scenario_types_check['missing_scenarios'])}")
            
            if stock_price_check:
                print(f"  Stock Price Disc:     {'✓ PASSED' if stock_price_check['passed'] else '✗ FAILED'}")
            
            if competitor_comp_check:
                print(f"  Competitor Comp:      {'✓ PASSED' if competitor_comp_check['passed'] else '✗ FAILED'}")
            
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
                "terms_check": terms_check,
                "scenarios_check": scenarios_check,
                # NEW METRICS
                "margins_check": margins_check,
                "market_share_check": market_share_check,
                "drivers_check": drivers_check,
                "opps_risks_check": opps_risks_check,
                "scenario_types_check": scenario_types_check,
                "stock_price_check": stock_price_check,
                "competitor_comp_check": competitor_comp_check
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


def run_full_evaluation(dataset_path: str = "dataset_reports.jsonl", output_path: str = "evals/results/reports_results.json"):
    """Run full report evaluation suite"""
    
    print("\n" + "="*70)
    print("ENHANCED REPORT GENERATION EVALUATION")
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
    
    summary = {
        "total_tests": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "pass_rate": accuracy,
        "avg_overall_score": avg_overall
    }
    
    # Save results
    output = {"summary": summary, "results": results}
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
    print(f"Avg Overall Score:    {avg_overall:.1%}")
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
    parser = argparse.ArgumentParser(description="Run enhanced report evaluation suite")
    parser.add_argument('--dataset', default='dataset_reports.jsonl')
    parser.add_argument('--output', default='results/reports_results.json')
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