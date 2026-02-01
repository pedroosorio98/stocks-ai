# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 20:13:20 2026

@author: Pedro
"""

import json
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_results(results_path: str = "evals/results.json"):
    """Analyze and visualize eval results"""
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    summary = data["summary"]
    results = data["results"]
    
    # Group by pass/fail
    passed = [r for r in results if r["passed"]]
    failed = [r for r in results if not r["passed"]]
    
    print(f"Passed tests: {len(passed)}")
    print(f"Failed tests: {len(failed)}")
    
    # Analyze failure reasons
    if failed:
        print("\nFailed Test Cases:")
        for r in failed:
            print(f"  - {r['id']}: {r['source_check']['missing']}")
    
    # Plot metrics
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pass/Fail pie chart
    axes[0].pie([len(passed), len(failed)], 
                labels=['Passed', 'Failed'],
                colors=['#4CAF50', '#F44336'],
                autopct='%1.1f%%')
    axes[0].set_title('Test Results')
    
    # Precision/Recall bar chart
    metrics = ['Precision', 'Recall']
    values = [summary['avg_precision'], summary['avg_recall']]
    axes[1].bar(metrics, values, color=['#2196F3', '#FF9800'])
    axes[1].set_ylim([0, 1])
    axes[1].set_title('Average Metrics')
    
    plt.tight_layout()
    plt.savefig('evals/results_chart.png')
    print("\nChart saved to evals/results_chart.png")

if __name__ == "__main__":
    analyze_results()
