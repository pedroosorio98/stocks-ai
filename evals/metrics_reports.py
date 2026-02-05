# -*- coding: utf-8 -*-
"""
Metrics for Report Generation Evaluation
Comprehensive metrics for PDF report quality
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Any
import re
from openai import OpenAI

client = OpenAI()


# Required sections for investment reports
REQUIRED_SECTIONS = {
    "core_description": [
        "what the company does",
        "business",
        "products",
        "services",
        "markets",
        "operates"
    ],
    "historical_context": [
        "revenue",
        "margin",
        "performance",
        "trends",
        "historical",
        "financial",
        "growth"
    ],
    "key_drivers": [
        "drivers",
        "factors",
        "performance",
        "metrics",
        "kpi"
    ],
    "outlook": [
        "outlook",
        "future",
        "opportunities",
        "risks",
        "threats",
        "challenges"
    ]
}


def check_section_presence(report_text: str) -> Dict[str, bool]:
    """
    Check if required sections are present in report
    """
    report_lower = report_text.lower()
    
    sections_found = {}
    for section_name, keywords in REQUIRED_SECTIONS.items():
        # Count how many keywords appear
        matches = sum(1 for kw in keywords if kw in report_lower)
        # Section is "found" if at least 2 keywords appear
        sections_found[section_name] = matches >= 2
    
    return sections_found


def calculate_section_completeness(report_text: str) -> float:
    """
    Calculate completeness score (0-1) based on sections present
    """
    sections = check_section_presence(report_text)
    return sum(sections.values()) / len(sections)


def verify_facts(report_text: str, ground_truth_facts: List[Dict]) -> Dict[str, Any]:
    """
    Verify factual claims against ground truth
    
    ground_truth_facts format:
    [
        {"fact": "NVIDIA revenue exceeded $60B in 2024", "verifiable": true},
        {"fact": "Main competitor is AMD", "verifiable": true}
    ]
    """
    verified = []
    missing = []
    
    for fact_obj in ground_truth_facts:
        fact = fact_obj['fact']
        
        # Extract key entities/numbers
        entities = re.findall(r'\b[A-Z]{2,}\b|\$?\d+[BM]?\b|\d{4}\b', fact)
        
        # Check if entities appear in report
        found_count = sum(1 for entity in entities if entity in report_text)
        
        # Fact is verified if at least 50% of entities present
        if found_count >= len(entities) * 0.5:
            verified.append(fact)
        else:
            missing.append(fact)
    
    accuracy = len(verified) / len(ground_truth_facts) if ground_truth_facts else None
    
    return {
        "verified_facts": verified,
        "missing_facts": missing,
        "accuracy": accuracy,
        "total": len(ground_truth_facts)
    }


def count_data_backed_claims(report_text: str) -> Dict[str, Any]:
    """
    Count claims that have data backing (numbers, dates, sources)
    """
    sentences = re.split(r'[.!?]+', report_text)
    
    total_claims = 0
    backed_claims = 0
    
    for sentence in sentences:
        # Skip very short sentences
        if len(sentence.split()) < 5:
            continue
        
        # Skip section headers
        if sentence.strip().startswith('#'):
            continue
        
        total_claims += 1
        
        # Check for data backing
        has_number = bool(re.search(r'\d+[%$]?|\d+\.\d+', sentence))
        has_date = bool(re.search(r'\b(20\d{2}|Q[1-4]|FY\d{2})\b', sentence))
        has_source = bool(re.search(r'\b(according to|source:|from|reported|10-K|10-Q|earnings)\b', sentence, re.I))
        
        if has_number or has_date or has_source:
            backed_claims += 1
    
    ratio = backed_claims / total_claims if total_claims > 0 else 0
    
    return {
        "total_claims": total_claims,
        "backed_claims": backed_claims,
        "ratio": ratio
    }


def evaluate_source_quality(sources: List[Dict]) -> Dict[str, Any]:
    """
    Evaluate quality and diversity of sources
    
    sources format:
    [
        {"type": "internal", "file": "NVDA_10K.pdf"},
        {"type": "web", "url": "https://...", "title": "..."}
    ]
    """
    if not sources:
        return {
            "score": 0,
            "internal_count": 0,
            "web_count": 0,
            "total": 0
        }
    
    internal = [s for s in sources if s.get('type') == 'internal']
    web = [s for s in sources if s.get('type') == 'web']
    
    # Quality score: prefer mix of both
    has_internal = len(internal) > 0
    has_web = len(web) > 0
    
    if has_internal and has_web:
        score = 1.0
    elif has_internal or has_web:
        score = 0.7
    else:
        score = 0.0
    
    return {
        "score": score,
        "internal_count": len(internal),
        "web_count": len(web),
        "total": len(sources)
    }


def check_length_appropriateness(report_text: str, min_words: int = 1500, max_words: int = 3000) -> Dict[str, Any]:
    """
    Check if report length is appropriate
    """
    word_count = len(report_text.split())
    
    if word_count < min_words:
        score = word_count / min_words
        status = "too_short"
    elif word_count > max_words:
        score = max(0.7, max_words / word_count)  # Minimum 70% if too long
        status = "too_long"
    else:
        score = 1.0
        status = "appropriate"
    
    return {
        "word_count": word_count,
        "status": status,
        "score": score,
        "min_expected": min_words,
        "max_expected": max_words
    }


def llm_judge_quality(report_text: str, query: str) -> Dict[str, Any]:
    """
    Use GPT-4 to evaluate report quality
    """
    
    # Truncate report if too long
    truncated_text = report_text[:4000] if len(report_text) > 4000 else report_text
    
    prompt = f"""You are evaluating a company analysis report for investment research quality.

Query: {query}

Report (truncated):
{truncated_text}

Rate the report on the following dimensions (1-5 scale):

1. **Professional Quality**: Does this read like professional investment research?
2. **Depth of Analysis**: Does it go beyond surface facts to provide insights?
3. **Data-Backed**: Are claims supported by specific data and numbers?
4. **Completeness**: Does it cover all material aspects?
5. **Clarity**: Is it well-structured and easy to understand?

Respond ONLY in JSON format:
{{
  "professional_quality": <score 1-5>,
  "depth": <score 1-5>,
  "data_backed": <score 1-5>,
  "completeness": <score 1-5>,
  "clarity": <score 1-5>,
  "overall": <average score>,
  "brief_feedback": "<1-2 sentence summary>"
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        
        return result
        
    except Exception as e:
        print(f"LLM judge failed: {e}")
        return {
            "professional_quality": 0,
            "depth": 0,
            "data_backed": 0,
            "completeness": 0,
            "clarity": 0,
            "overall": 0,
            "brief_feedback": f"Error: {str(e)}"
        }


def calculate_citation_accuracy(report_text: str, sources: List[Dict]) -> Dict[str, Any]:
    """
    Check if citations in report match actual sources
    """
    # Extract citations from report
    citations = re.findall(r'\[([^\]]+)\]|\(([^\)]+)\)', report_text)
    citations = [c for group in citations for c in group if c]
    
    # Get source identifiers
    source_names = []
    for s in sources:
        if 'file' in s:
            source_names.append(s['file'])
        if 'url' in s:
            source_names.append(s['url'])
        if 'title' in s:
            source_names.append(s['title'])
    
    # Check citations
    validated = []
    hallucinated = []
    
    for citation in citations:
        # Check if citation matches any source
        if any(citation.lower() in str(src).lower() for src in source_names):
            validated.append(citation)
        else:
            hallucinated.append(citation)
    
    accuracy = len(validated) / len(citations) if citations else 1.0
    
    return {
        "total_citations": len(citations),
        "validated": len(validated),
        "hallucinated": len(hallucinated),
        "accuracy": accuracy
    }


def check_must_contain_terms(report_text: str, required_terms: List[str]) -> Dict[str, Any]:
    """
    Check if report contains required keywords/concepts
    Similar to Q&A must_contain but for longer reports
    
    Args:
        report_text: The full report text
        required_terms: List of important terms/concepts that should appear
    
    Returns:
        {
            "found": ["GPU", "data center"],
            "missing": ["quantum computing"],
            "precision": 0.67,
            "passed": False
        }
    """
    if not required_terms:
        return {
            "found": [],
            "missing": [],
            "precision": 1.0,
            "passed": True
        }
    
    report_lower = report_text.lower()
    
    found = []
    missing = []
    
    for term in required_terms:
        if term.lower() in report_lower:
            found.append(term)
        else:
            missing.append(term)
    
    precision = len(found) / len(required_terms) if required_terms else 1.0
    
    return {
        "found": found,
        "missing": missing,
        "precision": precision,
        "passed": len(missing) == 0
    }


def check_probability_scenarios(report_text: str, expected_count: int = 3) -> Dict[str, Any]:
    """
    Check if report contains probability scenarios with percentages
    
    Looks for patterns like:
    - "30% probability"
    - "Base case (50%)"
    - "Bear case: 25%"
    - "Bullish scenario - 40%"
    
    Args:
        report_text: Full report text
        expected_count: Number of scenarios expected (default: 3)
    
    Returns:
        {
            "probabilities_found": ["30%", "50%", "20%"],
            "count": 3,
            "expected_count": 3,
            "sum": 100.0,
            "sums_to_100": True,
            "passed": True
        }
    """
    # Pattern to find percentages in scenario contexts
    # Looks for: number followed by % near scenario-related words
    scenario_keywords = r'(scenario|case|probability|likelihood|chance|outlook)'
    
    # Find all percentage values near scenario keywords
    # Pattern: finds "XX%" where XX is 1-3 digits
    percentage_pattern = r'(\d{1,3})%'
    
    # Split text into sentences/sections
    lines = report_text.lower().split('\n')
    
    probabilities_found = []
    scenario_lines = []
    
    for line in lines:
        # Check if line mentions scenarios
        if re.search(scenario_keywords, line, re.IGNORECASE):
            # Find percentages in this line
            percentages = re.findall(percentage_pattern, line)
            for pct in percentages:
                prob_value = int(pct)
                # Only consider reasonable probability values (1-100%)
                if 1 <= prob_value <= 100:
                    probabilities_found.append(f"{pct}%")
                    scenario_lines.append(line.strip()[:100])  # Keep first 100 chars
    
    # Remove duplicates while preserving order
    unique_probs = []
    seen = set()
    for p in probabilities_found:
        if p not in seen:
            unique_probs.append(p)
            seen.add(p)
    
    # Calculate sum of probabilities
    prob_values = [int(p.replace('%', '')) for p in unique_probs]
    prob_sum = sum(prob_values)
    
    # Check if they sum to ~100% (allow small rounding errors)
    sums_to_100 = 95 <= prob_sum <= 105
    
    # Passed if we found the expected count and they sum to 100
    count_match = len(unique_probs) == expected_count
    passed = count_match and sums_to_100
    
    return {
        "probabilities_found": unique_probs,
        "scenario_lines": scenario_lines[:expected_count],  # Show where found
        "count": len(unique_probs),
        "expected_count": expected_count,
        "sum": prob_sum,
        "sums_to_100": sums_to_100,
        "count_match": count_match,
        "passed": passed
    }