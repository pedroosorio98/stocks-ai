# -*- coding: utf-8 -*-

from typing import List, Dict, Any
import re


def calculate_answer_relevance(query: str, answer: str) -> float:
    """
    Calculate keyword overlap between query and answer
    Returns: 0.0 to 1.0
    """
    query_terms = set(re.findall(r'\b\w+\b', query.lower()))
    answer_terms = set(re.findall(r'\b\w+\b', answer.lower()))
    
    # Remove stopwords
    stopwords = {
        'what', 'is', 'the', 'and', 'or', 'from', 'to', 'a', 'an', 'how', 'why',
        'when', 'where', 'who', 'which', 'about', 'does', 'do', 'did', 'can',
        'could', 'would', 'should', 'tell', 'me', 'you', 'it', 'this', 'that'
    }
    query_terms -= stopwords
    
    if not query_terms:
        return 1.0
    
    overlap = query_terms & answer_terms
    return len(overlap) / len(query_terms)


def check_sources_cited(answer: str, expected_tickers: List[str]) -> Dict[str, Any]:
    """
    DEPRECATED: Use check_retrieval_accuracy instead
    This function checks if tickers appear in answer text
    But you should check if data FROM those tickers was retrieved
    """
    found = []
    missing = []
    
    answer_upper = answer.upper()
    
    for ticker in expected_tickers:
        if ticker.upper() in answer_upper:
            found.append(ticker)
        else:
            missing.append(ticker)
    
    precision = len(found) / len(expected_tickers) if expected_tickers else 1.0
    
    return {
        "found": found,
        "missing": missing,
        "precision": precision,
        "passed": len(missing) == 0
    }


def check_has_sources_section(answer: str) -> bool:
    """
    Check if answer includes a Sources/References section
    Simplified: just look for the word "sources" or "references" followed by colon
    """
    answer_lower = answer.lower()
    
    # Simple check: does it have "sources:" or "references:" anywhere?
    if 'sources:' in answer_lower:
        return True
    if 'references:' in answer_lower:
        return True
    if 'citations:' in answer_lower:
        return True
    
    return False


def check_must_contain(answer: str, required_terms: List[str]) -> Dict[str, Any]:
    """
    Check if answer contains required words/phrases
    
    Args:
        answer: The answer text to check
        required_terms: List of words/phrases that must appear (case-insensitive)
    
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
    
    answer_lower = answer.lower()
    
    found = []
    missing = []
    
    for term in required_terms:
        if term.lower() in answer_lower:
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


def check_retrieval_accuracy(chunks: List[Dict], expected_tickers: List[str]) -> Dict[str, Any]:
    """
    Check if retrieved chunks come from expected tickers' folders
    This is the CORRECT metric - checks if Data/TICKER/ was used
    """
    if not chunks:
        return {
            "retrieved_tickers": [],
            "expected_tickers": expected_tickers,
            "precision": 0.0,
            "passed": False
        }
    
    retrieved_tickers = set()
    
    for chunk in chunks:
        ticker = chunk.get("ticker", "")
        if ticker:
            retrieved_tickers.add(ticker.upper())
    
    # Check how many expected tickers were retrieved
    matches = 0
    found_tickers = []
    missing_tickers = []
    
    for ticker in expected_tickers:
        if ticker.upper() in retrieved_tickers:
            matches += 1
            found_tickers.append(ticker)
        else:
            missing_tickers.append(ticker)
    
    precision = matches / len(expected_tickers) if expected_tickers else 1.0
    
    return {
        "retrieved_tickers": list(retrieved_tickers),
        "expected_tickers": expected_tickers,
        "found": found_tickers,
        "missing": missing_tickers,
        "precision": precision,
        "passed": len(missing_tickers) == 0
    }


def calculate_response_quality(answer: str) -> Dict[str, Any]:
    """
    Calculate basic quality metrics for answer
    """
    word_count = len(answer.split())
    
    # Count numbers (indicates data-backed response)
    numbers = re.findall(r'\d+[.,]?\d*[%$]?', answer)
    has_numbers = len(numbers)
    
    # Count sentences
    sentences = re.split(r'[.!?]+', answer)
    sentence_count = len([s for s in sentences if s.strip()])
    
    # Check for citations
    has_internal_citations = bool(re.search(r'\[INTERNAL\]|\b10-[KQ]\b|Form \d+', answer, re.IGNORECASE))
    has_web_citations = bool(re.search(r'https?://|www\.', answer))
    
    return {
        "word_count": word_count,
        "has_numbers": has_numbers,
        "sentence_count": sentence_count,
        "has_internal_citations": has_internal_citations,
        "has_web_citations": has_web_citations
    }


def calculate_precision_recall(answer: str, expected_facts: List[str]) -> Dict[str, float]:
    """
    Calculate precision/recall for expected facts in answer
    """
    found = 0
    
    for fact in expected_facts:
        # Extract key terms from fact
        fact_terms = set(re.findall(r'\b\w+\b', fact.lower()))
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were'}
        fact_terms -= stopwords
        
        # Check if most terms appear in answer
        answer_lower = answer.lower()
        matches = sum(1 for term in fact_terms if term in answer_lower)
        
        if matches >= len(fact_terms) * 0.6:  # 60% of terms present
            found += 1
    
    precision = found / len(expected_facts) if expected_facts else 1.0
    recall = precision  # Same for this simple check
    
    return {
        "precision": precision,
        "recall": recall,
        "facts_found": found,
        "facts_expected": len(expected_facts)
    }