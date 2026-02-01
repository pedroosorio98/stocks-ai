# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 20:11:56 2026

@author: Pedro
"""

from typing import List, Dict
import re

def calculate_answer_relevance(query: str, answer: str) -> float:
    """
    Simple keyword-based relevance score
    Checks if key terms from query appear in answer
    """
    query_terms = set(re.findall(r'\w+', query.lower()))
    answer_terms = set(re.findall(r'\w+', answer.lower()))
    
    # Remove common stopwords
    stopwords = {'what', 'is', 'the', 'and', 'or', 'from', 'to', 'a', 'an'}
    query_terms -= stopwords
    
    if not query_terms:
        return 1.0
    
    overlap = query_terms & answer_terms
    return len(overlap) / len(query_terms)

def calculate_retrieval_accuracy(retrieved_chunks: List[Dict], expected_sources: List[str]) -> Dict:
    """
    Check if retrieved chunks actually contain expected companies
    """
    retrieved_sources = set()
    
    for chunk in retrieved_chunks:
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})
        
        # Extract company ticker from text or metadata
        for source in expected_sources:
            if source.upper() in text.upper() or source.upper() in str(metadata).upper():
                retrieved_sources.add(source)
    
    precision = len(retrieved_sources) / len(expected_sources) if expected_sources else 1.0
    
    return {
        "retrieved_sources": list(retrieved_sources),
        "expected_sources": expected_sources,
        "precision": precision
    }

def calculate_context_length(retrieved_chunks: List[Dict]) -> int:
    """Calculate total context length"""
    return sum(len(chunk.get("text", "")) for chunk in retrieved_chunks)
