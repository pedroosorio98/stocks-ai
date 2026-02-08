# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 20:12:58 2026

@author: Pedro
"""

from pathlib import Path
from openai import OpenAI
from rag import retrieve_internal, retrieve_web_exa, build_context

client = OpenAI()

def answer(query: str, use_web: bool = True):
    internal = retrieve_internal(query, k=12)
    web = retrieve_web_exa(query, k=4) if use_web else []

    # DEBUG: Show what was retrieved
    print("\n" + "="*60)
    print("RETRIEVED INTERNAL CHUNKS:")
    print("="*60)
    
    # Group by ticker for better visibility
    by_ticker = {}
    for hit in internal:
        ticker = hit.get('ticker', 'N/A')
        if ticker not in by_ticker:
            by_ticker[ticker] = []
        by_ticker[ticker].append(hit)
    
    for ticker, hits in by_ticker.items():
        print(f"\n{ticker} ({len(hits)} chunks):")
        for i, hit in enumerate(hits, 1):
            source = Path(hit.get('source', '')).name
            score = hit.get('score', 0)
            lang = hit.get('query_lang', '?')
            print(f"  {i:2d}. {source:30s} | score: {score:.3f} | via: {lang}")
    
    print("="*60 + "\n")

    context = build_context(internal, web)

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
    return resp.choices[0].message.content


if __name__ == "__main__":
    while True:
        q = input("\nAsk> ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break
        print("\n" + answer(q, use_web=True))