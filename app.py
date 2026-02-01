# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 16:52:36 2026

@author: Pedro
"""

from flask import Flask, render_template, request, jsonify
from pathlib import Path
from openai import OpenAI
from rag import retrieve_internal, retrieve_web_exa, build_context

app = Flask(__name__)
client = OpenAI()

def answer(query: str, use_web: bool = True):
    """RAG-powered answer function"""
    internal = retrieve_internal(query, k=12)
    web = retrieve_web_exa(query, k=4) if use_web else []

    # DEBUG: Show what was retrieved (will appear in Flask console)
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


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/prompt', methods=['POST'])
def handle_prompt():
    """Handle AI prompt with RAG system"""
    try:
        prompt_data = request.json.get('prompt')
        
        if not prompt_data or not prompt_data.strip():
            return jsonify({'error': 'Empty prompt'}), 400
        
        # Use the RAG system to generate response
        use_web = request.json.get('use_web', True)  # Allow frontend to toggle web search
        response = answer(prompt_data, use_web=use_web)
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
    
    except Exception as e:
        print(f"Error in handle_prompt: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/api/report', methods=['POST'])
def generate_report():
    """Generate company analysis report"""
    company_name = request.json.get('company')
    # TODO: Implement report generation logic using RAG
    report_link = f"/static/reports/{company_name}_analysis_report.pdf"
    return jsonify({'report_link': report_link})


@app.route('/api/chart', methods=['POST'])
def create_chart():
    """Create chart from data"""
    chart_data = request.json.get('data')
    # TODO: Process chart data with AI
    return jsonify({
        'data': [{
            'x': [1, 2, 3],
            'y': [2, 3, 5],
            'type': 'scatter'
        }]
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
