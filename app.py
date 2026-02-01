# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 16:52:36 2026

@author: Pedro
"""

from flask import Flask, render_template, request, jsonify, send_file
from pathlib import Path
from openai import OpenAI
from rag import retrieve_internal, retrieve_web_exa, build_context
from reports import CompanyReportGenerator
import threading
import uuid

app = Flask(__name__)
client = OpenAI()

# Store for tracking report generation jobs
report_jobs = {}

# Mapping of company names to tickers
COMPANY_TO_TICKER = {
    # US Companies
    "NVIDIA": "NVDA",
    "Microsoft": "MSFT",
    "Apple": "AAPL",
    "Alphabet": "GOOGL",
    "Amazon": "AMZN",
    "Meta Platforms": "META",
    "Broadcom": "AVGO",
    "Tesla": "TSLA",
    "Berkshire Hathaway": "BRK.B",
    "JPMorgan Chase": "JPM",
    
    # Brazilian Companies
    "Nu Holdings": "NU",
    "Petrobras": "PETR4",
    "Itaú Unibanco": "ITUB4",
    "Vale": "VALE3",
    "BTG Pactual": "BPAC11",
    "Banco Santander Brasil": "SANB11",
    "Ambev": "ABEV3",
    "Banco Bradesco": "BBDC4",
    "WEG": "WEGE3",
    "Klabin": "KLBN11",
}


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

def generate_report_async(job_id: str, ticker: str):
    """
    Background function to generate report
    Updates job_jobs dict with status
    """
    try:
        print(f"[REPORT JOB {job_id}] Starting generation for {ticker}")
        report_jobs[job_id]['status'] = 'generating'
        
        # Generate the report
        generator = CompanyReportGenerator(ticker)
        pdf_path = generator.generate_full_report()
        
        # Update job status
        report_jobs[job_id]['status'] = 'completed'
        report_jobs[job_id]['pdf_path'] = str(pdf_path)
        report_jobs[job_id]['filename'] = pdf_path.name
        
        print(f"[REPORT JOB {job_id}] Completed: {pdf_path.name}")
        
    except Exception as e:
        print(f"[REPORT JOB {job_id}] Error: {e}")
        report_jobs[job_id]['status'] = 'failed'
        report_jobs[job_id]['error'] = str(e)


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
    """
    Generate company analysis report
    Returns job_id for tracking progress
    """
    try:
        company_name = request.json.get('company')
        
        if not company_name:
            return jsonify({'error': 'No company specified'}), 400
        
        # Convert company name to ticker
        ticker = COMPANY_TO_TICKER.get(company_name)
        
        if not ticker:
            return jsonify({'error': f'Unknown company: {company_name}'}), 400
        
        # Create unique job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job tracking
        report_jobs[job_id] = {
            'status': 'pending',
            'company': company_name,
            'ticker': ticker,
            'pdf_path': None,
            'filename': None,
            'error': None
        }
        
        # Start report generation in background thread
        thread = threading.Thread(
            target=generate_report_async,
            args=(job_id, ticker)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'success',
            'job_id': job_id,
            'message': f'Report generation started for {company_name} ({ticker})'
        })
        
    except Exception as e:
        print(f"Error in generate_report: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/api/report/status/<job_id>', methods=['GET'])
def report_status(job_id):
    """
    Check status of report generation job
    """
    if job_id not in report_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = report_jobs[job_id]
    
    response = {
        'job_id': job_id,
        'status': job['status'],
        'company': job['company'],
        'ticker': job['ticker']
    }
    
    if job['status'] == 'completed':
        response['filename'] = job['filename']
        response['download_url'] = f'/api/report/download/{job_id}'
    
    if job['status'] == 'failed':
        response['error'] = job['error']
    
    return jsonify(response)


@app.route('/api/report/download/<job_id>', methods=['GET'])
def download_report(job_id):
    """
    Download generated report PDF
    """
    if job_id not in report_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = report_jobs[job_id]
    
    if job['status'] != 'completed':
        return jsonify({'error': 'Report not ready'}), 400
    
    pdf_path = Path(job['pdf_path'])
    
    if not pdf_path.exists():
        return jsonify({'error': 'Report file not found'}), 404
    
    return send_file(
        pdf_path,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=job['filename']
    )


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
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)