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
import traceback
import json
import sys

# Add charting_tool folder to path
sys.path.insert(0, str(Path(__file__).parent / "charting_tool"))

# Import chart agent
try:
    from chart_agent import create_chart_from_prompt
    print("[STARTUP] ✅ Chart agent loaded successfully")
except ImportError as e:
    print(f"[STARTUP] ⚠️ Chart agent not available: {e}")
    create_chart_from_prompt = None

app = Flask(__name__)
client = OpenAI()

# File-based storage for report jobs (survives Railway restarts)
JOBS_FILE = Path("report_jobs.json")

# Store for tracking report generation jobs
report_jobs = {}

def save_jobs():
    """Save jobs to file for persistence across Railway restarts"""
    try:
        with open(JOBS_FILE, 'w', encoding='utf-8') as f:
            json.dump(report_jobs, f, indent=2)
        print(f"[JOBS] Saved {len(report_jobs)} jobs to {JOBS_FILE}")
    except Exception as e:
        print(f"[JOBS] Error saving jobs: {e}")

def load_jobs():
    """Load jobs from file on startup"""
    global report_jobs
    try:
        if JOBS_FILE.exists():
            with open(JOBS_FILE, 'r', encoding='utf-8') as f:
                report_jobs = json.load(f)
            print(f"[JOBS] Loaded {len(report_jobs)} jobs from {JOBS_FILE}")
        else:
            print(f"[JOBS] No existing jobs file found, starting fresh")
    except Exception as e:
        print(f"[JOBS] Error loading jobs: {e}")
        report_jobs = {}

# Load existing jobs on startup
load_jobs()

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

    prompt = f"""Answer in ENGLISH only.
        
                You are a helpful research assistant.
                Answer the user using the provided context.
                If something is not in context, say you don't know.
                Always include a short 'Sources' section listing the INTERNAL and WEB citations you used.
                
                User question: {query}
                
                Context:
                {context}
                """

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful research assistant. "
                    "Always respond in English only, regardless of the user's language. "
                    "Do not translate citations; keep sources exactly as given."
                )
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content

def generate_report_async(job_id: str, ticker: str):
    """
    Background function to generate report
    Updates job_jobs dict with status and persists to file
    """
    try:
        print(f"[REPORT JOB {job_id}] Starting generation for {ticker}")
        report_jobs[job_id]['status'] = 'generating'
        save_jobs()  # Persist status change
        
        # Generate the report
        generator = CompanyReportGenerator(ticker)
        pdf_path = generator.generate_full_report()
        
        # Update job status
        report_jobs[job_id]['status'] = 'completed'
        report_jobs[job_id]['pdf_path'] = str(pdf_path)
        report_jobs[job_id]['filename'] = pdf_path.name
        save_jobs()  # Persist completion
        
        print(f"[REPORT JOB {job_id}] Completed: {pdf_path.name}")
        
    except Exception as e:
        print(f"[REPORT JOB {job_id}] ERROR: {e}")
        print(f"[REPORT JOB {job_id}] Full traceback:")
        traceback.print_exc()  # Print full stack trace
        
        report_jobs[job_id]['status'] = 'failed'
        report_jobs[job_id]['error'] = str(e)
        save_jobs()  # Persist error state


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/prompt', methods=['POST'])
def handle_prompt():
    try:
        data = request.get_json(silent=True) or {}
        prompt_data = (data.get('prompt') or '').strip()

        if not prompt_data:
            return jsonify({'error': 'Empty prompt'}), 400

        use_web = bool(data.get('use_web', True))
        response = answer(prompt_data, use_web=use_web)

        return jsonify({'response': response, 'status': 'success'})

    except Exception as e:
        print("ERROR in handle_prompt:", repr(e))
        traceback.print_exc()   # <<< this is the important line
        return jsonify({'error': str(e), 'status': 'error'}), 500

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
        save_jobs()  # Persist new job immediately
        
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
    print(f"[STATUS CHECK] Checking job: {job_id}")
    print(f"[STATUS CHECK] Current jobs in memory: {list(report_jobs.keys())}")
    
    if job_id not in report_jobs:
        print(f"[STATUS CHECK] Job {job_id} NOT FOUND")
        print(f"[STATUS CHECK] Available jobs: {list(report_jobs.keys())}")
        return jsonify({
            'error': 'Job not found - it may have been lost due to server restart',
            'job_id': job_id,
            'hint': 'Try generating the report again'
        }), 404
    
    job = report_jobs[job_id]
    print(f"[STATUS CHECK] Job {job_id} status: {job['status']}")
    
    response = {
        'job_id': job_id,
        'status': job['status'],
        'company': job['company'],
        'ticker': job['ticker']
    }
    
    if job['status'] == 'completed':
        response['filename'] = job['filename']
        response['download_url'] = f'/api/report/download/{job_id}'
        print(f"[STATUS CHECK] Job completed: {job['filename']}")
    
    if job['status'] == 'failed':
        response['error'] = job['error']
        print(f"[STATUS CHECK] Job failed: {job['error']}")
    
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
    """
    AI-powered chart generation from natural language prompt
    
    Request:
        {
            "prompt": "Plot NVDA revenue together with NVDA stock price, stock price on secondary axis"
        }
    
    Response:
        {
            "success": true,
            "data": {...},  // Plotly JSON
            "code": "...",  // Generated Python code (for debugging)
            "data_sources": {...}  // Identified data sources
        }
    """
    try:
        prompt = request.json.get('prompt', '').strip()
        
        if not prompt:
            return jsonify({
                'success': False,
                'error': 'Empty prompt. Please enter a chart description (e.g., "Plot NVDA revenue vs stock price")'
            }), 400
        
        # Check if chart agent is available
        if create_chart_from_prompt is None:
            return jsonify({
                'success': False,
                'error': 'Chart agent not loaded. Make sure chart_agent.py is in the charting_tool folder.'
            }), 500
        
        print(f"\n[CHART API] Received prompt: {prompt}")
        
        # Create chart using AI agent
        result = create_chart_from_prompt(prompt)
        
        if result['success']:
            print(f"[CHART API] ✅ Chart created successfully")
            return jsonify(result)
        else:
            print(f"[CHART API] ❌ Chart creation failed: {result.get('error')}")
            return jsonify(result), 500
            
    except Exception as e:
        import traceback
        print(f"[CHART API] ❌ Error: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


# WARNING: debug=True causes Flask to reload, which clears report_jobs dict!
# For production, set debug=False or use use_reloader=False
if __name__ == '__main__':
    #app.run(debug=True, host='0.0.0.0', port=5000)
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)