# Stocks-AI — RAG Research Reports Agent

Stocks-AI is a Python project that:
1) Scrapes data for every company from **SEC fillings, Alpha Vantage, Yahoo Finance and companies IR websites**.
1) Builds a local **RAG** index (FAISS + metadata) from all the files located inside the "Data" folders, saves the index.faiss and internal_meta.jsonl in S3.
2) Generates **investment-style research reports** using retrieved internal chunks + web search (Exa and OpenAI). This is the actual agent that performs the reports according to various instructions.
3) Runs **evals** from the evals folder to access how good is the reports generator performing. The evals assesses things like: "contains 3 scenarios", "contains market share", "contains all three margins" etc.
4) Includes a **Charting Tool** that turns prompts into interative Plotly charts by using Alpha Vantage and Yahoo Finance data.

---

## Project Structure (high-level)

- `rag.py`  
  Core retrieval: OpenAI embeddings + FAISS for internal search, optional Exa and OpenAI for web search, and context building.

- `build_index.py`  
  Builds your FAISS index (`internal.faiss`) and metadata file (`internal_meta.jsonl`) from the "Data" folder.

- `reports.py`  
  The report generator. Produces a full report with multiple sections, sources, and formatting. This is the actual agent that performs the reports according to various instructions.

- `run_evals_reports.py`  
  Runs report evals across a dataset and produces results JSON.

- `metrics_reports.py`  
  Metric functions used by the eval runner (section completeness, length, terms, scenario checks, etc.).

- `dataset_reports.jsonl`  
  The evaluation dataset (one test case per line).

- `reports_results.json`  
  Output produced by running evals (per-test scores, pass/fail, breakdown).

- `charting_tool/` (or `chart_agent.py`)  
  LLM-powered chart generation that executes safe-ish Plotly code to produce charts from prompts.

---

## Requirements

- Python 3.10+ (3.11 recommended)
- OpenAI API key (embeddings + report generation)
- Exa API key (web search)
- AWS credentials for FAISS index/metadata that are stored in S3

---

## Setup

### 1) Create environment
```bash
conda create -n stocks_ai python=3.11 -y
conda activate stocks_ai
