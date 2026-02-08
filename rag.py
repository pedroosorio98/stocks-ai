# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 20:13:50 2026

@author: Pedro
"""

import os, json
from pathlib import Path
import numpy as np
import time
import faiss
from openai import OpenAI
from typing import List, Set
import boto3
from botocore.exceptions import ClientError

client = OpenAI()

BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "index"
FAISS_PATH = INDEX_DIR / "internal.faiss"
META_PATH = INDEX_DIR / "internal_meta.jsonl"

# S3 Configuration
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "osorio-stocks-research")
S3_FAISS_KEY = "internal.faiss"
S3_META_KEY = "internal_meta.jsonl"  # Optional: if you upload metadata too

# Map common company names to tickers
COMPANY_TO_TICKER = {
    # US Companies
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "jpmorgan chase": "JPM",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "berkshire": "BRK.B",
    "berkshire hathaway": "BRK.B",
    "broadcom":"AVGO",
    "aapl": "AAPL",
    "amzn": "AMZN",
    "avgo": "AVGO",
    "brk.b": "BRK.B",
    "googl": "GOOGL",
    "jpm": "JPM",
    "meta": "META",
    "msft": "MSFT",
    "nvda": "NVDA",
    "tsla": "TSLA",
    
    # Brazilian Companies
    "itau": "ITUB4",
    "itaú": "ITUB4",
    "itau unibanco": "ITUB4",
    "bradesco": "BBDC4",
    "banco do brasil": "BBAS3",
    "petrobras": "PETR4",
    "vale": "VALE3",
    "ambev": "ABEV3",
    "weg": "WEGE3",
    "santander brasil": "SANB11",
    "klabin": "KLBN11",
    "nubank": "NU",
    "nu": "NU",
    "btg":"BPAC11",
    "btgpactual":"BPAC11",
    "btg pactual":"BPAC11",
    "abev3": "ABEV3",
    "bbdc4": "BBDC4",
    "bpac11": "BPAC11",
    "itub4": "ITUB4",
    "klbn11": "KLBN11",
    "petr4": "PETR4",
    "sanb11": "SANB11",
    "vale3": "VALE3",
    "wege3": "WEGE3",
    "nu": "NU",
}


VALID_TICKERS = {
    # US
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "AVGO", "BRK.B",
    # Brazil
    "ITUB4", "BBDC4", "BBAS3", "PETR4", "VALE3", "ABEV3", "WEGE3", "SANB11", "KLBN11", "NU", "BPAC11"
}


def download_from_s3():
    """
    Download FAISS index (and optionally metadata) from S3.
    Only downloads if file doesn't exist locally.
    """
    # Create index directory if it doesn't exist
    INDEX_DIR.mkdir(exist_ok=True)

    faiss_exists = FAISS_PATH.exists()
    meta_exists = META_PATH.exists()  # <-- requires META_PATH global

    # If both already exist, nothing to do
    if faiss_exists and meta_exists:
        print(f"[S3] FAISS index already exists locally: {FAISS_PATH}")
        print(f"[S3] Size: {FAISS_PATH.stat().st_size / (1024**3):.2f} GB")
        print(f"[S3] Metadata already exists locally: {META_PATH}")
        print(f"[S3] Size: {META_PATH.stat().st_size / (1024**2):.2f} MB")
        return True

    # If only FAISS exists, still try metadata
    if faiss_exists and not meta_exists:
        print(f"[S3] FAISS index already exists locally: {FAISS_PATH}")
        print(f"[S3] Size: {FAISS_PATH.stat().st_size / (1024**3):.2f} GB")
        print(f"[S3] Metadata not found locally. Will download from S3...")

    # If only metadata exists, still try FAISS
    if (not faiss_exists) and meta_exists:
        print(f"[S3] Metadata already exists locally: {META_PATH}")
        print(f"[S3] Size: {META_PATH.stat().st_size / (1024**2):.2f} MB")
        print(f"[S3] FAISS index not found locally. Will download from S3...")

    if not faiss_exists:
        print(f"[S3] FAISS index not found locally. Downloading from S3...")
        print(f"[S3] Bucket: {S3_BUCKET}, Key: {S3_FAISS_KEY}")

    if not meta_exists:
        print(f"[S3] Metadata not found locally. Downloading from S3...")
        print(f"[S3] Bucket: {S3_BUCKET}, Key: {S3_META_KEY}")  # <-- requires S3_META_KEY global

    try:
        # Initialize S3 client
        # For AWS S3: uses AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY env vars
        # For Cloudflare R2: set endpoint_url
        s3_config = {}

        # Check if using Cloudflare R2
        r2_endpoint = os.getenv("R2_ENDPOINT_URL")
        if r2_endpoint:
            s3_config["endpoint_url"] = r2_endpoint
            print(f"[S3] Using Cloudflare R2 endpoint: {r2_endpoint}")

        # Get region (default to auto for R2)
        region = os.getenv("AWS_REGION", "auto")
        s3_config["region_name"] = region

        s3 = boto3.client("s3", **s3_config)

        # Download FAISS index (if missing)
        if not faiss_exists:
            print(f"[S3] Downloading {S3_FAISS_KEY}... (this may take 1-2 minutes for 2.8GB)")
            start_time = time.time()

            s3.download_file(
                Bucket=S3_BUCKET,
                Key=S3_FAISS_KEY,
                Filename=str(FAISS_PATH)
            )

            elapsed = time.time() - start_time
            file_size = FAISS_PATH.stat().st_size / (1024**3)  # GB

            print(f"[S3] Downloaded FAISS successfully!")
            print(f"[S3] File: {FAISS_PATH}")
            print(f"[S3] Size: {file_size:.2f} GB")
            print(f"[S3] Time: {elapsed:.1f}s ({file_size/elapsed*60:.1f} MB/s)")

        # Download metadata (if missing)
        if not meta_exists:
            print(f"[S3] Downloading {S3_META_KEY}... (this may take a bit for a large JSONL)")
            start_time = time.time()

            s3.download_file(
                Bucket=S3_BUCKET,
                Key=S3_META_KEY,
                Filename=str(META_PATH)
            )

            elapsed = time.time() - start_time
            meta_size_mb = META_PATH.stat().st_size / (1024**2)

            print(f"[S3] Downloaded metadata successfully!")
            print(f"[S3] File: {META_PATH}")
            print(f"[S3] Size: {meta_size_mb:.2f} MB")
            print(f"[S3] Time: {elapsed:.1f}s ({meta_size_mb/elapsed:.1f} MB/s)")

        return True

    except ClientError as e:
        error_code = e.response['Error']['Code']
        print(f"[S3] Download failed: {error_code}")
        print(f"[S3] Error: {e}")

        if error_code == "NoSuchBucket":
            print(f"[S3] Bucket '{S3_BUCKET}' does not exist!")
        elif error_code == "NoSuchKey":
            # Could be either FAISS or META key — print both to help debugging
            print(f"[S3] One of these files was not found in bucket:")
            print(f"     - FAISS Key: {S3_FAISS_KEY}")
            print(f"     - META  Key: {S3_META_KEY}")
        elif error_code in ["InvalidAccessKeyId", "SignatureDoesNotMatch"]:
            print("[S3] Check your AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")

        return False

    except Exception as e:
        print(f"[S3] Unexpected error: {e}")
        return False

def detect_tickers_from_query(query: str) -> List[str]:
    """
    Detect MULTIPLE tickers or company names from query.
    Returns list of tickers found (can be empty).
    """
    if not query:
        return []
    
    query_lower = query.lower()
    detected = set()
    
    import re
    
    # 1. Detect explicit ticker format: $AAPL, $MSFT
    dollar_tickers = re.findall(r'\$([A-Z\.]{1,6})\b', query.upper())
    for ticker in dollar_tickers:
        if ticker in VALID_TICKERS:
            detected.add(ticker)
            print(f"[TICKER DETECT] Found $ ticker: {ticker}")
    
    # 2. Get all tickers from your metadata (existing tickers in your system)
    all_meta_tickers = set()
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                all_meta_tickers.add(doc["ticker"])
    except Exception:
        # Fallback to predefined list
        all_meta_tickers = VALID_TICKERS
    
    # 3. Check for exact ticker mentions in the query
    for ticker in all_meta_tickers:
        # Match ticker as whole word (e.g., "JPM" but not "JPMORGAN")
        pattern = r'\b' + re.escape(ticker) + r'\b'
        if re.search(pattern, query.upper()):
            detected.add(ticker)
            print(f"[TICKER DETECT] Found exact ticker: {ticker}")
    
    # 4. Check company name mappings
    for company_name, ticker in COMPANY_TO_TICKER.items():
        if company_name in query_lower:
            detected.add(ticker)
            print(f"[TICKER DETECT] Matched company '{company_name}' -> {ticker}")
    
    # 5. Special handling for comparison queries
    # Patterns like "AAPL vs MSFT", "compare AAPL and MSFT"
    comparison_words = ['vs', 'versus', 'compare', 'comparison', 'between', 'and']
    has_comparison = any(word in query_lower for word in comparison_words)
    
    if has_comparison:
        # More aggressive ticker detection in comparison mode
        words = query.upper().split()
        for word in words:
            cleaned = word.strip('.,;:!?()[]{}')
            if cleaned in all_meta_tickers:
                detected.add(cleaned)
    
    result = sorted(list(detected))
    
    if result:
        print(f"[TICKER DETECT] Total tickers detected: {', '.join(result)}")
    else:
        print("[TICKER DETECT] No specific tickers detected, searching all")
    
    return result


# Keep backward compatibility - single ticker detection
def detect_ticker_from_query(query: str) -> str | None:
    """
    Detect single ticker (backward compatible).
    Returns first ticker found, or None.
    """
    tickers = detect_tickers_from_query(query)
    return tickers[0] if tickers else None

def translate_to_portuguese(query: str):
    """
    Translate English query to Portuguese using GPT.
    Falls back to original query if translation fails.
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # Faster and cheaper for translation
            messages=[
                {
                    "role": "system",
                    "content": "Translate the following question to Brazilian Portuguese. Only output the translation, nothing else."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            temperature=0,
            max_tokens=200
        )
        translation = resp.choices[0].message.content.strip()
        print(f"[RAG] Translated query: {translation}")
        return translation
    except Exception as e:
        print(f"[RAG] Translation failed: {e}, using original query")
        return query

def load_meta():
    meta = []
    with open(META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return meta

def embed_query(q: str):
    r = client.embeddings.create(model="text-embedding-3-large", input=[q])
    v = np.array([r.data[0].embedding], dtype="float32")
    faiss.normalize_L2(v)
    return v

def retrieve_internal(query: str, k=6, filter_tickers: List[str] = None):
    """
    Dual-query multilingual retrieval with MULTI-ticker filtering.
    Downloads FAISS index from S3 if not present locally.
    
    Args:
        query: User question
        k: Number of results to return
        filter_tickers: List of tickers to filter by (optional)
    """
    # **NEW: Download from S3 if needed**
    if not FAISS_PATH.exists():
        print("[RAG] FAISS index not found locally. Attempting S3 download...")
        if not download_from_s3():
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_PATH} and S3 download failed. "
                "Check S3 credentials and bucket configuration."
            )
    
    index = faiss.read_index(str(FAISS_PATH))
    meta = load_meta()
    
    # Auto-detect tickers from query if not explicitly provided
    detected_tickers = detect_tickers_from_query(query)
    target_tickers = filter_tickers or detected_tickers
    
    if target_tickers:
        print(f"[RAG] Filtering results to tickers: {', '.join(target_tickers)}")
    
    # Search with English query
    print(f"[RAG] Searching with English query: {query}")
    v_en = embed_query(query)
    # Retrieve MORE results initially (we'll filter after)
    search_k = k * 10 if target_tickers else k
    scores_en, ids_en = index.search(v_en, search_k)
    
    # Search with Portuguese query
    pt_query = translate_to_portuguese(query)
    v_pt = embed_query(pt_query)
    scores_pt, ids_pt = index.search(v_pt, search_k)
    
    # Merge results and deduplicate
    seen = set()
    results = []
    
    # Add English results
    for score, idx in zip(scores_en[0], ids_en[0]):
        if idx != -1 and idx not in seen:
            d = meta[idx].copy()
            
            # Filter by tickers if specified (ANY match)
            if target_tickers and d["ticker"] not in target_tickers:
                continue
            
            d["score"] = float(score)
            d["query_lang"] = "en"
            results.append(d)
            seen.add(idx)
    
    # Add Portuguese results
    for score, idx in zip(scores_pt[0], ids_pt[0]):
        if idx != -1 and idx not in seen:
            d = meta[idx].copy()
            
            # Filter by tickers if specified (ANY match)
            if target_tickers and d["ticker"] not in target_tickers:
                continue
            
            d["score"] = float(score)
            d["query_lang"] = "pt"
            results.append(d)
            seen.add(idx)
    
    # Sort by score (highest first)
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # SMART BALANCING: If multiple tickers, try to get results from each
    if len(target_tickers) > 1:
        results = balance_ticker_results(results, target_tickers, k)
    else:
        results = results[:k]
    
    if target_tickers:
        ticker_counts = {}
        for r in results:
            ticker_counts[r["ticker"]] = ticker_counts.get(r["ticker"], 0) + 1
        print(f"[RAG] Found {len(results)} chunks: {dict(ticker_counts)}")
    else:
        print(f"[RAG] Found {len(results)} unique chunks")
    
    return results


def balance_ticker_results(results: List[dict], tickers: List[str], k: int) -> List[dict]:
    """
    Balance results across multiple tickers for fair comparison.
    
    Strategy: Try to get at least k/n results from each ticker (if available)
    """
    n_tickers = len(tickers)
    per_ticker = max(2, k // n_tickers)  # At least 2 per ticker
    
    # Group by ticker
    by_ticker = {ticker: [] for ticker in tickers}
    for r in results:
        if r["ticker"] in by_ticker:
            by_ticker[r["ticker"]].append(r)
    
    # Take top results from each ticker
    balanced = []
    for ticker in tickers:
        balanced.extend(by_ticker[ticker][:per_ticker])
    
    # Fill remaining slots with best scores
    remaining_slots = k - len(balanced)
    if remaining_slots > 0:
        remaining = [r for r in results if r not in balanced]
        balanced.extend(remaining[:remaining_slots])
    
    # Sort by score again
    balanced.sort(key=lambda x: x["score"], reverse=True)
    
    return balanced[:k]

def retrieve_web_exa(query: str, k: int = 4, retries: int = 3, backoff: float = 1.5):
    """
    Dual-language web search (English + Portuguese) matching retrieve_internal behavior.
    Searches both languages and merges unique results.
    """
    exa_key = os.getenv("EXA_API_KEY") or os.getenv("EXA_KEY") or os.getenv("EXA_AI_API_KEY")
    if not exa_key:
        print("[web] EXA_API_KEY not set. Skipping web.")
        return []

    try:
        from openai import OpenAI
        import re
    except Exception as e:
        print(f"[web] import failed: {e}")
        return []

    exa_client = OpenAI(
        base_url="https://api.exa.ai",
        api_key=exa_key
    )

    def search_exa_single(search_query: str, lang: str):
        """Helper to perform single Exa search and extract sources"""
        try:
            enhanced_query = f"""{search_query}

                            After providing your answer, list all sources you used in this format:
                            SOURCES:
                            1. [Title] - URL
                            2. [Title] - URL
                            etc."""
                                        
            resp = exa_client.chat.completions.create(
                model="exa",
                messages=[{"role": "user", "content": enhanced_query}],
                temperature=0,
                max_tokens=2000
            )
            
            content = resp.choices[0].message.content
            
            # Extract structured sources section
            sources_section = re.search(r'SOURCES?:?\s*\n((?:\d+\..*\n?)+)', content, re.IGNORECASE)
            
            results = []
            if sources_section:
                sources_text = sources_section.group(1)
                source_lines = re.findall(r'\d+\.\s*(.+?)(?:\s*-\s*(https?://\S+)|$)', sources_text)
                
                for title, url in source_lines:
                    title = title.strip('[]')
                    if url:
                        results.append({
                            "title": title,
                            "url": url.strip(),
                            "text": content[:800],  # Include snippet of content
                            "query_lang": lang
                        })
            
            # Fallback: extract any URLs from content
            if not results:
                urls = re.findall(r'https?://[^\s\)\]]+', content)
                for url in urls:
                    domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
                    title = domain_match.group(1) if domain_match else url
                    results.append({
                        "title": title,
                        "url": url,
                        "text": content[:800],
                        "query_lang": lang
                    })
            
            return results
            
        except Exception as e:
            print(f"[web] Exa {lang} search failed: {e}")
            return []

    # Perform searches with retries
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            # Search with English query
            print(f"[web] Searching with English query: {query}")
            en_results = search_exa_single(query, "en")
            
            # Search with Portuguese query
            pt_query = translate_to_portuguese(query)
            print(f"[web] Searching with Portuguese query: {pt_query}")
            pt_results = search_exa_single(pt_query, "pt")
            
            # Merge results and deduplicate by URL (like retrieve_internal deduplicates by idx)
            seen_urls = set()
            merged_results = []
            
            # Add English results first
            for result in en_results:
                url = result.get("url", "")
                if url and url not in seen_urls:
                    merged_results.append(result)
                    seen_urls.add(url)
            
            # Add Portuguese results
            for result in pt_results:
                url = result.get("url", "")
                if url and url not in seen_urls:
                    merged_results.append(result)
                    seen_urls.add(url)
            
            # Limit to k results
            final_results = merged_results[:k]
            
            if final_results:
                en_count = sum(1 for r in final_results if r.get("query_lang") == "en")
                pt_count = sum(1 for r in final_results if r.get("query_lang") == "pt")
                print(f"[web] Found {len(final_results)} unique sources ({en_count} from EN, {pt_count} from PT)")
                return final_results
            
            # If no results found, return empty (don't use fallback)
            print("[web] No sources extracted from Exa")
            return []

        except Exception as e:
            last_err = e
            print(f"[web] Exa failed (attempt {attempt}/{retries}): {e}")
            time.sleep(backoff ** attempt)

    print(f"[web] Exa giving up. Last error: {last_err}")
    return []

def build_context(internal_hits, web_hits, max_chars_per_chunk=800):
    """
    Build context with per-chunk truncation to prevent token overflow.
    """
    ctx = []

    # ----- Internal hits -----
    for h in internal_hits:
        src = h.get("source", "")
        fname = Path(src).name if src else "unknown"

        # Create citation
        if "page" in h:
            cite = f"{fname} p.{h['page']}"
        elif "row_start" in h and "row_end" in h:
            cite = f"{fname} rows {h['row_start']}-{h['row_end']}"
        elif "chunk" in h:
            cite = f"{fname} chunk {h['chunk']}"
        else:
            cite = f"{fname}"

        text = h.get("text", "")
        if text:
            # TRUNCATE each chunk to prevent overflow
            truncated = text[:max_chars_per_chunk]
            if len(text) > max_chars_per_chunk:
                truncated += "..."
            ctx.append(f"[INTERNAL | {cite}]\n{truncated}")

    # ----- Web hits -----
    for w in web_hits or []:
        title = w.get("title") or "Untitled"
        url = w.get("url") or ""
        text = w.get("text") or ""
        if text:
            # TRUNCATE web content too
            truncated = text[:max_chars_per_chunk]
            if len(text) > max_chars_per_chunk:
                truncated += "..."
            ctx.append(f"[WEB | {title} | {url}]\n{truncated}")

    return "\n\n".join(ctx)