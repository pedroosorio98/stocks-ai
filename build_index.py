# -*- coding: utf-8 -*-
"""
build_index.py (cache-aware)

- Indexes Data/<TICKER>/** (recursive)
- Skips any folder named "ir_data" (case-insensitive)
- Uses cached extracted text from cache/text/<md5>.txt whenever possible
- Stores chunk text in index/internal_meta.jsonl under key "text" (so RAG can use internal context)

Outputs:
  index/internal.faiss
  index/internal_meta.jsonl
  
Then uploads to S3 bucket using AWS credentials from environment variables.
"""

import json
import hashlib
import os
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from pypdf import PdfReader
from openai import OpenAI
import boto3

# ----------------------------
# Config
# ----------------------------
EMBED_MODEL = "text-embedding-3-large"   # change to -large later if you want
CHUNK_SIZE = 1800
OVERLAP = 150
BATCH_EMBED = 128
MAX_CHARS_PER_CHUNK = 8000

SKIP_DIR_NAMES = {}  # add more if you want

# S3 Config (from environment variables)
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'osorio-stocks-research')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
INDEX_DIR = BASE_DIR / "index"
CACHE_DIR = BASE_DIR / "cache"
CACHE_TEXT_DIR = CACHE_DIR / "text"

INDEX_DIR.mkdir(exist_ok=True)
CACHE_TEXT_DIR.mkdir(parents=True, exist_ok=True)

FAISS_OUT = INDEX_DIR / "internal.faiss"
META_OUT = INDEX_DIR / "internal_meta.jsonl"

client = OpenAI()

# ----------------------------
# Helpers
# ----------------------------
def should_skip(p: Path) -> bool:
    parts = [x.lower() for x in p.parts]
    return any(d in parts for d in SKIP_DIR_NAMES)

def get_ticker(path: Path) -> str:
    try:
        return path.relative_to(DATA_DIR).parts[0]
    except Exception:
        return "UNKNOWN"

def normalize_ws(s: str) -> str:
    return " ".join((s or "").split())

def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    text = normalize_ws(text)
    if not text:
        return []
    chunks = []
    i = 0
    step = max(1, chunk_size - overlap)
    while i < len(text):
        chunks.append(text[i:i + chunk_size])
        i += step
    return chunks

def md5_file_bytes(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def cache_path_for_file(path: Path) -> Path:
    return CACHE_TEXT_DIR / f"{md5_file_bytes(path)}.txt"

def read_cached_or_extract(path: Path) -> str:
    """
    Returns extracted text for file. Uses cache if available.
    If no cache exists, extracts quickly and writes cache.
    """
    cpath = cache_path_for_file(path)
    if cpath.exists():
        try:
            return cpath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            # If cache read fails, fall back to extraction
            pass

    # No cache => extract
    ext = path.suffix.lower()

    if ext == ".pdf":
        txt = extract_pdf_text(path)
    elif ext == ".csv":
        txt = extract_csv_text(path)
    elif ext in (".html", ".htm"):
        txt = extract_html_text(path)
    elif ext in (".txt", ".md"):
        txt = safe_read_text(path)
    elif ext == ".json":
        txt = safe_read_text(path)
    else:
        txt = ""

    txt = normalize_ws(txt)

    # Write cache (even if empty, you can decide; we'll only write if there's something)
    if txt:
        try:
            cpath.write_text(txt, encoding="utf-8")
        except Exception:
            pass

    return txt

def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            return path.read_text(encoding="latin-1", errors="ignore")
        except Exception:
            return ""

def extract_pdf_text(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
    except Exception:
        return ""
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(pages)

def extract_csv_text(path: Path, rows_limit: int = 500) -> str:
    """
    CSV can be huge. For extraction, we read only a limited number of rows
    as text so we don't blow up chunk counts. (You can raise if needed.)
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return ""
    if len(df) > rows_limit:
        df = df.iloc[:rows_limit].copy()
    return df.to_csv(index=False)

def extract_html_text(path: Path) -> str:
    raw = safe_read_text(path).replace("\x00", " ")
    return strip_html(raw)

def strip_html(html: str) -> str:
    import re
    html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
    html = re.sub(r"(?is)<[^>]+>", " ", html)
    html = html.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return " ".join(html.split())

def embed_texts(texts, batch_size=BATCH_EMBED, model=EMBED_MODEL, max_chars=MAX_CHARS_PER_CHUNK):
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = [t[:max_chars] for t in texts[i:i + batch_size]]
        resp = client.embeddings.create(model=model, input=batch)
        all_vecs.extend([d.embedding for d in resp.data])
        print(f"Embedded {min(i + batch_size, len(texts))}/{len(texts)}")
    return np.array(all_vecs, dtype="float32")

# ----------------------------
# S3 Upload Function
# ----------------------------
def upload_to_s3(local_path: Path, s3_key: str, bucket: str = S3_BUCKET_NAME):
    """
    Upload a local file to S3 bucket.
    
    Args:
        local_path: Path to local file
        s3_key: S3 object key (e.g., 'internal.faiss')
        bucket: S3 bucket name
    """
    try:
        # Get credentials from environment variables
        access_key = os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        region = os.getenv('AWS_REGION', 'us-east-1')
        
        if not access_key or not secret_key:
            print("⚠️  WARNING: AWS credentials not found in environment variables")
            print("   Skipping S3 upload. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")
            return False
        
        # Create S3 client
        s3 = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
        
        # Get file size for progress reporting
        file_size = local_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"\n📤 Uploading {local_path.name} to S3...")
        print(f"   Bucket: {bucket}")
        print(f"   Key: {s3_key}")
        print(f"   Size: {file_size_mb:.2f} MB")
        
        # Upload file
        s3.upload_file(
            str(local_path),
            bucket,
            s3_key
        )
        
        print(f"   ✅ Upload successful!")
        print(f"   S3 URI: s3://{bucket}/{s3_key}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Upload failed: {e}")
        return False

# ----------------------------
# Main
# ----------------------------
def main():
    if not DATA_DIR.exists():
        raise SystemExit(f"Data folder not found: {DATA_DIR}")

    # Rebuild outputs
    if FAISS_OUT.exists():
        FAISS_OUT.unlink()
    if META_OUT.exists():
        META_OUT.unlink()

    # Gather candidate files (any type you want to support)
    exts = {".pdf", ".csv", ".html", ".htm", ".txt", ".md", ".json"}
    files = [p for p in DATA_DIR.rglob("*") if p.is_file() and p.suffix.lower() in exts and not should_skip(p)]
    files = sorted(files)

    all_docs = []
    for p in files:
        ticker = get_ticker(p)
        rel = p.relative_to(BASE_DIR).as_posix()

        raw_text = read_cached_or_extract(p)
        if not raw_text or len(raw_text) < 200:
            continue

        for ci, ch in enumerate(chunk_text(raw_text)):
            all_docs.append({
                "ticker": ticker,
                "source": rel,          # relative path is nicer for display
                "type": p.suffix.lower().lstrip("."),
                "chunk": ci,
                "text": ch,
            })

    if not all_docs:
        raise SystemExit("No indexable text found (after skipping ir_data and applying min text length).")

    texts = [d["text"] for d in all_docs]
    vecs = embed_texts(texts)

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vecs)   # cosine
    index.add(vecs)

    faiss.write_index(index, str(FAISS_OUT))

    with open(META_OUT, "w", encoding="utf-8") as f:
        for d in all_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\n✅ DONE. Indexed {len(all_docs)} chunks.")
    print(f"- FAISS: {FAISS_OUT}")
    print(f"- META : {META_OUT}")
    print(f"- CACHE: {CACHE_TEXT_DIR}")
    
    # ----------------------------
    # Upload to S3
    # ----------------------------
    print("\n" + "="*60)
    print("📦 UPLOADING INDEX FILES TO S3")
    print("="*60)
    
    # Upload FAISS index
    upload_to_s3(FAISS_OUT, "internal.faiss")
    
    # Upload metadata
    upload_to_s3(META_OUT, "internal_meta.jsonl")
    
    print("\n" + "="*60)
    print("🎉 BUILD AND UPLOAD COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
