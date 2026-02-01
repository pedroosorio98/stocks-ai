# download_sec_filings_html.py
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import requests


SEC_TICKER_CIK_JSON = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik10}.json"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_dashes}/"

DEFAULT_USER_AGENT = "pedro pedro.osoriomn@gmail.com"  # prefer env SEC_USER_AGENT


def sec_get(url: str, user_agent: str, timeout: int = 90) -> requests.Response:
    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": "application/json,text/plain,text/html,*/*",
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r

def ticker_variants_sec(ticker: str) -> List[str]:
    t = ticker.upper().strip()
    vars_ = [t]

    if "." in t:
        vars_.append(t.replace(".", "-"))   # BRK.B -> BRK-B
        vars_.append(t.replace(".", ""))    # BRK.B -> BRKB (sometimes useful)

    if "-" in t:
        vars_.append(t.replace("-", "."))   # BRK-B -> BRK.B
        vars_.append(t.replace("-", ""))    # BRK-B -> BRKB

    # de-dupe preserve order
    seen = set()
    out = []
    for v in vars_:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out

def ticker_to_cik(ticker: str, user_agent: str) -> int:
    candidates = ticker_variants_sec(ticker)
    data = sec_get(SEC_TICKER_CIK_JSON, user_agent=user_agent).json()

    # Build fast lookup once
    lookup = {}
    for _, row in data.items():
        tk = str(row.get("ticker", "")).upper().strip()
        if tk:
            lookup[tk] = int(row["cik_str"])

    for c in candidates:
        if c in lookup:
            return lookup[c]

    raise ValueError(f"Ticker not found in SEC mapping: {ticker} (tried: {candidates})")

def load_submissions(cik: int, user_agent: str) -> Dict[str, Any]:
    cik10 = f"{cik:010d}"
    return sec_get(SEC_SUBMISSIONS.format(cik10=cik10), user_agent=user_agent).json()


def normalize_form(form: str) -> str:
    form = (form or "UNKNOWN").strip()
    form = re.sub(r"[^\w\-. ]+", "_", form).replace(" ", "_")
    return form or "UNKNOWN"


def extract_recent_filings(submissions_like: Dict[str, Any]) -> List[Dict[str, Any]]:
    recent = submissions_like.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    filing_dates = recent.get("filingDate", [])

    out = []
    for form, acc, primary, fdate in zip(forms, accessions, primary_docs, filing_dates):
        out.append({
            "form": form,
            "accession": acc,
            "accession_no_dashes": acc.replace("-", ""),
            "primary_document": primary,
            "filing_date": fdate,
        })
    return out


def extract_older_page_names(submissions: Dict[str, Any]) -> List[str]:
    files = submissions.get("filings", {}).get("files", [])
    return [f.get("name") for f in files if f.get("name")]


def load_older_page(page_name: str, user_agent: str) -> Dict[str, Any]:
    url = f"https://data.sec.gov/submissions/{page_name}"
    return sec_get(url, user_agent=user_agent).json()


def should_keep_form(form: str, allowed: Optional[set]) -> bool:
    if not allowed:
        return True
    return (form or "").upper() in allowed


def download_primary_doc_html(cik: int, filing: Dict[str, Any], out_dir: Path, user_agent: str) -> Path:
    acc_no_dashes = filing["accession_no_dashes"]
    primary = filing.get("primary_document")
    if not primary:
        raise ValueError("No primary_document in filing.")

    base = ARCHIVES_BASE.format(cik=cik, acc_no_dashes=acc_no_dashes)
    url = base + primary
    dest = out_dir / primary

    if dest.exists() and dest.stat().st_size > 0:
        return dest

    r = sec_get(url, user_agent=user_agent)
    dest.write_bytes(r.content)
    return dest


def parse_tickers_file(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"tickers file not found: {p}")
    tickers = []
    for line in p.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if not t:
            continue
        if t.startswith("#"):
            continue
        tickers.append(t.upper())
    # de-dupe preserving order
    seen = set()
    out = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def download_for_ticker(ticker: str, data_root: Path, user_agent: str, sleep_s: float,
                        max_filings: int, forms_filter: Optional[set]) -> None:
    cik = ticker_to_cik(ticker, user_agent)
    subs = load_submissions(cik, user_agent)

    filings: List[Dict[str, Any]] = []
    filings.extend(extract_recent_filings(subs))

    # Pull older pages so this becomes "all filings"
    for page_name in extract_older_page_names(subs):
        time.sleep(sleep_s)
        page = load_older_page(page_name, user_agent)
        recent_like = page.get("filings", {}).get("recent", {})
        filings.extend(extract_recent_filings({"filings": {"recent": recent_like}}))

    # De-dupe and filter
    seen = set()
    uniq = []
    for f in filings:
        acc = f["accession"]
        form = f.get("form", "")
        if acc in seen:
            continue
        if not should_keep_form(form, forms_filter):
            continue
        seen.add(acc)
        uniq.append(f)

    if max_filings and max_filings > 0:
        uniq = uniq[:max_filings]

    print(f"\n==== {ticker} | CIK {cik} | filings to download: {len(uniq)} ====")

    for i, f in enumerate(uniq, start=1):
        form = f.get("form", "UNKNOWN")
        fdate = f.get("filing_date", "unknown-date")
        acc = f.get("accession", "unknown-acc")

        form_dir = normalize_form(form)
        out_dir = data_root / ticker / "sec" / form_dir / f"{fdate}_{acc}"
        out_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "ticker": ticker,
            "cik": cik,
            "form": form,
            "filing_date": fdate,
            "accession": acc,
            "primary_document": f.get("primary_document"),
        }
        (out_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        print(f"[{i}/{len(uniq)}] {ticker} {form} {fdate} {acc}")

        try:
            primary_path = download_primary_doc_html(cik, f, out_dir, user_agent)
            print(f"  ✓ saved: {primary_path.name}")
        except Exception as e:
            (out_dir / "_error.txt").write_text(str(e), encoding="utf-8")
            print(f"  ✗ error: {e}")

        time.sleep(sleep_s)


def main():
    ap = argparse.ArgumentParser()

    # One ticker OR a file of tickers
    ap.add_argument("--ticker", default="", help="Single ticker, e.g. NVDA")
    ap.add_argument("--tickers_file", "--tickers-file", default="", help="Path to tickers.txt (one ticker per line)")

    ap.add_argument("--data_root", default="Data", help="Root Data folder (default: Data)")
    ap.add_argument("--user_agent", default=os.getenv("SEC_USER_AGENT", DEFAULT_USER_AGENT),
                    help='Set env SEC_USER_AGENT e.g. "Pedro Osorio pedro@email.com"')
    ap.add_argument("--sleep", type=float, default=0.2, help="Sleep between SEC requests (seconds)")
    ap.add_argument("--max_filings", type=int, default=0, help="Limit filings per ticker (0 = no limit)")
    ap.add_argument("--forms", default="", help='Optional filter: "10-K,10-Q,8-K,DEF 14A,20-F,6-K"')
    args = ap.parse_args()

    if "@" not in args.user_agent or " " not in args.user_agent:
        raise SystemExit(
            'Set a proper SEC user agent like: --user_agent "Pedro Osorio pedro@email.com" '
            "or set env SEC_USER_AGENT."
        )

    data_root = Path(args.data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    forms_filter = {f.strip().upper() for f in args.forms.split(",") if f.strip()} if args.forms else None

    # Decide tickers list
    tickers: List[str] = []
    if args.tickers_file:
        tickers = parse_tickers_file(args.tickers_file)

    if args.ticker:
        tickers.append(args.ticker.upper().strip())

    # de-dupe preserving order
    seen = set()
    tickers = [t for t in tickers if t and (t not in seen and not seen.add(t))]

    if not tickers:
        raise SystemExit("Provide --ticker NVDA or --tickers_file tickers.txt")

    for t in tickers:
        try:
            download_for_ticker(
                ticker=t,
                data_root=data_root,
                user_agent=args.user_agent,
                sleep_s=args.sleep,
                max_filings=args.max_filings,
                forms_filter=forms_filter,
            )
        except Exception as e:
            print(f"!! FAILED for {t}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
