# get_alpha_vantage_data.py
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import requests
import pandas as pd

BASE_URL = "https://www.alphavantage.co/query"


# ---------------------------
# HTTP + IO helpers
# ---------------------------

def av_get(params: Dict[str, Any], timeout: int = 60) -> Tuple[Optional[Dict[str, Any]], Optional[str], Dict[str, Any]]:
    """
    Returns (json_or_none, text_or_none, info)
    If response parses as JSON, json_or_none is dict/list and text_or_none is None.
    If response is CSV or non-json text, json_or_none is None and text_or_none is raw text.
    """
    r = requests.get(BASE_URL, params=params, timeout=timeout)
    r.raise_for_status()

    info = {
        "status_code": r.status_code,
        "content_type": r.headers.get("Content-Type", ""),
        "url": r.url,
    }

    text = r.text or ""
    t = text.strip()
    if t.startswith("{") or t.startswith("["):
        try:
            return r.json(), None, info
        except Exception:
            return None, text, info

    return None, text, info


def parse_csv(text: str) -> pd.DataFrame:
    from io import StringIO
    return pd.read_csv(StringIO(text))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


# ---------------------------
# Alpha Vantage response heuristics
# ---------------------------

def is_rate_limited(j: Optional[Dict[str, Any]], text: Optional[str]) -> bool:
    msg = ""
    if isinstance(j, dict):
        msg = (j.get("Note") or j.get("Information") or j.get("Error Message") or "")
    msg = (msg or "") + " " + (text or "")
    msg = msg.lower()
    return ("thank you for using alpha vantage" in msg) or ("frequency" in msg) or ("rate limit" in msg)


def is_invalid_symbol_response(function_name: str, j: Optional[Dict[str, Any]], text: Optional[str]) -> bool:
    """
    A light heuristic; the stronger invalid check is is_empty_or_invalid_av()
    """
    # JSON errors
    if isinstance(j, dict):
        if j.get("Error Message"):
            return True

        if function_name == "GLOBAL_QUOTE":
            gq = j.get("Global Quote")
            if not isinstance(gq, dict) or not gq:
                return True
            price = (gq.get("05. price") or "").strip()
            return price == ""

        if function_name == "OVERVIEW":
            sym = (j.get("Symbol") or "").strip()
            return sym == ""

    # CSV invalid
    if text:
        t = text.strip().lower()
        if "invalid api call" in t:
            return True
        if function_name.startswith("TIME_SERIES") or "WEEKLY" in function_name or "MONTHLY" in function_name:
            # valid CSV should include these
            if ("timestamp" not in t) and ("open" not in t) and ("close" not in t):
                return True

    return False


def is_empty_or_invalid_av(function_name: str, j: Optional[Dict[str, Any]], text: Optional[str]) -> bool:
    """
    Treat empty JSON {} as invalid, plus enforce expected keys for certain endpoints.
    """
    # If not JSON, use CSV/text heuristics
    if j is None:
        return is_invalid_symbol_response(function_name, None, text)

    # JSON: {} happens often with wrong symbol but HTTP 200
    if not isinstance(j, dict) or len(j) == 0:
        return True

    if j.get("Error Message"):
        return True

    if function_name in {"INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW"}:
        return not (isinstance(j.get("annualReports"), list) or isinstance(j.get("quarterlyReports"), list))

    if function_name == "EARNINGS":
        return not (isinstance(j.get("annualEarnings"), list) or isinstance(j.get("quarterlyEarnings"), list))

    if function_name == "OVERVIEW":
        return (j.get("Symbol") or "").strip() == ""

    if function_name == "GLOBAL_QUOTE":
        gq = j.get("Global Quote")
        if not isinstance(gq, dict) or not gq:
            return True
        return (gq.get("05. price") or "").strip() == ""

    # For other JSON endpoints we accept any dict unless it's clearly invalid
    return False


# ---------------------------
# Symbol variants per endpoint (BRK.B vs BRK-B etc.)
# ---------------------------

def symbol_variants(symbol: str) -> List[str]:
    s = symbol.strip().upper()
    variants = [s]

    if "." in s:
        variants.append(s.replace(".", "-"))   # BRK.B -> BRK-B
        variants.append(s.replace(".", ""))    # BRK.B -> BRKB

    if "-" in s:
        variants.append(s.replace("-", "."))   # BRK-B -> BRK.B
        variants.append(s.replace("-", ""))    # BRK-B -> BRKB

    # de-dupe preserve order
    out, seen = [], set()
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


# cache winner per (endpoint, original_symbol)
_SYMBOL_CACHE: Dict[Tuple[str, str], str] = {}


def av_get_with_symbol_fallback(
    function_name: str,
    base_params: Dict[str, Any],
    original_symbol: str,
    timeout: int = 60,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Dict[str, Any], str]:
    """
    Try multiple symbol variants for THIS endpoint until a valid response is found.
    Returns: (json_or_none, text_or_none, info, symbol_used)
    """
    orig = original_symbol.strip().upper()
    cache_key = (function_name, orig)

    variants = symbol_variants(orig)
    if cache_key in _SYMBOL_CACHE:
        cached = _SYMBOL_CACHE[cache_key]
        variants = [cached] + [v for v in variants if v != cached]

    last: Tuple[Optional[Dict[str, Any]], Optional[str], Dict[str, Any], str] = (None, None, {}, orig)

    for sym in variants:
        params = dict(base_params)
        params["symbol"] = sym

        try:
            j, text, info = av_get(params, timeout=timeout)
        except Exception as e:
            last = (None, None, {"error": str(e), "url": "", "status_code": None, "content_type": ""}, sym)
            continue

        # If rate-limited, don't spam variants
        if is_rate_limited(j, text):
            return j, text, info, sym

        if not is_empty_or_invalid_av(function_name, j, text):
            _SYMBOL_CACHE[cache_key] = sym
            return j, text, info, sym

        last = (j, text, info, sym)

    return last


def av_get_with_tickers_fallback(
    function_name: str,
    base_params: Dict[str, Any],
    original_symbol: str,
    timeout: int = 60,
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Dict[str, Any], str]:
    """
    Some endpoints use tickers= (NEWS_SENTIMENT). Do the same fallback logic.
    """
    orig = original_symbol.strip().upper()
    cache_key = (function_name + ":tickers", orig)

    variants = symbol_variants(orig)
    if cache_key in _SYMBOL_CACHE:
        cached = _SYMBOL_CACHE[cache_key]
        variants = [cached] + [v for v in variants if v != cached]

    last: Tuple[Optional[Dict[str, Any]], Optional[str], Dict[str, Any], str] = (None, None, {}, orig)

    for sym in variants:
        params = dict(base_params)
        params["tickers"] = sym

        try:
            j, text, info = av_get(params, timeout=timeout)
        except Exception as e:
            last = (None, None, {"error": str(e), "url": "", "status_code": None, "content_type": ""}, sym)
            continue

        if is_rate_limited(j, text):
            return j, text, info, sym

        # For NEWS_SENTIMENT, a valid response is typically a dict with 'feed' list or other fields
        if j is None:
            last = (j, text, info, sym)
            continue

        if isinstance(j, dict) and (("feed" in j and isinstance(j["feed"], list)) or len(j) > 0):
            _SYMBOL_CACHE[cache_key] = sym
            return j, text, info, sym

        last = (j, text, info, sym)

    return last


# ---------------------------
# Tabular exports
# ---------------------------

def try_tabular_exports(function_name: str, data: Dict[str, Any]) -> List[Tuple[str, pd.DataFrame]]:
    out: List[Tuple[str, pd.DataFrame]] = []

    if function_name in {"INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW"}:
        for k in ("annualReports", "quarterlyReports"):
            if isinstance(data.get(k), list):
                df = pd.DataFrame(data[k])
                if not df.empty:
                    out.append((k, df))

    if function_name == "EARNINGS":
        for k in ("annualEarnings", "quarterlyEarnings"):
            if isinstance(data.get(k), list):
                df = pd.DataFrame(data[k])
                if not df.empty:
                    out.append((k, df))

    if function_name in {"DIVIDENDS", "SPLITS", "SHARES_OUTSTANDING", "INSIDER_TRANSACTIONS"}:
        if isinstance(data.get("data"), list):
            df = pd.DataFrame(data["data"])
            if not df.empty:
                out.append(("data", df))

    if function_name == "NEWS_SENTIMENT":
        if isinstance(data.get("feed"), list):
            df = pd.DataFrame(data["feed"])
            if not df.empty:
                out.append(("feed", df))

    if function_name in {"REALTIME_OPTIONS", "HISTORICAL_OPTIONS", "EARNINGS_ESTIMATES", "EARNINGS_CALL_TRANSCRIPT"}:
        for k, v in data.items():
            if isinstance(v, list):
                df = pd.DataFrame(v)
                if not df.empty:
                    out.append((k, df))

    if function_name == "OVERVIEW" and isinstance(data, dict) and data:
        df = pd.DataFrame([data])
        out.append(("overview", df))

    if function_name == "GLOBAL_QUOTE" and isinstance(data.get("Global Quote"), dict):
        df = pd.DataFrame([data["Global Quote"]])
        out.append(("quote", df))

    return out


# ---------------------------
# Endpoints list
# ---------------------------

TIME_SERIES_CALLS = [
    ("TIME_SERIES_INTRADAY", {"interval": "60min", "outputsize": "compact", "datatype": "csv"}),
    ("TIME_SERIES_DAILY", {"outputsize": "compact", "datatype": "csv"}),
    ("TIME_SERIES_DAILY_ADJUSTED", {"outputsize": "compact", "datatype": "csv"}),
    ("TIME_SERIES_WEEKLY", {"datatype": "csv"}),
    ("TIME_SERIES_WEEKLY_ADJUSTED", {"datatype": "csv"}),
    ("TIME_SERIES_MONTHLY", {"datatype": "csv"}),
    ("TIME_SERIES_MONTHLY_ADJUSTED", {"datatype": "csv"}),
    ("GLOBAL_QUOTE", {}),
]

JSON_CALLS = [
    ("OVERVIEW", {}),
    ("INCOME_STATEMENT", {}),
    ("BALANCE_SHEET", {}),
    ("CASH_FLOW", {}),
    ("EARNINGS", {}),
    ("EARNINGS_ESTIMATES", {}),
    ("SHARES_OUTSTANDING", {}),
    ("DIVIDENDS", {}),
    ("SPLITS", {}),
    ("INSIDER_TRANSACTIONS", {}),
    ("NEWS_SENTIMENT", {"_use_tickers_param": True}),
    ("EARNINGS_CALL_TRANSCRIPT", {}),
    ("REALTIME_OPTIONS", {"require_greeks": "true"}),
    ("HISTORICAL_OPTIONS", {}),
]


# ---------------------------
# Main dump logic
# ---------------------------

def dump_symbol(symbol: str, api_key: str, data_root: Path, sleep_seconds: float = 15.0, timeout: int = 60) -> None:
    original_symbol = symbol.strip().upper()

    # Folder uses original symbol (Data/BRK.B/...)
    base_dir = data_root / original_symbol / "alpha_vantage"
    csv_dir = base_dir / "csv"
    json_dir = base_dir / "raw_json"
    err_dir = base_dir / "errors"
    txt_dir = base_dir / "raw_text"

    csv_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    err_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    # Record all variants (for debugging)
    write_json(base_dir / "_symbol_variants.json", {
        "original_symbol": original_symbol,
        "variants": symbol_variants(original_symbol),
        "note": "This script picks the best variant per endpoint and caches it.",
    })

    print(f"\n=== {original_symbol} -> {base_dir} ===")

    # ---- TIME SERIES (CSV or JSON for GLOBAL_QUOTE) ----
    for fn, extra in TIME_SERIES_CALLS:
        base_params = {"function": fn, "apikey": api_key, **extra}

        print(f"[{original_symbol}] {fn} ...")
        try:
            j, text, info, sym_used = av_get_with_symbol_fallback(fn, base_params, original_symbol, timeout=timeout)
        except Exception as e:
            write_json(err_dir / f"{fn}.json", {"error": str(e), "params": base_params})
            time.sleep(sleep_seconds)
            continue

        # Rate limit
        if is_rate_limited(j, text):
            write_json(err_dir / f"{fn}_rate_limited.json", {
                "params": {**base_params, "symbol": sym_used},
                "info": info,
                "preview": str(j or text)[:800]
            })
            print(f"  ! rate-limited; sleeping {sleep_seconds}s")
            time.sleep(sleep_seconds)
            continue

        # Save raw response (for audit)
        if j is not None:
            write_json(json_dir / f"{fn}.json", {"response": j, "request": {**base_params, "symbol": sym_used}, "info": info})
            try:
                for suffix, df in try_tabular_exports(fn, j if isinstance(j, dict) else {}):
                    write_csv(csv_dir / f"{fn}__{suffix}.csv", df)
            except Exception as e:
                write_json(err_dir / f"{fn}_export.json", {"error": str(e)})
        else:
            # CSV response
            if is_invalid_symbol_response(fn, None, text):
                write_text(txt_dir / f"{fn}.txt", text or "")
                write_json(err_dir / f"{fn}_invalid_symbol.csv.json", {
                    "params": {**base_params, "symbol": sym_used},
                    "info": info,
                    "preview": (text or "")[:800]
                })
            else:
                try:
                    df = parse_csv(text or "")
                    write_csv(csv_dir / f"{fn}.csv", df)
                except Exception as e:
                    write_json(err_dir / f"{fn}_csv_parse.json", {
                        "error": str(e),
                        "params": {**base_params, "symbol": sym_used},
                        "preview": (text or "")[:800],
                        "info": info
                    })
                    write_text(txt_dir / f"{fn}.txt", text or "")

        time.sleep(sleep_seconds)

    # ---- JSON CALLS ----
    for fn, extra0 in JSON_CALLS:
        extra = dict(extra0)  # copy
        base_params = {"function": fn, "apikey": api_key, **{k: v for k, v in extra.items() if not k.startswith("_")}}

        print(f"[{original_symbol}] {fn} ...")

        try:
            if extra.pop("_use_tickers_param", False):
                j, text, info, sym_used = av_get_with_tickers_fallback(fn, base_params, original_symbol, timeout=timeout)
                request_params = {**base_params, "tickers": sym_used}
            else:
                j, text, info, sym_used = av_get_with_symbol_fallback(fn, base_params, original_symbol, timeout=timeout)
                request_params = {**base_params, "symbol": sym_used}
        except Exception as e:
            write_json(err_dir / f"{fn}.json", {"error": str(e), "params": base_params})
            time.sleep(sleep_seconds)
            continue

        # Rate limit
        if is_rate_limited(j, text):
            write_json(err_dir / f"{fn}_rate_limited.json", {
                "params": request_params,
                "info": info,
                "preview": str(j or text)[:800]
            })
            print(f"  ! rate-limited; sleeping {sleep_seconds}s")
            time.sleep(sleep_seconds)
            continue

        if j is None:
            # Non-json response from a JSON endpoint -> keep for debugging
            write_text(txt_dir / f"{fn}.txt", text or "")
            write_json(err_dir / f"{fn}_nonjson.json", {"params": request_params, "info": info})
            time.sleep(sleep_seconds)
            continue

        # If endpoint still returned empty/invalid json, log it
        if is_empty_or_invalid_av(fn, j if isinstance(j, dict) else None, None):
            write_json(err_dir / f"{fn}_invalid_symbol.json", {"response": j, "params": request_params, "info": info})
            time.sleep(sleep_seconds)
            continue

        # Save raw JSON
        write_json(json_dir / f"{fn}.json", {"response": j, "request": request_params, "info": info})

        # Export to CSV if applicable
        try:
            exports = try_tabular_exports(fn, j if isinstance(j, dict) else {})
            for suffix, df in exports:
                write_csv(csv_dir / f"{fn}__{suffix}.csv", df)
        except Exception as e:
            write_json(err_dir / f"{fn}_export.json", {"error": str(e)})

        time.sleep(sleep_seconds)

    # Record cache decisions (nice debugging)
    write_json(base_dir / "_symbol_resolution_cache.json", {
        "cache": {f"{k[0]}::{k[1]}": v for k, v in _SYMBOL_CACHE.items() if k[1] == original_symbol},
        "note": "Chosen symbol variant per endpoint for this ticker.",
    })

    print(f"Done: {original_symbol}")


# ---------------------------
# CLI
# ---------------------------

def load_tickers_file(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"tickers file not found: {p}")
    lines = p.read_text(encoding="utf-8").splitlines()
    tickers: List[str] = []
    for line in lines:
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        tickers.append(t)
    return tickers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="", help="Comma-separated tickers, e.g. NVDA,MSFT,AAPL")
    parser.add_argument("--symbol", default="", help="Single ticker, e.g. NVDA")
    parser.add_argument("--tickers_file", "--tickers-file", default="", help="Text file with one ticker per line")
    parser.add_argument("--apikey", default=os.getenv("ALPHAVANTAGE_API_KEY"), help="Alpha Vantage key (or set env ALPHAVANTAGE_API_KEY)")
    parser.add_argument("--data_root", default="Data", help="Data root folder (default: Data)")
    parser.add_argument("--sleep", type=float, default=15.0, help="Sleep between requests (seconds)")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds")
    args = parser.parse_args()

    if not args.apikey:
        raise SystemExit("Missing API key. Set ALPHAVANTAGE_API_KEY or pass --apikey.")

    data_root = Path(args.data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    if args.tickers_file:
        syms = load_tickers_file(args.tickers_file)
    elif args.symbol:
        syms = [args.symbol.strip()]
    elif args.symbols:
        syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        raise SystemExit("Provide --tickers_file, --symbol, or --symbols.")

    # de-dupe while preserving order
    seen = set()
    ordered: List[str] = []
    for s in syms:
        u = s.strip().upper()
        if u and u not in seen:
            seen.add(u)
            ordered.append(u)

    for s in ordered:
        dump_symbol(s, args.apikey, data_root, sleep_seconds=args.sleep, timeout=args.timeout)

    print("\nAll done.")


if __name__ == "__main__":
    main()
