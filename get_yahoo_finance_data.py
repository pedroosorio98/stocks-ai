# get_yahoo_brazil_data.py
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf


# -------------------------
# Helpers
# -------------------------

def ensure_sa_suffix(ticker: str) -> str:
    """
    Yahoo Finance Brazil tickers are usually <CODE>.SA
    Examples:
      BPAC11 -> BPAC11.SA
      PETR4  -> PETR4.SA
      VALE3  -> VALE3.SA
    If user already passes .SA or another suffix, keep it.
    """
    t = ticker.strip().upper()
    if "." in t:  # already has suffix like .SA
        return t
    return f"{t}.SA"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True, encoding="utf-8")


def safe_df(x) -> Optional[pd.DataFrame]:
    try:
        if x is None:
            return None
        if isinstance(x, pd.DataFrame):
            return x
        if isinstance(x, pd.Series):
            return x.to_frame()
        return pd.DataFrame(x)
    except Exception:
        return None


def save_df_if_nonempty(path: Path, df: Optional[pd.DataFrame]) -> None:
    if df is None:
        return
    if isinstance(df, pd.DataFrame) and df.empty:
        return
    write_csv(path, df)


def try_getattr(obj, name: str):
    try:
        return getattr(obj, name)
    except Exception:
        return None


# -------------------------
# Main downloader
# -------------------------

def download_yahoo_for_ticker(
    ticker_input: str,
    data_root: Path,
    sleep_s: float = 0.2,
    intraday_period: str = "730d",   # max yfinance allows for some intervals
    intraday_interval: str = "60m",
    daily_period: str = "max",
) -> None:
    original = ticker_input.strip().upper()
    yahoo_ticker = ensure_sa_suffix(original)

    out_dir = data_root / original / "yahoo_finance"
    raw_dir = out_dir / "raw_json"
    csv_dir = out_dir / "csv"
    err_dir = out_dir / "errors"

    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    err_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {original} (Yahoo: {yahoo_ticker}) -> {out_dir} ===")

    t = yf.Ticker(yahoo_ticker)

    # 1) Meta / info
    try:
        info = t.get_info()
        write_json(raw_dir / "info.json", info)
    except Exception as e:
        write_json(err_dir / "info_error.json", {"error": str(e)})

    time.sleep(sleep_s)

    # 2) Fast info (lighter, sometimes works even when info fails)
    try:
        fast_info = try_getattr(t, "fast_info")
        if fast_info is not None:
            # fast_info is dict-like
            write_json(raw_dir / "fast_info.json", dict(fast_info))
    except Exception as e:
        write_json(err_dir / "fast_info_error.json", {"error": str(e)})

    time.sleep(sleep_s)

    # 3) Price history (daily)
    try:
        hist_daily = t.history(period=daily_period, auto_adjust=False)
        save_df_if_nonempty(csv_dir / "history_daily.csv", hist_daily)
    except Exception as e:
        write_json(err_dir / "history_daily_error.json", {"error": str(e)})

    time.sleep(sleep_s)

    # 4) Intraday (60m)
    try:
        hist_1h = t.history(period=intraday_period, interval=intraday_interval, auto_adjust=False)
        save_df_if_nonempty(csv_dir / f"history_{intraday_interval}.csv", hist_1h)
    except Exception as e:
        write_json(err_dir / "history_intraday_error.json", {"error": str(e), "interval": intraday_interval})

    time.sleep(sleep_s)

    # 5) Actions (dividends/splits)
    try:
        actions = t.actions
        save_df_if_nonempty(csv_dir / "actions.csv", safe_df(actions))
    except Exception as e:
        write_json(err_dir / "actions_error.json", {"error": str(e)})

    time.sleep(sleep_s)

    # 6) Dividends & splits separately
    try:
        divs = t.dividends
        save_df_if_nonempty(csv_dir / "dividends.csv", safe_df(divs))
    except Exception as e:
        write_json(err_dir / "dividends_error.json", {"error": str(e)})

    time.sleep(sleep_s)

    try:
        splits = t.splits
        save_df_if_nonempty(csv_dir / "splits.csv", safe_df(splits))
    except Exception as e:
        write_json(err_dir / "splits_error.json", {"error": str(e)})

    time.sleep(sleep_s)

    # 7) Financial statements (annual + quarterly) - availability varies a lot for Brazil tickers
    # Income statement
    for name, attr in [
        ("income_stmt_annual", "income_stmt"),
        ("income_stmt_quarterly", "quarterly_income_stmt"),
        ("balance_sheet_annual", "balance_sheet"),
        ("balance_sheet_quarterly", "quarterly_balance_sheet"),
        ("cashflow_annual", "cashflow"),
        ("cashflow_quarterly", "quarterly_cashflow"),
    ]:
        try:
            df = try_getattr(t, attr)
            save_df_if_nonempty(csv_dir / f"{name}.csv", safe_df(df))
        except Exception as e:
            write_json(err_dir / f"{name}_error.json", {"error": str(e), "attr": attr})
        time.sleep(sleep_s)

    # 8) Earnings / calendar
    try:
        cal = t.calendar
        save_df_if_nonempty(csv_dir / "calendar.csv", safe_df(cal))
    except Exception as e:
        write_json(err_dir / "calendar_error.json", {"error": str(e)})

    time.sleep(sleep_s)

    try:
        earnings_dates = t.earnings_dates
        save_df_if_nonempty(csv_dir / "earnings_dates.csv", safe_df(earnings_dates))
    except Exception as e:
        write_json(err_dir / "earnings_dates_error.json", {"error": str(e)})

    time.sleep(sleep_s)

    # 9) Holders (often missing for non-US)
    for name, attr in [
        ("major_holders", "major_holders"),
        ("institutional_holders", "institutional_holders"),
        ("mutualfund_holders", "mutualfund_holders"),
        ("insider_transactions", "insider_transactions"),
        ("insider_purchases", "insider_purchases"),
        ("insider_roster_holders", "insider_roster_holders"),
    ]:
        try:
            df = try_getattr(t, attr)
            save_df_if_nonempty(csv_dir / f"{name}.csv", safe_df(df))
        except Exception as e:
            write_json(err_dir / f"{name}_error.json", {"error": str(e), "attr": attr})
        time.sleep(sleep_s)

    # 10) News (list of dicts)
    try:
        news = t.news
        if isinstance(news, list):
            write_json(raw_dir / "news.json", news)
            # also a small CSV for convenience
            rows = []
            for it in news:
                if not isinstance(it, dict):
                    continue
                rows.append({
                    "provider": it.get("publisher"),
                    "title": it.get("title"),
                    "link": it.get("link"),
                    "published": it.get("providerPublishTime"),
                    "type": it.get("type"),
                })
            if rows:
                write_csv(csv_dir / "news.csv", pd.DataFrame(rows))
    except Exception as e:
        write_json(err_dir / "news_error.json", {"error": str(e)})

    time.sleep(sleep_s)

    # 11) Options chain (rare for B3 tickers)
    try:
        expirations = getattr(t, "options", [])
        if expirations:
            write_json(raw_dir / "options_expirations.json", list(expirations))
            # download first N expirations
            for exp in expirations[:6]:
                try:
                    chain = t.option_chain(exp)
                    calls = safe_df(chain.calls)
                    puts = safe_df(chain.puts)
                    save_df_if_nonempty(csv_dir / f"options_calls_{exp}.csv", calls)
                    save_df_if_nonempty(csv_dir / f"options_puts_{exp}.csv", puts)
                except Exception as e:
                    write_json(err_dir / f"options_chain_{exp}_error.json", {"error": str(e)})
                time.sleep(sleep_s)
        else:
            write_json(raw_dir / "options_expirations.json", [])
    except Exception as e:
        write_json(err_dir / "options_error.json", {"error": str(e)})

    # 12) Save a small manifest
    write_json(out_dir / "_manifest.json", {
        "input_ticker": original,
        "yahoo_ticker_used": yahoo_ticker,
        "folder": str(out_dir),
        "notes": "Some modules are not available for all Brazil tickers; see errors/ folder for failures.",
    })

    print(f"Done: {original}")


def load_tickers_file(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"tickers file not found: {p}")
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        out.append(t)
    # de-dupe preserving order
    seen = set()
    final = []
    for t in out:
        u = t.strip().upper()
        if u and u not in seen:
            seen.add(u)
            final.append(u)
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="", help="Single ticker, e.g. BPAC11")
    ap.add_argument("--tickers_file", "--tickers-file", default="", help="tickers.txt (one per line)")
    ap.add_argument("--data_root", default="Data", help="Data root folder (default: Data)")
    ap.add_argument("--sleep", type=float, default=0.2, help="Sleep between calls (seconds)")
    ap.add_argument("--daily_period", default="max", help='Daily history period (default: "max")')
    ap.add_argument("--intraday_period", default="730d", help='Intraday period (default: "730d")')
    ap.add_argument("--intraday_interval", default="60m", help='Intraday interval (default: "60m")')
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    tickers: List[str] = []
    if args.tickers_file:
        tickers.extend(load_tickers_file(args.tickers_file))
    if args.ticker:
        tickers.append(args.ticker.strip().upper())

    # de-dupe preserve order
    seen = set()
    tickers = [t for t in tickers if t and (t not in seen and not seen.add(t))]

    if not tickers:
        raise SystemExit("Provide --ticker BPAC11 or --tickers_file tickers.txt")

    for t in tickers:
        try:
            download_yahoo_for_ticker(
                ticker_input=t,
                data_root=data_root,
                sleep_s=args.sleep,
                daily_period=args.daily_period,
                intraday_period=args.intraday_period,
                intraday_interval=args.intraday_interval,
            )
        except Exception as e:
            print(f"!! FAILED for {t}: {e}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
