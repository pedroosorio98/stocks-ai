# pip install selenium webdriver-manager openai requests beautifulsoup4 tqdm

import os
import re
import json
import time
import gzip
import hashlib
from tqdm import tqdm
from pathlib import Path
from urllib.parse import urlparse, unquote, urljoin, urlunparse

import requests
from bs4 import BeautifulSoup
from openai import OpenAI

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service


# =========================
# CONFIG
# =========================

FILE_EXTS = (
    ".pdf", ".zip", ".xlsx", ".xls", ".csv", ".doc", ".docx", ".ppt", ".pptx",
    ".html", ".htm", ".mp3", ".wav", ".m4a"
)

EVENT_KEYWORDS = [
    "annual meeting", "shareholder meeting", "shareholders meeting",
    "annual shareholder", "meeting", "events", "event", "webcast",
    "presentation", "investor day", "earnings call", "conference",
    "agm", "extraordinary", "assembl", "assembleia", "assembleia geral",
    "annual-meeting", "shareholder", "/events", "investor/events", "webcast",
]

FIN_KEYWORDS = [
    "results", "quarter", "quarterly", "annual", "report", "reports",
    "financial", "earnings", "release", "news release", "press release",
    "10-k", "10q", "10-q", "8-k", "proxy", "sec", "filings", "form 20-f",
    "presentation", "investor presentation", "slides", "deck",
    "annual-reports", "quarterly-results", "financial-reports", "sec-filings",
]

FRE_KEYWORDS = [
    "formulário de referência", "formulario de referencia", "fre",
    "reference form", "reference report", "formulario referencia",
]

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    ),
    "Accept-Encoding": "identity",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive",
}

BLOCK_PATTERNS = [
    "access denied",
    "request blocked",
    "permission to access",
    "errors.edgesuite.net",
    "akamai",
    "reference #",
    "incident id",
    "you don't have permission to access",
]


# =========================
# HELPERS
# =========================

def force_https(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return url
    parsed = urlparse(url)
    if parsed.scheme == "":
        return "https://" + url.lstrip("/")
    if parsed.scheme.lower() == "http":
        return urlunparse(("https", parsed.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))
    return url

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def safe_filename(name: str, default="file"):
    name = normalize_whitespace(name)
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]+', "_", name)
    name = name.strip(" .")
    return name[:180] if name else default

def best_label(text: str, href: str) -> str:
    text = normalize_whitespace(text)
    if text:
        return text
    try:
        name = unquote(Path(urlparse(href).path).name)
        return name or href
    except Exception:
        return href

def url_path_ext(url: str) -> str:
    try:
        p = urlparse(url).path.lower()
        for ext in FILE_EXTS:
            if p.endswith(ext):
                return ext
    except Exception:
        pass
    return ""

def looks_fileish_url(url: str) -> bool:
    u = (url or "").lower()
    return (
        "static-files" in u
        or "mzfilemanager" in u
        or "api.mziq.com" in u
        or "download" in u
        or "filemanager" in u
        or "document" in u
        or "attachments" in u
        or ("media" in u and "pdf" in u)
    )

def classify_kind(href: str) -> str:
    if url_path_ext(href):
        return "file"
    if looks_fileish_url(href):
        return "file"
    return "webpage"

def is_block_page_text(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in BLOCK_PATTERNS)

def filename_from_response(final_url: str, headers: dict, label: str):
    cd = (headers.get("Content-Disposition") or headers.get("content-disposition") or "") or ""
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd, flags=re.IGNORECASE)
    if m:
        return safe_filename(unquote(m.group(1)))

    path_name = unquote(Path(urlparse(final_url).path).name)
    if path_name and "." in path_name:
        return safe_filename(path_name)

    h = hashlib.md5(final_url.encode("utf-8")).hexdigest()[:8]
    return safe_filename(f"{label}_{h}")

def guess_extension_from_content_type(ct: str) -> str:
    ct = (ct or "").lower()
    if "pdf" in ct:
        return ".pdf"
    if "zip" in ct:
        return ".zip"
    if "spreadsheet" in ct or "excel" in ct:
        return ".xlsx"
    if "csv" in ct:
        return ".csv"
    if "msword" in ct:
        return ".doc"
    if "officedocument.wordprocessingml" in ct:
        return ".docx"
    if "presentation" in ct:
        return ".pptx"
    if "html" in ct:
        return ".html"
    if "audio/mpeg" in ct or "mp3" in ct:
        return ".mp3"
    if "audio/wav" in ct:
        return ".wav"
    if "audio/mp4" in ct or "m4a" in ct:
        return ".m4a"
    return ""

def sniff_magic_extension(b: bytes) -> str:
    if not b:
        return ""
    if b.startswith(b"%PDF-"):
        return ".pdf"
    if b.startswith(b"PK\x03\x04"):
        return ".zip"
    if b.startswith(b"\x1F\x8B\x08"):
        return ".gz"
    if b.startswith(b"ID3") or (len(b) >= 2 and b[0] == 0xFF and (b[1] & 0xE0) == 0xE0):
        return ".mp3"
    if b.startswith(b"RIFF") and b[8:12] == b"WAVE":
        return ".wav"
    return ""

def looks_like_html_bytes(b: bytes) -> bool:
    if not b:
        return False
    head = b.lstrip()[:300].lower()
    return (
        head.startswith(b"<!doctype")
        or head.startswith(b"<html")
        or head.startswith(b"<head")
        or head.startswith(b"<body")
        or head.startswith(b"<script")
        or head.startswith(b"<meta")
        or head.startswith(b"<")
    )

def looks_like_text_bytes(b: bytes) -> bool:
    if not b:
        return False
    if b.count(b"\x00") > 0:
        return False
    sample = b[:2000]
    weird = sum((x < 9 or (13 < x < 32) or x > 126) for x in sample)
    return (weird / max(1, len(sample))) < 0.30

def normalize_item(year, text, href):
    href = force_https((href or "").strip())
    text = normalize_whitespace(text or "")
    return {
        "year": year,
        "text": text,
        "href": href,
        "kind": classify_kind(href),
        "ext": url_path_ext(href),
    }


# =========================
# SELENIUM (cookie accept + scrape + dropdown)
# =========================

COOKIE_BUTTON_XPATHS = [
    "//*[self::button or self::a][contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'accept')]",
    "//*[self::button or self::a][contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'agree')]",
    "//*[self::button or self::a][contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'allow')]",
    "//*[self::button or self::a][contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'aceitar')]",
    "//*[self::button or self::a][contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'concordo')]",
    "//*[self::button or self::a][contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'permitir')]",
    "//*[self::button or self::a][contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'aceitar todos')]",
    "//*[self::button or self::a][contains(translate(normalize-space(.),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'accept all')]",
]

def build_driver(headless=True, window_size="1400,900", user_data_dir=None):
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument(f"--window-size={window_size}")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--lang=en-US")
    opts.add_argument(f"user-agent={DEFAULT_HEADERS['User-Agent']}")
    if user_data_dir:
        opts.add_argument(f"--user-data-dir={user_data_dir}")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)

def try_accept_cookies(driver, timeout_s=6):
    try:
        wait = WebDriverWait(driver, timeout_s)
        for xp in COOKIE_BUTTON_XPATHS:
            try:
                el = wait.until(EC.element_to_be_clickable((By.XPATH, xp)))
                if el.is_displayed():
                    el.click()
                    time.sleep(0.8)
                    return True
            except Exception:
                continue
    except Exception:
        pass
    return False

def get_all_links_on_current_page(driver, year_tag):
    items = []
    for a in driver.find_elements(By.CSS_SELECTOR, "a[href]"):
        try:
            href = force_https((a.get_attribute("href") or "").strip())
            if not href.startswith("http"):
                continue
            text = (a.text or "").strip()
            items.append(normalize_item(year_tag, text, href))
        except Exception:
            continue
    return items

def try_select_year(driver, wait, year: int) -> bool:
    """
    Best-effort: tries to click a dropdown and then a year option.
    Returns True if it likely selected something, else False.
    """
    # candidates that often open year menus
    toggle_css = [
        "button", "[role='button']",
        "select", ".dropdown", ".select", ".year", ".filtro-ano",
        "[aria-haspopup='listbox']", "[aria-expanded]"
    ]

    # 1) try clicking something that looks like current year/has 20xx
    clicked = False
    for css in toggle_css:
        for e in driver.find_elements(By.CSS_SELECTOR, css):
            try:
                txt = (e.text or "").strip()
                if e.is_displayed() and re.search(r"\b20\d{2}\b", txt):
                    e.click()
                    clicked = True
                    break
            except Exception:
                pass
        if clicked:
            break

    # 2) if no obvious toggle, try a direct element containing year
    if not clicked:
        try:
            y_elems = driver.find_elements(By.XPATH, "//*[contains(text(),'20') and string-length(normalize-space(text()))<=12]")
            for e in y_elems:
                try:
                    txt = (e.text or "").strip()
                    if e.is_displayed() and re.search(r"\b20\d{2}\b", txt):
                        e.click()
                        clicked = True
                        break
                except Exception:
                    pass
        except Exception:
            pass

    if not clicked:
        return False

    # 3) click the year option
    try:
        option = wait.until(EC.element_to_be_clickable((By.XPATH, f"//*[normalize-space()='{year}']")))
        option.click()
        time.sleep(1.8)
        return True
    except Exception:
        return False

def wait_table_refresh(driver, previous_links_count: int, max_wait_s=12):
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        time.sleep(0.5)
        cnt = len(driver.find_elements(By.CSS_SELECTOR, "a[href]"))
        if cnt != previous_links_count:
            return True
    return False

def scrape_ir_links_with_years(url: str, YEARS_BACK=3, start_year=2025, headless=True, user_data_dir=None):
    """
    - Loads page
    - Accept cookies
    - If dropdown/year selection works: scrape each year
    - Else: scrape first page only
    """
    driver = build_driver(headless=headless, user_data_dir=user_data_dir)
    wait = WebDriverWait(driver, 18)
    out = {}

    try:
        url = force_https(url)
        driver.get(url)
        time.sleep(4)

        try_accept_cookies(driver, timeout_s=6)
        time.sleep(1.2)

        html = driver.page_source or ""
        if is_block_page_text(html):
            return {"__blocked__": True, "first_page": []}

        # always save first page
        out["first_page"] = get_all_links_on_current_page(driver, "first_page")

        # try dropdown years
        for y in range(start_year, start_year - YEARS_BACK, -1):
            prev_cnt = len(driver.find_elements(By.CSS_SELECTOR, "a[href]"))
            ok = try_select_year(driver, wait, y)
            if not ok:
                # no dropdown on this site; stop trying years
                break
            wait_table_refresh(driver, prev_cnt, max_wait_s=12)
            out[y] = get_all_links_on_current_page(driver, y)

        return out
    finally:
        driver.quit()

def scrape_ir_links_with_fallback(url: str, YEARS_BACK=3, start_year=2025, headless=True, fallback_headful=True, user_data_dir=None):
    r1 = scrape_ir_links_with_years(url, YEARS_BACK=YEARS_BACK, start_year=start_year, headless=headless, user_data_dir=user_data_dir)
    blocked = bool(r1.get("__blocked__"))
    got_links = sum(len(v) for k, v in r1.items() if k != "__blocked__") > 0

    if (blocked or not got_links) and fallback_headful and headless:
        print("[WARN] Selenium got blocked or saw 0 links in headless. Retrying NON-headless...")
        r2 = scrape_ir_links_with_years(url, YEARS_BACK=YEARS_BACK, start_year=start_year, headless=False, user_data_dir=user_data_dir)
        # drop marker
        r2.pop("__blocked__", None)
        return r2

    r1.pop("__blocked__", None)
    return r1


# =========================
# DEDUPE + CANDIDATES
# =========================

def flatten_and_dedupe(scraped: dict):
    seen = {}
    for year, items in scraped.items():
        for it in items:
            href = (it.get("href") or "").strip()
            if not href:
                continue
            if href not in seen:
                seen[href] = dict(it)
            else:
                if len(it.get("text", "")) > len(seen[href].get("text", "")):
                    seen[href]["text"] = it.get("text", "")
                    seen[href]["year"] = it.get("year", year)
    return list(seen.values())

def looks_like_candidate(it):
    href = (it.get("href") or "").lower()
    text = (it.get("text") or "").lower()
    kind = it.get("kind") or "webpage"

    if kind == "file":
        return True
    if looks_fileish_url(href):
        return True

    if any(k in text for k in [x.lower() for x in FRE_KEYWORDS]) or any(k in href for k in ["fre", "formulario", "referencia"]):
        return True

    if any(k in text for k in [x.lower() for x in EVENT_KEYWORDS]) or any(k in href for k in EVENT_KEYWORDS):
        return True

    if any(k in text for k in [x.lower() for x in FIN_KEYWORDS]) or any(k in href for k in FIN_KEYWORDS):
        return True

    if any(k in text for k in ["cvm", "sec", "filings", "resultados", "relatórios", "relatorios", "demonstra", "investor", "investors"]):
        return True

    return False


# =========================
# 1-HOP EXPANSION (requests)
# =========================

def is_hub_like(it):
    if it.get("kind") == "file":
        return False
    href = (it.get("href") or "").lower()
    text = (it.get("text") or "").lower()
    hub_words = EVENT_KEYWORDS + FIN_KEYWORDS + [
        "presentations", "presentation", "events", "documents", "reports", "filings",
        "annual-meeting", "investor", "investors", "sec", "cvm"
    ]
    return any(w in href for w in hub_words) or any(w in text for w in hub_words)

def scrape_links_requests(url: str, timeout=45) -> list:
    try:
        url = force_https(url)
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        ct = (r.headers.get("Content-Type", "") or "").lower()
        if "html" not in ct:
            return []
        html = r.text or ""
        if is_block_page_text(html):
            return []
        soup = BeautifulSoup(html, "html.parser")
        out = []
        for a in soup.select("a[href]"):
            href = (a.get("href") or "").strip()
            if not href:
                continue
            full = force_https(urljoin(r.url, href))
            if not full.startswith("http"):
                continue
            text = normalize_whitespace(a.get_text(" ", strip=True))
            out.append({"text": text, "href": full})
        return out
    except Exception:
        return []

def expand_one_hop(items, max_hubs=10, sleep=0.8):
    hubs = [it for it in items if is_hub_like(it)][:max_hubs]
    extra = []
    for h in hubs:
        hub_url = h["href"]
        print(f"[1-hop] expanding: {best_label(h.get('text',''), hub_url)}\n  {hub_url}")
        time.sleep(sleep)
        links = scrape_links_requests(hub_url)
        for lk in links:
            extra.append(normalize_item(h.get("year", "hub"), lk.get("text", ""), lk.get("href", "")))
    return extra


# =========================
# SCORING (FILES >>> WEBPAGES)
# =========================

def heuristic_score(it):
    href = (it.get("href") or "").lower()
    text = (it.get("text") or "").lower()
    kind = it.get("kind") or "webpage"
    ext = (it.get("ext") or "").lower()

    score = 0.0
    score += 90.0 if kind == "file" else 10.0

    if ext == ".pdf":
        score += 35.0
    elif ext in (".xlsx", ".xls", ".csv"):
        score += 30.0
    elif ext in (".ppt", ".pptx", ".doc", ".docx"):
        score += 26.0
    elif ext in (".html", ".htm"):
        score += 18.0

    if looks_fileish_url(href):
        score += 40.0

    for k in [x.lower() for x in FRE_KEYWORDS]:
        if k in text or k in href:
            score += 40.0

    for k in [x.lower() for x in FIN_KEYWORDS]:
        if k in text or k in href:
            score += 14.0

    for k in [x.lower() for x in EVENT_KEYWORDS]:
        if k in text or k in href:
            score += 12.0

    return score

def ensure_fre_included(items, top_n=20):
    def is_fre(it):
        t = (it.get("text") or "").lower()
        h = (it.get("href") or "").lower()
        return any(k in t for k in [x.lower() for x in FRE_KEYWORDS]) or any(k in h for k in ["fre", "formulario", "referencia"])

    fre = sorted([it for it in items if is_fre(it)], key=heuristic_score, reverse=True)
    non = sorted([it for it in items if not is_fre(it)], key=heuristic_score, reverse=True)

    forced = fre[:min(len(fre), top_n)]
    remaining_slots = top_n - len(forced)
    forced += non[:max(0, remaining_slots)]
    return forced


# =========================
# OPENAI RANKING (OPTIONAL)
# =========================

def pick_top_links_with_openai(candidates, top_n=20, model="gpt-5.2"):
    client = OpenAI()
    candidates = sorted(candidates, key=heuristic_score, reverse=True)[:250]

    payload = []
    for i, c in enumerate(candidates):
        label = best_label(c.get("text", ""), c.get("href", ""))
        payload.append({
            "id": i,
            "year": c.get("year"),
            "label": label,
            "href": c.get("href", ""),
            "kind": c.get("kind", "webpage"),
            "ext": c.get("ext", ""),
            "hint_score": heuristic_score(c),
        })

    instructions = (
        "You select the most important Investor Relations documents/links.\n"
        "HARD rule: prefer direct FILE links (PDF/XLS/XLSX/CSV/DOC/DOCX/PPT/PPTX/HTML files) over generic webpages.\n"
        "Only pick generic webpages if they are core IR hubs OR clearly lead to key docs.\n"
        "Always prioritize: annual/quarterly results, earnings releases, investor presentations (PDF decks), SEC/CVM filings, proxy statement, annual report.\n"
        "If anything looks like 'Formulário de Referência (FRE)' / 'reference form', treat it as VERY important and include it.\n"
        "Return STRICT JSON only in this format:\n"
        "{\"selected\":[{\"id\":123,\"reason\":\"...\"}, ...]}\n"
    )

    user_input = {"task": f"Select top {top_n}", "links": payload}

    resp = client.responses.create(
        model=model,
        reasoning={"effort": "low"},
        instructions=instructions,
        input=[{"role": "user", "content": json.dumps(user_input, ensure_ascii=False)}],
    )

    data = json.loads(resp.output_text.strip())

    selected_ids = []
    for x in data.get("selected", []):
        if isinstance(x, dict) and "id" in x:
            selected_ids.append(int(x["id"]))

    chosen = []
    for sid in selected_ids:
        if 0 <= sid < len(candidates):
            chosen.append(candidates[sid])

    if len(chosen) < top_n:
        remaining = [c for c in candidates if c not in chosen]
        chosen += remaining[: (top_n - len(chosen))]

    chosen = ensure_fre_included(chosen, top_n=top_n)
    return chosen[:top_n], data


# =========================
# DOWNLOAD (ROBUST)
# =========================

def fetch_raw_bytes(session: requests.Session, url: str, timeout=90):
    url = force_https(url)
    r = session.get(url, headers=DEFAULT_HEADERS, stream=True, timeout=timeout, allow_redirects=True)
    r.raise_for_status()

    try:
        r.raw.decode_content = False
    except Exception:
        pass

    raw = r.raw.read()

    enc = (r.headers.get("Content-Encoding") or "").lower().strip()
    if enc == "gzip" and raw.startswith(b"\x1F\x8B\x08"):
        try:
            raw = gzip.decompress(raw)
        except Exception:
            pass

    return r, raw

def download_items(items, out_dir="downloads_top20", timeout=90, max_retries=4, sleep_between=2.0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()

    saved = []
    for i, it in enumerate(items, start=1):
        href = it["href"]
        label = best_label(it.get("text", ""), href)

        print(f"[{i}/{len(items)}] downloading: {label}\n  {href}")

        ok = False
        last_err = None

        for attempt in range(1, max_retries + 1):
            try:
                if attempt > 1:
                    time.sleep(min(12, sleep_between * (2 ** (attempt - 2))))

                r, raw = fetch_raw_bytes(sess, href, timeout=timeout)

                final_url = r.url
                ct = (r.headers.get("Content-Type", "") or "").lower()
                ct_ext = guess_extension_from_content_type(ct)
                magic_ext = sniff_magic_extension(raw[:64])

                bytes_look_html = looks_like_html_bytes(raw[:4096])
                bytes_look_text = looks_like_text_bytes(raw[:4096])

                is_pdf = raw.startswith(b"%PDF-") or ("application/pdf" in ct) or (magic_ext == ".pdf")
                is_file_by_magic = magic_ext in [".pdf", ".zip", ".mp3", ".wav"]
                is_html = (bytes_look_html or bytes_look_text) and not is_file_by_magic and not is_pdf

                if is_html:
                    encoding = r.encoding or "utf-8"
                    try:
                        html = raw.decode(encoding, errors="replace")
                    except Exception:
                        html = raw.decode("utf-8", errors="replace")

                    fname = safe_filename(label) + "_" + hashlib.md5(final_url.encode("utf-8")).hexdigest()[:8] + ".html"
                    fpath = out_dir / fname
                    fpath.write_text(html, encoding="utf-8", errors="ignore")
                    saved.append(str(fpath))
                    ok = True
                    break

                fname = filename_from_response(final_url, r.headers, label)

                # extension priority
                final_ext = ct_ext or (magic_ext if magic_ext != ".gz" else "") or it.get("ext") or url_path_ext(final_url) or ".bin"

                if "." not in fname:
                    fname += final_ext
                else:
                    if raw.startswith(b"%PDF-") and not fname.lower().endswith(".pdf"):
                        fname = safe_filename(Path(fname).stem) + ".pdf"

                fpath = out_dir / fname
                with open(fpath, "wb") as f:
                    f.write(raw)

                print(f"  ✓ saved file: {fpath.name} (ct={ct or 'n/a'})")
                saved.append(str(fpath))
                ok = True
                break

            except requests.exceptions.ReadTimeout as e:
                last_err = e
                print(f"  !! timeout (attempt {attempt}/{max_retries})")
            except Exception as e:
                last_err = e
                print(f"  !! failed (attempt {attempt}/{max_retries}): {e}")

        if not ok:
            print(f"  !! giving up: {last_err}")

    return saved


# =========================
# PIPELINE
# =========================

def run_ir_pipeline(
    url: str,
    YEARS_BACK=3,
    start_year=2025,
    headless=True,
    fallback_headful=True,
    top_n=100,
    do_one_hop=True,
    openai_model="gpt-5.2",
    out_dir="downloads_top20",
    user_data_dir=None,
):
    scraped = scrape_ir_links_with_fallback(
        url=url,
        YEARS_BACK=YEARS_BACK,
        start_year=start_year,
        headless=headless,
        fallback_headful=fallback_headful,
        user_data_dir=user_data_dir,
    )

    flat = flatten_and_dedupe(scraped)
    cands = [x for x in flat if looks_like_candidate(x)]

    print(f"\nTotal deduped links: {len(flat)}")
    print(f"Candidate links: {len(cands)}")

    if do_one_hop and cands:
        extra = expand_one_hop(cands, max_hubs=10, sleep=0.8)
        if extra:
            merged = flatten_and_dedupe({"base": cands, "extra": extra})
            cands = [x for x in merged if looks_like_candidate(x)]
            print(f"After 1-hop expansion: candidates = {len(cands)}")

    use_llm = bool(os.getenv("OPENAI_API_KEY"))
    if use_llm and cands:
        top, _dbg = pick_top_links_with_openai(cands, top_n=top_n, model=openai_model)
    else:
        top = ensure_fre_included(sorted(cands, key=heuristic_score, reverse=True)[:top_n], top_n=top_n) if cands else []

    saved = download_items(top, out_dir=out_dir, timeout=90, max_retries=4, sleep_between=2.0) if top else []

    return top, saved

dict_ir_links = {
                #"NVDA":["https://investor.nvidia.com/events-and-presentations/events-and-presentations/default.aspx","https://investor.nvidia.com/financial-info/quarterly-results/default.aspx","https://investor.nvidia.com/events-and-presentations/presentations/default.aspx"],
                #"MSFT":["https://www.microsoft.com/en-us/Investor/events/default","https://www.microsoft.com/en-us/Investor/annual-reports","https://www.microsoft.com/en-us/investor/annual-meeting"],
                #"AAPL":["https://investor.apple.com/investor-relations/default.aspx"],
                #"GOOGL":["https://abc.xyz/investor/earnings/","https://abc.xyz/investor/events/default.aspx","https://abc.xyz/investor/news/default.aspx"],
                #"AMZN":["https://ir.aboutamazon.com/quarterly-results/default.aspx","https://press.aboutamazon.com/press-release-archive"],
                #"META":["https://investor.atmeta.com/investor-events/default.aspx","https://investor.atmeta.com/investor-news/default.aspx","https://investor.atmeta.com/annual-meeting/default.aspx"],
                #"AVGO":["https://investors.broadcom.com/financial-information/quarterly-results"],
                #"TSLA":["https://ir.tesla.com/press","https://ir.tesla.com/#quarterly-disclosure"],
                #"BRK.B":["https://www.berkshirehathaway.com/news/2025news.html"],
                #"JPM":["https://www.jpmorganchase.com/ir/events","https://www.jpmorganchase.com/ir/quarterly-earnings","https://www.jpmorganchase.com/ir/investor-day"],
                #"NU":["https://www.investidores.nu/financas/central-de-resultados/","https://www.investidores.nu/en/press-releases/"],
                #"BPAC11":["https://ri.btgpactual.com/documentos-cvm/","https://ri.btgpactual.com/principais-informacoes/informacoes-financeiras/","https://ri.btgpactual.com/principais-informacoes/apresentacoes-e-planilhas-series-historicas/"],
                #"BBDC4":["https://www.bradescori.com.br/informacoes-ao-mercado/central-de-resultados/","https://www.bradescori.com.br/informacoes-ao-mercado/relatorios-e-planilhas/cvm/","https://www.bradescori.com.br/informacoes-ao-mercado/apresentacoes/"],
                #"PETR4":["https://www.investidorpetrobras.com.br/apresentacoes-relatorios-e-eventos/relatorios-anuais/","https://www.investidorpetrobras.com.br/resultados-e-comunicados/central-de-resultados/"],
                #"KLBN11":["https://ri.klabin.com.br/informacoes-financeiras/documentos-entregues-a-cvm/"] # ["https://ri.klabin.com.br/divulgacoes-e-resultados/central-de-resultados/","https://ri.klabin.com.br/informacoes-financeiras/documentos-entregues-a-cvm/","https://ri.klabin.com.br/divulgacoes-e-resultados/apresentacoes/"],
                #"WEGE3":["https://ri.weg.net/informacoes-financeiras/central-de-resultados/","https://ri.weg.net/informacoes-financeiras/formulario-de-referencia-e-cadastral/","https://ri.weg.net/publicacoes-e-comunicados/apresentacoes-e-documentos/"],
                #"ABEV3":["https://ri.ambev.com.br/relatorios-publicacoes/divulgacao-de-resultados/","https://ri.ambev.com.br/relatorios-publicacoes/publicacoes-cvm-sec/"],
                #"SANB11":["https://www.santander.com.br/ri/resultados-e-relatorios/divulgacao-de-resultados/","https://www.santander.com.br/ri/resultados-e-relatorios/relatorios-e-apresentacoes/"],
                #"ITUB4":["https://www.itau.com.br/relacoes-com-investidores/resultados-e-relatorios/central-de-resultados/","https://www.itau.com.br/relacoes-com-investidores/resultados-e-relatorios/documentos-regulatorios/formulario-de-referencia/"],
                "VALE3":["https://ri-vale.mz-sites.com/informacoes-para-o-mercado/relatorios-anuais/formulario-de-referencia/","https://vale.com/pt/comunicados-resultados-apresentacoes-e-relatorios"],
                }

for ticker in tqdm(dict_ir_links.keys()):
    
    links = dict_ir_links[ticker]
    
    for url in links:
        
        top, saved = run_ir_pipeline(
                     url=url,
                     YEARS_BACK=3,
                     start_year=2025,
                     headless=True,
                     top_n=200,
                     do_one_hop=True,
                     openai_model="gpt-5.2",
                     out_dir=f"Data/{ticker}/ir_data")