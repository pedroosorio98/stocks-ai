import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, unquote
import time
import random
from pathlib import Path
import hashlib
import base64

def create_folder(ticker):
    """Create folder structure"""
    folder_path = Path(f"Data/{ticker}/ir_data")
    folder_path.mkdir(parents=True, exist_ok=True)
    
    # Create PDF subfolder
    pdf_folder = folder_path
    pdf_folder.mkdir(exist_ok=True)
    
    return folder_path

def setup_stealth_driver(download_path=None):
    """Setup undetected Chrome driver with download preferences"""
    print("  Setting up stealth Chrome driver (v144)...")
    
    options = uc.ChromeOptions()
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--start-maximized')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    # PDF download settings
    if download_path:
        prefs = {
            "download.default_directory": str(download_path),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "plugins.always_open_pdf_externally": True,  # Don't open in browser
            "safebrowsing.enabled": True
        }
        options.add_experimental_option("prefs", prefs)
    
    driver = uc.Chrome(
        version_main=144,
        options=options,
        use_subprocess=True
    )
    
    print(f"  ✓ Stealth driver initialized (Chrome 144)")
    return driver

def human_delay(min_sec=3, max_sec=7):
    """Random human-like delay"""
    time.sleep(random.uniform(min_sec, max_sec))

def wait_for_page_load(driver, timeout=90):
    """Wait for page to fully load"""
    print("  Waiting for page to load...")
    
    try:
        print("    [1/5] Document ready...")
        WebDriverWait(driver, timeout).until(
            lambda d: d.execute_script('return document.readyState') == 'complete'
        )
        print("    ✓ Document ready")
        
        print("    [2/5] Body element...")
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        print("    ✓ Body found")
        
        print("    [3/5] Anti-bot challenges...")
        human_delay(8, 12)
        print("    ✓ Challenge processing complete")
        
        print("    [4/5] JavaScript execution...")
        human_delay(5, 8)
        print("    ✓ JavaScript complete")
        
        print("    [5/5] Checking for links...")
        try:
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.TAG_NAME, "a"))
            )
            link_count = len(driver.find_elements(By.TAG_NAME, "a"))
            print(f"    ✓ Found {link_count} link elements")
        except:
            print("    ⚠️ No links found")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False

def scroll_and_load(driver, scrolls=25):
    """Scroll to load content"""
    print("  Scrolling...")
    
    last_height = driver.execute_script("return document.body.scrollHeight")
    
    for i in range(scrolls):
        if i % 3 == 0:
            scroll_position = random.randint(500, 1500)
            driver.execute_script(f"window.scrollBy(0, {scroll_position});")
        else:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        
        human_delay(2, 4)
        
        # Click Load More buttons
        try:
            load_buttons = driver.find_elements(By.XPATH, 
                "//button[contains(translate(., 'LOAD', 'load'), 'load')] | "
                "//button[contains(translate(., 'MORE', 'more'), 'more')]")
            
            for btn in load_buttons:
                try:
                    if btn.is_displayed() and btn.is_enabled():
                        print(f"    ✓ Clicking 'Load More'")
                        driver.execute_script("arguments[0].click();", btn)
                        human_delay(4, 7)
                except:
                    pass
        except:
            pass
        
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height != last_height:
            print(f"    ✓ New content loaded")
            last_height = new_height
        
        if (i + 1) % 5 == 0:
            print(f"    Progress: {i+1}/{scrolls}")
    
    driver.execute_script("window.scrollTo(0, 0);")
    human_delay(2, 4)
    print("  ✓ Scrolling complete")

def get_all_links(driver, base_url):
    """Extract all links including PDFs"""
    print("\n  Extracting links...")
    all_links = set()
    
    # Selenium
    try:
        elements = driver.find_elements(By.TAG_NAME, "a")
        print(f"  [1/3] Selenium: {len(elements)} elements")
        for elem in elements:
            try:
                href = elem.get_attribute('href')
                if href:
                    all_links.add(href)
            except:
                pass
        print(f"        → {len(all_links)} links")
    except Exception as e:
        print(f"  [1/3] Error: {e}")
    
    # JavaScript
    try:
        js_links = driver.execute_script("""
            var links = [];
            document.querySelectorAll('a[href]').forEach(a => links.push(a.href));
            return links;
        """)
        print(f"  [2/3] JavaScript: {len(js_links)} links")
        all_links.update(js_links)
        print(f"        → Total: {len(all_links)}")
    except Exception as e:
        print(f"  [2/3] Error: {e}")
    
    # BeautifulSoup
    try:
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        bs_links = soup.find_all('a', href=True)
        print(f"  [3/3] BeautifulSoup: {len(bs_links)} tags")
        for link in bs_links:
            href = link.get('href')
            if href:
                full_url = urljoin(base_url, href)
                all_links.add(full_url)
        print(f"        → Total: {len(all_links)}")
    except Exception as e:
        print(f"  [3/3] Error: {e}")
    
    print(f"\n  ✓ Total unique links: {len(all_links)}")
    return list(all_links)

def filter_relevant_links(links, base_url):
    """Filter for relevant links INCLUDING PDFs"""
    parsed_base = urlparse(base_url)
    base_domain = parsed_base.netloc
    relevant = []
    pdf_links = []
    
    for link in links:
        parsed = urlparse(link)
        
        if parsed.netloc != base_domain:
            continue
        
        link_lower = link.lower()
        
        # **CAPTURE PDF LINKS**
        if link_lower.endswith('.pdf') or '/pdf/' in link_lower:
            pdf_links.append(link)
            continue
        
        # Skip non-content
        skip = ['#', 'javascript:', 'mailto:', 'tel:', '.jpg', '.png', '.gif', 
                '.svg', '.ico', '.css', '.js', '.woff', '/search', '/login']
        
        if any(s in link_lower for s in skip):
            continue
        
        # Keep press releases and important pages
        if any(x in link_lower for x in ['press-release', 'sec-filing', 'corporate', '/press', 'quarterly']):
            relevant.append(link)
    
    return list(dict.fromkeys(relevant)), list(dict.fromkeys(pdf_links))

def extract_pdfs_from_page(html_content, base_url):
    """Extract PDF links from HTML content"""
    soup = BeautifulSoup(html_content, 'html.parser')
    pdf_links = set()
    
    # Find all links
    for link in soup.find_all('a', href=True):
        href = link.get('href')
        if href and (href.lower().endswith('.pdf') or '/pdf/' in href.lower()):
            full_url = urljoin(base_url, href)
            pdf_links.add(full_url)
    
    return list(pdf_links)

def sanitize_filename(url, is_pdf=False):
    """Create safe filename"""
    parsed = urlparse(url)
    path = unquote(parsed.path)
    
    if path and path != '/':
        parts = [p for p in path.split('/') if p]
        filename = '_'.join(parts[-3:]) if len(parts) >= 3 else '_'.join(parts)
    else:
        filename = hashlib.md5(url.encode()).hexdigest()[:12]
    
    for char in '<>:"/\\|?*&=% +':
        filename = filename.replace(char, '_')
    
    filename = filename[:200].strip('_.')
    
    if not filename:
        filename = hashlib.md5(url.encode()).hexdigest()[:12]
    
    # Keep .pdf extension if it's a PDF, otherwise use .html
    if is_pdf:
        if not filename.endswith('.pdf'):
            filename += '.pdf'
    else:
        if not filename.endswith('.html'):
            filename += '.html'
    
    return filename

def download_pdf_with_selenium(driver, url, save_path):
    """Download PDF using Selenium - fetches as base64 then saves"""
    try:
        # Navigate to PDF URL
        driver.get(url)
        human_delay(2, 4)
        
        # Get PDF as base64 using Chrome DevTools
        try:
            pdf_base64 = driver.execute_cdp_cmd("Page.printToPDF", {
                "printBackground": True
            })
            
            # This method doesn't work for external PDFs, try alternative
        except:
            pass
        
        # Alternative: Use JavaScript to fetch as blob
        pdf_content = driver.execute_async_script("""
            var url = arguments[0];
            var callback = arguments[1];
            
            var xhr = new XMLHttpRequest();
            xhr.open('GET', url, true);
            xhr.responseType = 'blob';
            
            xhr.onload = function() {
                if (this.status === 200) {
                    var reader = new FileReader();
                    reader.onloadend = function() {
                        callback(reader.result);
                    }
                    reader.readAsDataURL(xhr.response);
                } else {
                    callback(null);
                }
            };
            
            xhr.onerror = function() {
                callback(null);
            };
            
            xhr.send();
        """, url)
        
        if pdf_content and pdf_content.startswith('data:'):
            # Extract base64 content
            base64_data = pdf_content.split(',')[1]
            pdf_bytes = base64.b64decode(base64_data)
            
            # Save to file
            with open(save_path, 'wb') as f:
                f.write(pdf_bytes)
            
            size_kb = save_path.stat().st_size / 1024
            print(f"  ✓ PDF saved ({size_kb:.1f} KB)")
            return True
        else:
            print(f"  ✗ Failed to fetch PDF content")
            return False
    
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:100]}")
        return False

def download_page(driver, url, save_path):
    """Download page using Selenium"""
    try:
        driver.get(url)
        human_delay(3, 6)
        
        WebDriverWait(driver, 30).until(
            lambda d: d.execute_script('return document.readyState') == 'complete'
        )
        human_delay(2, 4)
        
        html_content = driver.page_source
        
        if 'access denied' in html_content.lower():
            print(f"  🚫 Blocked")
            return False, None
        
        with open(save_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(html_content)
        
        size_kb = save_path.stat().st_size / 1024
        print(f"  ✓ Saved ({size_kb:.1f} KB)")
        
        # Return the HTML content for PDF extraction
        return True, html_content
    
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:50]}")
        return False, None

def scrape_and_save(ticker, url):
    """Main function"""
    print(f"\n{'='*70}")
    print(f"Processing {ticker}: {url}")
    print(f"{'='*70}")
    
    folder_path = create_folder(ticker)
    pdf_folder = folder_path / "pdfs"
    
    parsed = urlparse(url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    
    driver = None
    
    try:
        driver = setup_stealth_driver(download_path=pdf_folder)
        
        if not url.startswith('https://'):
            url = url.replace('http://', 'https://')
        
        print(f"\n  Loading: {url}")
        driver.get(url)
        
        wait_for_page_load(driver)
        scroll_and_load(driver, scrolls=25)
        
        all_links = get_all_links(driver, base_url)
        
        if not all_links:
            print("\n⚠️  NO LINKS FOUND!")
            return
        
        relevant, initial_pdfs = filter_relevant_links(all_links, base_url)
        print(f"\n  Filtered to {len(relevant)} HTML pages")
        print(f"  Found {len(initial_pdfs)} PDF links on main page")
        
        if len(relevant) > 0:
            print("\n  Sample HTML links:")
            for i, link in enumerate(relevant[:5], 1):
                print(f"    {i}. {link}")
        
        if len(initial_pdfs) > 0:
            print("\n  Sample PDF links:")
            for i, link in enumerate(initial_pdfs[:5], 1):
                print(f"    {i}. {link}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        if driver:
            driver.quit()
        return
    
    if not relevant and not initial_pdfs:
        print("\n⚠️  No relevant links!")
        if driver:
            driver.quit()
        return
    
    # **DOWNLOAD HTML PAGES AND EXTRACT MORE PDFs**
    print(f"\n{'='*70}")
    print(f"DOWNLOADING {len(relevant)} HTML PAGES...")
    print(f"{'='*70}")
    
    success = 0
    failed = 0
    all_pdfs = set(initial_pdfs)
    
    for idx, link in enumerate(relevant, 1):
        filename = sanitize_filename(link, is_pdf=False)
        save_path = folder_path / filename
        
        if save_path.exists():
            print(f"[{idx}/{len(relevant)}] {link[:60]}... (exists)")
            success += 1
            
            # Still extract PDFs from existing file
            try:
                with open(save_path, 'r', encoding='utf-8', errors='ignore') as f:
                    html_content = f.read()
                page_pdfs = extract_pdfs_from_page(html_content, base_url)
                all_pdfs.update(page_pdfs)
            except:
                pass
            
            continue
        
        print(f"[{idx}/{len(relevant)}] {link[:60]}...")
        
        result, html_content = download_page(driver, link, save_path)
        if result:
            success += 1
            # Extract PDFs from this page
            if html_content:
                page_pdfs = extract_pdfs_from_page(html_content, base_url)
                if page_pdfs:
                    print(f"    → Found {len(page_pdfs)} PDFs on this page")
                    all_pdfs.update(page_pdfs)
        else:
            failed += 1
        
        time.sleep(random.uniform(2, 4))
    
    # **DOWNLOAD ALL PDFs USING SELENIUM**
    all_pdfs = list(all_pdfs)
    print(f"\n{'='*70}")
    print(f"DOWNLOADING {len(all_pdfs)} PDF FILES...")
    print(f"{'='*70}")
    
    pdf_success = 0
    pdf_failed = 0
    
    for idx, pdf_url in enumerate(all_pdfs, 1):
        filename = sanitize_filename(pdf_url, is_pdf=True)
        save_path = pdf_folder / filename
        
        if save_path.exists():
            print(f"[{idx}/{len(all_pdfs)}] {pdf_url[:60]}... (exists)")
            pdf_success += 1
            continue
        
        print(f"[{idx}/{len(all_pdfs)}] {pdf_url[:60]}...")
        
        if download_pdf_with_selenium(driver, pdf_url, save_path):
            pdf_success += 1
        else:
            pdf_failed += 1
        
        time.sleep(random.uniform(2, 4))
    
    if driver:
        driver.quit()
        print("\n  Browser closed")
    
    print(f"\n{'='*70}")
    print(f"{ticker} SUMMARY")
    print(f"{'='*70}")
    print(f"  HTML Pages:")
    print(f"    Links found: {len(relevant)}")
    print(f"    ✓ Downloaded: {success}")
    print(f"    ✗ Failed: {failed}")
    print(f"\n  PDF Files:")
    print(f"    Links found: {len(all_pdfs)}")
    print(f"    ✓ Downloaded: {pdf_success}")
    print(f"    ✗ Failed: {pdf_failed}")
    print(f"\n  📁 HTML Location: {folder_path}")
    print(f"  📄 PDF Location: {pdf_folder}")
    print(f"{'='*70}")

def main():
    """Main execution"""
    targets = {
        #'TSLA': 'https://ir.tesla.com/press',
        'TSLA': 'https://ir.tesla.com/#quarterly-disclosure',
    }
    
    print("="*70)
    print("TESLA SCRAPER WITH PDF DOWNLOAD (SELENIUM METHOD)")
    print("="*70)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    for ticker, url in targets.items():
        try:
            scrape_and_save(ticker, url)
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    print("\n" + "="*70)
    print("SCRAPING COMPLETED!")
    print("="*70)

if __name__ == "__main__":
    main()
