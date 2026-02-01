import os
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from urllib.parse import urljoin, urlparse
import time
from pathlib import Path
from datetime import datetime

def create_folder(ticker):
    """Create folder structure for storing files"""
    folder_path = Path(f"Data/{ticker}/ir_data")
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path

def setup_driver():
    """Setup Selenium WebDriver with Chrome"""
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def scroll_page(driver, num_scrolls=5):
    """Scroll page to trigger lazy loading"""
    print("  Scrolling page to load dynamic content...")
    last_height = driver.execute_script("return document.body.scrollHeight")
    
    for i in range(num_scrolls):
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        
        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        
        if new_height == last_height:
            # Try scrolling up and down again
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
            time.sleep(1)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
        
        last_height = new_height
    
    # Scroll back to top
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(1)

def get_past_years(n=3):
    """Get list of past n years including current year"""
    current_year = datetime.now().year
    return [str(current_year - i) for i in range(n)]

def find_year_dropdown(driver):
    """Find year dropdown/select element using multiple strategies"""
    wait = WebDriverWait(driver, 10)
    
    # Strategy 1: Look for select elements with year-related names
    selectors = [
        "select[name*='year' i]",
        "select[name*='ano' i]",
        "select[id*='year' i]",
        "select[id*='ano' i]",
        "select[class*='year' i]",
        "select[class*='ano' i]",
        "select[aria-label*='year' i]",
        "select"
    ]
    
    for selector in selectors:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            for element in elements:
                options = element.find_elements(By.TAG_NAME, "option")
                if any(opt.text.strip().isdigit() and len(opt.text.strip()) == 4 
                       for opt in options):
                    return element, 'select'
        except:
            continue
    
    # Strategy 2: Look for button groups or div-based dropdowns
    button_selectors = [
        "button[class*='year' i]",
        "div[class*='year' i]",
        "a[class*='year' i]",
        "button[class*='ano' i]",
        "div[class*='ano' i]",
        "button[data-year]",
        "div[data-year]"
    ]
    
    for selector in button_selectors:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            if elements:
                return elements, 'buttons'
        except:
            continue
    
    return None, None

def select_year_from_dropdown(driver, year, dropdown_element, dropdown_type):
    """Select a specific year from dropdown"""
    try:
        if dropdown_type == 'select':
            select = Select(dropdown_element)
            
            try:
                select.select_by_visible_text(year)
            except:
                try:
                    select.select_by_value(year)
                except:
                    for option in select.options:
                        if year in option.text:
                            option.click()
                            break
            
            time.sleep(3)
            return True
            
        elif dropdown_type == 'buttons':
            for element in dropdown_element:
                if year in element.text:
                    driver.execute_script("arguments[0].click();", element)
                    time.sleep(3)
                    return True
        
        return False
    except Exception as e:
        print(f"  Warning: Error selecting year {year}: {e}")
        return False

def get_all_links_selenium(driver, base_url):
    """Extract all links from current page state using Selenium"""
    try:
        # Scroll to load all content
        scroll_page(driver, num_scrolls=5)
        
        # Wait for any dynamic content
        time.sleep(3)
        
        links = []
        
        # Method 1: Get links from Selenium directly
        print("  Extracting links from page...")
        link_elements = driver.find_elements(By.TAG_NAME, "a")
        
        for element in link_elements:
            try:
                href = element.get_attribute('href')
                if href:
                    full_url = urljoin(base_url, href)
                    parsed_url = urlparse(full_url)
                    parsed_base = urlparse(base_url)
                    
                    # Include same domain links and document links
                    if (parsed_url.netloc == parsed_base.netloc or 
                        any(full_url.lower().endswith(ext) for ext in 
                            ['.pdf', '.xlsx', '.xls', '.csv', '.doc', '.docx', '.zip', '.ppt', '.pptx'])):
                        links.append(full_url)
            except:
                continue
        
        # Method 2: Parse with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            parsed_url = urlparse(full_url)
            parsed_base = urlparse(base_url)
            
            if (parsed_url.netloc == parsed_base.netloc or 
                any(full_url.lower().endswith(ext) for ext in 
                    ['.pdf', '.xlsx', '.xls', '.csv', '.doc', '.docx', '.zip', '.ppt', '.pptx'])):
                links.append(full_url)
        
        # Remove duplicates
        links = list(set(links))
        
        # Filter out navigation links, images, and non-content URLs
        filtered_links = []
        exclude_patterns = ['#', 'javascript:', 'mailto:', '.jpg', '.png', '.gif', '.svg', '.ico']
        
        for link in links:
            if not any(pattern in link.lower() for pattern in exclude_patterns):
                filtered_links.append(link)
        
        return filtered_links
    
    except Exception as e:
        print(f"  Error extracting links: {e}")
        return []

def get_file_extension(url, content_type):
    """Determine the appropriate file extension based on URL and content type"""
    url_path = urlparse(url).path.lower()
    
    extensions = {
        '.pdf': '.pdf',
        '.xlsx': '.xlsx',
        '.xls': '.xls',
        '.csv': '.csv',
        '.doc': '.doc',
        '.docx': '.docx',
        '.ppt': '.ppt',
        '.pptx': '.pptx',
        '.txt': '.txt',
        '.xml': '.xml',
        '.json': '.json',
        '.zip': '.zip'
    }
    
    for ext in extensions:
        if url_path.endswith(ext):
            return extensions[ext]
    
    if content_type:
        content_type = content_type.lower()
        
        if 'pdf' in content_type:
            return '.pdf'
        elif 'excel' in content_type or 'spreadsheet' in content_type:
            return '.xlsx' if 'openxmlformats' in content_type else '.xls'
        elif 'csv' in content_type:
            return '.csv'
        elif 'word' in content_type or 'msword' in content_type:
            return '.docx' if 'openxmlformats' in content_type else '.doc'
        elif 'powerpoint' in content_type or 'presentation' in content_type:
            return '.pptx' if 'openxmlformats' in content_type else '.ppt'
        elif 'text/plain' in content_type:
            return '.txt'
        elif 'json' in content_type:
            return '.json'
        elif 'xml' in content_type:
            return '.xml'
        elif 'html' in content_type:
            return '.html'
    
    return '.html'

def sanitize_filename(url, extension, year=None):
    """Create a valid filename from URL"""
    parsed = urlparse(url)
    path = parsed.path
    
    if path and path != '/':
        filename = path.split('/')[-1]
        if '.' in filename:
            filename = '.'.join(filename.split('.')[:-1])
    else:
        filename = parsed.netloc + parsed.path.replace('/', '_')
        if parsed.query:
            filename += '_' + parsed.query[:50]
    
    filename = filename.replace('?', '_').replace('&', '_').replace('=', '_')
    filename = filename.replace(' ', '_').replace('|', '_').replace(':', '_')
    
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    filename = filename[:200]
    filename = filename.rstrip('_.')
    
    if not filename:
        filename = f"page_{abs(hash(url)) % 10000}"
    
    if year:
        filename = f"{year}_{filename}"
    
    return filename + extension

def download_and_save(url, save_path):
    """Download and save a file in its original format"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '')
        extension = get_file_extension(url, content_type)
        
        if not str(save_path).endswith(extension):
            save_path = save_path.with_suffix(extension)
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        return True, extension, content_type
    
    except Exception as e:
        print(f"  Error downloading: {str(e)[:50]}")
        return False, None, None

def scrape_with_year_selection(ticker, url, num_years=3):
    """Main function to scrape links with year selection support"""
    print(f"\n{'='*70}")
    print(f"Processing {ticker}: {url}")
    print(f"{'='*70}")
    
    folder_path = create_folder(ticker)
    parsed = urlparse(url)
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    
    driver = None
    all_links = []
    links_by_year = {}
    
    try:
        print("Initializing browser...")
        driver = setup_driver()
        driver.get(url)
        
        print("Waiting for page to load...")
        time.sleep(5)
        
        target_years = get_past_years(num_years)
        print(f"Target years: {', '.join(target_years)}")
        
        print("Looking for year selector...")
        dropdown, dropdown_type = find_year_dropdown(driver)
        
        if dropdown:
            print(f"Found year selector (type: {dropdown_type})")
            
            for year in target_years:
                print(f"\nProcessing year: {year}")
                
                if select_year_from_dropdown(driver, year, dropdown, dropdown_type):
                    print(f"  Selected year {year}")
                    
                    year_links = get_all_links_selenium(driver, base_url)
                    links_by_year[year] = year_links
                    all_links.extend(year_links)
                    
                    print(f"  Found {len(year_links)} links for {year}")
                else:
                    print(f"  Could not select year {year}")
        else:
            print("No year selector found - using static scraping")
            year_links = get_all_links_selenium(driver, base_url)
            links_by_year['all'] = year_links
            all_links = year_links
            print(f"Found {len(year_links)} total links")
        
    except Exception as e:
        print(f"Error during scraping: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if driver:
            driver.quit()
    
    all_links = list(set(all_links))
    print(f"\nTotal unique links found: {len(all_links)}")
    
    if len(all_links) == 0:
        print("WARNING: No links found. The page might require different scraping approach.")
        print("Please check if the URL is correct and accessible.")
        return
    
    print(f"\n{'='*70}")
    print("Starting downloads...")
    print(f"{'='*70}")
    
    successful = 0
    failed = 0
    file_types = {}
    
    for idx, link in enumerate(all_links, 1):
        link_year = None
        for year, year_links in links_by_year.items():
            if link in year_links:
                link_year = year
                break
        
        initial_filename = sanitize_filename(link, '', link_year)
        save_path = folder_path / initial_filename
        
        print(f"[{idx}/{len(all_links)}] {link[:65]}...")
        
        success, extension, content_type = download_and_save(link, save_path)
        
        if success:
            successful += 1
            file_types[extension] = file_types.get(extension, 0) + 1
            print(f"  Saved as: {extension}")
        else:
            failed += 1
        
        time.sleep(1)
    
    print(f"\n{'='*70}")
    print(f"{ticker} Summary:")
    print(f"{'='*70}")
    print(f"  Successfully downloaded: {successful}")
    print(f"  Failed: {failed}")
    print(f"\n  File Types:")
    for ext, count in sorted(file_types.items()):
        print(f"    {ext}: {count} files")
    print(f"\n  Saved in: {folder_path}")
    print(f"{'='*70}")

def main():
    """Main execution function"""
    targets = {
        #'AVGO': 'https://investors.broadcom.com/financial-information/quarterly-results',
        #'ITUB4':'https://www.itau.com.br/relacoes-com-investidores/resultados-e-relatorios/central-de-resultados/',
        'ITUB4':'https://www.itau.com.br/relacoes-com-investidores/resultados-e-relatorios/documentos-regulatorios/formulario-de-referencia/'
        #'TSLA': 'https://ir.tesla.com/press'
    }
    
    print("="*70)
    print("ADVANCED INVESTOR RELATIONS WEB SCRAPER")
    print("="*70)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Scraping past 3 years of data")
    print(f"Supported formats: PDF, Excel, CSV, Word, PPT, HTML, JSON, XML, ZIP")
    print("="*70)
    
    for ticker, url in targets.items():
        try:
            scrape_with_year_selection(ticker, url, num_years=3)
        except Exception as e:
            print(f"\nCritical error processing {ticker}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("SCRAPING COMPLETED!")
    print("="*70)

if __name__ == "__main__":
    main()
