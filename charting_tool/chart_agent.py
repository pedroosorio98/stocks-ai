# chart_agent.py
"""
AI-Powered Charting Agent for Financial Data

Architecture:
1. Scan all available CSV files for requested tickers
2. Extract metadata (filenames, column names, row names)
3. AI identifies which columns/rows contain requested data
4. AI generates Python code to load, transform, and plot data
5. Execute code safely and return Plotly JSON

Example Flow:
User: "Plot NVDA revenue together with NVDA stock price, stock price on secondary axis"
→ Scan Data/NVDA/ files
→ Find: income_stmt_quarterly.csv has "Total Revenue" row
→ Find: historical_data.csv has "Close" column
→ Generate code to load both, create dual-axis plot
→ Execute and return chart
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import numpy as np
import re
from openai import OpenAI
import plotly
import datetime

client = OpenAI()


class DataCatalog:
    """
    Scans CSV files and creates a metadata catalog for AI to search
    """
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.catalog = {
            'ticker': self.ticker,
            'alpha_vantage': {},
            'yahoo_finance': {}
        }
        self._scan_files()
    
    def _scan_files(self):
        """Scan all CSV files and extract metadata"""
        ticker_folder = Path("Data") / self.ticker
        
        # Scan Alpha Vantage files
        av_path = ticker_folder / "alpha_vantage" / "csv"
        if av_path.exists():
            for csv_file in av_path.glob("*.csv"):
                self.catalog['alpha_vantage'][csv_file.name] = self._get_file_metadata(csv_file)
        
        # Scan Yahoo Finance files
        yf_path = ticker_folder / "yahoo_finance" / "csv"
        if yf_path.exists():
            for csv_file in yf_path.glob("*.csv"):
                self.catalog['yahoo_finance'][csv_file.name] = self._get_file_metadata(csv_file)
    
    def _get_file_metadata(self, filepath: Path) -> Dict:
        """Extract metadata from CSV file"""
        try:
            # Read first few rows to get structure
            df = pd.read_csv(filepath, nrows=5)
            
            metadata = {
                'path': str(filepath),
                'columns': list(df.columns),
                'num_rows': len(df),
                'sample_data': df.head(3).to_dict('records')
            }
            
            # For files with index column (Yahoo Finance format)
            if df.columns[0] in ['Breakdown', 'Unnamed: 0']:
                df_indexed = pd.read_csv(filepath, index_col=0, nrows=5)
                metadata['row_names'] = list(df_indexed.index)
                metadata['format'] = 'rows_are_metrics'  # Rows = metrics, columns = dates
            else:
                metadata['format'] = 'columns_are_metrics'  # Columns = metrics, rows = dates
            
            return metadata
            
        except Exception as e:
            return {'error': str(e)}
    
    def to_json(self) -> str:
        """Convert catalog to JSON for AI"""
        return json.dumps(self.catalog, indent=2)


def create_multi_ticker_catalog(tickers: List[str]) -> Dict:
    """Create catalog for multiple tickers"""
    catalogs = {}
    for ticker in tickers:
        catalog = DataCatalog(ticker)
        catalogs[ticker.upper()] = catalog.catalog
    return catalogs


def identify_data_sources(user_prompt: str, tickers: List[str]) -> Dict[str, Any]:
    """
    Use AI to identify which CSV files and columns contain requested data
    
    This function:
    1. Scans actual CSV files that exist for each ticker
    2. Returns REAL filenames and paths from the file system
    3. AI picks the right file based on what actually exists
    
    Args:
        user_prompt: User's natural language request
        tickers: List of tickers mentioned in prompt
    
    Returns:
        Dict with identified data sources and column mappings (using real filenames)
    """
    # Create catalog by scanning actual files
    catalogs = create_multi_ticker_catalog(tickers)
    catalog_json = json.dumps(catalogs, indent=2)
    
    print(f"[DATA SOURCES] Scanned files for {tickers}")
    print(f"[DATA SOURCES] Catalog: {catalog_json[:500]}...")  # Show first 500 chars
    
    system_prompt = """You are a financial data expert. Your job is to identify which CSV files and columns contain the data the user is requesting.

Given a catalog of available CSV files with their columns/rows, identify:
1. Which file contains each piece of requested data
2. Which column or row name contains it (even if name is abbreviated/different)
3. The data type (stock price, revenue, margin, etc.)
4. What transformations the user EXPLICITLY requested (if any)

IMPORTANT: Only suggest transformations if the user explicitly asked for them:
- User says "plot stock price" → transformations: []
- User says "normalize to 1" → transformations: ["normalize_to_1"]
- User says "YoY growth" → transformations: ["yoy_growth"]

For example:
- "revenue" could be in column "Total Revenue", "totalRevenue", or "RevTotal"
- "stock price" could be in column "Close", "Adj Close", or "close"
- "gross margin" could be "Gross Profit" / "Total Revenue"

IMPORTANT: Use the EXACT filenames from the catalog provided, not generic names!

Return a JSON object with this structure:
{
  "data_items": [
    {
      "requested": "NVDA revenue",
      "ticker": "NVDA",
      "source": "yahoo_finance",
      "filename": "income_stmt_quarterly.csv",  // EXACT filename from catalog
      "path": "Data/NVDA/yahoo_finance/csv/income_stmt_quarterly.csv",  // Full path from catalog
      "column_or_row": "Total Revenue",
      "data_type": "revenue",
      "format": "rows_are_metrics"
    },
    {
      "requested": "NVDA stock price",
      "ticker": "NVDA",
      "source": "alpha_vantage",
      "filename": "TIME_SERIES_DAILY__full.csv",  // Use ACTUAL filename from catalog
      "path": "Data/NVDA/alpha_vantage/csv/TIME_SERIES_DAILY__full.csv",  // Full path
      "column_or_row": "close",
      "data_type": "stock_price",
      "format": "columns_are_metrics"
    }
  ],
  "transformations": ["normalize_to_1"],
  "chart_type": "line"
}

CRITICAL: The "path" field MUST match exactly what's in the catalog metadata!

Be smart about matching:
- "Rev" → "Total Revenue"
- "EPS" → "Basic EPS" or "Diluted EPS"
- "Margin" → Calculate from revenue and profit
- "P/E" → Calculate from price and EPS
"""

    user_message = f"""User request: "{user_prompt}"

Available data catalog:
{catalog_json}

Identify which files and columns/rows contain the requested data."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )
    
    return json.loads(response.choices[0].message.content)


def generate_chart_code(user_prompt: str, data_sources: Dict, tickers: List[str]) -> str:
    """
    Generate Python code to create the chart based on identified data sources
    
    Args:
        user_prompt: Original user request
        data_sources: Output from identify_data_sources()
        tickers: List of tickers
    
    Returns:
        Python code string to execute
    """
    system_prompt = """You are a Python code generator for financial charting.

IMPORTANT: You will be given the EXACT file paths to use from the data_sources.
Use these paths EXACTLY as provided - they are real files that exist on the system.

Given identified data sources, generate Python code to:
1. Load the data from CSV files
2. Apply ONLY the transformations explicitly requested by the user
3. Create Plotly chart that matches EXACTLY what the user asked for

CRITICAL: Do NOT add transformations the user didn't ask for!
- If user says "plot stock price" → plot raw stock price (no normalization)
- If user says "normalize to 1" → THEN use normalize_to_1()
- If user says "YoY growth" → THEN use yoy_growth()
- Default: plot raw data unless transformation is explicitly requested

Available helper functions (ONLY use if requested):
- normalize_to_1(series) - Use ONLY if user asks to "normalize" or "start at 1"
- yoy_growth(series) - Use ONLY if user asks for "year-over-year" or "YoY"
- qoq_growth(series) - Use ONLY if user asks for "quarter-over-quarter" or "QoQ"
- rolling_sum(series, periods) - Use ONLY if user asks for "trailing" or "rolling sum"
- moving_average(series, days) - Use ONLY if user asks for "moving average" or "MA"

Check the "transformations" field in data_sources - if empty, plot raw data!

Data loading patterns:

CRITICAL: All CSV files are in a /csv/ subfolder!

IMPORTANT (Timezone safety): Yahoo Finance timestamps can contain mixed timezones. Always parse dates using to_utc_naive_datetime(...) (preferred) or pd.to_datetime(..., utc=True), never bare pd.to_datetime(...).

- Alpha Vantage: Data/{TICKER}/alpha_vantage/csv/FILENAME.csv
- Yahoo Finance: Data/{TICKER}/yahoo_finance/csv/FILENAME.csv

For rows_are_metrics format (Yahoo Finance income/balance):
```python
# MUST include /csv/ in path!
df = pd.read_csv('Data/NVDA/yahoo_finance/csv/income_stmt_quarterly.csv', index_col=0)
revenue = df.loc['Total Revenue']  # Get row
revenue.index = to_utc_naive_datetime(revenue.index)
revenue = pd.to_numeric(revenue, errors='coerce')
```

For columns_are_metrics format (Alpha Vantage, Yahoo prices):
```python
# Load data - MUST include /csv/ in path!
df = pd.read_csv('Data/NVDA/alpha_vantage/csv/TIME_SERIES_DAILY__full.csv')

# Find date column (handles: date, Date, timestamp, fiscalDateEnding, etc.)
date_candidates = [c for c in df.columns if any(x in c.lower() for x in ['date', 'time'])]
date_col = date_candidates[0] if date_candidates else df.columns[0]
print(f"DEBUG: Selected date column: '{date_col}' from {df.columns.tolist()}")

# Find price column (handles: close, adjusted_close, adj close, 4. close, etc.)
# IMPORTANT: Check user's request for "adjusted" or "adj" keywords
price_candidates = []

# If user wants adjusted close
if 'adjusted' in user_prompt.lower() or 'adj' in user_prompt.lower():
    # Look for adjusted close (with space, underscore, or combined)
    price_candidates = [c for c in df.columns if 'adj' in c.lower() and 'close' in c.lower().replace('_', ' ')]
    if not price_candidates:
        print("DEBUG: User requested adjusted close but column not found, trying alternatives")

# If no adjusted close found or user didn't request it, try regular close
if not price_candidates:
    price_candidates = [c for c in df.columns if 'close' in c.lower() and 'adj' not in c.lower()]

# Fallback to any close column
if not price_candidates:
    price_candidates = [c for c in df.columns if 'close' in c.lower()]

# Last resort fallback
if not price_candidates:
    for col_name in ['Close', 'close', 'CLOSE', 'price', 'Price']:
        if col_name in df.columns:
            price_candidates = [col_name]
            break

price_col = price_candidates[0] if price_candidates else 'close'
print(f"DEBUG: Selected price column: '{price_col}' from {df.columns.tolist()}")

# Create clean DataFrame for plotting
plot_df = pd.DataFrame({
    'Date': to_utc_naive_datetime(df[date_col]),
    'Price': pd.to_numeric(df[price_col], errors='coerce')
})

# Clean and sort
plot_df = plot_df.dropna().sort_values('Date')

# CRITICAL: Use DataFrame COLUMNS for x and y, NOT index!
# x=plot_df['Date'] gets the Date column
# y=plot_df['Price'] gets the Price column
```




FILE PATH STRUCTURE - EXTREMELY IMPORTANT:
All CSV files are in a /csv/ subfolder!

Correct paths:
✅ Data/NVDA/alpha_vantage/csv/TIME_SERIES_DAILY__full.csv
✅ Data/NVDA/yahoo_finance/csv/historical_data.csv
✅ Data/NVDA/yahoo_finance/csv/income_stmt_quarterly.csv

Wrong paths (NEVER use these):
❌ Data/NVDA/alpha_vantage/TIME_SERIES_DAILY.csv (missing /csv/)
❌ Data/NVDA/yahoo_finance/income_stmt_quarterly.csv (missing /csv/)

ALWAYS include the /csv/ subfolder in all file paths!

IMPORTANT: CSV files have these column name variations:

Yahoo Finance:
- Date, Open, High, Low, Close, Adj Close, Volume
- OR: timestamp, open, high, low, close, adjusted_close, volume (with underscores!)

Alpha Vantage TIME_SERIES_DAILY:
- timestamp, open, high, low, close, adjusted_close, volume, dividend_amount
- All lowercase with underscores

Alpha Vantage (older format):
- 1. open, 2. high, 3. low, 4. close, 5. volume

CRITICAL: 
- "Adj Close" and "adjusted_close" are THE SAME (just different naming)
- Always use .lower() and replace('_', ' ') when searching column names
- If user says "adjusted", look for columns containing both "adj" AND "close" (case-insensitive)

Always check column names and use flexible matching:
```python
# Find date column
date_cols = [col for col in df.columns if 'date' in col.lower()]
date_col = date_cols[0] if date_cols else df.columns[0]

# Find close price column (flexible for 'close', 'Close', '4. close', etc.)
close_cols = [col for col in df.columns if 'close' in col.lower() and 'adj' not in col.lower()]
if not close_cols:  # Fallback to any column with 'close'
    close_cols = [col for col in df.columns if 'close' in col.lower()]
close_col = close_cols[0] if close_cols else 'Close'

# ALWAYS convert to numeric and handle errors
prices = pd.to_numeric(df[close_col], errors='coerce')
```


FILE PATH RULES (CRITICAL):
1. ALL CSV files are inside a /csv/ subfolder
2. Alpha Vantage path: Data/{TICKER}/alpha_vantage/csv/{FILENAME}.csv
3. Yahoo Finance path: Data/{TICKER}/yahoo_finance/csv/{FILENAME}.csv
4. NEVER use paths like Data/NVDA/alpha_vantage/FILENAME.csv (missing /csv/)
5. Common Alpha Vantage files: TIME_SERIES_DAILY__full.csv, INCOME_STATEMENT__quarterlyReports.csv
6. Common Yahoo Finance files: historical_data.csv, income_stmt_quarterly.csv

CRITICAL RULES:
1. Store final figure in variable 'fig'
2. NO markdown formatting
3. MANDATORY: Always include fig.update_traces(hovertemplate='%{x}<br>%{y} <extra></extra>') after creating all traces
4. MANDATORY: Apply default styling (width=1500, height=650, white bg, Calibri 16px, legend on top, axis borders, color palette)
5. MANDATORY: Apply color palette to traces: colors[0]=#0D3512, colors[1]=#003868, etc.
6. NEVER include fig.show() - the figure is returned automatically
7. End code after applying colors - do NOT add fig.show() or any display commands
3. Handle missing data gracefully (dropna, fillna)
4. Use go.Figure() for dual-axis charts
5. Always include title, axis labels, legend
6. Use hovermode='x unified'
7. Return ONLY executable Python code


MANDATORY: Use DataFrame with explicit columns, NOT Series with index!

Example for stock price chart:
```python
# Load data
df = pd.read_csv('Data/NVDA/alpha_vantage/csv/TIME_SERIES_DAILY__full.csv')

# Find columns
date_col = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()][0]
price_col = [c for c in df.columns if 'close' in c.lower() and 'adj' not in c.lower()][0]

# Create plotting DataFrame
plot_df = pd.DataFrame({
    'Date': to_utc_naive_datetime(df[date_col]),
    'Price': pd.to_numeric(df[price_col], errors='coerce')
}).dropna().sort_values('Date')

# DEBUG: Print what we're actually plotting
print(f"DEBUG: Plotting {len(plot_df)} data points")
print(f"DEBUG: Date column: {date_col}")
print(f"DEBUG: Price column: {price_col}")
print(f"DEBUG: Price range: ${plot_df['Price'].min():.2f} to ${plot_df['Price'].max():.2f}")
print(f"DEBUG: First 5 prices: {plot_df['Price'].head().tolist()}")
print(f"DEBUG: Last 5 prices: {plot_df['Price'].tail().tolist()}")

# Create chart using DataFrame COLUMNS
fig = go.Figure()

# CRITICAL: Convert to lists to ensure Plotly gets raw values
fig.add_trace(go.Scatter(
    x=plot_df['Date'].tolist(),     # Convert to list
    y=plot_df['Price'].tolist(),    # Convert to list  
    mode='lines',
    name='NVDA'
))
fig.update_layout(title='NVDA Stock Price', xaxis_title='Date', yaxis_title='Price ($)')
```

Example for dual-axis chart:
fig = go.Figure()
fig.add_trace(go.Scatter(x=revenue.index, y=revenue.values, name='Revenue', yaxis='y'))
fig.add_trace(go.Scatter(x=prices.index, y=prices.values, name='Stock Price', yaxis='y2'))

fig.update_layout(
    title='NVDA Revenue vs Stock Price',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Revenue ($)', side='left'),
    yaxis2=dict(title='Stock Price ($)', side='right', overlaying='y'),
    hovermode='x unified'
)
```
"""

    user_message = f"""User request: "{user_prompt}"

Identified data sources (USE EXACT PATHS PROVIDED):
{json.dumps(data_sources, indent=2)}

IMPORTANT: 
- Use the "path" field from each data_item EXACTLY as shown
- These paths point to REAL files that exist on the system
- Do NOT modify the paths or filenames
- Check the "transformations" field - if empty [], plot raw data without transformations
- Only apply transformations that are explicitly listed in the transformations array
- Apply default styling (1500x650, white bg, Calibri 16px, legend on top, color palette) UNLESS user requests different styling
- Add default hovertemplate UNLESS user specifically requested a custom tooltip format

Generate Python code to create this chart."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.1
    )
    
    code = response.choices[0].message.content.strip()
    
    # Remove markdown code blocks if present
    if code.startswith('```'):
        code = code.split('```')[1]
        if code.startswith('python'):
            code = code[6:]
        code = code.strip()
    
    # CRITICAL: Remove ALL import statements (pd and go are already imported)
    lines = code.split('\n')
    filtered_lines = []
    for line in lines:
        # Skip any line that starts with 'import' or 'from ... import'
        stripped = line.strip()
        if not stripped.startswith('import ') and not stripped.startswith('from '):
            # Also skip fig.show() - it opens a new window instead of returning JSON
            if 'fig.show()' not in stripped:
                filtered_lines.append(line)
            else:
                print(f"[CODE GEN] Removed fig.show() - chart is returned via JSON")
        else:
            print(f"[CODE GEN] Removed import: {stripped}")
    
    code = '\n'.join(filtered_lines)
    
    # FORCE FIX: Replace problematic plotting patterns
    print("[CODE GEN] Applying forced fixes...")
    
    # Fix 1: Replace .index with .values in y-axis
    if 'y=prices.index' in code or 'y = prices.index' in code:
        print("[CODE GEN] Fixed: Replaced y=prices.index with y=prices.values")
        code = code.replace('y=prices.index', 'y=prices.values')
        code = code.replace('y = prices.index', 'y = prices.values')
    
    # Fix 2: If using set_index pattern, force DataFrame approach
    if '.set_index(' in code and 'go.Scatter' in code:
        print("[CODE GEN] WARNING: Code uses set_index pattern - forcing DataFrame fix")
        # Add a conversion before plotting
        fix_code = """
# FORCED FIX: Convert to DataFrame for proper plotting
if isinstance(prices, pd.Series):
    plot_df = pd.DataFrame({'Date': prices.index, 'Price': prices.values})
    prices = plot_df
"""
        # Insert before fig = go.Figure()
        code = code.replace('fig = go.Figure()', fix_code + '\nfig = go.Figure()')
        # Also fix the Scatter call
        code = code.replace('x=prices.index, y=prices.values', "x=prices['Date'], y=prices['Price']")
        code = code.replace('x=prices.index, y=prices', "x=prices['Date'], y=prices['Price']")
    
    return code


def execute_chart_code(code: str, user_prompt: str = '') -> Dict[str, Any]:
    """
    Execute generated code safely and return Plotly JSON
    
    Args:
        code: Python code string
    
    Returns:
        dict with 'success', 'data' (Plotly JSON), or 'error'
    """
    import traceback
    
    # Store user prompt for conditional styling
    
    # Define helper functions
    def normalize_to_1(series):
        series = series.dropna()
        if len(series) == 0 or series.iloc[0] == 0:
            return series
        return series / series.iloc[0]
    
    def yoy_growth(series):
        periods = 4 if len(series) > 8 else 1
        return (series / series.shift(periods)) - 1
    
    def qoq_growth(series):
        return (series / series.shift(1)) - 1
    
    def rolling_sum(series, periods):
        return series.rolling(window=periods, min_periods=1).sum()
    
    def moving_average(series, days):
        ma = series.rolling(window=days, min_periods=1).mean()
        print(f"DEBUG MA: Input length={len(series)}, Output length={len(ma)}, Days={days}")
        print(f"DEBUG MA: First 5 values: {ma.head().tolist()}")
        print(f"DEBUG MA: Last 5 values: {ma.tail().tolist()}")
        return ma
    
    # Helper function to find date column flexibly
    def find_date_column(df):
        """Find the date column regardless of capitalization"""
        for col in df.columns:
            if col.lower() in ['date', 'timestamp', 'fiscaldateending']:
                return col
        return None
    def to_utc_naive_datetime(x):
        """
        Convert a Series/Index/list of date-like values to datetime64[ns] (tz-naive),
        robust to:
          - tz-naive inputs (Alpha Vantage often)
          - tz-aware inputs (Yahoo often)
          - mixed/invalid strings (coerced to NaT)

        Strategy:
          1) Parse with pd.to_datetime(errors='coerce') (no forced utc here)
          2) If tz-aware -> convert to UTC and strip tz
          3) If tz-naive -> return as-is
        """
        from pandas.api.types import is_datetime64tz_dtype

        dt = pd.to_datetime(x, errors="coerce")

        # Series case
        if isinstance(dt, pd.Series):
            try:
                if is_datetime64tz_dtype(dt):
                    return dt.dt.tz_convert("UTC").dt.tz_localize(None)
            except Exception:
                pass
            # tz-naive Series[datetime64[ns]]
            return dt

        # DatetimeIndex / Index case
        try:
            if is_datetime64tz_dtype(dt):
                return dt.tz_convert("UTC").tz_localize(None)
        except Exception:
            pass

        return dt


    # Create namespace with allowed functions
    namespace = {
        'pd': pd,
        'np': np,
        'go': __import__('plotly.graph_objects', fromlist=['go']),
        'Path': Path,
        'normalize_to_1': normalize_to_1,
        'yoy_growth': yoy_growth,
        'qoq_growth': qoq_growth,
        'rolling_sum': rolling_sum,
        'moving_average': moving_average,
        'find_date_column': find_date_column,
        'to_utc_naive_datetime': to_utc_naive_datetime,
        'user_prompt': user_prompt,  # Pass user prompt for conditional styling
    }
    
    try:
        # Preprocess generated code: prevent it from redefining helpers that we provide
        def _strip_conflicting_helpers(src: str) -> str:
            # Remove any function definition named to_utc_naive_datetime from generated code
            # so the executor-provided version is used (avoids tz_convert on tz-naive data).
            pattern = r"(?m)^def\s+to_utc_naive_datetime\s*\(.*?\):\n(?:^[ \t].*\n)+"
            return re.sub(pattern, "", src)

        code = _strip_conflicting_helpers(code)

        # Execute code (defensive timezone handling for Yahoo Finance mixed tz data)
        _orig_to_datetime = pd.to_datetime

        def _make_utc_naive(dt_obj):
            # Convert tz-aware -> UTC naive; leave naive as-is
            try:
                if getattr(dt_obj, "tz", None) is not None:
                    return dt_obj.tz_convert("UTC").tz_localize(None)
            except Exception:
                pass
            # Series case
            try:
                if hasattr(dt_obj, "dt") and getattr(dt_obj.dt, "tz", None) is not None:
                    return dt_obj.dt.tz_convert("UTC").dt.tz_localize(None)
            except Exception:
                pass
            return dt_obj

        def _safe_to_datetime(*args, **kwargs):
            kwargs.setdefault("utc", True)
            kwargs.setdefault("errors", "coerce")
            return _make_utc_naive(_orig_to_datetime(*args, **kwargs))

        pd.to_datetime = _safe_to_datetime
        try:
            exec(code, namespace)
        finally:
            pd.to_datetime = _orig_to_datetime
        
        # Get figure
        if 'fig' in namespace:
            fig = namespace['fig']
            user_prompt = namespace.get('user_prompt', '')
            
            # CONDITIONAL DEFAULT STYLING (apply only if user didn't specify otherwise)
            print("[STYLING] Applying conditional default styling...")
            
            # Always force hovertemplate (unless user wants custom format)
            if 'hovertemplate' not in user_prompt.lower() and 'tooltip' not in user_prompt.lower():
                fig.update_traces(hovertemplate='%{x}<br>%{y} <extra></extra>')
                print("[STYLING] ✓ Applied default hovertemplate")
            
            # Apply axis borders ONLY if user didn't say "remove borders" or "no borders"
            if 'remove' not in user_prompt.lower() or 'border' not in user_prompt.lower():
                if 'no border' not in user_prompt.lower() and 'without border' not in user_prompt.lower():
                    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=False)
                    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=False)
                    print("[STYLING] ✓ Applied axis borders")
                else:
                    print("[STYLING] ⊘ Skipped borders (user requested removal)")
            else:
                print("[STYLING] ⊘ Skipped borders (user requested removal)")
            
            # Apply color palette ONLY if user didn't specify colors
            if 'color' not in user_prompt.lower() and 'red' not in user_prompt.lower() and 'blue' not in user_prompt.lower():
                colors = ["#0D3512","#003868","#545919","#13501B","#404040","#B1B395","#156082","#000000","#10857B"]
                for i, trace in enumerate(fig.data):
                    color = colors[i % len(colors)]
                    trace.marker.color = color
                    if hasattr(trace, 'line'):
                        trace.line.color = color
                print("[STYLING] ✓ Applied color palette")
            else:
                print("[STYLING] ⊘ Skipped colors (user specified custom colors)")
            
            # Always apply layout defaults (unless specified otherwise)
            layout_updates = {}
            
            if 'width' not in user_prompt.lower():
                layout_updates['width'] = 1500
            if 'height' not in user_prompt.lower():
                layout_updates['height'] = 650
            
            layout_updates.update({
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'font': dict(family='Calibri', size=16, color='black'),
                'legend': dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                )
            })
            
            fig.update_layout(**layout_updates)
            print("[STYLING] ✓ Applied layout defaults")
            
            plotly_json = fig.to_json()
            return {
                'success': True,
                'data': json.loads(plotly_json)
            }
        else:
            return {
                'success': False,
                'error': 'Code must define "fig" variable with Plotly figure'
            }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def extract_tickers_from_prompt(prompt: str) -> List[str]:
    """
    Use AI to extract ticker symbols from user prompt
    
    Args:
        prompt: User's natural language request
    
    Returns:
        List of ticker symbols
    """
    system_prompt = """Extract all stock ticker symbols mentioned in the user's request.

Return a JSON object with a list of tickers:
{"tickers": ["NVDA", "GOOGL", "AAPL"]}

Recognize these company name to ticker mappings:

US COMPANIES:
- "NVIDIA" → "NVDA"
- "Microsoft" → "MSFT"
- "Apple" → "AAPL"
- "Google" or "Alphabet" → "GOOGL"
- "Amazon" → "AMZN"
- "Meta" or "Facebook" → "META"
- "Broadcom" → "AVGO"
- "Tesla" → "TSLA"
- "Berkshire Hathaway" or "Berkshire" → "BRK.B"
- "JPMorgan" or "JPMorgan Chase" → "JPM"

BRAZILIAN COMPANIES:
- "Nu Holdings" or "Nu" or "Nubank" → "NU"
- "Petrobras" → "PETR4"
- "Itaú" or "Itau Unibanco" or "Itaú Unibanco" → "ITUB4"
- "Vale" → "VALE3"
- "BTG Pactual" or "BTG" → "BPAC11"
- "Santander Brasil" or "Banco Santander Brasil" → "SANB11"
- "Ambev" → "ABEV3"
- "Bradesco" or "Banco Bradesco" → "BBDC4"
- "WEG" → "WEGE3"
- "Klabin" → "KLBN11"

If ticker is already provided (like "NVDA", "BPAC11"), keep it as-is.
If company name is mentioned, convert to ticker using the mappings above."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract tickers from: {prompt}"}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )
    
    result = json.loads(response.choices[0].message.content)
    return result.get('tickers', [])


def create_chart_from_prompt(user_prompt: str) -> Dict[str, Any]:
    """
    Main orchestration function - creates chart from natural language prompt
    
    Args:
        user_prompt: User's chart request
    
    Returns:
        dict with Plotly JSON or error
    
    Example:
        >>> result = create_chart_from_prompt("Plot NVDA revenue vs stock price")
        >>> if result['success']:
        >>>     plotly_json = result['data']
    """
    try:
        print(f"\n[CHART AGENT] User prompt: {user_prompt}")
        
        # Step 1: Extract tickers
        print("[CHART AGENT] Step 1: Extracting tickers...")
        tickers = extract_tickers_from_prompt(user_prompt)
        print(f"[CHART AGENT] Found tickers: {tickers}")
        
        if not tickers:
            return {
                'success': False,
                'error': 'No ticker symbols found in prompt. Please mention at least one company (e.g., NVDA, GOOGL, AAPL)'
            }
        
        # Step 2: Identify data sources
        print("[CHART AGENT] Step 2: Scanning CSV files and identifying data sources...")
        data_sources = identify_data_sources(user_prompt, tickers)
        print(f"[CHART AGENT] Identified {len(data_sources.get('data_items', []))} data items")
        
        # Step 3: Generate code
        print("[CHART AGENT] Step 3: Generating Python code...")
        code = generate_chart_code(user_prompt, data_sources, tickers)
        print(f"[CHART AGENT] Generated {len(code)} characters of code")
        print(f"\n--- Generated Code ---\n{code}\n--- End Code ---\n")
        
        # Step 4: Execute code
        print("[CHART AGENT] Step 4: Executing code...")
        result = execute_chart_code(code, user_prompt)
        
        if result['success']:
            print("[CHART AGENT] ✅ Chart created successfully!")
            # Add the code to result for debugging
            result['code'] = code
            result['data_sources'] = data_sources
        else:
            print(f"[CHART AGENT] ❌ Execution failed: {result['error']}")
        
        return result
        
    except Exception as e:
        import traceback
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }