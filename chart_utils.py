# chart_utils.py
"""
Utility functions for loading financial data and performing transformations
Used by AI-generated code in the charting tool

Data Structure:
Data/
  ├── NVDA/
  │   ├── alpha_vantage/csv/
  │   │   ├── INCOME_STATEMENT__quarterlyReports.csv
  │   │   ├── BALANCE_SHEET__quarterlyReports.csv
  │   │   ├── CASH_FLOW__quarterlyReports.csv
  │   │   └── TIME_SERIES_DAILY__full.csv (or similar)
  │   └── yahoo_finance/csv/
  │       ├── income_stmt_quarterly.csv
  │       ├── balance_sheet_quarterly.csv
  │       ├── cash_flow_quarterly.csv
  │       └── historical_data.csv (or price data file)
  └── GOOGL/
      └── ... (same structure)
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import os

def list_available_files(ticker: str) -> dict:
    """
    List all CSV files available for a ticker
    Useful for debugging
    
    Returns:
        dict with 'alpha_vantage' and 'yahoo_finance' lists of available files
    """
    ticker = ticker.upper()
    ticker_folder = Path("Data") / ticker
    
    result = {
        'alpha_vantage': [],
        'yahoo_finance': []
    }
    
    av_path = ticker_folder / "alpha_vantage" / "csv"
    if av_path.exists():
        result['alpha_vantage'] = [f.name for f in av_path.glob("*.csv")]
    
    yf_path = ticker_folder / "yahoo_finance" / "csv"
    if yf_path.exists():
        result['yahoo_finance'] = [f.name for f in yf_path.glob("*.csv")]
    
    return result


def load_stock_prices(ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.Series:
    """
    Load daily stock prices (close prices) from Alpha Vantage or Yahoo Finance
    
    Expected files:
    - Alpha Vantage: TIME_SERIES_DAILY__full.csv (or any file with "TIME_SERIES_DAILY" in name)
        Columns: date, open, high, low, close, volume
    - Yahoo Finance: Any CSV with historical prices (e.g., historical_data.csv, prices.csv)
        Columns: Date, Open, High, Low, Close, Adj Close, Volume
    
    Args:
        ticker: Company ticker symbol (e.g., 'NVDA', 'GOOGL', 'ITUB4')
        start: Start date in 'YYYY-MM-DD' format (optional)
        end: End date in 'YYYY-MM-DD' format (optional)
    
    Returns:
        pandas Series with datetime index and close prices as values
    
    Example:
        >>> prices = load_stock_prices('NVDA', start='2020-01-01')
        >>> print(prices.head())
    """
    ticker = ticker.upper()
    ticker_folder = Path("Data") / ticker
    
    # Try Alpha Vantage first - look for any TIME_SERIES_DAILY file
    av_path = ticker_folder / "alpha_vantage" / "csv"
    if av_path.exists():
        # Find any file with TIME_SERIES_DAILY in the name
        daily_files = list(av_path.glob("*TIME_SERIES_DAILY*.csv"))
        if daily_files:
            df = pd.read_csv(daily_files[0])
            
            # Alpha Vantage uses 'date' column
            date_col = 'date' if 'date' in df.columns else 'timestamp'
            close_col = 'close' if 'close' in df.columns else 'Close'
            
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
            prices = pd.to_numeric(df[close_col], errors='coerce').sort_index()
            
            if start:
                prices = prices[prices.index >= pd.to_datetime(start)]
            if end:
                prices = prices[prices.index <= pd.to_datetime(end)]
            
            return prices
    
    # Try Yahoo Finance - look for any historical data file
    yf_path = ticker_folder / "yahoo_finance" / "csv"
    if yf_path.exists():
        # Common Yahoo Finance file names
        possible_names = [
            'historical_data.csv',
            'historical_data_full.csv',
            'prices.csv',
            'stock_prices.csv',
        ]
        
        for filename in possible_names:
            file_path = yf_path / filename
            if file_path.exists():
                df = pd.read_csv(file_path)
                
                # Yahoo Finance uses 'Date' column
                date_col = 'Date' if 'Date' in df.columns else 'date'
                # Try different close column names
                close_col = None
                for col in ['Close', 'Adj Close', 'close', 'adj_close']:
                    if col in df.columns:
                        close_col = col
                        break
                
                if not close_col:
                    raise ValueError(f"Could not find close price column in {filename}")
                
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
                prices = pd.to_numeric(df[close_col], errors='coerce').sort_index()
                
                if start:
                    prices = prices[prices.index >= pd.to_datetime(start)]
                if end:
                    prices = prices[prices.index <= pd.to_datetime(end)]
                
                return prices
        
        # If no standard names found, try the first CSV in the directory
        csv_files = list(yf_path.glob("*.csv"))
        if csv_files:
            # Filter out income/balance/cash flow files
            price_files = [f for f in csv_files if not any(x in f.name.lower() for x in ['income', 'balance', 'cash_flow', 'stmt'])]
            if price_files:
                df = pd.read_csv(price_files[0])
                # Try to find date and close columns
                date_col = None
                for col in ['Date', 'date', 'timestamp', 'Timestamp']:
                    if col in df.columns:
                        date_col = col
                        break
                
                close_col = None
                for col in ['Close', 'Adj Close', 'close', 'adj_close', 'price', 'Price']:
                    if col in df.columns:
                        close_col = col
                        break
                
                if date_col and close_col:
                    df[date_col] = pd.to_datetime(df[date_col])
                    df = df.set_index(date_col)
                    prices = pd.to_numeric(df[close_col], errors='coerce').sort_index()
                    
                    if start:
                        prices = prices[prices.index >= pd.to_datetime(start)]
                    if end:
                        prices = prices[prices.index <= pd.to_datetime(end)]
                    
                    return prices
    
    # If we get here, no data was found
    available = list_available_files(ticker)
    raise FileNotFoundError(
        f"No stock price data found for {ticker}.\n"
        f"Available Alpha Vantage files: {available['alpha_vantage']}\n"
        f"Available Yahoo Finance files: {available['yahoo_finance']}\n"
        f"Make sure you have a TIME_SERIES_DAILY file (Alpha Vantage) or historical_data.csv (Yahoo Finance)"
    )


def load_income_metric(ticker: str, metric: str, quarterly: bool = True) -> pd.Series:
    """
    Load income statement metric from quarterly or annual reports
    
    Expected files:
    - Alpha Vantage: INCOME_STATEMENT__quarterlyReports.csv or INCOME_STATEMENT__annualReports.csv
        Columns: fiscalDateEnding, totalRevenue, netIncome, grossProfit, operatingIncome, ebitda, etc.
    - Yahoo Finance: income_stmt_quarterly.csv or income_stmt_annual.csv
        First column: metric names (rows), other columns: dates (quarters/years)
    
    Args:
        ticker: Company ticker symbol
        metric: Metric name
            Alpha Vantage: 'totalRevenue', 'netIncome', 'grossProfit', 'operatingIncome', 'ebitda'
            Yahoo Finance: 'Total Revenue', 'Net Income', 'Gross Profit', 'Operating Income', 'EBITDA'
        quarterly: True for quarterly data, False for annual
    
    Returns:
        pandas Series with datetime index and metric values
    
    Example:
        >>> revenue = load_income_metric('NVDA', 'totalRevenue', quarterly=True)
        >>> # Or for Yahoo Finance:
        >>> revenue = load_income_metric('NVDA', 'Total Revenue', quarterly=True)
    """
    ticker = ticker.upper()
    ticker_folder = Path("Data") / ticker
    
    # Try Alpha Vantage
    filename = "INCOME_STATEMENT__quarterlyReports.csv" if quarterly else "INCOME_STATEMENT__annualReports.csv"
    av_path = ticker_folder / "alpha_vantage" / "csv" / filename
    
    if av_path.exists():
        df = pd.read_csv(av_path)
        df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
        df = df.set_index('fiscalDateEnding').sort_index()
        
        if metric not in df.columns:
            available_metrics = [col for col in df.columns if col not in ['fiscalDateEnding', 'reportedCurrency']]
            raise KeyError(
                f"Metric '{metric}' not found in Alpha Vantage income statement.\n"
                f"Available metrics: {available_metrics[:10]}..."
            )
        
        return pd.to_numeric(df[metric], errors='coerce')
    
    # Try Yahoo Finance
    filename = "income_stmt_quarterly.csv" if quarterly else "income_stmt_annual.csv"
    yf_path = ticker_folder / "yahoo_finance" / "csv" / filename
    
    if yf_path.exists():
        df = pd.read_csv(yf_path, index_col=0)
        # Yahoo Finance has metrics as rows, dates as columns
        
        if metric not in df.index:
            # Try common variations
            metric_variations = {
                'revenue': 'Total Revenue',
                'totalRevenue': 'Total Revenue',
                'netIncome': 'Net Income',
                'grossProfit': 'Gross Profit',
                'operatingIncome': 'Operating Income',
                'ebitda': 'EBITDA',
            }
            
            if metric in metric_variations:
                metric = metric_variations[metric]
            
            if metric not in df.index:
                available_metrics = df.index.tolist()
                raise KeyError(
                    f"Metric '{metric}' not found in Yahoo Finance income statement.\n"
                    f"Available metrics: {available_metrics[:10]}..."
                )
        
        series = df.loc[metric]
        # Filter out non-date columns like 'TTM'
        date_cols = [col for col in series.index if col not in ['TTM', 'Breakdown']]
        series = series[date_cols]
        series.index = pd.to_datetime(series.index)
        return pd.to_numeric(series, errors='coerce').sort_index()
    
    available = list_available_files(ticker)
    raise FileNotFoundError(
        f"No income statement data found for {ticker}.\n"
        f"Available files: {available}"
    )


def load_balance_metric(ticker: str, metric: str, quarterly: bool = True) -> pd.Series:
    """
    Load balance sheet metric from quarterly or annual reports
    
    Expected files:
    - Alpha Vantage: BALANCE_SHEET__quarterlyReports.csv or BALANCE_SHEET__annualReports.csv
        Columns: fiscalDateEnding, totalAssets, totalShareholderEquity, totalLiabilities, etc.
    - Yahoo Finance: balance_sheet_quarterly.csv or balance_sheet_annual.csv
        First column: metric names (rows), other columns: dates
    
    Args:
        ticker: Company ticker symbol
        metric: Metric name
            Alpha Vantage: 'totalAssets', 'totalShareholderEquity', 'totalLiabilities'
            Yahoo Finance: 'Total Assets', 'Total Equity Gross Minority Interest', 'Total Liabilities Net Minority Interest'
        quarterly: True for quarterly data, False for annual
    
    Returns:
        pandas Series with datetime index and metric values
    """
    ticker = ticker.upper()
    ticker_folder = Path("Data") / ticker
    
    # Try Alpha Vantage
    filename = "BALANCE_SHEET__quarterlyReports.csv" if quarterly else "BALANCE_SHEET__annualReports.csv"
    av_path = ticker_folder / "alpha_vantage" / "csv" / filename
    
    if av_path.exists():
        df = pd.read_csv(av_path)
        df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
        df = df.set_index('fiscalDateEnding').sort_index()
        
        if metric not in df.columns:
            available_metrics = [col for col in df.columns if col not in ['fiscalDateEnding', 'reportedCurrency']]
            raise KeyError(f"Metric '{metric}' not found. Available: {available_metrics[:10]}...")
        
        return pd.to_numeric(df[metric], errors='coerce')
    
    # Try Yahoo Finance
    filename = "balance_sheet_quarterly.csv" if quarterly else "balance_sheet_annual.csv"
    yf_path = ticker_folder / "yahoo_finance" / "csv" / filename
    
    if yf_path.exists():
        df = pd.read_csv(yf_path, index_col=0)
        
        if metric not in df.index:
            # Try common variations
            metric_variations = {
                'totalAssets': 'Total Assets',
                'totalShareholderEquity': 'Total Equity Gross Minority Interest',
                'totalLiabilities': 'Total Liabilities Net Minority Interest',
            }
            
            if metric in metric_variations:
                metric = metric_variations[metric]
            
            if metric not in df.index:
                available_metrics = df.index.tolist()
                raise KeyError(f"Metric '{metric}' not found. Available: {available_metrics[:10]}...")
        
        series = df.loc[metric]
        date_cols = [col for col in series.index if col not in ['TTM', 'Breakdown']]
        series = series[date_cols]
        series.index = pd.to_datetime(series.index)
        return pd.to_numeric(series, errors='coerce').sort_index()
    
    available = list_available_files(ticker)
    raise FileNotFoundError(f"No balance sheet data found for {ticker}. Available: {available}")


def load_cash_flow_metric(ticker: str, metric: str, quarterly: bool = True) -> pd.Series:
    """
    Load cash flow statement metric
    
    Expected files:
    - Alpha Vantage: CASH_FLOW__quarterlyReports.csv or CASH_FLOW__annualReports.csv
    - Yahoo Finance: cash_flow_quarterly.csv or cash_flow_annual.csv
    
    Common metrics:
    - Alpha Vantage: 'operatingCashflow', 'capitalExpenditures', 'freeCashFlow'
    - Yahoo Finance: 'Operating Cash Flow', 'Capital Expenditure', 'Free Cash Flow'
    """
    ticker = ticker.upper()
    ticker_folder = Path("Data") / ticker
    
    # Try Alpha Vantage
    filename = "CASH_FLOW__quarterlyReports.csv" if quarterly else "CASH_FLOW__annualReports.csv"
    av_path = ticker_folder / "alpha_vantage" / "csv" / filename
    
    if av_path.exists():
        df = pd.read_csv(av_path)
        df['fiscalDateEnding'] = pd.to_datetime(df['fiscalDateEnding'])
        df = df.set_index('fiscalDateEnding').sort_index()
        return pd.to_numeric(df[metric], errors='coerce')
    
    # Try Yahoo Finance
    filename = "cash_flow_quarterly.csv" if quarterly else "cash_flow_annual.csv"
    yf_path = ticker_folder / "yahoo_finance" / "csv" / filename
    
    if yf_path.exists():
        df = pd.read_csv(yf_path, index_col=0)
        
        # Metric variations
        if metric not in df.index:
            metric_variations = {
                'operatingCashflow': 'Operating Cash Flow',
                'capitalExpenditures': 'Capital Expenditure',
                'freeCashFlow': 'Free Cash Flow',
            }
            if metric in metric_variations:
                metric = metric_variations[metric]
        
        series = df.loc[metric]
        date_cols = [col for col in series.index if col not in ['TTM', 'Breakdown']]
        series = series[date_cols]
        series.index = pd.to_datetime(series.index)
        return pd.to_numeric(series, errors='coerce').sort_index()
    
    raise FileNotFoundError(f"No cash flow data found for {ticker}")


# ========== TRANSFORMATION FUNCTIONS ==========

def normalize_to_1(series: pd.Series) -> pd.Series:
    """
    Normalize series to start at 1
    
    Example:
        >>> prices = pd.Series([100, 110, 105, 120])
        >>> normalized = normalize_to_1(prices)
        >>> # Result: [1.0, 1.1, 1.05, 1.2]
    """
    if len(series) == 0 or pd.isna(series.iloc[0]) or series.iloc[0] == 0:
        return series
    return series / series.iloc[0]


def yoy_growth(series: pd.Series) -> pd.Series:
    """
    Calculate year-over-year growth rate
    For quarterly data: compares to 4 periods back
    For annual data: compares to 1 period back
    
    Returns growth as decimals (multiply by 100 for percentage)
    """
    # Assume quarterly if we have many periods
    periods = 4 if len(series) > 8 else 1
    return (series / series.shift(periods)) - 1


def qoq_growth(series: pd.Series) -> pd.Series:
    """Calculate quarter-over-quarter growth rate"""
    return (series / series.shift(1)) - 1


def rolling_sum(series: pd.Series, periods: int) -> pd.Series:
    """
    Calculate rolling sum over N periods (trailing sum)
    
    Example:
        >>> quarterly_revenue = load_income_metric('NVDA', 'totalRevenue')
        >>> ttm_revenue = rolling_sum(quarterly_revenue, 4)  # Trailing 12 months
    """
    return series.rolling(window=periods, min_periods=1).sum()


def moving_average(series: pd.Series, days: int) -> pd.Series:
    """
    Calculate N-day moving average
    
    Example:
        >>> prices = load_stock_prices('NVDA')
        >>> ma_50 = moving_average(prices, 50)
        >>> ma_200 = moving_average(prices, 200)
    """
    return series.rolling(window=days, min_periods=1).mean()