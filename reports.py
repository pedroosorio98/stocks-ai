# reports.py
"""
Company Report Generator using RAG system
Generates comprehensive PDF reports with financial analysis
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional
from openai import OpenAI
import pandas as pd

# Import your existing RAG functions
from rag import retrieve_internal, retrieve_web_exa, build_context, detect_tickers_from_query

# PDF Generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image, ListFlowable, ListItem
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib import colors

client = OpenAI()

# Setup directories
BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


class CompanyReportGenerator:
    """Generate comprehensive company reports using RAG"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.report_data = {}
        self.timestamp = datetime.now()
        
        # Track all sources used
        self.internal_sources = set()  # Set of (filename, page/chunk) tuples
        self.web_sources = set()       # Set of (title, url) tuples
        
    def _track_sources(self, internal_hits: List[dict], web_hits: List[dict]):
        """Track sources from retrieved context"""
        
        # Track internal sources
        for hit in internal_hits:
            src = hit.get("source", "")
            if src:
                fname = Path(src).name
                
                # Create citation reference
                if "page" in hit:
                    ref = (fname, f"p.{hit['page']}")
                elif "row_start" in hit and "row_end" in hit:
                    ref = (fname, f"rows {hit['row_start']}-{hit['row_end']}")
                elif "chunk" in hit:
                    ref = (fname, f"chunk {hit['chunk']}")
                else:
                    ref = (fname, "")
                
                self.internal_sources.add(ref)
        
        # Track web sources
        for web in web_hits:
            title = web.get("title", "Untitled")
            url = web.get("url", "")
            if url:
                self.web_sources.add((title, url))
    
    def _markdown_to_html(self, text: str) -> str:
        """Convert markdown **bold** to HTML <b>bold</b> for ReportLab"""
        import re
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        return text
    
    def _load_financial_data(self) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Load quarterly income statement data from Alpha Vantage or Yahoo Finance
        
        Returns:
            Tuple of (DataFrame, source_name) or (None, None) if not found
        """
        import os
        
        ticker_folder = Path("Data") / self.ticker
        
        # ========== ENHANCED LOGGING FOR RAILWAY DEBUG ==========
        print(f"\n{'='*60}")
        print(f"[LOAD DATA] Attempting to load data for ticker: {self.ticker}")
        print(f"[LOAD DATA] Current working directory: {os.getcwd()}")
        print(f"[LOAD DATA] Ticker folder path: {ticker_folder}")
        print(f"[LOAD DATA] Ticker folder exists: {ticker_folder.exists()}")
        
        if ticker_folder.exists():
            try:
                contents = list(ticker_folder.iterdir())
                print(f"[LOAD DATA] Ticker folder contents: {[c.name for c in contents]}")
            except Exception as e:
                print(f"[LOAD DATA] Error listing ticker folder: {e}")
        else:
            print(f"[LOAD DATA] ⚠️ Ticker folder does NOT exist!")
            data_folder = Path("Data")
            print(f"[LOAD DATA] Data folder exists: {data_folder.exists()}")
            if data_folder.exists():
                try:
                    available = [d.name for d in data_folder.iterdir() if d.is_dir()]
                    print(f"[LOAD DATA] Available tickers: {available}")
                except Exception as e:
                    print(f"[LOAD DATA] Error listing Data/: {e}")
            else:
                print(f"[LOAD DATA] ⚠️ Data/ folder NOT FOUND!")
                try:
                    print(f"[LOAD DATA] Root contents: {os.listdir('.')[:20]}")
                except:
                    pass
        
        # Try Alpha Vantage first
        alpha_vantage_path = ticker_folder / "alpha_vantage" / "csv" / "INCOME_STATEMENT__quarterlyReports.csv"
        print(f"[LOAD DATA] AV path: {alpha_vantage_path}")
        print(f"[LOAD DATA] AV exists: {alpha_vantage_path.exists()}")
        
        if alpha_vantage_path.exists():
            print(f"  → ✅ Loading Alpha Vantage data for {self.ticker}")
            df = pd.read_csv(alpha_vantage_path)
            print(f"[LOAD DATA] Success! Shape: {df.shape}")
            print(f"{'='*60}\n")
            return df, "alpha_vantage"
        
        # Try Yahoo Finance
        yahoo_finance_path = ticker_folder / "yahoo_finance" / "csv" / "income_stmt_quarterly.csv"
        print(f"[LOAD DATA] YF path: {yahoo_finance_path}")
        print(f"[LOAD DATA] YF exists: {yahoo_finance_path.exists()}")
        
        if yahoo_finance_path.exists():
            print(f"  → ✅ Loading Yahoo Finance data for {self.ticker}")
            df = pd.read_csv(yahoo_finance_path, index_col=0)
            print(f"[LOAD DATA] Success! Shape: {df.shape}")
            print(f"{'='*60}\n")
            return df, "yahoo_finance"
        
        print(f"  ⚠ No financial data found for {self.ticker}")
        print(f"{'='*60}\n")
        return None, None
    
    def _load_balance_sheet_data(self, source: str) -> Optional[pd.DataFrame]:
        """
        Load quarterly balance sheet data from Alpha Vantage or Yahoo Finance
        
        Args:
            source: 'alpha_vantage' or 'yahoo_finance' (determined from income statement)
        
        Returns:
            DataFrame with balance sheet data or None if not found
        """
        ticker_folder = Path("Data") / self.ticker
        
        print(f"[LOAD BALANCE] Loading balance sheet for {self.ticker} from {source}")
        
        if source == "alpha_vantage":
            balance_path = ticker_folder / "alpha_vantage" / "csv" / "BALANCE_SHEET__quarterlyReports.csv"
            print(f"[LOAD BALANCE] Path: {balance_path}")
            print(f"[LOAD BALANCE] Exists: {balance_path.exists()}")
            if balance_path.exists():
                print(f"  → ✅ Loading Alpha Vantage balance sheet")
                return pd.read_csv(balance_path)
        
        elif source == "yahoo_finance":
            balance_path = ticker_folder / "yahoo_finance" / "csv" / "balance_sheet_quarterly.csv"
            print(f"[LOAD BALANCE] Path: {balance_path}")
            print(f"[LOAD BALANCE] Exists: {balance_path.exists()}")
            if balance_path.exists():
                print(f"  → ✅ Loading Yahoo Finance balance sheet")
                return pd.read_csv(balance_path, index_col=0)
        
        print(f"  ⚠ No balance sheet data found for {self.ticker}")
        return None
    
    def _calculate_financial_metrics(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """
        Calculate financial metrics from income statement data
        
        Args:
            df: Income statement DataFrame
            source: 'alpha_vantage' or 'yahoo_finance'
        
        Returns:
            DataFrame with calculated metrics (metrics as rows, quarters as columns)
        """
        metrics = {}
        
        if source == "alpha_vantage":
            # Alpha Vantage: columns are metrics, rows are quarters
            df = df.sort_values('fiscalDateEnding', ascending=True)
            df = df.tail(12)  # Get last 12 quarters for YoY calculation
            
            quarters = df['fiscalDateEnding'].tolist()
            
            # Total Revenue
            total_revenue = pd.to_numeric(df['totalRevenue'], errors='coerce')
            metrics['Total Revenue'] = total_revenue.tolist()
            
            # YoY Revenue Growth
            yoy_growth = []
            for i in range(len(total_revenue)):
                if i >= 4 and not pd.isna(total_revenue.iloc[i]) and not pd.isna(total_revenue.iloc[i-4]) and total_revenue.iloc[i-4] != 0:
                    growth = (total_revenue.iloc[i] / total_revenue.iloc[i-4]) - 1
                    yoy_growth.append(growth)
                else:
                    yoy_growth.append(None)
            metrics['YoY Revenue Growth'] = yoy_growth
            
            # Margins
            gross_profit = pd.to_numeric(df['grossProfit'], errors='coerce')
            metrics['Gross Margin'] = (gross_profit / total_revenue).tolist()
            
            operating_income = pd.to_numeric(df['operatingIncome'], errors='coerce')
            metrics['Operating Margin'] = (operating_income / total_revenue).tolist()
            
            ebitda = pd.to_numeric(df['ebitda'], errors='coerce')
            metrics['EBITDA Margin'] = (ebitda / total_revenue).tolist()
            
            net_income = pd.to_numeric(df['netIncome'], errors='coerce')
            metrics['Net Margin'] = (net_income / total_revenue).tolist()
            
        elif source == "yahoo_finance":
            # Yahoo Finance: rows are metrics (now in index after index_col=0), columns are quarters
            # IMPORTANT: Yahoo Finance columns are in REVERSE chronological order (newest first)
            date_columns = [col for col in df.columns if col not in ['TTM']]
            
            # Reverse to get oldest → newest for proper YoY calculation
            date_columns_oldest_first = list(reversed(date_columns))
            
            # Take last 12 quarters (most recent) for calculation
            quarters_all = date_columns_oldest_first[-12:] if len(date_columns_oldest_first) >= 12 else date_columns_oldest_first
            quarters_display = quarters_all[-8:] if len(quarters_all) >= 8 else quarters_all
            
            def get_metric_values(metric_name: str, quarters_to_use) -> List[float]:
                # Use index to find rows (after index_col=0)
                if metric_name not in df.index:
                    return [None] * len(quarters_to_use)
                row = df.loc[metric_name]  # This returns a Series
                values = []
                for quarter in quarters_to_use:
                    try:
                        # row is a Series, access by column name directly
                        val = row[quarter] if quarter in row else None
                        values.append(float(val) if val is not None else None)
                    except (ValueError, TypeError, KeyError):
                        values.append(None)
                return values
            
            # Total Revenue (get 12 quarters for YoY calculation)
            total_revenue_all = get_metric_values('Total Revenue', quarters_all)
            total_revenue = total_revenue_all[-8:]  # Keep last 8 for display
            metrics['Total Revenue'] = total_revenue
            
            # YoY Revenue Growth (calculate from all 12 quarters, oldest → newest)
            yoy_growth_all = []
            for i in range(len(total_revenue_all)):
                if i >= 4 and total_revenue_all[i] and total_revenue_all[i-4] and total_revenue_all[i-4] != 0:
                    growth = (total_revenue_all[i] / total_revenue_all[i-4]) - 1
                    yoy_growth_all.append(growth)
                else:
                    yoy_growth_all.append(None)
            metrics['YoY Revenue Growth'] = yoy_growth_all[-8:]  # Keep last 8
            
            # Margins (only need last 8 quarters for display)
            gross_profit = get_metric_values('Gross Profit', quarters_display)
            metrics['Gross Margin'] = [gp/rev if gp and rev and rev != 0 else None for gp, rev in zip(gross_profit, total_revenue)]
            
            operating_income = get_metric_values('Operating Income', quarters_display)
            metrics['Operating Margin'] = [oi/rev if oi and rev and rev != 0 else None for oi, rev in zip(operating_income, total_revenue)]
            
            ebitda = get_metric_values('EBITDA', quarters_display)
            metrics['EBITDA Margin'] = [eb/rev if eb and rev and rev != 0 else None for eb, rev in zip(ebitda, total_revenue)]
            
            net_income = get_metric_values('Net Income', quarters_display)
            metrics['Net Margin'] = [ni/rev if ni and rev and rev != 0 else None for ni, rev in zip(net_income, total_revenue)]
            
            # Keep quarters in oldest → newest order for display (data matches)
            quarters = quarters_display
            
        # Create DataFrame with all quarters
        metrics_df = pd.DataFrame(metrics, index=quarters)
        
        # Keep only last 8 quarters for display (after YoY calculation)
        metrics_df = metrics_df.tail(8)
        
        # Drop columns where ALL values are NaN (empty quarters with no data)
        metrics_df = metrics_df.dropna(axis=1, how='all')
        
        return metrics_df.T
    
    def _calculate_balance_sheet_metrics(self, income_df: pd.DataFrame, balance_df: pd.DataFrame, source: str, quarters: list) -> pd.DataFrame:
        """
        Calculate balance sheet metrics (ROA, ROE) using income statement and balance sheet data
        
        Args:
            income_df: Income statement DataFrame (already loaded)
            balance_df: Balance sheet DataFrame
            source: 'alpha_vantage' or 'yahoo_finance'
            quarters: List of quarters to match (same as income statement)
        
        Returns:
            DataFrame with ROA and ROE metrics (metrics as rows, quarters as columns)
        """
        metrics = {}
        
        if source == "alpha_vantage":
            # Alpha Vantage: columns are metrics, rows are quarters
            balance_df = balance_df.sort_values('fiscalDateEnding', ascending=True)
            income_df_sorted = income_df.sort_values('fiscalDateEnding', ascending=True)
            
            # Get ALL net income data (need extra quarters for trailing 4Q sum)
            net_income_series = pd.to_numeric(income_df_sorted['netIncome'], errors='coerce')
            
            # Calculate trailing 4-quarter (TTM) net income
            ttm_net_income = net_income_series.rolling(window=4, min_periods=4).sum()
            
            # Filter to quarters we want to display
            income_df_filtered = income_df_sorted[income_df_sorted['fiscalDateEnding'].isin(quarters)]
            balance_df_filtered = balance_df[balance_df['fiscalDateEnding'].isin(quarters)]
            
            # Get TTM net income for display quarters
            ttm_net_income_filtered = ttm_net_income[income_df_sorted['fiscalDateEnding'].isin(quarters)].tolist()
            
            # Get balance sheet items
            total_assets = pd.to_numeric(balance_df_filtered['totalAssets'], errors='coerce').tolist()
            total_equity = pd.to_numeric(balance_df_filtered['totalShareholderEquity'], errors='coerce').tolist()
            
            # Calculate ROA and ROE using TTM net income
            roa = []
            roe = []
            for ttm_ni, ta, te in zip(ttm_net_income_filtered, total_assets, total_equity):
                # ROA = TTM Net Income / Total Assets
                if ttm_ni and not pd.isna(ttm_ni) and ta and ta != 0:
                    roa.append(ttm_ni / ta)
                else:
                    roa.append(None)
                
                # ROE = TTM Net Income / Total Equity
                if ttm_ni and not pd.isna(ttm_ni) and te and te != 0:
                    roe.append(ttm_ni / te)
                else:
                    roe.append(None)
            
            metrics['Return on Assets'] = roa
            metrics['Return on Equity'] = roe
            
        elif source == "yahoo_finance":
            # Yahoo Finance: rows are metrics (index), columns are quarters
            # quarters list is already in oldest → newest order
            
            def get_metric_values(df, metric_name: str, quarters_to_use) -> List[float]:
                if metric_name not in df.index:
                    return [None] * len(quarters_to_use)
                row = df.loc[metric_name]
                values = []
                for quarter in quarters_to_use:
                    try:
                        val = row[quarter] if quarter in row else None
                        values.append(float(val) if val is not None else None)
                    except (ValueError, TypeError, KeyError):
                        values.append(None)
                return values
            
            # Get Net Income from income statement for ALL available quarters
            all_quarters_income = income_df.loc['Net Income'] if 'Net Income' in income_df.index else None
            
            if all_quarters_income is not None:
                # Get all date columns (exclude TTM)
                date_cols = [col for col in all_quarters_income.index if col not in ['TTM', 'Breakdown']]
                # Sort oldest to newest
                date_cols_sorted = sorted([pd.to_datetime(col) for col in date_cols])
                
                # Convert net income to Series
                ni_series = pd.Series(dtype=float)
                for date in date_cols_sorted:
                    date_str = date.strftime('%Y-%m-%d')
                    if date_str in all_quarters_income.index:
                        ni_series[date] = float(all_quarters_income[date_str]) if all_quarters_income[date_str] else None
                
                # Calculate TTM net income (trailing 4 quarters)
                ttm_net_income_series = ni_series.rolling(window=4, min_periods=4).sum()
                
                # Get TTM values for our display quarters
                ttm_net_income = []
                for quarter in quarters:
                    quarter_dt = pd.to_datetime(quarter)
                    if quarter_dt in ttm_net_income_series.index:
                        ttm_net_income.append(ttm_net_income_series[quarter_dt])
                    else:
                        ttm_net_income.append(None)
            else:
                ttm_net_income = [None] * len(quarters)
            
            # Get balance sheet items
            total_assets = get_metric_values(balance_df, 'Total Assets', quarters)
            total_equity = get_metric_values(balance_df, 'Total Equity Gross Minority Interest', quarters)
            
            # Calculate ROA and ROE using TTM net income
            roa = []
            roe = []
            for ttm_ni, ta, te in zip(ttm_net_income, total_assets, total_equity):
                # ROA = TTM Net Income / Total Assets
                if ttm_ni and not pd.isna(ttm_ni) and ta and ta != 0:
                    roa.append(ttm_ni / ta)
                else:
                    roa.append(None)
                
                # ROE = TTM Net Income / Total Equity
                if ttm_ni and not pd.isna(ttm_ni) and te and te != 0:
                    roe.append(ttm_ni / te)
                else:
                    roe.append(None)
            
            metrics['Return on Assets'] = roa
            metrics['Return on Equity'] = roe
        
        # Create DataFrame
        balance_metrics_df = pd.DataFrame(metrics, index=quarters)
        
        # Drop columns where ALL values are NaN (empty quarters with no data)
        balance_metrics_df = balance_metrics_df.dropna(axis=1, how='all')
        
        return balance_metrics_df.T
    
    def _format_financial_table_for_pdf(self, metrics_df: pd.DataFrame) -> Table:
        """
        Format financial metrics DataFrame as ReportLab Table
        
        Args:
            metrics_df: DataFrame with metrics as rows, quarters as columns
        
        Returns:
            ReportLab Table object
        """
        # Prepare header - clean column names
        clean_columns = []
        for col in metrics_df.columns:
            col_str = str(col)
            # Remove time info if present (e.g., "2024-03-31 00:00:00" -> "2024-03-31")
            if ' ' in col_str:
                col_str = col_str.split(' ')[0]
            # Truncate to 10 chars
            clean_columns.append(col_str[:10])
        
        header = ['Metric'] + clean_columns
        data = [header]
        
        # Add metric rows
        for metric_name in metrics_df.index:
            row = [metric_name]
            for value in metrics_df.loc[metric_name]:
                if value is None or pd.isna(value):
                    row.append('-')
                elif metric_name == 'Total Revenue':
                    if abs(value) >= 1e9:
                        row.append(f'${value/1e9:.1f}B')
                    elif abs(value) >= 1e6:
                        row.append(f'${value/1e6:.0f}M')
                    else:
                        row.append(f'${value:,.0f}')
                elif 'Growth' in metric_name or 'Margin' in metric_name or 'Return' in metric_name:
                    row.append(f'{value*100:.1f}%')
                else:
                    row.append(f'{value:,.0f}')
            data.append(row)
        
        # Create table with column widths
        # IMPORTANT: Keep total width consistent regardless of number of columns
        # Target total width: 8.0 inches (maximizes page usage on 8.5" page with margins)
        num_quarters = len(metrics_df.columns)
        
        target_total_width = 8.0 * inch # 8.0
        metric_col_width = 1.6 * inch
        
        # Calculate quarter column width to reach target total width
        # total_width = metric_col_width + (num_quarters × quarter_col_width)
        # quarter_col_width = (total_width - metric_col_width) / num_quarters
        quarter_col_width = (target_total_width - metric_col_width) / num_quarters
        
        col_widths = [metric_col_width] + [quarter_col_width] * num_quarters
        table = Table(data, colWidths=col_widths)
        
        # Style the table
        table.setStyle(TableStyle([
            # Header styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#043A22')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            
            # First column styling
            ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#F0F0F0')),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            
            # Data cells styling
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
            ('FONTNAME', (1, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (1, 1), (-1, -1), 9),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            
            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            
            # Alternating rows
            ('ROWBACKGROUNDS', (1, 1), (-1, -1), [colors.white, colors.HexColor('#F9F9F9')]),
        ]))
        
        return table
    
    def generate_section(self, section_name: str, prompt: str, context_k: int = 8) -> str:
        """
        Generate a single report section using RAG + LLM
        
        Args:
            section_name: Name of the section (e.g., "Core Description")
            prompt: The query to retrieve relevant context
            context_k: Number of context chunks to retrieve
            
        Returns:
            Generated section content
        """
        print(f"\n[REPORT] Generating section: {section_name}")
        
        # Retrieve relevant context from RAG
        internal_hits = retrieve_internal(
            query=prompt,
            k=context_k,
            filter_tickers=[self.ticker]
        )
        
        # Optionally get web context for recent info
        web_hits = retrieve_web_exa(f"{self.ticker} {prompt}", k=2)
        
        # Track sources used
        self._track_sources(internal_hits, web_hits)
        
        # Build context
        context = build_context(internal_hits, web_hits, max_chars_per_chunk=1000)
        
        if not context:
            print(f"[REPORT] No context found for {section_name}")
            return f"Insufficient data available for {section_name}."
        
        # Generate content using LLM
        system_prompt = f"""You are a financial analyst writing a professional company report section.

                            Section: {section_name}
                            Company Ticker: {self.ticker}
                            
                            Instructions:
                            - Write in clear, professional language
                            - Use specific data and metrics from the provided context
                            - Be objective and analytical
                            - Cite specific numbers, dates, and facts
                            - Keep the section focused and concise (300-500 words)
                            - Do not use markdown formatting
                            - Write in paragraph form suitable for PDF"""

        user_prompt = f"""Based on the following context, write the '{section_name}' section for {self.ticker}:

                        CONTEXT:
                        {context}
                        
                        Write a comprehensive {section_name} section."""

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content.strip()
            print(f"[REPORT] Generated {len(content)} chars for {section_name}")
            return content
            
        except Exception as e:
            print(f"[REPORT] Error generating {section_name}: {e}")
            return f"Error generating {section_name}: {str(e)}"
    
    def generate_all_sections(self) -> Dict[str, str]:
        """Generate all report sections"""
        
        sections = {
            "Core Description": f'''Provide a comprehensive description of {self.ticker}:
        
        REQUIRED CONTENT:
        1. Core Business: Describe what the company does, its main products/services, and business model
        2. Markets: List the specific geographic markets and industry segments where it operates
        3. Value Proposition: Explain the company's unique value proposition in 2-3 sentences
        4. Competitors: you must name at least 3 main competitors
        5. Differentiation: Explain how {self.ticker} differentiates itself from these competitors
        6. Management: Describe {self.ticker} management team, key people inside the company
        
        Ensure you cover all 6 points above.''',
            
            "Historical Context and Competitive Positioning": f'''Provide historical and competitive analysis for {self.ticker}:
        
        REQUIRED CONTENT:
        1. Historical Background: Summarize the company's history and major milestones
        2. Market Share: You MUST state the estimated market share percentage (format: "market share: XX%")
        3. Competitive Advantages: List at least 2 specific competitive advantages
        4. Structural Changes: Identify any significant M&A deals, management changes, product launches, or strategy pivots
        5. Performance Metrics:
           - Revenue growth rate over past 2-3 years (state specific percentage)
           - Profitability trend (state specific metric like "ROE increased from 15% to 18%")
        
        You must include numerical percentages for market share and growth rates. Do not skip the quantitative data.''',
            
            "Key Drivers of Performance": f'''Analyze the key drivers of {self.ticker}'s performance:
        
        REQUIRED FORMAT - YOU MUST INCLUDE ALL OF THE FOLLOWING:
        
        1. Key Performance Indicators (KPIs):
           - List 3-5 specific KPIs that matter most for this company
           - For each KPI, provide the recent value/metric
        
        2. Margin Analysis (MANDATORY - INCLUDE ALL THREE):
           - Gross Margin: [You MUST state as "Gross margin: XX.X%"]
           - EBITDA Margin: [You MUST state as "EBITDA margin: XX.X%"]
           - Net Margin: [You MUST state as "Net margin: XX.X%"]
           - For each margin above, describe the trend (improving/declining) over past 2-3 years
        
        3. Competitor Margin Comparison:
           - Name at least 2 competitors
           - State their gross/EBITDA/net margins for comparison
           - Example: "Competitor A has gross margin of 45% vs {self.ticker}'s 38%"
        
        4. Economic Moat:
           - Identify 2-3 specific competitive advantages/moats
           - Explain why each is defensible
        
        5. Market Share:
           - State the estimated market share percentage (format: "market share: XX%")
        
        6. Critical Success Factors:
           - List EXACTLY 3-5 key factors (not fewer than 3, not more than 5)
           - Number them as 1., 2., 3., etc.
        
        CRITICAL: Do not proceed to write this section unless you can provide specific numerical values for all three margins (gross, EBITDA, net). If data is unavailable, state "Data not available" but attempt to find estimates.''',
            
            "Company Outlook": f'''What is the future outlook for {self.ticker}?
        
        REQUIRED CONTENT:
        1. Growth Prospects: Describe expected growth trajectory with specific drivers
        2. Strategic Initiatives: List at least 2-3 current strategic initiatives or focus areas
        3. Expected Timeline: When are these initiatives expected to impact results?
        
        Provide specific details, not vague statements.''',
            
            "Opportunities": f'''Identify growth opportunities for {self.ticker}:
        
        REQUIRED FORMAT:
        List AT LEAST 2 specific opportunities. Format as:
        
        1. [Opportunity Name]: [Detailed description with specifics]
           - Potential impact: [quantify if possible]
           
        2. [Opportunity Name]: [Detailed description with specifics]
           - Potential impact: [quantify if possible]
        
        You must cite at least 2 examples. Include specific market names, product categories, or expansion vectors.''',
            
            "Risks": f'''Identify key risks facing {self.ticker}:
        
        REQUIRED FORMAT:
        List AT LEAST 2 specific risks. Format as:
        
        1. [Risk Name]: [Detailed description of the risk]
           - Potential impact: [describe severity]
           
        2. [Risk Name]: [Detailed description of the risk]  
           - Potential impact: [describe severity]
        
        You must cite at least 2 examples. Be specific about regulatory, competitive, operational, or market risks.''',
            
            "Scenarios": f'''Develop probability-weighted scenarios for {self.ticker}:
        
        REQUIRED FORMAT - THIS IS MANDATORY:
        
        **Base Case Scenario (Probability: XX%):**
        [Description of base case]
        Stock price implication: [discuss expected price movement/valuation]
        
        **Bull Case Scenario (Probability: XX%):**
        [Description of bull case]
        Stock price implication: [discuss upside potential]
        
        **Bear Case Scenario (Probability: XX%):**
        [Description of bear case]
        Stock price implication: [discuss downside risk]
        
        CRITICAL REQUIREMENTS:
        1. You MUST provide 3 numerical probabilities (one for each scenario)
        2. The 3 probabilities MUST sum to 100%
        3. Format probabilities clearly (e.g., "Probability: 50%")
        4. Each scenario must discuss stock price implications
        5. Use the exact labels: "Base Case", "Bull Case", "Bear Case"
        
        Example format:
        - Base Case Scenario (Probability: 50%)
        - Bull Case Scenario (Probability: 30%)
        - Bear Case Scenario (Probability: 20%)
        
        Do not write this section unless you include all 3 probabilities that sum to 100%.'''
        }
        
        print(f"\n{'='*60}")
        print(f"GENERATING REPORT FOR: {self.ticker}")
        print(f"{'='*60}")
        
        for section_name, query in sections.items():
            content = self.generate_section(section_name, query, context_k=10)
            self.report_data[section_name] = content
        
        return self.report_data
    
    def create_pdf(self, filename: str = None) -> Path:
        """
        Create professional PDF report
        
        Args:
            filename: Optional custom filename
            
        Returns:
            Path to generated PDF
        """
        if not filename:
            timestamp = self.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.ticker}_Report_{timestamp}.pdf"
        
        filepath = REPORTS_DIR / filename
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Container for PDF elements
        story = []
        
        # Define styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#043A22'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.grey,
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#043A22'),
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold',
            borderWidth=1,
            borderColor=colors.HexColor('#043A22'),
            borderPadding=5
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=11,
            leading=16,
            alignment=TA_JUSTIFY,
            spaceAfter=12,
            fontName='Helvetica'
        )
        
        source_heading_style = ParagraphStyle(
            'SourceHeading',
            parent=styles['Heading3'],
            fontSize=13,
            textColor=colors.HexColor('#043A22'),
            spaceAfter=8,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        )
        
        source_item_style = ParagraphStyle(
            'SourceItem',
            parent=styles['Normal'],
            fontSize=9,
            leading=12,
            leftIndent=20,
            spaceAfter=6,
            fontName='Helvetica'
        )
        
        # Title Page
        story.append(Spacer(1, 1.5*inch))
        story.append(Paragraph(f"Company Analysis Report", title_style))
        story.append(Paragraph(f"{self.ticker}", title_style))
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph(
            f"Generated: {self.timestamp.strftime('%B %d, %Y at %H:%M UTC')}",
            subtitle_style
        ))
        story.append(Spacer(1, 0.2*inch))
        
        # Disclaimer box
        disclaimer_data = [[
            Paragraph(
                "<b>Disclaimer:</b> This report is for informational purposes only and does not constitute investment advice. "
                "Please conduct your own research before making investment decisions.",
                ParagraphStyle('Disclaimer', parent=styles['Normal'], fontSize=9, textColor=colors.grey)
            )
        ]]
        disclaimer_table = Table(disclaimer_data, colWidths=[6.5*inch])
        disclaimer_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
            ('BOX', (0, 0), (-1, -1), 1, colors.grey),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(disclaimer_table)
        
        story.append(PageBreak())
        
        # ========== KEY FINANCIAL METRICS TABLE ==========
        try:
            df, source = self._load_financial_data()
            if df is not None:
                # Section heading
                story.append(Paragraph("Key Financial Metrics", heading_style))
                story.append(Spacer(1, 0.1*inch))
                
                # Source note
                source_name = "Alpha Vantage" if source == "alpha_vantage" else "Yahoo Finance"
                story.append(Paragraph(
                    f"<i>Quarterly Income Statement Metrics (Source: {source_name})</i>",
                    ParagraphStyle('SourceNote', parent=styles['Normal'], 
                                 fontSize=9, textColor=colors.grey, alignment=TA_LEFT)
                ))
                story.append(Spacer(1, 0.1*inch))
                
                # Calculate and format metrics
                metrics_df = self._calculate_financial_metrics(df, source)
                financial_table = self._format_financial_table_for_pdf(metrics_df)
                
                # Add table
                story.append(financial_table)
                story.append(Spacer(1, 0.3*inch))
                
                print(f"  → Added financial table with {len(metrics_df.columns)} quarters")
                
                # ========== BALANCE SHEET METRICS TABLE ==========
                try:
                    # Load balance sheet data
                    balance_df = self._load_balance_sheet_data(source)
                    if balance_df is not None:
                        # Calculate balance sheet metrics (ROA, ROE)
                        quarters_list = metrics_df.columns.tolist()
                        balance_metrics_df = self._calculate_balance_sheet_metrics(df, balance_df, source, quarters_list)
                        
                        # Remove columns where BOTH tables have ALL NaN values
                        # First, identify columns to keep
                        columns_to_keep = []
                        for col in metrics_df.columns:
                            # Check if income table has ANY non-null value in this column
                            income_has_data = metrics_df[col].notna().any()
                            # Check if balance table has ANY non-null value in this column
                            balance_has_data = balance_metrics_df[col].notna().any() if col in balance_metrics_df.columns else False
                            
                            # Keep column if EITHER table has data
                            if income_has_data or balance_has_data:
                                columns_to_keep.append(col)
                        
                        # Filter both tables to only include columns with data
                        metrics_df = metrics_df[columns_to_keep]
                        balance_metrics_df = balance_metrics_df[columns_to_keep]
                        
                        # Recreate tables with filtered columns
                        financial_table = self._format_financial_table_for_pdf(metrics_df)
                        balance_table = self._format_financial_table_for_pdf(balance_metrics_df)
                        
                        # Replace the previously added income table with filtered version
                        story.pop()  # Remove spacer
                        story.pop()  # Remove old financial_table
                        
                        # Add filtered income table
                        story.append(financial_table)
                        story.append(Spacer(1, 0.2*inch))
                        
                        # Add source note for balance sheet table
                        source_name = "Alpha Vantage" if source == "alpha_vantage" else "Yahoo Finance"
                        story.append(Paragraph(
                            f"<i>Trailing 4-Quarter Measures (Source: {source_name})</i>",
                            ParagraphStyle('BalanceSourceNote', parent=styles['Normal'], 
                                         fontSize=9, textColor=colors.grey, alignment=TA_LEFT)
                        ))
                        story.append(Spacer(1, 0.1*inch))
                        
                        # Add balance sheet table
                        story.append(balance_table)
                        story.append(Spacer(1, 0.2*inch))
                        
                        print(f"  → Final tables have {len(columns_to_keep)} quarters (removed {len(quarters_list) - len(columns_to_keep)} empty columns)")
                    else:
                        print(f"  ⚠ Skipping balance sheet table - no data found")
                except Exception as e:
                    print(f"  ⚠ Error generating balance sheet table: {e}")
                    import traceback
                    traceback.print_exc()
                # ========== END BALANCE SHEET TABLE ==========
                
            else:
                print(f"  ⚠ Skipping financial table - no data found")
        except Exception as e:
            print(f"  ⚠ Error generating financial table: {e}")
            import traceback
            traceback.print_exc()
        # ========== END FINANCIAL TABLE ==========
        
        story.append(PageBreak())
        
        # Report sections
        section_order = [
            "Core Description",
            "Historical Context and Competitive Positioning",
            "Key Drivers of Performance",
            "Company Outlook",
            "Opportunities",
            "Risks",
            "Scenarios"
        ]
        
        for section_name in section_order:
            if section_name in self.report_data:
                # Section heading
                story.append(Paragraph(section_name, heading_style))
                story.append(Spacer(1, 0.1*inch))
                
                # Section content
                content = self.report_data[section_name]
                # Split into paragraphs for better formatting
                paragraphs = content.split('\n\n')
                for para in paragraphs:
                    if para.strip():
                        para_html = self._markdown_to_html(para.strip())
                        story.append(Paragraph(para_html, body_style))
                
                story.append(Spacer(1, 0.2*inch))
        
        # ========== SOURCES & REFERENCES SECTION ==========
        story.append(PageBreak())
        story.append(Paragraph("Sources & References", heading_style))
        story.append(Spacer(1, 0.15*inch))
        
        story.append(Paragraph(
            "This report was generated using the following sources:",
            body_style
        ))
        story.append(Spacer(1, 0.1*inch))
        
        # Internal Sources
        if self.internal_sources:
            story.append(Paragraph("Internal Documents", source_heading_style))
            
            # Group by filename
            sources_by_file = {}
            for fname, location in sorted(self.internal_sources):
                if fname not in sources_by_file:
                    sources_by_file[fname] = []
                if location:
                    sources_by_file[fname].append(location)
            
            for fname, locations in sorted(sources_by_file.items()):
                if locations:
                    locations_str = ", ".join(sorted(locations))
                    source_text = f"• <b>{fname}</b> ({locations_str})"
                else:
                    source_text = f"• <b>{fname}</b>"
                
                story.append(Paragraph(source_text, source_item_style))
            
            story.append(Spacer(1, 0.15*inch))
        else:
            story.append(Paragraph("Internal Documents", source_heading_style))
            story.append(Paragraph("• No internal sources used", source_item_style))
            story.append(Spacer(1, 0.15*inch))
        
        # Web Sources
        if self.web_sources:
            story.append(Paragraph("Web Sources", source_heading_style))
            
            for idx, (title, url) in enumerate(sorted(self.web_sources), 1):
                # Make URL clickable
                source_text = f"• <b>{title}</b><br/><link href='{url}' color='blue'>{url}</link>"
                story.append(Paragraph(source_text, source_item_style))
            
            story.append(Spacer(1, 0.15*inch))
        else:
            story.append(Paragraph("Web Sources", source_heading_style))
            story.append(Paragraph("• No web sources used", source_item_style))
            story.append(Spacer(1, 0.15*inch))
        
        # Summary statistics
        total_sources = len(self.internal_sources) + len(self.web_sources)
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(
            f"<i>Total sources referenced: {total_sources} "
            f"({len(self.internal_sources)} internal, {len(self.web_sources)} web)</i>",
            ParagraphStyle('SourceSummary', parent=styles['Normal'], 
                         fontSize=9, textColor=colors.grey, alignment=TA_CENTER)
        ))
        
        # Footer with generation info
        story.append(PageBreak())
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph(
            f"End of Report - {self.ticker}",
            subtitle_style
        ))
        story.append(Paragraph(
            f"Powered by RAG-Enhanced Financial Analysis System",
            ParagraphStyle('Footer', parent=styles['Normal'], fontSize=9, 
                         textColor=colors.grey, alignment=TA_CENTER)
        ))
        
        # Build PDF
        doc.build(story)
        
        print(f"\n[REPORT] PDF generated: {filepath}")
        print(f"[REPORT] Sources: {len(self.internal_sources)} internal, {len(self.web_sources)} web")
        return filepath
    
    def generate_full_report(self, filename: str = None) -> Path:
        """
        Complete pipeline: Generate all sections and create PDF
        
        Returns:
            Path to generated PDF
        """
        # Generate all sections
        self.generate_all_sections()
        
        # Create PDF
        pdf_path = self.create_pdf(filename)
        
        print(f"\n{'='*60}")
        print(f"✓ REPORT COMPLETE: {pdf_path.name}")
        print(f"{'='*60}\n")
        
        return pdf_path


# Convenience function
def generate_company_report(ticker: str, filename: str = None) -> Path:
    """
    Quick function to generate a complete report
    
    Args:
        ticker: Company ticker symbol
        filename: Optional custom filename for PDF
        
    Returns:
        Path to generated PDF file
    """
    generator = CompanyReportGenerator(ticker)
    return generator.generate_full_report(filename)


# CLI interface for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python reports.py <TICKER>")
        print("Example: python reports.py AAPL")
        sys.exit(1)
    
    ticker = sys.argv[1]
    
    print(f"\n{'='*60}")
    print(f"Company Report Generator")
    print(f"{'='*60}\n")
    
    try:
        pdf_path = generate_company_report(ticker)
        print(f"\n✓ Success! Report saved to: {pdf_path}")
    except Exception as e:
        print(f"\n✗ Error generating report: {e}")
        import traceback
        traceback.print_exc()