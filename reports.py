# reports.py
"""
Company Report Generator using RAG system
Generates comprehensive PDF reports with financial analysis
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple
from openai import OpenAI

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
        4. Competitors: Name at least 3 main competitors
        5. Differentiation: Explain how {self.ticker} differentiates itself from these competitors
        
        Ensure you cover all 5 points above.''',
            
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