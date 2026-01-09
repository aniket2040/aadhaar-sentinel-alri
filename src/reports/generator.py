"""
PDF Report Generator Module

Generates automated PDF reports for UIDAI stakeholders with ALRI scores,
reason codes, recommendations, and trend visualizations.

Requirements: 12.1, 12.2, 12.3, 12.4
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
from fpdf import FPDF
import io

from src.scoring.alri_calculator import ALRIResult
from src.explainability.reason_codes import ReasonCode, Severity
from src.recommendations.engine import Intervention, CostLevel


@dataclass
class ReportMetadata:
    """Metadata for generated reports."""
    title: str
    generated_at: datetime
    generated_by: str = "Aadhaar Sentinel Platform"
    version: str = "1.0"


class PDFReportGenerator:
    """
    Generates automated PDF reports for UIDAI stakeholders.
    
    Supports two report types:
    - State reports: Top-10 at-risk districts per state
    - District reports: Detailed metrics for a single district
    
    Requirements: 12.1, 12.2, 12.3, 12.4
    """
    
    # Color scheme for severity levels
    SEVERITY_COLORS = {
        Severity.LOW: (144, 238, 144),       # Light green
        Severity.MEDIUM: (255, 255, 153),    # Light yellow
        Severity.HIGH: (255, 165, 0),        # Orange
        Severity.CRITICAL: (255, 99, 71),    # Tomato red
    }
    
    # Color scheme for ALRI score ranges
    SCORE_COLORS = {
        'low': (144, 238, 144),      # 0-25: Green
        'medium': (255, 255, 153),   # 25-50: Yellow
        'high': (255, 165, 0),       # 50-75: Orange
        'critical': (255, 99, 71),   # 75-100: Red
    }
    
    def __init__(self):
        """Initialize the PDF report generator."""
        self._pdf: Optional[FPDF] = None
    
    def _create_pdf(self) -> FPDF:
        """Create a new PDF document with standard settings."""
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        return pdf
    
    def _add_header(self, pdf: FPDF, title: str, subtitle: str = "") -> None:
        """Add report header with title and subtitle."""
        # Title
        pdf.set_font("Helvetica", "B", 20)
        pdf.set_text_color(0, 51, 102)  # Dark blue
        pdf.cell(0, 15, title, ln=True, align="C")
        
        # Subtitle
        if subtitle:
            pdf.set_font("Helvetica", "I", 12)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 8, subtitle, ln=True, align="C")
        
        # Timestamp
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(128, 128, 128)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pdf.cell(0, 8, f"Generated: {timestamp}", ln=True, align="C")
        
        # Separator line
        pdf.ln(5)
        pdf.set_draw_color(0, 51, 102)
        pdf.set_line_width(0.5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(10)
    
    def _add_section_header(self, pdf: FPDF, title: str) -> None:
        """Add a section header."""
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(0, 51, 102)
        pdf.cell(0, 10, title, ln=True)
        pdf.ln(2)
    
    def _get_score_color(self, score: float) -> tuple:
        """Get color based on ALRI score."""
        if score < 25:
            return self.SCORE_COLORS['low']
        elif score < 50:
            return self.SCORE_COLORS['medium']
        elif score < 75:
            return self.SCORE_COLORS['high']
        else:
            return self.SCORE_COLORS['critical']
    
    def _get_risk_level(self, score: float) -> str:
        """Get risk level label based on ALRI score."""
        if score < 25:
            return "Low Risk"
        elif score < 50:
            return "Medium Risk"
        elif score < 75:
            return "High Risk"
        else:
            return "Critical Risk"

    def _add_alri_score_box(self, pdf: FPDF, result: ALRIResult) -> None:
        """Add a visual ALRI score box."""
        # Score box background
        color = self._get_score_color(result.alri_score)
        pdf.set_fill_color(*color)
        
        # Draw score box
        x_start = 70
        pdf.rect(x_start, pdf.get_y(), 70, 25, "F")
        
        # Score value
        pdf.set_font("Helvetica", "B", 24)
        pdf.set_text_color(0, 0, 0)
        pdf.set_xy(x_start, pdf.get_y() + 2)
        pdf.cell(70, 12, f"{result.alri_score:.1f}", align="C")
        
        # Risk level label
        pdf.set_font("Helvetica", "", 10)
        pdf.set_xy(x_start, pdf.get_y() + 12)
        pdf.cell(70, 8, self._get_risk_level(result.alri_score), align="C")
        
        pdf.ln(30)
    
    def _add_sub_scores_table(self, pdf: FPDF, result: ALRIResult) -> None:
        """Add sub-scores breakdown table."""
        self._add_section_header(pdf, "Risk Component Breakdown")
        
        # Table header
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(0, 51, 102)
        pdf.set_text_color(255, 255, 255)
        
        col_widths = [60, 40, 40, 50]
        headers = ["Component", "Score", "Weight", "Contribution"]
        
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 8, header, border=1, fill=True, align="C")
        pdf.ln()
        
        # Table rows
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(0, 0, 0)
        
        sub_scores = [
            ("Coverage Risk", result.coverage_risk, 0.30),
            ("Data Instability Risk", result.instability_risk, 0.30),
            ("Biometric Compliance Risk", result.biometric_risk, 0.30),
            ("Anomaly Factor", result.anomaly_factor, 0.10),
        ]
        
        for name, score, weight in sub_scores:
            contribution = score * weight * 100
            color = self._get_score_color(score * 100)
            pdf.set_fill_color(*color)
            
            pdf.cell(col_widths[0], 7, name, border=1, align="L")
            pdf.cell(col_widths[1], 7, f"{score:.3f}", border=1, fill=True, align="C")
            pdf.cell(col_widths[2], 7, f"{weight:.0%}", border=1, align="C")
            pdf.cell(col_widths[3], 7, f"{contribution:.2f}", border=1, align="C")
            pdf.ln()
        
        pdf.ln(5)
    
    def _add_reason_codes_section(self, pdf: FPDF, reason_codes: List[ReasonCode]) -> None:
        """Add reason codes section."""
        self._add_section_header(pdf, "Risk Drivers (Reason Codes)")
        
        if not reason_codes:
            pdf.set_font("Helvetica", "I", 10)
            pdf.set_text_color(128, 128, 128)
            pdf.cell(0, 8, "No significant risk drivers identified.", ln=True)
            pdf.ln(5)
            return
        
        # Table header
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(0, 51, 102)
        pdf.set_text_color(255, 255, 255)
        
        col_widths = [50, 80, 30, 30]
        headers = ["Code", "Description", "Severity", "Contrib."]
        
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 8, header, border=1, fill=True, align="C")
        pdf.ln()
        
        # Table rows
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(0, 0, 0)
        
        for rc in reason_codes:
            severity_color = self.SEVERITY_COLORS.get(rc.severity, (255, 255, 255))
            
            pdf.cell(col_widths[0], 7, rc.code[:25], border=1, align="L")
            
            # Truncate description if too long
            desc = rc.description[:40] + "..." if len(rc.description) > 40 else rc.description
            pdf.cell(col_widths[1], 7, desc, border=1, align="L")
            
            pdf.set_fill_color(*severity_color)
            pdf.cell(col_widths[2], 7, rc.severity.value, border=1, fill=True, align="C")
            pdf.set_fill_color(255, 255, 255)
            
            pdf.cell(col_widths[3], 7, f"{rc.contribution:.1%}", border=1, align="C")
            pdf.ln()
        
        pdf.ln(5)

    def _add_recommendations_section(self, pdf: FPDF, recommendations: List[Intervention]) -> None:
        """Add recommendations section."""
        self._add_section_header(pdf, "Recommended Interventions")
        
        if not recommendations:
            pdf.set_font("Helvetica", "I", 10)
            pdf.set_text_color(128, 128, 128)
            pdf.cell(0, 8, "No interventions recommended at this time.", ln=True)
            pdf.ln(5)
            return
        
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(0, 0, 0)
        
        for i, intervention in enumerate(recommendations[:5], 1):  # Top 5 recommendations
            # Priority badge
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(8, 7, f"{i}.", align="R")
            
            # Action name
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(80, 7, f" {intervention.action}", align="L")
            
            # Cost indicator
            cost_colors = {
                CostLevel.LOW: (144, 238, 144),
                CostLevel.MEDIUM: (255, 255, 153),
                CostLevel.HIGH: (255, 165, 0),
            }
            color = cost_colors.get(intervention.estimated_cost, (200, 200, 200))
            pdf.set_fill_color(*color)
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(25, 7, f"Cost: {intervention.estimated_cost.value}", border=1, fill=True, align="C")
            
            # Impact
            pdf.set_fill_color(255, 255, 255)
            pdf.cell(40, 7, f"Impact: {intervention.estimated_impact:,}", align="C")
            pdf.ln()
            
            # Description
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(80, 80, 80)
            pdf.set_x(18)
            pdf.multi_cell(175, 5, intervention.description)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)
        
        pdf.ln(5)
    
    def _add_trend_visualization(self, pdf: FPDF, trend_data: Dict[str, Any]) -> None:
        """Add simple text-based trend visualization."""
        self._add_section_header(pdf, "Trend Analysis")
        
        if not trend_data:
            pdf.set_font("Helvetica", "I", 10)
            pdf.set_text_color(128, 128, 128)
            pdf.cell(0, 8, "No trend data available.", ln=True)
            pdf.ln(5)
            return
        
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(0, 0, 0)
        
        # Display trend indicators (using ASCII-compatible symbols)
        metrics = ['enrollment', 'demographic', 'biometric']
        trend_symbols = {
            'increasing': '[+]',
            'decreasing': '[-]',
            'stable': '[=]'
        }
        trend_colors = {
            'increasing': (144, 238, 144),  # Green
            'decreasing': (255, 99, 71),    # Red
            'stable': (255, 255, 153)       # Yellow
        }
        
        for metric in metrics:
            if metric in trend_data:
                trend = trend_data[metric].get('trend', 'stable')
                symbol = trend_symbols.get(trend, '[=]')
                color = trend_colors.get(trend, (200, 200, 200))
                
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(60, 8, f"{metric.title()} Trend:", align="L")
                
                pdf.set_fill_color(*color)
                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(35, 8, f"{symbol} {trend.title()}", border=1, fill=True, align="C")
                
                # Forecast values if available
                if 'forecast_values' in trend_data[metric]:
                    values = trend_data[metric]['forecast_values']
                    if values:
                        pdf.set_font("Helvetica", "", 9)
                        pdf.cell(80, 8, f"  Next {len(values)} months avg: {sum(values)/len(values):,.0f}", align="L")
                
                pdf.ln()
        
        pdf.ln(5)
    
    def _add_district_table(self, pdf: FPDF, results: List[ALRIResult], title: str = "District Rankings") -> None:
        """Add a table of districts with ALRI scores."""
        self._add_section_header(pdf, title)
        
        if not results:
            pdf.set_font("Helvetica", "I", 10)
            pdf.set_text_color(128, 128, 128)
            pdf.cell(0, 8, "No district data available.", ln=True)
            return
        
        # Table header
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(0, 51, 102)
        pdf.set_text_color(255, 255, 255)
        
        col_widths = [10, 50, 30, 25, 25, 25, 25]
        headers = ["#", "District", "ALRI", "Cov.", "Inst.", "Bio.", "Anom."]
        
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 8, header, border=1, fill=True, align="C")
        pdf.ln()
        
        # Table rows
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(0, 0, 0)
        
        for rank, result in enumerate(results, 1):
            color = self._get_score_color(result.alri_score)
            
            pdf.cell(col_widths[0], 7, str(rank), border=1, align="C")
            pdf.cell(col_widths[1], 7, result.district[:25], border=1, align="L")
            
            pdf.set_fill_color(*color)
            pdf.cell(col_widths[2], 7, f"{result.alri_score:.1f}", border=1, fill=True, align="C")
            pdf.set_fill_color(255, 255, 255)
            
            pdf.cell(col_widths[3], 7, f"{result.coverage_risk:.2f}", border=1, align="C")
            pdf.cell(col_widths[4], 7, f"{result.instability_risk:.2f}", border=1, align="C")
            pdf.cell(col_widths[5], 7, f"{result.biometric_risk:.2f}", border=1, align="C")
            pdf.cell(col_widths[6], 7, f"{result.anomaly_factor:.2f}", border=1, align="C")
            pdf.ln()
        
        pdf.ln(5)

    def _add_footer(self, pdf: FPDF) -> None:
        """Add report footer."""
        pdf.set_y(-25)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 5, "Aadhaar Sentinel - ALRI Platform | Confidential", align="C", ln=True)
        pdf.cell(0, 5, f"Page {pdf.page_no()}", align="C")
    
    def generate_state_report(
        self,
        state: str,
        alri_results: List[ALRIResult],
        trend_data: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> bytes:
        """
        Generate PDF report with top-10 at-risk districts for a state.
        
        Requirements: 12.1, 12.2, 12.3
        
        Args:
            state: State name for the report
            alri_results: List of ALRIResult objects for districts in the state
            trend_data: Optional dict mapping district names to trend information
            
        Returns:
            PDF document as bytes
        """
        pdf = self._create_pdf()
        
        # Header
        self._add_header(
            pdf,
            f"ALRI State Report: {state.title()}",
            "Top At-Risk Districts Analysis"
        )
        
        # Sort by ALRI score (highest first) and take top 10
        sorted_results = sorted(alri_results, key=lambda r: r.alri_score, reverse=True)
        top_10 = sorted_results[:10]
        
        # Executive Summary
        self._add_section_header(pdf, "Executive Summary")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(0, 0, 0)
        
        total_districts = len(alri_results)
        critical_count = sum(1 for r in alri_results if r.alri_score >= 75)
        high_count = sum(1 for r in alri_results if 50 <= r.alri_score < 75)
        avg_score = sum(r.alri_score for r in alri_results) / total_districts if total_districts > 0 else 0
        
        summary_text = (
            f"This report analyzes {total_districts} districts in {state.title()}. "
            f"The average ALRI score is {avg_score:.1f}. "
            f"There are {critical_count} districts in critical risk and {high_count} in high risk categories. "
            f"Immediate attention is recommended for the top-ranked districts listed below."
        )
        pdf.multi_cell(0, 6, summary_text)
        pdf.ln(5)
        
        # Top 10 Districts Table
        self._add_district_table(pdf, top_10, "Top 10 At-Risk Districts")
        
        # Detailed analysis for top 3 districts
        pdf.add_page()
        self._add_section_header(pdf, "Detailed Analysis - Top 3 Districts")
        
        for i, result in enumerate(top_10[:3], 1):
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(0, 51, 102)
            pdf.cell(0, 10, f"{i}. {result.district.title()}", ln=True)
            
            # Mini score display
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(0, 0, 0)
            color = self._get_score_color(result.alri_score)
            pdf.set_fill_color(*color)
            pdf.cell(40, 7, f"ALRI: {result.alri_score:.1f}", border=1, fill=True, align="C")
            pdf.cell(40, 7, f"Risk: {self._get_risk_level(result.alri_score)}", align="L")
            pdf.ln(10)
            
            # Sub-scores summary
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(0, 5, f"Coverage: {result.coverage_risk:.2f} | Instability: {result.instability_risk:.2f} | Biometric: {result.biometric_risk:.2f} | Anomaly: {result.anomaly_factor:.2f}", ln=True)
            
            # Reason codes if available
            if result.reason_codes:
                pdf.set_font("Helvetica", "I", 9)
                codes = [rc.code if hasattr(rc, 'code') else str(rc) for rc in result.reason_codes[:3]]
                pdf.cell(0, 5, f"Key Drivers: {', '.join(codes)}", ln=True)
            
            # Trend if available
            if trend_data and result.district in trend_data:
                district_trend = trend_data[result.district]
                trends = []
                for metric in ['enrollment', 'demographic', 'biometric']:
                    if metric in district_trend:
                        trend = district_trend[metric].get('trend', 'stable')
                        trends.append(f"{metric}: {trend}")
                if trends:
                    pdf.cell(0, 5, f"Trends: {' | '.join(trends)}", ln=True)
            
            pdf.ln(5)
        
        # Recommendations summary
        pdf.add_page()
        self._add_section_header(pdf, "Recommended Actions Summary")
        
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(0, 0, 0)
        
        # Aggregate recommendations from top districts
        all_recommendations = []
        for result in top_10[:5]:
            if result.recommendations:
                all_recommendations.extend(result.recommendations)
        
        if all_recommendations:
            # Deduplicate by action name
            seen = set()
            unique_recs = []
            for rec in all_recommendations:
                action = rec.action if hasattr(rec, 'action') else str(rec)
                if action not in seen:
                    seen.add(action)
                    unique_recs.append(rec)
            
            self._add_recommendations_section(pdf, unique_recs[:5])
        else:
            pdf.cell(0, 8, "Generate reason codes and recommendations for detailed action items.", ln=True)
        
        # Footer
        self._add_footer(pdf)
        
        return bytes(pdf.output())

    def generate_district_report(
        self,
        district: str,
        alri_result: ALRIResult,
        trend_data: Optional[Dict[str, Any]] = None,
        historical_scores: Optional[List[float]] = None
    ) -> bytes:
        """
        Generate detailed PDF report for a single district.
        
        Requirements: 12.2, 12.3, 12.4
        
        Args:
            district: District name
            alri_result: ALRIResult object with scores and analysis
            trend_data: Optional dict with trend information for each metric
            historical_scores: Optional list of historical ALRI scores
            
        Returns:
            PDF document as bytes
        """
        pdf = self._create_pdf()
        
        # Header
        self._add_header(
            pdf,
            f"ALRI District Report: {district.title()}",
            f"State: {alri_result.state.title()}"
        )
        
        # ALRI Score Box
        self._add_section_header(pdf, "Overall ALRI Score")
        self._add_alri_score_box(pdf, alri_result)
        
        # Sub-scores breakdown
        self._add_sub_scores_table(pdf, alri_result)
        
        # Reason codes
        reason_codes = []
        if alri_result.reason_codes:
            for rc in alri_result.reason_codes:
                if isinstance(rc, ReasonCode):
                    reason_codes.append(rc)
                elif isinstance(rc, dict):
                    # Handle dict format
                    reason_codes.append(ReasonCode(
                        code=rc.get('code', 'Unknown'),
                        description=rc.get('description', ''),
                        severity=Severity(rc.get('severity', 'Low')),
                        contribution=rc.get('contribution', 0.0)
                    ))
        
        self._add_reason_codes_section(pdf, reason_codes)
        
        # Recommendations
        recommendations = []
        if alri_result.recommendations:
            for rec in alri_result.recommendations:
                if isinstance(rec, Intervention):
                    recommendations.append(rec)
                elif isinstance(rec, dict):
                    # Handle dict format
                    recommendations.append(Intervention(
                        action=rec.get('action', 'Unknown'),
                        description=rec.get('description', ''),
                        estimated_cost=CostLevel(rec.get('estimated_cost', 'Medium')),
                        estimated_impact=rec.get('estimated_impact', 0),
                        priority=rec.get('priority', 1)
                    ))
        
        self._add_recommendations_section(pdf, recommendations)
        
        # New page for trends
        pdf.add_page()
        
        # Trend visualization
        self._add_trend_visualization(pdf, trend_data or {})
        
        # Historical ALRI scores
        if historical_scores:
            self._add_section_header(pdf, "Historical ALRI Scores")
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(0, 0, 0)
            
            # Simple text-based visualization
            pdf.cell(0, 6, f"Last {len(historical_scores)} periods:", ln=True)
            
            for i, score in enumerate(historical_scores[-12:], 1):  # Last 12 periods max
                color = self._get_score_color(score)
                pdf.set_fill_color(*color)
                
                # Draw bar
                bar_width = score * 1.5  # Scale to fit
                pdf.cell(20, 6, f"T-{len(historical_scores)-i+1}:", align="R")
                pdf.cell(bar_width, 6, "", fill=True)
                pdf.cell(20, 6, f" {score:.1f}", align="L")
                pdf.ln()
            
            pdf.ln(5)
        
        # Key metrics summary
        self._add_section_header(pdf, "Key Metrics Summary")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(0, 0, 0)
        
        metrics = [
            ("Coverage Risk Score", f"{alri_result.coverage_risk:.3f}", "Measures enrollment coverage gaps"),
            ("Data Instability Score", f"{alri_result.instability_risk:.3f}", "Measures demographic update frequency"),
            ("Biometric Compliance Score", f"{alri_result.biometric_risk:.3f}", "Measures biometric update rates"),
            ("Anomaly Factor", f"{alri_result.anomaly_factor:.3f}", "Detects unusual data patterns"),
        ]
        
        for name, value, description in metrics:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(70, 6, name + ":", align="L")
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(30, 6, value, align="L")
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(90, 6, description, align="L")
            pdf.set_text_color(0, 0, 0)
            pdf.ln()
        
        pdf.ln(5)
        
        # Computation timestamp
        self._add_section_header(pdf, "Report Information")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(100, 100, 100)
        
        computed_at = alri_result.computed_at
        if isinstance(computed_at, datetime):
            computed_str = computed_at.strftime("%Y-%m-%d %H:%M:%S")
        else:
            computed_str = str(computed_at)
        
        pdf.cell(0, 5, f"ALRI Computed At: {computed_str}", ln=True)
        pdf.cell(0, 5, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.cell(0, 5, "Data Source: Aadhaar Sentinel Platform", ln=True)
        
        # Footer
        self._add_footer(pdf)
        
        return bytes(pdf.output())
    
    def generate_summary_report(
        self,
        all_results: List[ALRIResult],
        title: str = "ALRI National Summary Report"
    ) -> bytes:
        """
        Generate a summary report across all states/districts.
        
        Args:
            all_results: List of all ALRIResult objects
            title: Report title
            
        Returns:
            PDF document as bytes
        """
        pdf = self._create_pdf()
        
        # Header
        self._add_header(pdf, title, "Aadhaar Lifecycle Risk Index Overview")
        
        # National statistics
        self._add_section_header(pdf, "National Overview")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(0, 0, 0)
        
        if all_results:
            total = len(all_results)
            avg_score = sum(r.alri_score for r in all_results) / total
            max_score = max(r.alri_score for r in all_results)
            min_score = min(r.alri_score for r in all_results)
            
            critical = sum(1 for r in all_results if r.alri_score >= 75)
            high = sum(1 for r in all_results if 50 <= r.alri_score < 75)
            medium = sum(1 for r in all_results if 25 <= r.alri_score < 50)
            low = sum(1 for r in all_results if r.alri_score < 25)
            
            stats = [
                ("Total Districts Analyzed", str(total)),
                ("Average ALRI Score", f"{avg_score:.1f}"),
                ("Highest ALRI Score", f"{max_score:.1f}"),
                ("Lowest ALRI Score", f"{min_score:.1f}"),
                ("Critical Risk Districts", f"{critical} ({critical/total*100:.1f}%)"),
                ("High Risk Districts", f"{high} ({high/total*100:.1f}%)"),
                ("Medium Risk Districts", f"{medium} ({medium/total*100:.1f}%)"),
                ("Low Risk Districts", f"{low} ({low/total*100:.1f}%)"),
            ]
            
            for label, value in stats:
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(80, 7, label + ":", align="L")
                pdf.set_font("Helvetica", "", 10)
                pdf.cell(50, 7, value, align="L")
                pdf.ln()
            
            pdf.ln(5)
            
            # Top 10 at-risk districts nationally
            sorted_results = sorted(all_results, key=lambda r: r.alri_score, reverse=True)
            self._add_district_table(pdf, sorted_results[:10], "Top 10 At-Risk Districts (National)")
        else:
            pdf.cell(0, 8, "No district data available for analysis.", ln=True)
        
        # Footer
        self._add_footer(pdf)
        
        return bytes(pdf.output())
