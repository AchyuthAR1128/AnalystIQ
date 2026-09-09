"""PDF report generation for the management-ready export."""

from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _table(data, widths=None):
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def _chart_image(fig):
    """Convert a Plotly chart to an in-memory image using Kaleido."""
    png = fig.to_image(format="png", width=1200, height=650, scale=1)
    image = Image(BytesIO(png), width=6.8 * inch, height=3.68 * inch)
    return image


def build_pdf_report(analysis: dict, charts: list, summary: str, segmentation_result: dict | None = None) -> bytes:
    """Build a concise report from existing dashboard state without writing files."""
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("ReportTitle", parent=styles["Title"], textColor=colors.HexColor("#1E3A8A"), alignment=TA_CENTER)
    heading = ParagraphStyle("ReportHeading", parent=styles["Heading2"], textColor=colors.HexColor("#2563EB"), spaceBefore=12)
    story = [Paragraph("AnalystIQ — AI-Powered Data Analytics Copilot", title), Paragraph("Management Summary", styles["Heading3"]), Spacer(1, 10)]

    report, quality, kpis = analysis["quality_report"], analysis["quality"], analysis["kpis"]
    story += [Paragraph("Dataset and data quality", heading)]
    story.append(_table([
        ["Metric", "Value"],
        ["Source file", analysis["name"]],
        ["Original rows / cleaned rows", f"{report['original_rows']:,} / {report['cleaned_rows']:,}"],
        ["Missing values (before / after)", f"{report['original_missing']:,} / {report['cleaned_missing']:,}"],
        ["Duplicate rows detected", f"{quality['duplicate_rows']:,}"],
        ["Potential outliers", f"{sum(quality['outlier_counts'].values()):,}"],
    ], [2.8 * inch, 3.8 * inch]))

    story += [Paragraph("Key business KPIs", heading)]
    story.append(_table([
        ["Revenue", "Profit", "Orders", "Average order value", "Profit margin"],
        [f"₹{kpis['total_revenue']:,.0f}", f"₹{kpis['total_profit']:,.0f}", f"{kpis['total_orders']:,}", f"₹{kpis['average_order_value']:,.0f}", f"{kpis['profit_margin']:.2f}%"],
    ], [1.3 * inch] * 5))

    story += [Paragraph("Business insights", heading), Paragraph(summary.replace("\n", "<br/>"), styles["BodyText"])]
    chat_insights = [item.get("insight") for item in analysis.get("chat_history", []) if item.get("insight")][-3:]
    if chat_insights:
        story.append(Spacer(1, 6))
        story.append(Paragraph("Recent AI findings", styles["Heading3"]))
        for insight in chat_insights:
            story.append(Paragraph(f"• {insight}", styles["BodyText"]))

    if segmentation_result:
        story += [Paragraph("Segmentation summary", heading)]
        profile = segmentation_result["profile"].reset_index()
        headers = profile.columns.tolist()
        values = [[str(value) for value in row] for row in profile.head(8).itertuples(index=False, name=None)]
        story.append(_table([headers] + values))
        story.append(Paragraph("Illustrative K-means segmentation: clusters are exploratory and should not be used alone for operational decisions.", styles["BodyText"]))

    valid_charts = [chart for chart in charts if chart is not None][:3]
    if valid_charts:
        story += [Paragraph("Selected visualizations", heading)]
        for chart in valid_charts:
            try:
                story += [_chart_image(chart), Spacer(1, 8)]
            except Exception as error:
                story.append(Paragraph(f"Chart image unavailable in this export: {error}", styles["BodyText"]))

    document.build(story)
    return buffer.getvalue()
