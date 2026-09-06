import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

def generate_pdf_report(raw_img_path, cam_img_path, verdict, confidence, inspector_id="OFFICER-704", cuisine="Indian"):
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = os.path.join(REPORTS_DIR, f"audit_report_{file_id}.pdf")

    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1A237E")
    )
    story.append(Paragraph("Food Safety & Quality Inspection Record", title_style))
    story.append(Paragraph("<b>System:</b> Edge Deep-CNN Optical Screener (Offline Mode) | <b>Standard:</b> FSSAI Hygiene Audit", styles['Normal']))
    story.append(Spacer(1, 12))

    verdict_color = colors.HexColor("#2E7D32") if "Fresh" in verdict else colors.HexColor("#C62828")
    
    meta_data = [
        [Paragraph("<b>Audit ID:</b>", styles['Normal']), Paragraph(f"AUD-{file_id}", styles['Normal']),
         Paragraph("<b>Date/Time:</b>", styles['Normal']), Paragraph(timestamp_str, styles['Normal'])],
        [Paragraph("<b>Inspector ID:</b>", styles['Normal']), Paragraph(inspector_id, styles['Normal']),
         Paragraph("<b>Detected Cuisine:</b>", styles['Normal']), Paragraph(cuisine, styles['Normal'])],
        [Paragraph("<b>Final Verdict:</b>", styles['Normal']), 
         Paragraph(f"<font color='{verdict_color.hexval()}'><b>{verdict.upper()}</b></font>", styles['Normal']),
         Paragraph("<b>Confidence:</b>", styles['Normal']), Paragraph(f"<b>{confidence:.2f}%</b>", styles['Normal'])]
    ]

    meta_table = Table(meta_data, colWidths=[100, 170, 100, 170])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F5F5F5")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 15))

    story.append(Paragraph("<b>Visual Evidence & Grad-CAM Heatmap Localization</b>", styles['Heading3']))
    
    evidence_table = Table([
        [Paragraph("<b>Captured Frame</b>", styles['Normal']), Paragraph("<b>Grad-CAM Activation</b>", styles['Normal'])],
        [RLImage(raw_img_path, width=250, height=188),
         RLImage(cam_img_path, width=250, height=188)]
    ], colWidths=[270, 270])

    evidence_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
    ]))
    story.append(evidence_table)
    story.append(Spacer(1, 15))

    findings_text = (
        "<b>Notice:</b> Grad-CAM highlights specific pixel clusters responsible for the model's output. "
        "Warm colors (red/yellow) indicate focal areas of microbial or surface degradation."
    )
    story.append(Paragraph(findings_text, styles['Normal']))

    doc.build(story)
    return pdf_filename
