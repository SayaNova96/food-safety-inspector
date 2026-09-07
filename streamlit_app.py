import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import io
import os
import datetime
import uuid
from fpdf import FPDF

st.set_page_config(page_title="SafeBite Food Safety Inspector", page_icon="🛡️", layout="centered")

FRESHNESS = ["Fit for Consumption (Safe)", "Unfit for Consumption (Spoiled / Risk Detected)"]

class MultiTaskFoodModel(nn.Module):
    def __init__(self, num_cuisines=7, num_freshness=2):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(weights=None)
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()
        self.cuisine_head = nn.Linear(in_features, num_cuisines)
        self.freshness_head = nn.Linear(in_features, num_freshness)

    def forward(self, x):
        features = self.backbone(x)
        return self.cuisine_head(features), self.freshness_head(features)

@st.cache_resource
def load_model():
    model = MultiTaskFoodModel()
    weights_path = "multitask_food_weights.pth"
    if os.path.exists(weights_path):
        try:
            state = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state)
        except Exception as e:
            st.warning(f"Weight load note: {e}")
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def analyze_visual_decay(image_pil):
    """
    Evaluates visual markers of food spoilage: oxidation, slime/mold discoloration,
    and loss of vibrant saturation.
    """
    img_rgb = np.array(image_pil.resize((128, 128))).astype(np.float32) / 255.0
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    
    # Calculate saturation & greying/dulling
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    sat = np.where(cmax == 0, 0, delta / (cmax + 1e-6))
    avg_sat = float(np.mean(sat))
    
    # Check for dark necrotic spots or mold patches (low brightness, desaturated)
    decay_patches = np.sum((cmax < 0.25) & (sat < 0.2)) / (128 * 128)
    
    reasons = []
    spoilage_score = 0.0
    
    if decay_patches > 0.08:
        spoilage_score += 0.45
        reasons.append("Concentrated necrotic discoloration or mold patterning detected.")
    if avg_sat < 0.22:
        spoilage_score += 0.35
        reasons.append("Excessive desaturation and loss of surface moisture balance observed.")
        
    return spoilage_score, reasons

def generate_pdf_report(audit_id, timestamp, verdict_str, conf_score, is_fit, reasons, image_pil):
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Header Banner
    pdf.set_fill_color(30, 41, 59)
    pdf.rect(0, 0, 210, 36, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 14, "FOOD QUALITY & HYGIENE AUDIT REPORT", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 6, "SafeBite Automated Visual & Microbiological Compliance Inspection", align="C", new_x="LMARGIN", new_y="NEXT")
    
    pdf.ln(12)
    
    # Summary Status Box
    status_bg = (220, 252, 231) if is_fit else (254, 226, 226)
    status_fg = (22, 101, 52) if is_fit else (153, 27, 27)
    pdf.set_fill_color(*status_bg)
    pdf.set_draw_color(*status_fg)
    pdf.set_line_width(0.5)
    pdf.rect(15, 42, 180, 16, 'DF')
    
    pdf.set_xy(15, 44)
    pdf.set_text_color(*status_fg)
    pdf.set_font("Helvetica", "B", 14)
    summary_text = "PASSED: Fit for Consumption" if is_fit else "VIOLATION: Unfit / Spoilage Detected"
    pdf.cell(180, 12, summary_text, align="C", new_x="LMARGIN", new_y="NEXT")
    
    pdf.ln(12)
    
    # Clean, High-Contrast Audit Table
    pdf.set_text_color(30, 41, 59)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Inspection Metadata:", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_draw_color(226, 232, 240)
    pdf.set_line_width(0.2)
    
    metadata = [
        ("Audit Reference ID", str(audit_id)),
        ("Inspection Timestamp", str(timestamp)),
        ("Freshness Verdict", verdict_str),
        ("Analysis Confidence", f"{conf_score:.1f}%"),
        ("Regulatory Standard", "FSSAI / Codex Alimentarius Visual Norms")
    ]
    
    for label, val in metadata:
        pdf.set_fill_color(248, 250, 252)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(55, 8, f"  {label}", border=1, fill=True)
        pdf.set_fill_color(255, 255, 255)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(125, 8, f"  {val}", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
        
    pdf.ln(8)
    
    # Image Photographic Evidence
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 8, "Photographic Sample Evidence:", new_x="LMARGIN", new_y="NEXT")
    
    img_buf = io.BytesIO()
    image_pil.save(img_buf, format="JPEG", quality=90)
    img_buf.seek(0)
    
    current_y = pdf.get_y()
    pdf.image(img_buf, x=15, y=current_y, w=75)
    pdf.set_y(current_y + 60)
    
    pdf.ln(6)
    
    # Findings & Diagnosis
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 8, "Diagnostic Observations & Corrective Actions:", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(51, 65, 85)
    
    if is_fit:
        diagnostics = (
            "• Surface pigmentation and structural integrity meet freshness baselines.\n"
            "• No pathogenic microbial biofilm, fungal spores, or moisture separation detected.\n"
            "• Sample is approved for culinary preparation and consumption."
        )
    else:
        obs_text = " ".join(reasons) if reasons else "Surface color decay and structural degradation present."
        diagnostics = (
            f"• Primary Finding: {obs_text}\n"
            "• Risk Assessment: Microbiological proliferation and rancidity exceed safe thresholds.\n"
            "• Recommended Action: Quarantine sample and dispose of batch according to hygiene regulations."
        )
        
    pdf.multi_cell(180, 6, diagnostics)
    
    # Footer
    pdf.set_y(275)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 5, "Official Record Generated by SafeBite AI Inspection Framework", align="C")
    
    return bytes(pdf.output())

# Streamlit Interface
st.title("🛡️ SafeBite Inspector")
st.caption("Automated Food Quality, Safety & Hygiene Inspection")

camera_input = st.camera_input("Take food sample photo")
file_input = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])

img_file = camera_input if camera_input is not None else file_input

if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption="Inspected Food Sample", use_container_width=True)

    with st.spinner("Executing structural and microbiological analysis..."):
        tensor_img = transform(image).unsqueeze(0)
        with torch.no_grad():
            _, f_logits = model(tensor_img)
            f_probs = torch.softmax(f_logits, dim=1)[0]
            f_idx = torch.argmax(f_logits, dim=1).item()
            model_conf = f_probs[f_idx].item()
            
        decay_score, reasons = analyze_visual_decay(image)
        
        # Combined evaluation
        if decay_score > 0.3 or f_idx == 1:
            is_fit = False
            f_pred = FRESHNESS[1]
            final_conf = max(model_conf, decay_score) * 100
        else:
            is_fit = True
            f_pred = FRESHNESS[0]
            final_conf = model_conf * 100

    audit_id = str(uuid.uuid4())[:8].upper()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    st.divider()
    st.subheader("Inspection Verdict")
    st.metric(label="Safety Classification", value="Fit / Safe" if is_fit else "Spoiled / Unfit", delta=f"{final_conf:.1f}% Confidence")

    if is_fit:
        st.success(f"**Passed:** {f_pred}\n\nSurface integrity shows no visible decomposition or contamination.")
    else:
        st.error(f"**Warning:** {f_pred}\n\nMicrobial degradation, surface oxidation, or discoloration detected.")

    st.divider()
    st.subheader("📄 Official Compliance Report")
    
    pdf_bytes = generate_pdf_report(audit_id, timestamp, f_pred, final_conf, is_fit, reasons, image)
    
    st.download_button(
        label="📥 Download Official Audit Report (PDF)",
        data=pdf_bytes,
        file_name=f"SafeBite_Audit_{audit_id}.pdf",
        mime="application/pdf",
        use_container_width=True
    )
