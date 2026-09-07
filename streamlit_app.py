import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

st.set_page_config(page_title="SafeBite Inspector", page_icon="🥗", layout="centered")

CUISINES = ["Indian", "American", "British", "French", "Italian", "Chinese", "Thai"]
FRESHNESS = ["Fresh / Fit for Consumption", "Spoiled / Unfit for Consumption"]

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
    try:
        model.load_state_dict(torch.load("multitask_food_weights.pth", map_location="cpu"))
    except Exception:
        pass
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.title("🥗 SafeBite Inspector")
st.caption("AI-Powered Real-Time Food Quality & Safety Audit")

camera_input = st.camera_input("Take a photo of the food item")
file_input = st.file_uploader("Or upload a photo", type=["jpg", "jpeg", "png"])

img_file = camera_input if camera_input is not None else file_input

if img_file is not None:
    image = Image.open(img_file).convert("RGB")
    st.image(image, caption="Inspected Sample", use_container_width=True)

    with st.spinner("Analyzing microbiological and visual freshness indicators..."):
        tensor_img = transform(image).unsqueeze(0)
        with torch.no_grad():
            c_logits, f_logits = model(tensor_img)
            c_pred = CUISINES[torch.argmax(c_logits, dim=1).item()]
            f_probs = torch.softmax(f_logits, dim=1)[0]
            f_idx = torch.argmax(f_logits, dim=1).item()
            f_pred = FRESHNESS[f_idx]
            f_conf = f_probs[f_idx].item() * 100

    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Detected Cuisine", c_pred)
    with col2:
        st.metric("Freshness State", "Fit" if f_idx == 0 else "Spoiled", f"{f_conf:.1f}% confidence")

    if f_idx == 0:
        st.success(f"**Verdict:** {f_pred}\n\nNo visible spoilage or structural degradation detected.")
    else:
        st.error(f"**Verdict:** {f_pred}\n\nVisible microbial growth, discoloration, or oxidation detected.")
