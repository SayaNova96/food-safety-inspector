import os
import cv2
import numpy as np
import torch
from datetime import datetime
from torchvision import transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

from multitask_food_model import MultiTaskFoodCNN
from multitask_gradcam import MultiTaskGradCAM, overlay_heatmap
from audit_logger import generate_pdf_report

EVIDENCE_DIR = "evidence"
os.makedirs(EVIDENCE_DIR, exist_ok=True)

CUISINE_MAP = {
    0: "Indian", 1: "American", 2: "British", 
    3: "French", 4: "Italian", 5: "Chinese", 6: "Thai"
}
FRESHNESS_MAP = {0: "Fresh / Fit", 1: "Spoiled / Unfit"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Multi-Task Model
model = MultiTaskFoodCNN(num_cuisines=len(CUISINE_MAP), num_freshness=len(FRESHNESS_MAP))
try:
    model.load_state_dict(torch.load("multitask_food_weights.pth", map_location=device))
except FileNotFoundError:
    pass

model.to(device)
model.eval()

# Grad-CAM Engine hooked into final feature layer
cam_engine = MultiTaskGradCAM(model, model.features[-1])

tensor_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ----------------- FOOD DETECTION GATE -----------------
def is_food(frame, threshold=0.65):
    """
    Evaluates whether the frame contains a valid food object before running
    full multi-task inference and Grad-CAM backward propagation.
    
    Production implementation:
      - Validates minimum color/texture variance (rejects flat walls, ceilings, blank desks)
      - Computes maximum prediction probability/entropy from the backbone features
    """
    if frame is None:
        return False
    
    # Heuristic Check 1: Contrast / Variance Check (rejects blank walls or covered lenses)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < 20.0:  # Image is too blurry, dark, or featureless
        return False

    # Heuristic Check 2: Pre-screen using backbone feature confidence
    # If the network's top confidence is completely uncertain across all heads,
    # the object is likely out-of-distribution (non-food).
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    temp_tensor = tensor_transform(rgb_frame).unsqueeze(0).to(device)
    with torch.no_grad():
        logits_c, logits_f = model(temp_tensor)
        prob_c = torch.softmax(logits_c, dim=1).max().item()
        prob_f = torch.softmax(logits_f, dim=1).max().item()
    
    # Combined confidence heuristic (can be replaced by a dedicated Food vs Non-Food binary classifier)
    confidence_score = (prob_c + prob_f) / 2.0
    return confidence_score >= threshold
# --------------------------------------------------------

class MultiTaskInspectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Task Food Safety Field Inspection Tool")
        self.cap = cv2.VideoCapture(0)

        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky="NSEW")

        # Top Control Row
        top_bar = ttk.Frame(main_frame)
        top_bar.grid(row=0, column=0, columnspan=2, pady=5, sticky="W")

        ttk.Label(top_bar, text="Officer ID:").pack(side="left", padx=5)
        self.entry_officer = ttk.Entry(top_bar, width=14)
        self.entry_officer.insert(0, "OFFICER-704")
        self.entry_officer.pack(side="left", padx=5)

        # Viewports: Live Camera and Grad-CAM Overlay
        self.panel_live = ttk.Label(main_frame)
        self.panel_live.grid(row=1, column=0, padx=5, pady=5)

        self.panel_cam = ttk.Label(main_frame)
        self.panel_cam.grid(row=1, column=1, padx=5, pady=5)

        # Grad-CAM Head Selection
        focus_box = ttk.LabelFrame(main_frame, text="Grad-CAM Focus", padding="5")
        focus_box.grid(row=2, column=0, columnspan=2, pady=5, sticky="EW")

        self.task_var = tk.StringVar(value="freshness")
        ttk.Radiobutton(focus_box, text="Spoilage Focus (Freshness Head)", variable=self.task_var, 
                        value="freshness", command=self.update_cam_overlay).pack(side="left", padx=15)
        ttk.Radiobutton(focus_box, text="Dish Focus (Cuisine Head)", variable=self.task_var, 
                        value="cuisine", command=self.update_cam_overlay).pack(side="left", padx=15)

        # Status Display
        self.lbl_status = tk.Label(
            main_frame, text="Ready for Inspection", font=("Arial", 11, "bold"),
            bg="#263238", fg="#FFFFFF", padx=8, pady=6
        )
        self.lbl_status.grid(row=3, column=0, columnspan=2, sticky="EW", pady=6)

        # Action Buttons
        btn_box = ttk.Frame(main_frame)
        btn_box.grid(row=4, column=0, columnspan=2, pady=5)

        self.btn_audit = ttk.Button(btn_box, text="Capture & Audit Frame", command=self.audit_frame)
        self.btn_audit.pack(side="left", padx=8)

        self.btn_pdf = ttk.Button(btn_box, text="Export Audit Log to PDF", command=self.export_pdf, state="disabled")
        self.btn_pdf.pack(side="left", padx=8)

        self.current_frame = None
        self.frozen_tensor = None
        self.latest_results = {}
        self.last_blended = None
        self.last_raw = None

        self.update_feed()

    def update_feed(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame
            disp = cv2.resize(frame, (360, 270))
            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.panel_live.configure(image=img)
            self.panel_live.image = img
            
        self.root.after(30, self.update_feed)

    def audit_frame(self):
        if self.current_frame is None:
            return

        # ------------------- FOOD VERIFICATION GATE -------------------
        if not is_food(self.current_frame, threshold=0.50):
            self.lbl_status.config(
                text="REJECTED: Target is not food or image is unreadable. Inspection aborted.",
                bg="#C62828"
            )
            # Disable PDF generation and clear previous heatmap
            self.btn_pdf.config(state="disabled")
            blank = np.zeros((270, 360, 3), dtype=np.uint8)
            img_blank = ImageTk.PhotoImage(Image.fromarray(blank))
            self.panel_cam.configure(image=img_blank)
            self.panel_cam.image = img_blank
            return
        # -------------------------------------------------------------

        # Proceed only when verified as food
        self.last_raw = self.current_frame.copy()
        rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        self.frozen_tensor = tensor_transform(rgb_frame).unsqueeze(0).to(device)
        
        self.update_cam_overlay()
        self.btn_pdf.config(state="normal")

    def update_cam_overlay(self):
        if self.frozen_tensor is None:
            return

        selected_task = self.task_var.get()

        with torch.enable_grad():
            cam_map, target_class, (logits_c, logits_f) = cam_engine.generate_heatmap(
                self.frozen_tensor, task=selected_task
            )

        prob_c = torch.softmax(logits_c, dim=1)[0].detach().cpu().numpy()
        prob_f = torch.softmax(logits_f, dim=1)[0].detach().cpu().numpy()

        pred_c_idx = int(np.argmax(prob_c))
        pred_f_idx = int(np.argmax(prob_f))

        pred_cuisine = CUISINE_MAP[pred_c_idx]
        pred_fresh = FRESHNESS_MAP[pred_f_idx]
        conf_fresh = prob_f[pred_f_idx] * 100

        self.latest_results = {
            "cuisine": pred_cuisine,
            "verdict": pred_fresh,
            "confidence": conf_fresh
        }

        self.last_blended = overlay_heatmap(self.last_raw, cam_map, alpha=0.45)
        disp_cam = cv2.resize(self.last_blended, (360, 270))
        cam_rgb = cv2.cvtColor(disp_cam, cv2.COLOR_BGR2RGB)
        img_cam = ImageTk.PhotoImage(Image.fromarray(cam_rgb))
        self.panel_cam.configure(image=img_cam)
        self.panel_cam.image = img_cam

        bg_color = "#2E7D32" if pred_f_idx == 0 else "#C62828"
        self.lbl_status.config(
            text=f"Detected: {pred_cuisine} | Status: {pred_fresh} ({conf_fresh:.1f}%)",
            bg=bg_color
        )

    def export_pdf(self):
        if self.last_raw is None or self.last_blended is None:
            return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = os.path.join(EVIDENCE_DIR, f"raw_{stamp}.jpg")
        cam_path = os.path.join(EVIDENCE_DIR, f"cam_{stamp}.jpg")
        cv2.imwrite(raw_path, self.last_raw)
        cv2.imwrite(cam_path, self.last_blended)

        officer = self.entry_officer.get().strip() or "OFFICER-704"
        pdf_path = generate_pdf_report(
            raw_path, cam_path, 
            self.latest_results["verdict"], 
            self.latest_results["confidence"],
            inspector_id=officer, 
            cuisine=self.latest_results["cuisine"]
        )
        messagebox.showinfo("Report Saved", f"Audit PDF generated:\n{pdf_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MultiTaskInspectorApp(root)
    root.mainloop()
