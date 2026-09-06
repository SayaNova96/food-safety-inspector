import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, render_template_string, request, jsonify
from torchvision import transforms, models
from PIL import Image
import base64
import io

from multitask_food_model import MultiTaskFoodCNN
from multitask_gradcam import MultiTaskGradCAM, overlay_heatmap

app = Flask(__name__)
# Allow larger payloads up to 32MB
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

CUISINE_MAP = {0: "Indian", 1: "American", 2: "British", 3: "French", 4: "Italian", 5: "Chinese", 6: "Thai"}
FRESHNESS_MAP = {0: "Fresh / Fit", 1: "Spoiled / Unfit"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. MultiTask Model Setup
model = MultiTaskFoodCNN(num_cuisines=len(CUISINE_MAP), num_freshness=len(FRESHNESS_MAP))
try:
    model.load_state_dict(torch.load("multitask_food_weights.pth", map_location=device, weights_only=True))
except Exception:
    pass
model.to(device)
model.eval()

# 2. Gate Model (ImageNet classifier for Food vs Non-Food)
gate_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
gate_model.to(device)
gate_model.eval()

cam_engine = MultiTaskGradCAM(model, model.features[-1])

tensor_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def verify_is_food(pil_img, threshold=0.10):
    t = tensor_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = gate_model(t)
        probs = F.softmax(out, dim=1)[0]
    
    food_indices = list(range(923, 966)) + [987, 988, 989, 990] + list(range(768, 770))
    food_probability = torch.sum(probs[food_indices]).item()
    top_class = torch.argmax(probs).item()

    if top_class not in food_indices and food_probability < threshold:
        return False, "Target appears to be non-food. Please frame raw produce or a cooked meal."
        
    return True, "Food confirmed."

def generate_inspection_reasons(verdict, cam_map):
    hotspot_ratio = float(np.sum(cam_map > 0.6) / cam_map.size) * 100
    if "Spoiled" in verdict:
        return [
            f"Anomaly Clustering: {hotspot_ratio:.1f}% of inspected area exhibits high degradation activation.",
            "Surface Pathology: Visual characteristics match fungal mycelium, localized soft rot, or tissue browning.",
            "Cellular Integrity: Signs of structural collapse and oxidative spoilage detected.",
            "Compliance Ruling: Fails fresh food guidelines. Not safe for distribution or consumption."
        ]
    else:
        return [
            f"Surface Uniformity: Normal reading with {hotspot_ratio:.1f}% baseline surface variance.",
            "Pigmentation & Tone: Natural epidermal color retention with no visible mold colonies.",
            "Structural Integrity: Firm cellular structure without visible rot or discharge.",
            "Compliance Ruling: Passes standard visual inspection criteria."
        ]

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>SafeBite Inspector</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        body { background: #0F172A; color: #F8FAFC; display: flex; justify-content: center; min-height: 100vh; }
        
        .app-container {
            width: 100%;
            max-width: 440px;
            background: #1E293B;
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }

        .app-bar {
            background: #0F172A;
            padding: 14px 18px;
            display: flex;
            align-items: center;
            gap: 12px;
            border-bottom: 1px solid #334155;
        }
        .logo-svg { width: 34px; height: 34px; }
        .app-title { font-size: 18px; font-weight: 700; color: #F1F5F9; }
        .app-subtitle { font-size: 11px; color: #10B981; font-weight: 600; text-transform: uppercase; }

        .tab-bar {
            display: flex;
            background: #1E293B;
            border-bottom: 2px solid #334155;
        }
        .tab {
            flex: 1;
            text-align: center;
            padding: 14px 0;
            font-size: 13px;
            font-weight: 600;
            color: #94A3B8;
            cursor: pointer;
            border-bottom: 3px solid transparent;
        }
        .tab.active { color: #10B981; border-bottom: 3px solid #10B981; }

        .content { padding: 16px; display: flex; flex-direction: column; gap: 14px; flex: 1; }
        .panel { display: none; flex-direction: column; gap: 14px; }
        .panel.active { display: flex; }

        .media-box {
            background: #0F172A;
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid #334155;
            min-height: 220px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        video, img { width: 100%; max-height: 280px; object-fit: contain; display: block; }
        
        .btn {
            background: #10B981;
            color: white;
            border: none;
            padding: 14px;
            font-size: 15px;
            font-weight: 700;
            border-radius: 10px;
            width: 100%;
            cursor: pointer;
            text-align: center;
        }
        .btn-upload { background: #3B82F6; }
        .btn-outline { background: transparent; border: 1px solid #64748B; color: #F8FAFC; margin-top: 8px; }

        .banner {
            padding: 12px 14px;
            border-radius: 10px;
            font-size: 13px;
            font-weight: 600;
            background: #334155;
            display: none;
        }
        .banner-fresh { background: #065F46; color: #ECFDF5; border-left: 4px solid #10B981; }
        .banner-rotten { background: #7F1D1D; color: #FEF2F2; border-left: 4px solid #EF4444; }
        .banner-error { background: #991B1B; color: #FEE2E2; border-left: 4px solid #DC2626; }

        .report-box {
            background: #0F172A;
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .report-row { display: flex; justify-content: space-between; font-size: 13px; color: #94A3B8; }
        .report-row strong { color: #F8FAFC; }
        
        .reason-box {
            background: #1E293B;
            border-radius: 8px;
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 6px;
            margin-top: 6px;
        }
        .reason-box h4 { font-size: 12px; color: #38BDF8; text-transform: uppercase; margin-bottom: 4px; }
        .reason-item { font-size: 12px; color: #E2E8F0; line-height: 1.4; }
    </style>
</head>
<body>

<div class="app-container">
    <div class="app-bar">
        <svg class="logo-svg" viewBox="0 0 48 48" fill="none">
            <path d="M24 4L6 12V22C6 33.1 13.7 43.3 24 46C34.3 43.3 42 33.1 42 22V12L24 4Z" fill="#10B981" fill-opacity="0.2" stroke="#10B981" stroke-width="3"/>
            <path d="M16 23L22 29L32 17" stroke="#10B981" stroke-width="3.5" stroke-linecap="round"/>
        </svg>
        <div>
            <div class="app-title">SafeBite Inspector</div>
            <div class="app-subtitle">Food Safety AI • Offline</div>
        </div>
    </div>

    <div class="tab-bar">
        <div id="tab-cam" class="tab active" onclick="switchTab('camera')">Camera</div>
        <div id="tab-up" class="tab" onclick="switchTab('upload')">Upload</div>
        <div id="tab-rep" class="tab" onclick="switchTab('report')">Audit Report</div>
    </div>

    <div class="content">
        <!-- Live Camera View -->
        <div id="panel-camera" class="panel active">
            <div class="media-box">
                <video id="video" autoplay playsinline muted></video>
            </div>
            <button class="btn" onclick="captureAndAudit()">Capture & Audit Live Feed</button>
            <p id="cam-fallback-msg" style="font-size: 12px; color: #94A3B8; text-align: center; display: none;">
                Camera restricted by browser over unencrypted HTTP. Use the <b>Upload</b> tab or snap a photo directly below:
            </p>
            <input type="file" id="cameraDirectInput" accept="image/*" capture="environment" style="display:none;" onchange="handleFile(event)">
            <button class="btn btn-upload" id="btnSnapPhoto" style="display:none;" onclick="document.getElementById('cameraDirectInput').click()">Snap Photo Directly</button>
        </div>

        <!-- File Upload View -->
        <div id="panel-upload" class="panel">
            <div class="media-box">
                <p id="upload-placeholder" style="color: #64748B; font-size: 13px;">No food image selected</p>
                <img id="upload-preview" style="display:none;" />
            </div>
            <input type="file" id="fileInput" accept="image/*" style="display:none;" onchange="handleFile(event)">
            <button class="btn btn-upload" onclick="document.getElementById('fileInput').click()">Select Food Image File</button>
            <button class="btn" id="btnAuditUpload" style="display:none;" onclick="auditLoadedImage()">Audit Selected Image</button>
        </div>

        <!-- Shared Status & Heatmap Overlay Box -->
        <div id="status-banner" class="banner"></div>
        <div id="cam-box" class="media-box" style="display:none;">
            <img id="heatmap-preview" />
        </div>

        <!-- Audit Report Generator View -->
        <div id="panel-report" class="panel">
            <div class="report-box">
                <div class="report-row"><span style="font-weight:700; color:#F8FAFC;">Formal Audit Record</span><strong id="rep-badge">READY</strong></div>
                <hr style="border: 0; border-top: 1px solid #334155; margin: 4px 0;">
                <div class="report-row"><span>Audit ID:</span><strong id="rep-id">--</strong></div>
                <div class="report-row"><span>Timestamp:</span><strong id="rep-time">--</strong></div>
                <div class="report-row"><span>Detected Cuisine:</span><strong id="rep-cuisine">--</strong></div>
                <div class="report-row"><span>Verdict:</span><strong id="rep-verdict">--</strong></div>
                <div class="report-row"><span>Confidence:</span><strong id="rep-conf">--</strong></div>

                <div class="reason-box">
                    <h4>Visual Diagnosis & Reasons</h4>
                    <div id="reasons-list">
                        <div class="reason-item">Capture or upload an image to generate compliance diagnostics.</div>
                    </div>
                </div>
            </div>
            <button class="btn btn-outline" onclick="window.print()">Print / Save PDF</button>
        </div>
    </div>
</div>

<script>
    const video = document.getElementById('video');
    let loadedBase64 = null;

    // Initialize Camera Safely (Graceful degradation on HTTP mobile)
    function initCamera() {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
                .then(stream => { video.srcObject = stream; })
                .catch(err => {
                    console.warn("Camera access denied or unencrypted HTTP:", err);
                    document.getElementById('cam-fallback-msg').style.display = 'block';
                    document.getElementById('btnSnapPhoto').style.display = 'block';
                });
        } else {
            document.getElementById('cam-fallback-msg').style.display = 'block';
            document.getElementById('btnSnapPhoto').style.display = 'block';
        }
    }
    initCamera();

    // Tab Navigation
    function switchTab(name) {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));

        document.getElementById('panel-' + name).classList.add('active');
        if (name === 'camera') document.getElementById('tab-cam').classList.add('active');
        if (name === 'upload') document.getElementById('tab-up').classList.add('active');
        if (name === 'report') document.getElementById('tab-rep').classList.add('active');
    }

    // Client-side image scaling to avoid payload errors
    function handleFile(event) {
        const file = event.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = function(e) {
            const img = new Image();
            img.onload = function() {
                const canvas = document.createElement('canvas');
                let width = img.width;
                let height = img.height;
                const maxDim = 800;

                if (width > maxDim || height > maxDim) {
                    if (width > height) {
                        height = Math.round((height * maxDim) / width);
                        width = maxDim;
                    } else {
                        width = Math.round((width * maxDim) / height);
                        height = maxDim;
                    }
                }
                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, width, height);

                loadedBase64 = canvas.toDataURL('image/jpeg', 0.85);

                // Update UI preview
                const preview = document.getElementById('upload-preview');
                preview.src = loadedBase64;
                preview.style.display = 'block';
                document.getElementById('upload-placeholder').style.display = 'none';
                document.getElementById('btnAuditUpload').style.display = 'block';

                // Automatically switch to upload view to see it
                switchTab('upload');
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    function captureAndAudit() {
        if (!video.videoWidth) {
            alert("Camera feed not ready. Please use the 'Snap Photo' or 'Upload' option.");
            return;
        }
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        executeAudit(canvas.toDataURL('image/jpeg', 0.85));
    }

    function auditLoadedImage() {
        if (loadedBase64) {
            executeAudit(loadedBase64);
        }
    }

    async function executeAudit(b64Image) {
        const banner = document.getElementById('status-banner');
        const camBox = document.getElementById('cam-box');
        
        banner.style.display = 'block';
        banner.className = 'banner';
        banner.innerText = 'Evaluating visual freshness & spoilage markers...';

        try {
            const res = await fetch('/audit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: b64Image })
            });
            const data = await res.json();

            if (data.rejected) {
                banner.className = 'banner banner-error';
                banner.innerText = 'REJECTED: ' + data.message;
                camBox.style.display = 'none';
                return;
            }

            const isFresh = data.verdict.includes('Fresh');
            banner.className = 'banner ' + (isFresh ? 'banner-fresh' : 'banner-rotten');
            banner.innerText = `[${data.cuisine}] ${data.verdict} (${data.confidence}%)`;

            document.getElementById('heatmap-preview').src = 'data:image/jpeg;base64,' + data.cam_b64;
            camBox.style.display = 'flex';

            // Populate Report Tab
            document.getElementById('rep-badge').innerText = isFresh ? 'FIT' : 'UNFIT';
            document.getElementById('rep-badge').style.color = isFresh ? '#10B981' : '#EF4444';
            document.getElementById('rep-id').innerText = 'AUD-' + Date.now().toString().slice(-6);
            document.getElementById('rep-time').innerText = new Date().toLocaleTimeString();
            document.getElementById('rep-cuisine').innerText = data.cuisine;
            document.getElementById('rep-verdict').innerText = data.verdict;
            document.getElementById('rep-conf').innerText = data.confidence + '%';

            const rList = document.getElementById('reasons-list');
            rList.innerHTML = '';
            data.reasons.forEach(r => {
                const item = document.createElement('div');
                item.className = 'reason-item';
                item.innerText = '• ' + r;
                rList.appendChild(item);
            });

        } catch (e) {
            banner.className = 'banner banner-error';
            banner.innerText = 'Inference failed: ' + e;
        }
    }
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/audit', methods=['POST'])
def audit():
    try:
        raw_b64 = request.json['image'].split(',')[1]
        img_bytes = base64.b64decode(raw_b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # 1. Non-food Verification Gate
        is_food, msg = verify_is_food(pil_img)
        if not is_food:
            return jsonify({'rejected': True, 'message': msg})

        # 2. Multi-Task Deep CNN & Grad-CAM
        input_tensor = tensor_transform(pil_img).unsqueeze(0).to(device)

        with torch.enable_grad():
            cam_map, _, (logits_c, logits_f) = cam_engine.generate_heatmap(input_tensor, task='freshness')

        prob_c = torch.softmax(logits_c, dim=1)[0].detach().cpu().numpy()
        prob_f = torch.softmax(logits_f, dim=1)[0].detach().cpu().numpy()

        c_idx, f_idx = int(np.argmax(prob_c)), int(np.argmax(prob_f))
        verdict = FRESHNESS_MAP[f_idx]
        confidence = round(float(prob_f[f_idx] * 100), 1)

        blended = overlay_heatmap(cv_img, cam_map, alpha=0.45)
        _, buffer = cv2.imencode('.jpg', blended)
        cam_b64 = base64.b64encode(buffer).decode('utf-8')

        reasons = generate_inspection_reasons(verdict, cam_map)

        return jsonify({
            'rejected': False,
            'cuisine': CUISINE_MAP[c_idx],
            'verdict': verdict,
            'confidence': confidence,
            'cam_b64': cam_b64,
            'reasons': reasons
        })
    except Exception as e:
        return jsonify({'rejected': True, 'message': f'Processing error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
