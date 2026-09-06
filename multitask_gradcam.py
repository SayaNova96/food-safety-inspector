import cv2
import numpy as np
import torch
import torch.nn.functional as F

class MultiTaskGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, task='freshness', target_class=None):
        self.model.eval()
        self.model.zero_grad()

        logits_cuisine, logits_freshness = self.model(input_tensor)

        selected_logits = logits_freshness if task == 'freshness' else logits_cuisine

        if target_class is None:
            target_class = torch.argmax(selected_logits, dim=1).item()

        score = selected_logits[0, target_class]
        score.backward(retain_graph=True)

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)

        return cam, target_class, (logits_cuisine, logits_freshness)

def overlay_heatmap(original_bgr, cam, alpha=0.45, colormap=cv2.COLORMAP_JET):
    h, w, _ = original_bgr.shape
    heatmap = cv2.resize(cam, (w, h))
    heatmap = np.uint8(255 * heatmap)
    colored_cam = cv2.applyColorMap(heatmap, colormap)
    blended = cv2.addWeighted(original_bgr, 1 - alpha, colored_cam, alpha, 0)
    return blended
