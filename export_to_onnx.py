import os
import torch
import onnx
import onnxruntime as ort
import numpy as np
from multitask_food_model import MultiTaskFoodCNN

def export_and_optimize():
    weights_path = "multitask_food_weights.pth"
    onnx_output_path = "food_safety_mobile.onnx"
    
    # Define classes consistent with the training setup
    num_cuisines = 7
    num_freshness = 2

    print("[1/5] Initializing PyTorch MultiTaskFoodCNN model...")
    model = MultiTaskFoodCNN(num_cuisines=num_cuisines, num_freshness=num_freshness)
    
    if os.path.exists(weights_path):
        print(f"      Loading weights from {weights_path}...")
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        print(f"      [Notice] {weights_path} not found. Exporting base/initialized weights.")
        
    model.eval()

    # Create dummy tensor matching standard mobile camera preprocessed frame: (Batch, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    print(f"[2/5] Exporting graph to {onnx_output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input_image'],
        output_names=['cuisine_logits', 'freshness_logits'],
        dynamic_axes={
            'input_image': {0: 'batch_size'},
            'cuisine_logits': {0: 'batch_size'},
            'freshness_logits': {0: 'batch_size'}
        }
    )
    print("      ONNX export completed.")

    # Validate the exported model structural integrity
    print("[3/5] Validating ONNX graph integrity...")
    onnx_model = onnx.load(onnx_output_path)
    onnx.checker.check_model(onnx_model)
    print("      ONNX model is structurally valid.")

    # Cross-verify outputs between PyTorch and ONNX Runtime
    print("[4/5] Cross-verifying PyTorch vs ONNX Runtime inference outputs...")
    with torch.no_grad():
        torch_cuisine, torch_freshness = model(dummy_input)

    ort_session = ort.InferenceSession(onnx_output_path, providers=['CPUExecutionProvider'])
    ort_inputs = {'input_image': dummy_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    ort_cuisine, ort_freshness = ort_outputs[0], ort_outputs[1]

    # Verify numerical alignment
    np.testing.assert_allclose(torch_cuisine.numpy(), ort_cuisine, rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(torch_freshness.numpy(), ort_freshness, rtol=1e-03, atol=1e-05)
    print("      Verification successful: PyTorch and ONNX output matrices match.")

    # File size metrics
    file_size_mb = os.path.getsize(onnx_output_path) / (1024 * 1024)
    print(f"[5/5] Deployment asset ready:")
    print(f"      - File: {onnx_output_path}")
    print(f"      - Size: {file_size_mb:.2f} MB")
    print("\nNext: Copy 'food_safety_mobile.onnx' into your Android Studio project at:")
    print("      app/src/main/assets/food_safety_mobile.onnx")

if __name__ == "__main__":
    export_and_optimize()
