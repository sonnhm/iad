import os
import sys

import gradio as gr
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# Add project root to path so we can import models and utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app_utils.checkpoint import load_checkpoint
from models.patchcore import PatchCore

# Try importing visualization utilities gracefully
try:
    from visualization.gradcam import generate_heatmap
except ImportError:
    generate_heatmap = None

# Global variable for the loaded model
model = None


def load_patchcore():
    global model
    if model is None:
        try:
            model = PatchCore(
                backbone_name="resnet18", layers=["layer2", "layer3"], device="cpu"
            )  # Force CPU for HF Spaces Free Tier
            print("Model initialized for HF CPU usage.")
        except Exception as e:
            print(f"Error loading model setup: {e}")
            return None
    return model


def predict(image):
    if image is None:
        return None, "Please upload an image."

    transform = T.Compose(
        [
            T.Resize((256, 256)),
            T.CenterCrop((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img_tensor = transform(image.convert("RGB")).unsqueeze(0)

    net = load_patchcore()
    if net is None:
        return image, "Model initialization failed."

    try:
        # Note: If no memory_bank is loaded via .fit(), predict will fail.
        # This script acts as a stub structure showing where HF connects.
        if (
            hasattr(net, "memory_bank")
            and getattr(net, "memory_bank", None) is not None
        ):
            image_score, pixel_scores = net.predict(img_tensor)
            score_val = float(image_score.squeeze().item())

            # Simple heatmap visualization placeholder
            heatmap_img = image  # Replace with actual colorized heatmap
            return heatmap_img, f"Anomaly Score: {score_val:.4f}"
        else:
            return (
                image,
                "Model is ready to train but Memory Bank is not loaded for inference.",
            )
    except Exception as e:
        return image, f"Error during inference: {str(e)}"


interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(label="Anomaly Heatmap"), gr.Textbox(label="Result")],
    title="Industrial Anomaly Detection - PatchCore",
    description="Demo Gradio Interface. Upload an industrial image to detect defects.",
    allow_flagging="never",
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
