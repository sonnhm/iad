"""
ONNX Export Tool cho IAD Project

Script này xuất mô hình Convolutional Feature Extractor (Backbone)
sang định dạng chuẩn công nghiệp ONNX để deploy lên thiết bị không có GPU (ONNX Runtime).
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.patchcore import PatchCoreFeatureExtractor


def export_backbone_to_onnx(checkpoint_path, export_path, image_size=256):
    print(f"Loading backbone from: {checkpoint_path}")

    # Khởi tạo mô hình
    model = PatchCoreFeatureExtractor()
    model.eval()

    # Load trọng số nếu có
    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_backbone_weights(checkpoint_path)
    else:
        print("Warning: Checkpoint not found. Exporting initialized weights.")

    dummy_input = torch.randn(1, 3, image_size, image_size)

    print(f"Exporting ONNX to {export_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_image"],
        output_names=["feature_map"],
        dynamic_axes={
            "input_image": {0: "batch_size"},
            "feature_map": {0: "batch_size"},
        },
    )

    print(f"ONNX Export Successful: {export_path}")
    print("Ready for deployment on C++, C#, or Edge Computing Devices.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PatchCore Backbone to ONNX")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to backbone best.pth"
    )
    parser.add_argument(
        "--export", type=str, default="backbone.onnx", help="Destination path"
    )
    parser.add_argument("--size", type=int, default=256, help="Image input size")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.export)), exist_ok=True)
    export_backbone_to_onnx(args.checkpoint, args.export, args.size)
