"""
Models package — Industrial Anomaly Detection.

3 models chính:
    - PatchCore: patch-level feature matching (TỰ DEFINE TỪ GỐC)
    - Autoencoder: reconstruction-based anomaly detection
    - CNN+OC-SVM: feature extraction + One-Class SVM
"""

from models.autoencoder import Autoencoder
from models.cnn_feature import CNNFeatureExtractor
from models.custom_resnet import CustomResNet18, custom_resnet18
from models.patchcore import PatchCore, PatchCoreFeatureExtractor

__all__ = [
    "Autoencoder",
    "CustomResNet18",
    "custom_resnet18",
    "PatchCore",
    "PatchCoreFeatureExtractor",
    "CNNFeatureExtractor",
]
