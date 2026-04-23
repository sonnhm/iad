# IAD Repository Ground Truth Map
*Version: 1.0 (Post Phase 3 / Pre Phase 5)*
*Purpose: Persistent context representation to enforce zero-hallucination for AI agents.*

---

## 1. Data Processing Pipeline (`data_processing/mvtec.py`)
- **Input Format:** `(B, 3, 256, 256)` RGB Tensors.
- **Transformations:** `transforms.Resize((256, 256))` -> `transforms.ToTensor()` -> `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`
- **Splits:** `train` (80% good), `valid` (20% good), `test` (good + anomalous, includes GT masks).

## 2. YOLO Classifier (`app_utils/yolo_detector.py`)
- **Model:** Ultralytics YOLOv8n.
- **Task:** 15-class classification (Primary routing).
- **Threshold:** `conf > 0.65`. Fallback to manual selection if lower.
- **Checkpoint:** `runs/detect/yolo_product_detection/mvtec_15_cat/weights/best.pt`

## 3. Knowledge Distillation Backbone (`models/custom_resnet.py` & `training/backbone_trainer.py`)
- **Teacher:** `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)` - **FROZEN**.
- **Student:** `CustomResNet18` (self-built, 4 BasicBlocks) - **LEARNABLE**.
- **Loss Strategy:** 4-layer Hook-based MSE.
  - `MSE(Teacher.layer1, Student.layer1)` 
  - `MSE(Teacher.layer2, Student.layer2)`
  - `MSE(Teacher.layer3, Student.layer3)`
  - `MSE(Teacher.layer4, Student.layer4)`
- **Optimization:** Adam (lr=1e-3), `torch.cuda.amp.GradScaler` (FP16 targeting). Reduces VRAM by ~30% and speeds training by ~1.5x while retaining architectural parameters.
- **Network Dimensions (BasicBlock = Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN + Shortcut):**
  - Stem: `(B, 64, 64, 64)` via Conv7x7(s=2) + MaxPool
  - Layer 1: `(B, 64, 64, 64)`
  - Layer 2: `(B, 128, 32, 32)`
  - Layer 3: `(B, 256, 16, 16)`
  - Layer 4: `(B, 512, 8, 8)`

## 4. PatchCore Model (`models/patchcore.py`)
- **Feature Extraction:**
  - Extracts from Student Backbone: `layer2` (128 channels) + `layer3` (256 channels).
  - Upsamples `layer3` bilinearly to match `layer2` spatial resolution.
  - Concatenation: `128 + 256 = 384` channels. Tensor Final: `(B, 384, H', W')`.
- **Coreset Subsampling (Training):**
  - Ratio: `0.1` (keeps 10%).
  - Logic: Internal Random Projection (384-D -> 128-D) to speed up `torch.cdist` -> k-Center Greedy algorithm.
  - The final selected features remain `384-D` in the Memory Bank.
- **LSH & k-NN (Inference):**
  - Hash Bucket: `n_bits=12` (4096 buckets) via `PyTorchLSHIndex` (Cosine LSH Random Projection Hyperplanes).
  - Search: Vectorized, Multi-probe (XOR 1-bit flip, Hamming Distance = 1).
  - Distance Metric: `k=1`, exact minimum Euclidean distance on matched candidates subset.
- **Thresholding:** `99.5th` Percentile learned from train split. Production applies `x 0.8` tolerance factor.

## 5. Convolutional Autoencoder (`models/autoencoder.py`)
- **Encoder (Compress):** 5 Sequential Blocks
  1. `Conv2d(3->32, k=3, s=2, p=1)` + BN + ReLU -> `128x128`
  2. `Conv2d(32->64, k=3, s=2, p=1)` + BN + ReLU -> `64x64`
  3. `Conv2d(64->64, k=3, s=2, p=1)` + BN + ReLU -> `32x32`
  4. `Conv2d(64->64, k=3, s=2, p=1)` + BN + ReLU -> `16x16`
  5. `Conv2d(64->64, k=3, s=2, p=1)` + BN + ReLU -> `8x8`
- **Latent Bottleneck:** Dimension: `(B, 64, 8, 8)` (4,096 scalar params).
- **Decoder (Reconstruct):** 5 Sequential Blocks
  1. `ConvTranspose2d(64->64, k=4, s=2, p=1)` + BN + ReLU -> `16x16`
  2. `ConvTranspose2d(64->64, k=4, s=2, p=1)` + BN + ReLU -> `32x32`
  3. `ConvTranspose2d(64->64, k=4, s=2, p=1)` + BN + ReLU -> `64x64`
  4. `ConvTranspose2d(64->32, k=4, s=2, p=1)` + BN + ReLU -> `128x128`
  5. `ConvTranspose2d(32->3, k=4, s=2, p=1)` **(No BN, No ReLU)** -> `256x256`
- **Scoring:** `MSE(Input, Reconstruction)` pixel-wise map.

## 6. CNN + OC-SVM Baseline (`models/cnn_feature.py` & `app.py`)
- **Feature Extractor:** `torchvision` ResNet18 (Standard pretrained, distinct from KD backbone).
- **Pooling:** Slices off FC layer, uses `AdaptiveAvgPool2d` -> Flattens to `(B, 512)` vector.
- **Detector:** `sklearn.svm.OneClassSVM(gamma="auto")`.
- **Persistence:** Trained once via `DataLoader(batch_size=32)` on 100% normal data, then saved to disk via `joblib.dump()` to sever dependency on the 5GB dataset during inference.
- **Interpretability:** Generates Grad-CAM heatmaps by hooking target layer `model.features[-2]` (Layer 4).

## 7. Application Pipeline (`app.py`)
- **Blur Mitigation:** OpenCV CLAHE + Unsharp Masking applied *before* YOLO detection.
- **Concurrency:** Uses `ThreadPoolExecutor(max_workers=3)` to launch PatchCore, AE, and CNN+OC-SVM simultaneously.
- **Safety:** Implements `threading.Lock()` (Double-check locking) for thread-safe model caching.
- **Outputs (`numpy_to_base64`):** Patch map (PC), MSE error map (AE), Grad-CAM (OC-SVM).
