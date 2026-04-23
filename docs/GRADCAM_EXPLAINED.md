# Grad-CAM — Giải thích chi tiết

## 1. Grad-CAM là gì?

**Grad-CAM** (Gradient-weighted Class Activation Mapping) là kỹ thuật **giải thích trực quan** cho mạng CNN. Nó tạo ra **heatmap** cho thấy **vùng nào trong ảnh** mà model "chú ý" nhiều nhất khi đưa ra quyết định.

> **Ý tưởng chính**: Gradient lớn tại 1 vùng → vùng đó ảnh hưởng mạnh tới output → model "chú ý" ở đó.

## 2. Cách hoạt động — Từng bước

### Bước 1: Forward Pass
Đưa ảnh qua CNN → thu được **feature maps** (activation maps) tại target layer.

```
Ảnh (3, 256, 256)  →  CNN layers  →  Feature maps (512, 8, 8)
                                      ↑ target layer output
```

Mỗi feature map (1 channel) "phát hiện" 1 đặc trưng khác nhau (cạnh, góc, texture...).

### Bước 2: Backward Pass
Lan truyền ngược (backpropagation) từ output → tính **gradient** tại target layer.

```python
output = model(image)
target = output.mean()    # scalar target
target.backward()         # → tính gradients
```

Gradient cho biết: nếu activation thay đổi 1 chút, output thay đổi bao nhiêu?

### Bước 3: Global Average Pooling → Weights
Lấy trung bình gradient trên toàn bộ spatial dimensions (H, W) → được **1 weight cho mỗi channel**.

```python
# gradients shape: (1, 512, 8, 8)
weights = gradients.mean(dim=[2, 3])  # → (1, 512)
```

Weight cao → feature map đó quan trọng với output.

### Bước 4: Weighted Sum
Nhân mỗi feature map với weight tương ứng, rồi cộng lại:

```python
# weights: (1, 512, 1, 1) × activations: (1, 512, 8, 8)
cam = (weights * activations).sum(dim=1)  # → (1, 8, 8)
```

### Bước 5: ReLU
Chỉ giữ giá trị dương (vùng ảnh hưởng tích cực):

```python
cam = torch.relu(cam)    # Bỏ vùng ảnh hưởng tiêu cực
cam = cam / cam.max()    # Normalize về [0, 1]
```

### Bước 6: Resize + Overlay
Resize heatmap (8×8) lên kích thước ảnh gốc (256×256) → overlay lên ảnh.

## 3. Trong project này

Chúng ta dùng Grad-CAM với **CNNFeatureExtractor** (ResNet18 pretrained) để xem model "nhìn vào đâu":

```python
from models.cnn_feature import CNNFeatureExtractor
from visualization.gradcam import GradCAM, show_gradcam

model = CNNFeatureExtractor()
model.eval()

# Target layer = layer cuối của feature extractor (trước avgpool)
# Đây là layer conv có spatial information (H×W) → tạo heatmap có nghĩa
target_layer = model.features[-3]  # Layer cuối trước avgpool

show_gradcam(model, target_layer, img_tensor, save_path="gradcam_result.png")
```

## 4. Cách đọc Grad-CAM heatmap

| Màu | Ý nghĩa |
|-----|---------|
|  Đỏ/Vàng | Model chú ý **nhiều** → vùng quan trọng cho quyết định |
|  Xanh | Model chú ý **ít** → vùng ít ảnh hưởng |
|  Đen | Không có ảnh hưởng |

**Trong anomaly detection**:
- Vùng đỏ trên ảnh bất thường → model phát hiện ra **vùng bị lỗi**
- Vùng đỏ trên ảnh bình thường → vùng model dùng để **xác nhận ảnh OK**

## 5. Tại sao dùng Grad-CAM?

1. **Giải thích được** (Explainability): biết model "nghĩ" gì, không phải hộp đen
2. **Debug model**: nếu heatmap tập trung sai chỗ → model học sai pattern
3. **Tăng độ tin cậy**: khi deploy thực tế, cần biết model phát hiện đúng vùng lỗi
4. **Yêu cầu công nghiệp**: nhiều ngành (y tế, sản xuất) yêu cầu AI giải thích được

## 6. Sơ đồ tổng quan

```
Ảnh đầu vào
    │
    ▼
┌───────────────────────────────────┐
│  CNN Feature Extractor (ResNet18) │
│                                   │
│  conv1 → layer1 → layer2 → ...   │
│                          │        │
│              ┌───────────┘        │
│              ▼                    │
│     Target Layer (layer4)         │
│     ┌─────────────────┐          │
│     │ Feature Maps    │ ──────── Hook lưu activations
│     │ (512, 8, 8)     │          │
│     └─────────────────┘          │
│              │                    │
│              ▼                    │
│     avgpool → flatten → output   │
└──────────────│────────────────────┘
               │
               ▼
          Backward Pass
               │
               ▼
     Gradients tại Target Layer ──── Hook lưu gradients
               │
               ▼
     Global Average Pooling
     → weights (512,)
               │
               ▼
     Weighted Sum + ReLU
     → Heatmap (8, 8)
               │
               ▼
     Resize → (256, 256)
               │
               ▼
     Overlay lên ảnh gốc
```
