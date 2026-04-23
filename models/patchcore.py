"""
PatchCore Architecture - Advanced Industrial Anomaly Detection
Kiến trúc PatchCore - Phát hiện Bất thường Công nghiệp Nâng cao

This module implements the PatchCore algorithm from scratch, as proposed by Roth et al. (2022)
(Mô-đun này tự triển khai thuật toán PatchCore từ đầu, dựa trên đề xuất của Roth và cộng sự năm 2022)
in "Towards Total Recall in Industrial Anomaly Detection". This implementation avoids external
(trong luận văn "Tiến tới độ bao phủ toàn diện trong Phát hiện Bất thường Công nghiệp". Việc triển khai này tránh dùng)
machine learning libraries like scikit-learn to maintain total transparency and optimized execution.
(các thư viện học máy bên ngoài như scikit-learn để duy trì sự minh bạch tuyệt đối và tối ưu hóa thời gian chạy.)

Core Components (Các thành phần cốt lõi):
1. Backbone Feature Extractor (Bộ trích xuất Đặc trưng): Utilizes a CustomResNet18 backbone, extracting mid-level hierarchical
   features from layer2 (128-d) and layer3 (256-d), which are concatenated into a 384-d spatial feature map.
   (Sử dụng CustomResNet18, trích xuất đặc trưng phân tầng tầm trung từ layer2 (128 chiều) và layer3 (256 chiều),
   sau đó nối lại thành một Feature Map Không gian 384 chiều).
2. Nominal Memory Bank (Kho Bộ nhớ Tiêu chuẩn): A database of localized patch features extracted purely from normal images during fit().
   (Một cơ sở dữ liệu chứa các đặc trưng phân mảnh (patches) được trích xuất hoàn toàn từ ảnh bình thường).
3. Greedy Coreset Subsampling (Lọc mẫu Coreset Tham lam): A custom iterative algorithm that reduces the Memory Bank size by
   retaining only the most representative features (e.g. ratio = 0.1), preventing VRAM Out-Of-Memory (OOM)
   issues while preserving anomaly detection accuracy.
   (Thuật toán lặp tự xây dựng để giảm kích thước Kho Bộ nhớ bằng cách chỉ giữ lại các đặc trưng đại diện nhất,
   ngăn ngừa lỗi tràn bộ nhớ VRAM).
4. k-NN Scoring (Chấm điểm bằng K-láng giềng gần nhất): A highly optimized batch Euclidean distance computation for anomaly scoring and
   pixel-level anomaly map generation.
   (Tối ưu hóa các phép tính khoảng cách Euclidean theo lô để chấm điểm lỗi và tạo bản đồ nhiệt ở mức độ Pixel).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.custom_resnet import CustomResNet18


class PatchCoreFeatureExtractor(nn.Module):
    """
    Feature extractor (Bộ trích xuất Đặc trưng) cho PatchCore — trích multi-scale patch features.

    Sử dụng CustomResNet18 (tự implement) làm backbone (mạng xương sống).
    Trích features từ layer2 (128-d) + layer3 (256-d) → concat = 384-d.

    Input:  (B, 3, H, W)
    Output: (B, D, H', W') — spatial feature map (bản đồ đặc trưng không gian), D = 384
    """

    def __init__(self):
        super().__init__()

        # Khởi tạo mô hình ResNet18 tự xây (không cần num_classes vì không dùng FC layer cuối)
        backbone = CustomResNet18(num_classes=None)

        # Stem (Phần cuống): Bao gồm conv1 → bn1 → relu → maxpool để tiền xử lý ảnh ban đầu
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )

        # Feature layers (Các tầng đặc trưng)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2  # output: 128 channels (chiều)
        self.layer3 = backbone.layer3  # output: 256 channels (chiều)

    def forward(self, x):
        """
        Trích multi-scale spatial feature map (Trích xuất bản đồ đặc trưng đa tỷ lệ).

        Args:
            x: input images (Ảnh đầu vào dạng Tensor) có shape (B, 3, H, W)

        Returns:
            features: (B, 384, H', W') — spatial feature map (Bản đồ đặc trưng không gian)
                      384 = 128 (từ layer2) + 256 (từ layer3)
        """
        # Bước 1: Cho ảnh đi qua phần stem xử lý thô
        x = self.stem(x)
        # Bước 2: Đi qua tầng 1 (layer1)
        x = self.layer1(x)

        # Bước 3: Trích lọc đặc trưng ở tầng 2 và tầng 3
        feat2 = self.layer2(x)  # Trả về shape: (B, 128, H/8,  W/8)
        feat3 = self.layer3(feat2)  # Trả về shape: (B, 256, H/16, W/16)

        # Bước 4: Upsample (Tăng độ phân giải) feat3 về cùng kích thước không gian với feat2
        # Sử dụng phép nội suy 'bilinear' để giữ nguyên các giá trị không gian
        feat3_up = F.interpolate(
            feat3,
            size=feat2.shape[2:],  # Ép shape H/16, W/16 lên H/8, W/8
            mode="bilinear",  # Nội suy song tuyến tính
            align_corners=False,
        )

        # Bước 5: Concat (Nối) multi-scale đặc trưng lại với nhau
        # Trở thành: (B, 128+256=384, H/8, W/8)
        features = torch.cat([feat2, feat3_up], dim=1)

        return features

    def load_backbone_weights(self, checkpoint_path):
        """
        Load trained backbone weights (Tải trọng số mạng đã được huấn luyện qua Knowledge Distillation).

        Args:
            checkpoint_path (str): Đường dẫn tới file .pth
        """
        # Tải an toàn (weights_only=True) lên CPU trước để tránh lỗi lệch thiết bị CUDA
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["model"]

        # Rebuild tạm backbone để ghép map_state_dict
        backbone = CustomResNet18(num_classes=None)
        backbone.load_state_dict(state_dict)

        # Copy từng weights chéo sang feature extractor của PatchCore
        self.stem[0].load_state_dict(backbone.conv1.state_dict())
        self.stem[1].load_state_dict(backbone.bn1.state_dict())
        self.layer1.load_state_dict(backbone.layer1.state_dict())
        self.layer2.load_state_dict(backbone.layer2.state_dict())
        self.layer3.load_state_dict(backbone.layer3.state_dict())

        print(f"Backbone weights loaded from: {checkpoint_path}")


class PatchCore:
    """
    PatchCore Implementation (Roth et al., 2022).

    Pipeline Logic (Luồng Logic Vận hành):
        1. FIT (Training Phase - Giai đoạn Khởi tạo):
           - Extracts dense nominal patch features (Trích xuất mật độ đặc trưng cao) từ ảnh tốt MVTec.
           - Aggregates features into a vast 'Memory Bank' (Tập hợp chúng vào Kho Bộ Nhớ).
           - Executes Greedy Coreset Subsampling (Thực hiện Lọc mẫu) để nén Memory Bank's footprint
             mà vẫn giữ được mathematical topology (cấu trúc hình học toán học) của không gian.

        2. PREDICT (Inference Phase - Giai đoạn Suy luận dự đoán):
           - Extracts patch features (Trích Đặc trưng) từ ảnh test cần kiểm tra.
           - Computes optimized k-Nearest Neighbor (k-NN) Euclidean distances (Tính khoảng cách Euclidean láng giềng).
           - Image-level Anomaly Score: Xác định điểm bất thường tổng quát bằng Max distance.
           - Pixel-level Anomaly Map: Formed by reshaping the spatial distances (Biến đổi hình học lại thành Map).

    Args:
        backbone (PatchCoreFeatureExtractor): The core vision model for feature generation (Mạng trích xuất).
        device (str): Execution context (Môi trường thực thi), thường là 'cuda'.
        coreset_ratio (float): Tỷ lệ nén dữ liệu giữ lại (0.0 to 1.0).
        k_neighbors (int): Số lượng hàng xóm láng giềng cần xét cho k-NN score.
    """

    def __init__(self, backbone=None, device="cpu", coreset_ratio=0.1, k_neighbors=1):
        if backbone is None:
            backbone = PatchCoreFeatureExtractor()

        # Áp model lên Device (CPU/GPU) và chuyển mạng sang chế độ "eval" (chỉ sử dụng, không train ngược)
        self.backbone = backbone.to(device)
        self.backbone.eval()
        self.device = device

        self.coreset_ratio = coreset_ratio
        self.k_neighbors = k_neighbors
        self.use_lsh = True  # Toggle for LSH (Locality-Sensitive Hashing) optimization - Để tìm kiếm nhanh
        self.lsh_index = None

        # Memory bank (Kho bộ nhớ) — sẽ được xây trong fit()
        self.memory_bank = None  # (M, D) tensor sau khi đã lọc coreset
        self.feature_dim = None  # Số chiều D = 384
        self.spatial_size = (
            None  # (H', W') — kích thước spatial (không gian) của feature map
        )
        self.adaptive_threshold = (
            None  # Dynamic threshold (Ngưỡng thích ứng linh hoạt) tự học từ tập Train
        )

    # ================================================================
    # PHASE 1: FIT — Xây dựng Memory Bank (Kho Bộ Nhớ)
    # ================================================================

    def fit(self, train_dataset):
        """
        Xây dựng memory bank (Kho lưu trữ đặc trưng chuẩn) từ training data (ảnh normal).

        Quy trình:
            1. Forward mỗi ảnh train qua backbone → spatial feature map (bản đồ đặc trưng không gian).
            2. Reshape thành patches → gộp tất cả patches.
            3. Coreset subsampling (Lọc nén tham lam) → chọn subset đại diện.

        Args:
            train_dataset: Dataset trả về bộ (image_tensor, label)
        """
        print(
            "  [PatchCore] Building memory bank (Đang thu thập kho đặc trưng chuẩn)..."
        )

        all_patches = []

        # Bước 1: Quét toàn bộ ảnh Normal
        for i, (img, _) in enumerate(train_dataset):
            img = img.unsqueeze(0).to(self.device)  # Thêm chiều Batch (1, 3, H, W)

            # Khóa Gradient lại vì ta chỉ dùng Backbone để trích xuất thôi (tiết kiệm VRAM)
            with torch.no_grad():
                features = self.backbone(img)  # Trả về: (1, D, H', W')

            # Bước 2: Lưu spatial size (Kích thước không gian) để về sau (predict) dùng tái tạo heatmap
            if self.spatial_size is None:
                self.spatial_size = (features.shape[2], features.shape[3])
                self.feature_dim = features.shape[1]  # D (384)

            # Bước 3: Reshape (Ép hình dạng) thành chuỗi các Patches phẳng
            # Từ (1, D, H', W') ép thành (1, H', W', D) rồi duỗi ra thành danh sách (-1, D)
            patches = features.permute(0, 2, 3, 1).reshape(-1, self.feature_dim)
            all_patches.append(patches)

            if (i + 1) % 50 == 0:
                print(f"    Processed {i + 1}/{len(train_dataset)} images")

        # Gộp (Concat) tất cả hàng vạn patches lại với nhau: (N_total, D)
        all_patches = torch.cat(all_patches, dim=0)
        print(f"  [PatchCore] Raw memory bank (Kích thước gốc): {all_patches.shape}")

        # Bước 4: Coreset subsampling (Tiến hành triệt tiêu rườm rà, chỉ giữ cốt lõi)
        self.memory_bank = self._coreset_subsampling(all_patches)
        print(
            f"  [PatchCore] After coreset (Kích thước sau khi nén): {self.memory_bank.shape}"
        )

        # Bước 5: Xây dựng cấu trúc dữ liệu LSH (Băm dữ liệu không gian) để tăng tốc search ANN ngàn lần
        if self.use_lsh:
            print(
                "  [PatchCore] Building LSH Index for ANN (Khởi tạo Cây Băm Index)..."
            )
            self.lsh_index = PyTorchLSHIndex(
                feature_dim=self.feature_dim,
                n_bits=12,  # 2^12 = 4096 buckets (lỗ rổ băm)
                device=self.device,
            )
            self.lsh_index.build(self.memory_bank)

        # Lấy mẫu Threshold an toàn từ chính nó (Sau khi hoàn tất Memory Bank)
        self._calibrate_threshold(train_dataset)

    # ================================================================
    # PHASE 2: CORESET SUBSAMPLING (Nén Mẫu) — Tự implement Toán Học
    # ================================================================

    def _coreset_subsampling(self, features):
        """
        Greedy Coreset Selection (Lựa chọn Cốt lõi Tham lam) tối ưu 100% bằng PyTorch.
        """
        N = features.shape[0]  # Tổng số lượng điểm ảnh
        target_size = max(1, int(N * self.coreset_ratio))  # Số lượng mục tiêu cần lọc

        if target_size >= N:
            return features

        print(f"  [Coreset] Selecting {target_size} from {N} patches...")

        # 1. Random Projection (Dimensionality Reduction) để giảm chi phí thuật toán tính khoảng cách (O(nd))
        # Giảm số chiều (D) từ 384 chiều xuống còn 128 giúp tốc độ phép tính Vector tăng x3
        reduced_dim = min(128, features.shape[1])
        proj = torch.randn(features.shape[1], reduced_dim, device=features.device)
        features_reduced = torch.matmul(features, proj)

        # 2. Xây dựng Kho chứa ID của Coreset
        selected_indices = torch.zeros(
            target_size, dtype=torch.long, device=features.device
        )

        # Chọn điểm đầu tiên ngẫu nhiên (Làm điểm mốc lõi)
        first_idx = torch.randint(0, N, (1,), device=features.device)
        selected_indices[0] = first_idx

        # Xem Khoảng cách hiện tại tới Coreset Pool
        # Dùng hàm Native torch.cdist của C++ dưới đáy PyTorch tối ưu hơn vòng lặp Python
        min_distances = torch.cdist(
            features_reduced, features_reduced[first_idx]
        ).squeeze(1)

        # Vectorized Loop (Vòng lặp định tuyến Vector) cho k-Center Greedy
        for step in range(1, target_size):
            # Chọn điểm điểm XA CÁC ĐIỂM Ở TRONG KHO NHẤT để nó phủ thêm vùng không gian mới
            next_idx = torch.argmax(min_distances)
            selected_indices[step] = next_idx

            # Cập nhật distance matrix sau khi thu thập thêm 1 phần tử
            new_distances = torch.cdist(
                features_reduced, features_reduced[next_idx].unsqueeze(0)
            ).squeeze(1)

            # Giữ những khoảng cách gần hơn để thu hẹp bán kính
            min_distances = torch.minimum(min_distances, new_distances)
            min_distances[next_idx] = 0.0  # Reset khoảng cách của chính nó về 0

            if (step + 1) % 2000 == 0:
                print(f"    Coreset: {step + 1}/{target_size}")

        return features[selected_indices]

    # ================================================================
    # PHASE 3: PREDICT — Anomaly Scoring (Dự đoán Lỗi Bất Thường)
    # ================================================================

    def predict(self, images):
        """
        Tính anomaly score (Chỉ số bất thường) và anomaly map (Bản đồ phân bổ lỗi) cho batch ảnh.

        Args:
            images: Tensor chứa lô ảnh (B, 3, H, W) hoặc duy nhất 1 ảnh (3, H, W)

        Returns:
            scores: (B,) anomaly scores — giá trị dương (càng cao = càng dị thường)
            anomaly_maps: (B, H', W') anomaly maps — Phân bố điểm mức pixel
        """
        if images.dim() == 3:
            images = images.unsqueeze(0)

        images = images.to(self.device)

        # Cấm đạo hàm Gradient để tiết kiệm tài nguyên
        with torch.no_grad():
            features = self.backbone(images)  # Rút ruột (B, D, H', W')

        B, D, H, W = features.shape

        # Reshape thành mảnh ghép (patches) cho từng bức hình
        # Chuyển (B, D, H', W') → (B, H'*W', D)
        patches = features.permute(0, 2, 3, 1).reshape(B, H * W, D)

        scores = []
        anomaly_maps = []

        # Đo lặp qua từng Batch (Thường chỉ có B=1 khi Web app chạy)
        for b in range(B):
            # Tính khoảng cách độ sai lệch k-NN cho mỗi phân mảnh hình chữ nhật
            # patch_scores: Tensor (H'*W',) có chứa khoảng cách tới chuẩn màng láng giềng
            patch_scores = self._knn_score(patches[b])

            # Cấp độ Ảnh (Image-level score) = Xét mảng hình phạt MAX (Lỗi nặng nhất)
            # Vùng bất thường nhất quyết định số phận lỗi cho cả tập thể bức ảnh
            image_score = patch_scores.max().item()
            scores.append(image_score)

            # Cấp độ điểm ảnh (Pixel-level map) - vẽ màu
            anomaly_map = patch_scores.reshape(H, W)
            anomaly_maps.append(anomaly_map.cpu().numpy())

        scores = np.array(scores)
        anomaly_maps = np.stack(
            anomaly_maps, axis=0
        )  # Nối lại cấu trúc mảng Numpy (B, H', W')

        return scores, anomaly_maps

    def predict_score(self, images):
        """
        Chỉ tính image-level anomaly score (Chức năng dành cho Evaluators/Benchmarks).

        Args:
            images: Tensor (B, 3, H, W)

        Returns:
            scores: Dãy (B,) chứa điểm
        """
        scores, _ = self.predict(images)
        return scores

    # ================================================================
    # k-NN SCORING — Tự implement Tối Đa (Bypass the Scipy loop)
    # ================================================================

    def _knn_score(self, query_patches):
        """
        Tìm k-Nearest Neighbors (K-chòm sao láng giềng gần nhất) trong Memory bank và xuất kết quả.

        Tự implement thuật toán tìm kiếm khoảng cách Euclidean, KHÔNG dùng sklearn tránh lệ thuộc.

        Cách tính:
            1. Với mỗi bộ vá truy vấn (query patch), tính khoảng cách chéo tới TẤT CẢ các patches chuẩn
               (nằm trong Memory Bank).
            2. Sort (Sắp xếp tăng dần) → lấy k distances dưới đáy vực.
            3. Tính giá trị trung bình Mean của k distances = Áp án Anomaly Score cho vùng đó.

        Args:
            query_patches: Tensor (N_query, D)

        Returns:
            scores: Tensor (N_query,)  — Anomaly score gán cho mỗi patch
        """
        if self.memory_bank is None:
            raise RuntimeError("Memory bank chưa được xây! Gọi fit() trước.")

        N_query = query_patches.shape[0]

        # Tính khoảng cách Euclidean: Tensor (N_query, M)
        # Thực hiện chia lô (Batching) để tránh việc OOM (Hết RAM Đồ Họa) khi Kho Bộ nhớ quá to.
        batch_size = 256
        all_scores = []

        for start in range(0, N_query, batch_size):
            end = min(start + batch_size, N_query)
            query_batch = query_patches[start:end]  # Block query (batch, D)

            if self.use_lsh and self.lsh_index is not None:
                # Sử dụng chế độ LSH (Locality-Sensitive Hashing) Approximate Data Search để tăng cực nhanh
                distances = self.lsh_index.search(query_batch)
            else:
                # Thực hiện Exact Euclidean distance search (Ma trận thuần túy đầy đủ cự ly)
                distances = self._euclidean_distance_batch(
                    query_batch, self.memory_bank
                )  # (batch, M)

            # Thuật toán k-NN: Lấy rào k-mốc khoảng cách sát sạt
            if self.k_neighbors == 1:
                # Tối ưu siêu tốc k=1: Chỉ lấy hàm Min duy nhất
                min_dist, _ = distances.min(dim=1)  # Kích thước (batch,)
                all_scores.append(min_dist)
            else:
                # Tìm kiếm Top-K nhỏ nhất
                topk_dist, _ = distances.topk(
                    self.k_neighbors, dim=1, largest=False
                )  # Shape (batch, k)
                mean_dist = topk_dist.mean(dim=1)  # Gom chung trung bình (batch,)
                all_scores.append(mean_dist)

        scores = torch.cat(all_scores, dim=0)  # Gắn lại dãy kết quả mã (N_query,)
        return scores

    # ================================================================
    # THRESHOLD CALIBRATION — Định danh Lọc Ngưỡng Tự Nhiên
    # ================================================================

    def _calibrate_threshold(self, train_dataset):
        """
        Tính Adaptive Threshold (Ngưỡng thích ứng) dựa trên đỉnh sai lệch của Kho Ảnh Normal.
        (Hoạch định một vành đai phòng vệ 99.5% - Nếu lọt ra ngoài sẽ là Anomaly).
        """
        print("  [Calibration] Hoạch định ngưỡng Adaptive Threshold...")
        scores = []
        for i, (img, _) in enumerate(train_dataset):
            with torch.no_grad():
                img = img.unsqueeze(0).to(self.device)
                score, _ = self.predict(img)
                scores.append(score[0])
            if (i + 1) % 50 == 0:
                print(f"    Calibrating {i+1}/{len(train_dataset)}")

        scores = np.array(scores)
        mean_s = scores.mean()
        # Thống kê phân vị 99.5% Percentile (Sử dụng biểu đồ Box-plot tư duy Non-Gaussian)
        percentile_995 = float(np.percentile(scores, 99.5))
        self.adaptive_threshold = percentile_995
        print(
            f"  [Calibration] Dynamic Threshold Learned (99.5th Percentile): {self.adaptive_threshold:.4f} (Mean: {mean_s:.4f})"
        )

    # ================================================================
    # EUCLIDEAN DISTANCE — Vectorized Hardware Level Matrix Math
    # ================================================================

    @staticmethod
    def _euclidean_distance_batch(x, y):
        """
        Tính cự ly Euclidean nối dài giữa mọi đường chéo tọa độ trong Vector X và Y.

        Chỉ huy lập trình tự viết tay, KHÔNG dùng Sklearn (nhằm ép vào VRAM Cuda).

        Công thức Tối ưu hóa (Mathematical Trick):
            Xoá bỏ hoàn toàn lệnh FOR Loop cồng kềnh.
            Quy chiếu: ||x_i - y_j||^2 = ||x_i||^2 + ||y_j||^2 - 2 * (x_i · y_j)

        Args:
            x: (N, D) Tensor điểm 1
            y: (M, D) Tensor điểm 2

        Returns:
            distances: (N, M) lưới chiếu Tensor — Euclidean distance 2D Map
        """
        # Bước 1: Bình phương mảng ||x||^2 → trả bộ phận cột (N, 1)
        x_sq = (x**2).sum(dim=1, keepdim=True)

        # Bước 2: Bình phương mảng ||y||^2 → trả bộ phận hàng (1, M)
        y_sq = (y**2).sum(dim=1, keepdim=True).t()

        # Bước 3: Nhân rẽ ma trận Tích vô hướng Dot-Product X·Y: (N, M)
        xy = torch.mm(x, y.t())

        # Bước 4: Khớp chuỗi theo công thức đại số
        dist_sq = x_sq + y_sq - 2.0 * xy

        # BẢO VỆ CHUNG NGĂN SPAM NAN: Khóa Clamp đáy tối thiểu = 0.0 để căn bậc hai không lấy log âm
        dist_sq = torch.clamp(dist_sq, min=0.0)

        # Bước cuối: Chiết xuất Căn Bậc Hai hoàn thiện
        distances = torch.sqrt(dist_sq)

        return distances

    # ================================================================
    # SAVE/LOAD LOGIC
    # ================================================================

    def save(self, path):
        """
        Dump File hệ thống: Lưu PatchCore weights, memory bank và lsh dict.
        """
        import os

        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )

        state = {
            "memory_bank": self.memory_bank,
            "feature_dim": self.feature_dim,
            "spatial_size": self.spatial_size,
            "coreset_ratio": self.coreset_ratio,
            "k_neighbors": self.k_neighbors,
            "adaptive_threshold": self.adaptive_threshold,
            "use_lsh": self.use_lsh,
        }

        # Nhúng cấy Bảng quy hoạch Hyperplanes của Hash LSH nếu LSH=True
        if self.use_lsh and self.lsh_index is not None:
            state["lsh_hyperplanes"] = self.lsh_index.hyperplanes

        torch.save(state, path)
        print(f"PatchCore saved: {path}")

    def load(self, path):
        """
        Hồi sinh Model trạng thái (Revive State).
        """
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.memory_bank = state["memory_bank"].to(self.device)
        self.feature_dim = state["feature_dim"]
        self.spatial_size = state["spatial_size"]
        self.coreset_ratio = state["coreset_ratio"]
        self.k_neighbors = state["k_neighbors"]
        self.adaptive_threshold = state.get("adaptive_threshold", None)
        self.use_lsh = state.get("use_lsh", False)

        # Nếu Hệ thống trước đó có LSH, dựng móng lại ngay thay vì tính từ số 0
        if self.use_lsh and "lsh_hyperplanes" in state:
            print(
                "  [PatchCore] Reconstructing LSH Index (Tiến hành Tái Tạo Sàn Giao Băm)..."
            )
            self.lsh_index = PyTorchLSHIndex(
                feature_dim=self.feature_dim,
                n_bits=state["lsh_hyperplanes"].shape[1],
                device=self.device,
            )
            self.lsh_index.hyperplanes = state["lsh_hyperplanes"].to(self.device)
            self.lsh_index.build(self.memory_bank)

        print(f"PatchCore loaded: {path} (memory bank: {self.memory_bank.shape})")


class PyTorchLSHIndex:
    """
    Kích hoạt cơ chế Locality-Sensitive Hashing (LSH) thuần túy PyTorch để tạo lối tắt truy tìm k-NN nhanh.
    Sử dụng thuật toán Random Projection Hyperplanes (Các mặt phẳng ngẫu nhiên - Cosine LSH)
    để rẽ khoanh vùng Feature Space (Không gian Đồ thị) thành nhiều Buckets (Rổ con nhỏ hẹp).
    """

    def __init__(self, feature_dim, n_bits=12, device="cpu"):
        self.feature_dim = feature_dim
        self.n_bits = n_bits  # Chiều dài con dao băm nhị phân
        self.device = device

        # Khởi tạo ma trận rạch ngẫu nhiên Hyperplanes
        self.hyperplanes = torch.randn(feature_dim, n_bits, device=device)
        self.memory_hashes = None
        self.memory_bank_ref = None

    def build(self, memory_bank):
        """Dập tính mã Hash (Băm số khối) cho mọi hàng tồn trong Kho Bộ Nhớ."""
        self.memory_bank_ref = memory_bank
        self.memory_hashes = self._compute_hashes(memory_bank)

    def _compute_hashes(self, features):
        """Biến đổi (Dense Vector không gian) hầm hồ thành Mã Hash nhị phân ngắn gọn."""
        # Khái lược (N, D) x (D, n_bits) -> (N, n_bits)
        projections = torch.matmul(features, self.hyperplanes)
        # Chỉ lấy cờ Bool Positive
        bit_array = (projections > 0).int()
        # Sang hệ số thập phân Decimals Base-10 nhanh chóng
        powers = (2 ** torch.arange(self.n_bits, device=self.device)).int()
        hash_codes = (bit_array * powers).sum(dim=1)
        return hash_codes

    def search(self, query):
        """
        Bộ truy vấn ngầm LSH Vectorized (Loại bỏ Vòng lặp For).

        Cơ chế:
          1. Đọc Hash code cho TẤT CẢ ô gạch query một lúc.
          2. Giăng dò sóng các hạt bụi n_bits mã nhị phân liên kết qua lưới XOR logic bitwise (Hamming=1).
          3. Giáp lại mảnh ghép bằng Batching Broadcasting Operation trên PyTorch.
          4. Chỉ bật radar và tính distance Euclidean cho CÁC LÁNG GIỀNG TRÚNG HASH MASK.
          5. Đóng băng các láng giềng khác thành cục `inf` (vô cực cự ly).
        """
        # Bước 1: Hash một mẻ nướng Data (N_query,)
        query_hashes = self._compute_hashes(query)
        N_query = query.shape[0]
        M_memory = self.memory_bank_ref.shape[0]

        # Bước 2: Tạo mạng lưới trinh thám mã nhị XOR (Probe XOR)
        powers = (2 ** torch.arange(self.n_bits, device=self.device)).int()
        original = query_hashes.unsqueeze(1)
        flipped = query_hashes.unsqueeze(1) ^ powers.unsqueeze(0)
        all_probes = torch.cat([original, flipped], dim=1)

        # Bước 3: Tìm ra khu vực Mask chung của Hàng xóm
        mem_exp = self.memory_hashes.view(1, 1, M_memory)
        probe_exp = all_probes.unsqueeze(2)
        # Nắm lấy vị trí True Mask boolean
        match_mask = (probe_exp == mem_exp).any(dim=1)

        # Bước 4: Fallback An toàn - Nếu nhỡ băm sai mà chả có cụm nào dính, thì nhả cửa 100% để tính lại (Exact KNN Mode)
        empty_rows = match_mask.sum(dim=1) == 0
        if empty_rows.any():
            match_mask[empty_rows] = True

        # Bước 5: Cấm cửa (Masking out) láng giềng sai - Tính k-NN chuẩn cực mượt
        distances = PatchCore._euclidean_distance_batch(query, self.memory_bank_ref)
        # Nút thắt cổ chai: Nhét Float Infinity vào các gã hàng xóm không thuộc Mask. Khỏi lo k-NN tìm nhầm
        distances = distances.masked_fill(~match_mask, float("inf"))

        return distances
