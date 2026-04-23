"""
Systematic formatting fixes for KHOA_LUAN_TOT_NGHIEP_EN.md
- Add Chapter 1 heading
- Fix section 1.5 -> 1.4 numbering
- Remove duplicate chapter title lines
- Fix sub-section numbering (X.Y -> X.Y.Z)
- Fix image width and references
"""
import re

FILE = r"c:\AIP\iad\docs\KHOA_LUAN_TOT_NGHIEP_EN.md"

with open(FILE, "r", encoding="utf-8") as f:
    content = f.read()

# ============================================================
# 1. Add "# CHAPTER 1: INTRODUCTION" heading before 1.1
# ============================================================
content = content.replace(
    '<div style="page-break-after: always;"></div>\r\n\r\n## 1.1 The Industrial Quality Assurance Problem',
    '<div style="page-break-after: always;"></div>\r\n\r\n# CHAPTER 1: INTRODUCTION\r\n\r\n## 1.1 The Industrial Quality Assurance Problem'
)

# ============================================================
# 2. Fix section 1.5 -> 1.4 (was missing 1.4)
# ============================================================
content = content.replace("## 1.5 Proposed System & Contributions", "## 1.4 Proposed System & Contributions")
content = content.replace("## 1.6 Thesis Organization", "## 1.5 Thesis Organization")

# ============================================================
# 3. Remove duplicate chapter title text lines
# Pattern: after "# CHAPTER X: TITLE", there's a duplicate plain text line
# ============================================================
duplicate_titles = [
    "\r\nRELATED WORK \u2014 A Critical Analysis of Unsupervised Anomaly Detection Paradigms\r\n",
    "\r\nFEATURE BACKBONE \u2014 Hierarchical Representation Learning via ResNet-18\r\n",
    "\r\nTHE PATCHCORE MEMORY ENGINE \u2014 Coreset Construction and Anomaly Scoring\r\n",
    "\r\nOPTIMIZATION INTERNALS \u2014 Approximate Search and Hardware-Aware Distance Computation\r\n",
    "\r\nSYSTEM PIPELINE & CALIBRATION \u2014 From Raw Images to Actionable Decisions\r\n",
    "\r\nAGENTIC ML & EXPLAINABLE AI \u2014 Closing the Optimization Loop\r\n",
    "\r\nEXPERIMENTS & EVALUATION \u2014 Empirical Validation on MVTec AD\r\n",
    "\r\nCONCLUSION & FUTURE WORK\r\n",
]
for dup in duplicate_titles:
    content = content.replace(dup, "\r\n")

# Also handle the --- lines around chapter transitions (consolidate double ---)
content = re.sub(r'\r\n---\r\n\r\n\r\n---\r\n', '\r\n---\r\n', content)

# ============================================================
# 4. Fix sub-section numbering per chapter
# The pattern is: inside Chapter X, sub-sections use ### X.Y 
# but they should be ### X.parent.Y
# ============================================================

# Chapter 2: Related Work
# Section 2.1 Reconstruction-Based Methods contains ### 2.1, ### 2.2, ### 2.3, ### 2.4
# These should be ### 2.1.1, ### 2.1.2, etc.
# But the ### headings currently say "### 2.1 Core Hypothesis", "### 2.2 The Bottleneck"...
# The actual pattern is: ## 2.1 has sub ### labeled as "### 2.1", "### 2.2", "### 2.3", "### 2.4"

# Let me fix these systematically chapter by chapter.

# Chapter 2 fixes
replacements = [
    # Section 2.1 sub-sections
    ("### 2.1 Core Hypothesis: The Manifold Assumption", "### 2.1.1 Core Hypothesis: The Manifold Assumption"),
    ("### 2.2 The Bottleneck as an Information Filter", "### 2.1.2 The Bottleneck as an Information Filter"),
    ("### 2.3 Failure Mode Analysis\r\n\r\nReconstruction-based methods", "### 2.1.3 Failure Mode Analysis\r\n\r\nReconstruction-based methods"),
    ("### 2.4 Variants and Extensions", "### 2.1.4 Variants and Extensions"),
    # Section 2.2 sub-sections
    ("### 2.1 Core Hypothesis: The Support Boundary", "### 2.2.1 Core Hypothesis: The Support Boundary"),
    ("### 2.2 The Feature Extraction Bottleneck", "### 2.2.2 The Feature Extraction Bottleneck"),
    ("### 2.3 Failure Mode Analysis\r\n\r\n**Failure Mode 1: Spatial blindness", "### 2.2.3 Failure Mode Analysis\r\n\r\n**Failure Mode 1: Spatial blindness"),
    ("### 2.4 Variants: Deep SVDD", "### 2.2.4 Variants: Deep SVDD"),
    # Section 2.3 sub-sections
    ("### 2.1 Core Hypothesis: Normal Patterns Form a Discrete Reference Set", "### 2.3.1 Core Hypothesis: Normal Patterns Form a Discrete Reference Set"),
    ("### 2.2 Evolution: From SPADE to PatchCore", "### 2.3.2 Evolution: From SPADE to PatchCore"),
    ("### 2.3 The Remaining Gap: Computational Efficiency", "### 2.3.3 The Remaining Gap: Computational Efficiency"),
    
    # Chapter 3 fixes
    ("### 3.1 Spatial Normalization: The 256", "### 3.1.1 Spatial Normalization: The 256"),
    ("### 3.2 Channel Normalization: ImageNet Statistics", "### 3.1.2 Channel Normalization: ImageNet Statistics"),
    ("### 3.1 Conv1: The 7", "### 3.2.1 Conv1: The 7"),
    ("### 3.2 Batch Normalization + ReLU", "### 3.2.2 Batch Normalization + ReLU"),
    ("### 3.3 MaxPool: Translation Invariance", "### 3.2.3 MaxPool: Translation Invariance"),
    ("### 3.1 The Degradation Problem and Its Solution", "### 3.3.1 The Degradation Problem and Its Solution"),
    ("### 3.2 Why Skip Connections Are Critical for Anomaly Detection", "### 3.3.2 Why Skip Connections Are Critical for Anomaly Detection"),
    ("### 3.3 Layer 1: Low-Level Feature Dictionary", "### 3.3.3 Layer 1: Low-Level Feature Dictionary"),
    ("### 3.4 Layer 2: Mid-Level Structural Features", "### 3.3.4 Layer 2: Mid-Level Structural Features"),
    ("### 3.1 The KD Objective", "### 3.5.1 The KD Objective"),
    ("### 3.2 Rationale for Knowledge Distillation in Anomaly Detection", "### 3.5.2 Rationale for Knowledge Distillation in Anomaly Detection"),
    
    # Chapter 4 fixes
    ("### 4.1 The Multi-Scale Representation Problem", "### 4.1.1 The Multi-Scale Representation Problem"),
    ("### 4.2 Bilinear Interpolation for Spatial Alignment", "### 4.1.2 Bilinear Interpolation for Spatial Alignment"),
    ("### 4.3 Channel Concatenation: The 384-Dimensional", "### 4.1.3 Channel Concatenation: The 384-Dimensional"),
    ("### 4.1 The Single-Point Fragility Problem", "### 4.2.1 The Single-Point Fragility Problem"),
    ("### 4.2 Spatial Average Pooling as a Smoothing Operator", "### 4.2.2 Spatial Average Pooling as a Smoothing Operator"),
    ("### 4.1 The Nominal Reference Set", "### 4.3.1 The Nominal Reference Set"),
    ("### 4.2 Why Not Use All Patches? The Redundancy Argument", "### 4.3.2 Why Not Use All Patches? The Redundancy Argument"),
    ("### 4.1 Problem Statement", "### 4.4.1 Problem Statement"),
    ("### 4.2 NP-Hardness and the Greedy Approximation", "### 4.4.2 NP-Hardness and the Greedy Approximation"),
    ("### 4.3 Complexity Analysis", "### 4.4.3 Complexity Analysis"),
    ("### 4.4 The Random Projection Acceleration", "### 4.4.4 The Random Projection Acceleration"),
    ("### 4.5 Coreset Ratio: The Accuracy-Efficiency Trade-Off", "### 4.4.5 Coreset Ratio: The Accuracy-Efficiency Trade-Off"),
    ("### 4.1 Patch-Level Scoring", "### 4.5.1 Patch-Level Scoring"),
    ("### 4.2 Image-Level Scoring: Why Maximum, Not Average?", "### 4.5.2 Image-Level Scoring: Why Maximum, Not Average?"),
    ("### 4.3 Failure Modes of the Max-Score Approach", "### 4.5.3 Failure Modes of the Max-Score Approach"),
    
    # Chapter 5 fixes
    ("### 5.1 The Hashing Principle", "### 5.2.1 The Hashing Principle"),
    ("### 5.2 Random Hyperplane Hashing", "### 5.2.2 Random Hyperplane Hashing"),
    ("### 5.3 Bucket Population Analysis", "### 5.2.3 Bucket Population Analysis"),
    ("### 5.1 The Hash Collision Failure Mode", "### 5.3.1 The Hash Collision Failure Mode"),
    ("### 5.2 XOR-Probe: Expanding the Search Radius", "### 5.3.2 XOR-Probe: Expanding the Search Radius"),
    ("### 5.3 Recall Analysis Under XOR-Probe", "### 5.3.3 Recall Analysis Under XOR-Probe"),
    ("### 5.1 The Naive Distance Computation Problem", "### 5.4.1 The Naive Distance Computation Problem"),
    ("### 5.2 Algebraic Decomposition", "### 5.4.2 Algebraic Decomposition"),
    ("### 5.3 Numerical Stability: The `torch.clamp` Guard", "### 5.4.3 Numerical Stability: The `torch.clamp` Guard"),
    ("### 5.4 Batched Matrix Multiplication on GPU", "### 5.4.4 Batched Matrix Multiplication on GPU"),
    ("### 5.5 Combined Speedup Analysis", "### 5.4.5 Combined Speedup Analysis"),
    ("### 5.1 LSH", "### 5.5.1 LSH"),
    ("### 5.2 LSH", "### 5.5.2 LSH"),
    ("### 5.1 Hash Collision Clustering", "### 5.6.1 Hash Collision Clustering"),
    ("### 5.2 Dimensional Mismatch", "### 5.6.2 Dimensional Mismatch"),
    
    # Chapter 6 fixes
    ("### 6.1 Illumination Stabilization via CLAHE", "### 6.1.1 Illumination Stabilization via CLAHE"),
    ("### 6.2 Automated Product Routing via YOLOv8", "### 6.1.2 Automated Product Routing via YOLOv8"),
    ("### 6.3 Concurrent Model Initialization", "### 6.1.3 Concurrent Model Initialization"),
    ("### 6.1 Score Distribution Heterogeneity", "### 6.2.1 Score Distribution Heterogeneity"),
    ("### 6.2 Category-Specific Score Distributions", "### 6.2.2 Category-Specific Score Distributions"),
    ("### 6.1 ROC Curve Construction", "### 6.3.1 ROC Curve Construction"),
    ("### 6.2 The Area Under the ROC Curve (AUROC)", "### 6.3.2 The Area Under the ROC Curve (AUROC)"),
    ("### 6.3 Youden's J Statistic: The Optimal Operating Point", "### 6.3.3 Youden's J Statistic: The Optimal Operating Point"),
    ("### 6.4 Why Thresholds Are High-Precision Decimals", "### 6.3.4 Why Thresholds Are High-Precision Decimals"),
    ("### 6.1 Definition and Mathematical Properties", "### 6.4.1 Definition and Mathematical Properties"),
    ("### 6.2 Limitations of Linear Normalization", "### 6.4.2 Limitations of Linear Normalization"),
    ("### 6.1 Three-Zone Classification", "### 6.5.1 Three-Zone Classification"),
    ("### 6.1 The Memory Budget Problem", "### 6.6.1 The Memory Budget Problem"),
    ("### 6.2 Sequential Processing Strategy", "### 6.6.2 Sequential Processing Strategy"),
    ("### 6.3 Cross-Module Interaction: Batching", "### 6.6.3 Cross-Module Interaction: Batching"),
    ("### 6.1 Localized Inference Interface", "### 6.8.1 Localized Inference Interface"),
    ("### 6.2 Synchronous Triage and Async Explanations", "### 6.8.2 Synchronous Triage and Async Explanations"),
    ("### 6.3 Containerization", "### 6.8.3 Containerization"),
    
    # Chapter 7 fixes
    ("### 7.1 Category-Specific Optimal Configurations", "### 7.1.1 Category-Specific Optimal Configurations"),
    ("### 7.2 The Limitations of Conventional AutoML", "### 7.1.2 The Limitations of Conventional AutoML"),
    ("### 7.1 LLM as a Research Agent", "### 7.2.1 LLM as a Research Agent"),
    ("### 7.2 The Reasoning Advantage Over Black-Box Optimizers", "### 7.2.2 The Reasoning Advantage Over Black-Box Optimizers"),
    ("### 7.3 Structured Prompt Engineering", "### 7.2.3 Structured Prompt Engineering"),
    ("### 7.1 When Does the Loop Stop?", "### 7.3.1 When Does the Loop Stop?"),
    ("### 7.2 Convergence Properties", "### 7.3.2 Convergence Properties"),
    ("### 7.1 The Interpretation Gap", "### 7.4.1 The Interpretation Gap"),
    ("### 7.2 The Semantic Bridge Architecture", "### 7.4.2 The Semantic Bridge Architecture"),
    ("### 7.3 Limitations of LLM-Based Diagnosis", "### 7.4.3 Limitations of LLM-Based Diagnosis"),
    ("### 7.1 Relationship to AutoML", "### 7.5.1 Relationship to AutoML"),
    ("### 7.2 Relationship to Human-in-the-Loop ML", "### 7.5.2 Relationship to Human-in-the-Loop ML"),
    
    # Chapter 8 fixes
    ("### 8.1 Dataset: MVTec AD", "### 8.1.1 Dataset: MVTec AD"),
    ("### 8.2 Hardware Configuration", "### 8.1.2 Hardware Configuration"),
    ("### 8.3 Evaluation Metrics", "### 8.1.3 Evaluation Metrics"),
    ("### 8.4 Implementation Details", "### 8.1.4 Implementation Details"),
    ("### 8.1 Image-Level AUROC by Category", "### 8.3.1 Image-Level AUROC by Category"),
    ("### 8.2 Analysis of Results", "### 8.3.2 Analysis of Results"),
    ("### 8.1 Ablation 1: Effect of Coreset Ratio", "### 8.5.1 Ablation 1: Effect of Coreset Ratio"),
    ("### 8.2 Ablation 2: Effect of LSH Configuration", "### 8.5.2 Ablation 2: Effect of LSH Configuration"),
    ("### 8.3 Ablation 3: Effect of Knowledge Distillation", "### 8.5.3 Ablation 3: Effect of Knowledge Distillation"),
    ("### 8.4 Ablation 4: Dimensionality Reduction via Random Projection", "### 8.5.4 Ablation 4: Dimensionality Reduction via Random Projection"),
    ("### 8.5 Ablation 5: Agentic Optimization Effectiveness", "### 8.5.5 Ablation 5: Agentic Optimization Effectiveness"),
    ("### 8.1 Systematic Failure:", "### 8.7.1 Systematic Failure:"),
    ("### 8.2 Systematic Failure: \"Screw\"", "### 8.7.2 Systematic Failure: \"Screw\""),
    ("### 8.3 Edge Cases: False Positives", "### 8.7.3 Edge Cases: False Positives"),
    ("### 8.4 Edge Cases: Missed Detections", "### 8.7.4 Edge Cases: Missed Detections"),
    
    # Chapter 11 fixes
    ("### 11.1 Architectural Limitations", "### 11.3.1 Architectural Limitations"),
    ("### 11.2 Methodological Limitations", "### 11.3.2 Methodological Limitations"),
    ("### 11.1 FP16/INT8 Mixed Precision Inference", "### 11.4.1 FP16/INT8 Mixed Precision Inference"),
    ("### 11.2 Multi-Modal Fusion", "### 11.4.2 Multi-Modal Fusion"),
    ("### 11.3 Online Memory Bank Updates", "### 11.4.3 Online Memory Bank Updates"),
    ("### 11.4 Edge Deployment via ONNX/TensorRT", "### 11.4.4 Edge Deployment via ONNX/TensorRT"),
]

for old, new in replacements:
    if old in content:
        content = content.replace(old, new, 1)
    else:
        print(f"  WARNING: Could not find: {old[:60]}...")

# ============================================================
# 5. Fix image width="800" -> responsive style
# ============================================================
content = content.replace(
    'alt="CustomResNet-18 Architecture via Knowledge Distillation" width="800"',
    'alt="CustomResNet-18 Architecture via Knowledge Distillation" style="max-width: 90%; width: auto;"'
)
content = content.replace(
    'alt="Enhanced PatchCore Engine Architecture" width="800"',
    'alt="Enhanced PatchCore Engine Architecture" style="max-width: 90%; width: auto;"'
)
content = content.replace(
    'alt="End-to-End System Pipeline and Calibration" width="800"',
    'alt="End-to-End System Pipeline and Calibration" style="max-width: 90%; width: auto;"'
)
content = content.replace(
    'alt="MVTec AD Dataset Samples - Nominal vs Defective" width="800"',
    'alt="MVTec AD Dataset Samples - Nominal vs Defective" style="max-width: 90%; width: auto;"'
)
content = content.replace(
    'alt="ROC Curve Comparison on MVTec AD" width="800"',
    'alt="ROC Curve Comparison on MVTec AD" style="max-width: 90%; width: auto;"'
)
content = content.replace(
    'alt="Anomaly Localization Heatmap Comparison" width="800"',
    'alt="Anomaly Localization Heatmap Comparison" style="max-width: 90%; width: auto;"'
)
content = content.replace(
    'alt="Pareto Frontier (Latency vs AUROC)" width="800"',
    'alt="Pareto Frontier (Latency vs AUROC)" style="max-width: 90%; width: auto;"'
)

# ============================================================
# 6. Fix Chapter 9 image references (remove _placeholder, remove TODO)
# ============================================================
content = content.replace(
    '  <!-- TODO: Replace with the actual screenshot of your main web application interface -->\r\n  <img src="assets/demo/main_interface_placeholder.png" alt="IAD Web Interface Screenshot" width="800" style="border: 1px solid #ccc;"/>',
    '  <img src="assets/demo/main_interface.png" alt="IAD Web Interface Screenshot" style="max-width: 90%; width: auto; border: 1px solid #ccc;"/>'
)
content = content.replace(
    '  <!-- TODO: Replace with the actual screenshot demonstrating an anomaly being detected (Grad-CAM heatmap overlaid) -->\r\n  <img src="assets/demo/heatmap_detection_placeholder.png" alt="Heatmap Anomaly Detection" width="800" style="border: 1px solid #ccc;"/>',
    '  <img src="assets/demo/heatmap_detection.png" alt="Heatmap Anomaly Detection" style="max-width: 60%; width: auto; border: 1px solid #ccc;"/>'
)
content = content.replace(
    '  <!-- TODO: Replace with a screenshot showing the chatbot explaining the anomaly score or interacting with the user -->\r\n  <img src="assets/demo/chatbot_placeholder.png" alt="XAI Chatbot Assistant" width="800" style="border: 1px solid #ccc;"/>',
    '  <img src="assets/demo/chatbot_assistant.png" alt="XAI Chatbot Assistant" style="max-width: 50%; width: auto; border: 1px solid #ccc;"/>'
)

# ============================================================
# 7. Fix Chapter 6 title (was "SYSTEM ARCHITECTURE" in heading but "SYSTEM PIPELINE" in subtitle)
# ============================================================
content = content.replace(
    "SYSTEM PIPELINE & CALIBRATION \u2014 From Raw Images to Actionable Decisions\r\n\r\n<div",
    "<div"
)

# ============================================================
# 8. Update Figure 8.3 caption to match new chart content
# ============================================================
content = content.replace(
    '<em>Figure 8.3: Qualitative comparison of pixel-level anomaly localization. Enhanced PatchCore produces sharp, precise boundary delineations compared to noisy Autoencoder reconstructions.</em>',
    '<em>Figure 8.3: Anomaly Localization Capability: Image-Level AUROC comparison (left) and PatchCore Pixel-Level AUROC demonstrating spatial localization precision (right).</em>'
)

# ============================================================
# Write output
# ============================================================
with open(FILE, "w", encoding="utf-8") as f:
    f.write(content)

print(f"[OK] All formatting fixes applied to {FILE}")
print(f"     File size: {len(content):,} bytes")
