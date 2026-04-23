"""
Add table captions to KHOA_LUAN_TOT_NGHIEP_EN.md.
Each content table gets a "Table X.Y: Description" caption below it.
"""
import re

FILE = r"c:\AIP\iad\docs\KHOA_LUAN_TOT_NGHIEP_EN.md"

with open(FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Define table captions mapped by the HEADER LINE content (first |..| row before separator)
# Format: (header_line_number, last_row_line_number, caption)
# Line numbers are 1-indexed from the analysis
table_captions = [
    # Skip line 16 (cover page team) and 97 (acronyms) - front matter, no captions needed
    
    # Chapter 2
    (353, 359, "Table 2.1: Paradigm comparison and positioning of contributions across reconstruction, distribution, and embedding-based methods."),
    
    # Chapter 3
    (412, 416, "Table 3.1: ImageNet normalization constants per channel used for input conditioning."),
    (504, 509, "Table 3.2: Layer selection trade-off for PatchCore feature extraction."),
    (551, 557, "Table 3.3: Computational cost analysis by backbone stage (MAC operations for 256x256 input)."),
    
    # Chapter 4
    (727, 732, "Table 4.1: Coreset ratio vs accuracy-efficiency trade-off for the memory bank."),
    
    # Chapter 5
    (926, 930, "Table 5.1: Decomposed squared Euclidean distance computation strategy."),
    (969, 973, "Table 5.2: Combined speedup analysis of LSH + XOR-Probe vs exact search."),
    
    # Chapter 6
    (1065, 1069, "Table 6.1: Score distribution heterogeneity across the three deployed models."),
    (1165, 1169, "Table 6.2: Risk severity mapping for industrial deployment based on the Anomaly Index."),
    (1186, 1196, "Table 6.3: VRAM memory budget breakdown for concurrent three-model deployment."),
    
    # Chapter 7
    (1285, 1289, "Table 7.1: Category-specific hyperparameter sensitivity analysis."),
    (1478, 1483, "Table 7.2: Cross-module interaction analysis for the Agentic optimization loop."),
    
    # Chapter 8
    (1511, 1518, "Table 8.1: MVTec AD dataset statistics (15 categories, 5,354 images)."),
    (1532, 1538, "Table 8.2: Hardware configuration for all experiments."),
    (1544, 1550, "Table 8.3: Evaluation metrics summary and their interpretation."),
    (1554, 1562, "Table 8.4: Implementation details per model architecture."),
    (1570, 1576, "Table 8.5: Pipeline evolution from V1 (baseline) to V5 (production)."),
    (1588, 1605, "Table 8.6: Image-level AUROC comparison across all 15 MVTec AD categories."),
    (1644, 1652, "Table 8.7: SOTA benchmarking comparison with published methods."),
    (1680, 1684, "Table 8.8a: Ablation 1 -- Effect of coreset ratio on PatchCore performance."),
    (1690, 1696, "Table 8.8b: Ablation 2 -- Effect of LSH configuration on search performance."),
    (1702, 1707, "Table 8.8c: Ablation 3 -- Effect of Knowledge Distillation on backbone quality."),
    (1715, 1720, "Table 8.8d: Ablation 4 -- Dimensionality reduction via random projection."),
    (1731, 1747, "Table 8.8e: Ablation 5 -- Agentic optimization effectiveness per category."),
    (1774, 1786, "Table 8.9: Latency profiling breakdown for single-image inference."),
    
    # Chapter 10
    (1877, 1881, "Table 10.1: Research team structure and responsibilities."),
    (1887, 1894, "Table 10.2: Project schedule and milestone orchestration."),
    
    # Appendices
    (2069, 2081, "Table A.1: Complete system configuration parameters and tuning ranges."),
    (2114, 2129, "Table A.2: Extended glossary of technical terms."),
]

# Process in reverse order so line numbers don't shift
added = 0
for header_line, last_row, caption in reversed(table_captions):
    # Convert to 0-indexed
    insert_idx = last_row  # Insert after last row (0-indexed = last_row since last_row is 1-indexed)
    
    # Check bounds
    if insert_idx >= len(lines):
        print(f"  SKIP: Line {last_row} out of range for: {caption[:50]}")
        continue
    
    # Verify this line is actually a table row
    if not lines[insert_idx - 1].strip().startswith("|"):
        print(f"  WARN: Line {last_row} is not a table row: {lines[insert_idx-1].strip()[:60]}")
        # Try to find the actual last row nearby
        for offset in range(-2, 3):
            check = insert_idx - 1 + offset
            if 0 <= check < len(lines) and lines[check].strip().startswith("|"):
                insert_idx = check + 1
                break
    
    # Build caption HTML
    caption_html = f'\n<p align="center"><em>{caption}</em></p>\n\n'
    
    # Insert after the last table row
    lines.insert(insert_idx, caption_html)
    added += 1
    print(f"  OK: L{last_row} -> {caption[:60]}...")

with open(FILE, "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"\n[DONE] Added {added} table captions to {FILE}")
