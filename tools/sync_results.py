import csv
import json
import os

header = [
    "Category",
    "AE_AUROC",
    "AE_AP",
    "AE_F1",
    "AE_Precision",
    "AE_Recall",
    "AE_Specificity",
    "OCSVM_AUROC",
    "OCSVM_AP",
    "OCSVM_F1",
    "OCSVM_Precision",
    "OCSVM_Recall",
    "OCSVM_Specificity",
    "PC_AUROC",
    "PC_AP",
    "PC_F1",
    "PC_Precision",
    "PC_Recall",
    "PC_Specificity",
    "PC_Pixel_AUROC",
    "PC_PRO",
]

results_dir = "results"
output_file = "results.csv"


def sync():
    if not os.path.exists(results_dir):
        print("Results dir not found.")
        return

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for cat_file in sorted(os.listdir(results_dir)):
            if cat_file.endswith(".json"):
                path = os.path.join(results_dir, cat_file)
                with open(path) as j:
                    d = json.load(j)
                    row = [d["category"]]

                    # AE
                    ae = d.get("autoencoder") or {}
                    row.extend(
                        [
                            f"{ae.get('auroc', 0):.4f}",
                            f"{ae.get('average_precision', 0):.4f}",
                            f"{ae.get('f1', 0):.4f}",
                            f"{ae.get('precision', 0):.4f}",
                            f"{ae.get('recall', 0):.4f}",
                            f"{ae.get('specificity', 0):.4f}",
                        ]
                    )
                    # OCSVM
                    oc = d.get("cnn_ocsvm") or {}
                    row.extend(
                        [
                            f"{oc.get('auroc', 0):.4f}",
                            f"{oc.get('average_precision', 0):.4f}",
                            f"{oc.get('f1', 0):.4f}",
                            f"{oc.get('precision', 0):.4f}",
                            f"{oc.get('recall', 0):.4f}",
                            f"{oc.get('specificity', 0):.4f}",
                        ]
                    )
                    # PatchCore
                    pc = d.get("patchcore") or {}
                    row.extend(
                        [
                            f"{pc.get('auroc', 0):.4f}",
                            f"{pc.get('average_precision', 0):.4f}",
                            f"{pc.get('f1', 0):.4f}",
                            f"{pc.get('precision', 0):.4f}",
                            f"{pc.get('recall', 0):.4f}",
                            f"{pc.get('specificity', 0):.4f}",
                            f"{pc.get('pixel_auroc', 0):.4f}",
                            f"{pc.get('pro_score', 0):.4f}",
                        ]
                    )
                    writer.writerow(row)
    print(f"Successfully synced {output_file}")


if __name__ == "__main__":
    sync()
