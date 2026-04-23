"""
MVTec AD Dataset Downloader Helper

This script provides instructions and commands to download the MVTec AD dataset.
Due to the large size of the dataset (~5GB), automatic download is disabled to prevent
accidental bandwidth and storage consumption.
"""

import os
import sys


def print_instructions():
    print("=" * 60)
    print("MVTec AD Dataset Download Instructions")
    print("=" * 60)
    print("The dataset is approximately 5GB in size.")
    print("Download link: https://www.mvtec.com/company/research/datasets/mvtec-ad")
    print(
        "\nOr use the following direct curl/wget command (requires registration/auth if they changed the CDN, though historically it was open):"
    )
    print(
        "  wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"
    )
    print("\nExpected MD5 Checksum: ecd364a51e605d336fe0addabcedb7c2")
    print(
        "\nAfter downloading, extract the archive to the `datasets/mvtec/` directory."
    )
    print("The structure should look like:")
    print("  datasets/mvtec/")
    print("    ├── bottle/")
    print("    ├── cable/")
    print("    ├── capsule/")
    print("    └── ... (15 categories)")
    print("=" * 60)


if __name__ == "__main__":
    print_instructions()
    sys.exit(0)
