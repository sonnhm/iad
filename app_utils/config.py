"""
Config loader — đọc YAML configuration files.

Centralized constants and YAML config reader for the IAD project.
"""

import yaml

# ---------------------------------------------------------------------------
# Centralized project constants
# ---------------------------------------------------------------------------

DATA_ROOT = "datasets/mvtec"

ALL_CATEGORIES = [
    # Objects (10)
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
    # Textures (5)
    "carpet",
    "grid",
    "leather",
    "tile",
    "wood",
]


def load_config(config_path):
    """
    Đọc file YAML config.

    Args:
        config_path: đường dẫn tới file .yaml

    Returns:
        dict chứa các config parameters
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    return config
