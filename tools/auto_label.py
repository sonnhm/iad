import os

import cv2
import numpy as np
from tqdm import tqdm


def get_bbox_from_image(image_path, category):
    """
    Generate a bounding box for the object in the image.
    For textures, returns the full image.
    For objects, uses thresholding and contour detection.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]

    # Categories that are textures (full frame)
    textures = ["carpet", "grid", "leather", "tile", "wood"]
    if category in textures:
        return [
            0.5,
            0.5,
            1.0,
            1.0,
        ]  # YOLO format: x_center, y_center, width, height (normalized)

    # For objects, try to find the foreground
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if category == "cable":
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    else:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [0.5, 0.5, 1.0, 1.0]

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest_contour)

    x_center = (x + bw / 2) / w
    y_center = (y + bh / 2) / h
    width = bw / w
    height = bh / h

    return [x_center, y_center, width, height]


def main():
    base_path = r"C:\AIP\iad\datasets\mvtec"
    # We will store labels in a separate folder but reference original images
    dataset_path = r"C:\AIP\iad\datasets\yolo_dataset"
    os.makedirs(dataset_path, exist_ok=True)

    categories = sorted(
        [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    )

    train_paths = []
    val_paths = []

    # Labels directory matching image path structure or flat?
    # YOLO prefers labels and images to have the same directory structure.
    # Since original images are scattered, we'll create a flat labels folder and a text file listing images.

    labels_root = os.path.join(dataset_path, "labels")
    os.makedirs(labels_root, exist_ok=True)

    for cat_idx, category in enumerate(categories):
        print(f"Processing category: {category}")
        cat_path = os.path.join(base_path, category)

        # Collect train/test images
        for split in ["train", "test"]:
            split_path = os.path.join(cat_path, split)
            if not os.path.exists(split_path):
                continue

            for root, _, files in os.walk(split_path):
                for f in files:
                    if f.endswith(".png"):
                        img_path = os.path.join(root, f)
                        bbox = get_bbox_from_image(img_path, category)
                        if bbox is None:
                            continue

                        # Generate a unique name for the label file
                        # replace separators to avoid issues
                        rel_path = os.path.relpath(img_path, base_path).replace(
                            os.sep, "_"
                        )
                        label_filename = rel_path.replace(".png", ".txt")
                        label_path = os.path.join(labels_root, label_filename)

                        with open(label_path, "w") as lf:
                            lf.write(
                                f"{cat_idx} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n"
                            )

                        # Store image path and label path mapping?
                        # YOLOv8 expects labels to be in ../labels/ relative to images/
                        # This is tricky with existing scattered images.

                        # Alternative: Copy ONLY labels and use a list of image paths.
                        # BUT YOLOv8 needs the label file to be in the same place as the image
                        # OR in a 'labels' folder adjacent to the 'images' folder.

                        # Final Decision: Copy images using shutil.copy.
                        # It's MUCH faster than cv2.imwrite.
                        target_images_dir = os.path.join(dataset_path, "images", "all")
                        target_labels_dir = os.path.join(dataset_path, "labels", "all")
                        os.makedirs(target_images_dir, exist_ok=True)
                        os.makedirs(target_labels_dir, exist_ok=True)

                        import shutil

                        shutil.copy(
                            img_path,
                            os.path.join(
                                target_images_dir,
                                label_filename.replace(".txt", ".png"),
                            ),
                        )
                        shutil.move(
                            label_path, os.path.join(target_labels_dir, label_filename)
                        )

                        final_img_path = os.path.join(
                            target_images_dir, label_filename.replace(".txt", ".png")
                        )
                        if split == "train":
                            train_paths.append(final_img_path)
                        else:
                            val_paths.append(final_img_path)

    # Write train.txt and val.txt
    with open(os.path.join(dataset_path, "train.txt"), "w") as f:
        for p in train_paths:
            f.write(f"{p}\n")
    with open(os.path.join(dataset_path, "val.txt"), "w") as f:
        for p in val_paths:
            f.write(f"{p}\n")

    # Generate dataset.yaml
    yaml_content = f"""
path: {dataset_path}
train: train.txt
val: val.txt

names:
"""
    for i, cat in enumerate(categories):
        yaml_content += f"  {i}: {cat}\n"

    with open(os.path.join(dataset_path, "dataset.yaml"), "w") as f:
        f.write(yaml_content)

    print(f"Dataset prepared: {len(train_paths)} train, {len(val_paths)} val images.")


if __name__ == "__main__":
    main()
