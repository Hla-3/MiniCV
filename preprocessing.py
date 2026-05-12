import os
import csv
import numpy as np
import sys

# ============================================================
# Add lib path for importing custom modules
# ============================================================
sys.path.append(os.path.abspath("lib"))

from Io import IO
from transformations import Transformations
from core import Core

# ============================================================
# Settings
# ============================================================
TARGET_SIZE = (64, 64)      # Target size for resizing images
OUTPUT_DIR = "preprocessing_Output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Class label mapping
LABEL_MAP = {
    'buildings': 0,
    'forest':    1,
    'glacier':   2,
    'mountain':  3,
    'sea':       4,
    'street':    5
}

# ============================================================
# Function to load and preprocess dataset splits
# ============================================================
def load_split(csv_path: str, split_name: str):
    """
    Load images from a CSV file and apply preprocessing:

    1. Resize images to 64x64
    2. Normalize pixel values to [0,1]
    3. Convert labels to integer IDs

    Returns:
        X : np.ndarray
            Shape: (N, 64, 64, 3)
            Type : float32
            Range: [0,1]

        y : np.ndarray
            Shape: (N,)
            Type : int32
    """

    X_list = []
    y_list = []
    errors = 0

    # --------------------------------------------------------
    # Read CSV file
    # --------------------------------------------------------
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)

    print(f"\n{'=' * 50}")
    print(f"Loading: {split_name} ({total} images)")
    print(f"{'=' * 50}")

    # --------------------------------------------------------
    # Process images one by one
    # --------------------------------------------------------
    for i, row in enumerate(rows):

        img_path = row['image_path']
        label = row['label']

        # Print progress every 200 images
        if (i + 1) % 200 == 0 or (i + 1) == total:
            print(f"  {i + 1}/{total} images loaded...", end='\r')

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------
        try:
            img = IO.read_image(img_path, normalize=False)

        except Exception as e:
            errors += 1
            continue

        # ----------------------------------------------------
        # Handle RGBA images (remove alpha channel)
        # ----------------------------------------------------
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]

        # ----------------------------------------------------
        # Handle grayscale images
        # ----------------------------------------------------
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)

        # ----------------------------------------------------
        # Resize image
        # ----------------------------------------------------
        img_resized = Transformations.resize(
            img,
            TARGET_SIZE,
            method='bilinear'
        )

        # ----------------------------------------------------
        # Normalize image
        # uint8 [0,255] → float32 [0,1]
        # ----------------------------------------------------
        img_norm = img_resized.astype(np.float32) / 255.0

        # ----------------------------------------------------
        # Store processed image and label
        # ----------------------------------------------------
        X_list.append(img_norm)
        y_list.append(LABEL_MAP[label])

    # --------------------------------------------------------
    # Print loading summary
    # --------------------------------------------------------
    print(f"\n  Successfully loaded {len(X_list)} images")

    if errors > 0:
        print(f"  Skipped {errors} corrupted images")

    # Convert lists to NumPy arrays
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    return X, y


# ============================================================
# Process all dataset splits
# ============================================================
CSV_DIR = "prepare_dataset_Output"

splits = [
    (os.path.join(CSV_DIR, 'train.csv'), 'Train'),
    (os.path.join(CSV_DIR, 'val.csv'),   'Validation'),
    (os.path.join(CSV_DIR, 'test.csv'),  'Test'),
]

for csv_file, name in splits:

    X, y = load_split(csv_file, name)

    # --------------------------------------------------------
    # Save processed arrays
    # --------------------------------------------------------
    prefix = name.lower()

    np.save(
        os.path.join(OUTPUT_DIR, f'X_{prefix}.npy'),
        X
    )

    np.save(
        os.path.join(OUTPUT_DIR, f'y_{prefix}.npy'),
        y
    )

    print(f"  X_{prefix}.npy saved → shape: {X.shape}")
    print(f"  y_{prefix}.npy saved → shape: {y.shape}")


# ============================================================
# Final Summary
# ============================================================
print(f"\n{'=' * 50}")
print("Preprocessing Completed")
print(f"{'=' * 50}")

print(f"Output folder: {OUTPUT_DIR}/")
print()

print("Saved Files:")
print("  X_train.npy  → (N_train, 64, 64, 3) float32 [0,1]")
print("  y_train.npy  → (N_train,)            int32")

print("  X_val.npy    → (N_val, 64, 64, 3)")
print("  y_val.npy    → (N_val,)")

print("  X_test.npy   → (N_test, 64, 64, 3)")
print("  y_test.npy   → (N_test,)")

print()
print("Label Mapping:")

for class_name, label_id in LABEL_MAP.items():
    print(f"  {label_id} = {class_name}")