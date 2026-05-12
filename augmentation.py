import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# Paths
# ============================================================
sys.path.append(os.path.abspath("lib"))
from core import Core
from transformations import Transformations
from image_processing import ImageProcessing

INPUT_DIR = "preprocessing_Output"
OUTPUT_DIR = "augmentation_Output"
IMAGES_DIR = "augmentation_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Label mapping (integer → class name)
LABEL_MAP_INV = {
    0: 'buildings',
    1: 'forest',
    2: 'glacier',
    3: 'mountain',
    4: 'sea',
    5: 'street'
}

# ============================================================
# Helper Functions
# Convert image:
# float32 [0,1] ↔ uint8 [0,255]
# ============================================================
def to_uint8(img):
    return (Core.clip_pixel(img, 0, 1) * 255).astype(np.uint8)


def to_float32(img):
    return img.astype(np.float32) / 255.0


# ============================================================
# Augmentation Functions
# All augmentations use the provided library functions
# ============================================================

def aug_rotate(img_f32):
    """
    Augmentation 1:
    Rotate image by +20 degrees
    """
    img_u8 = to_uint8(img_f32)

    rotated = Transformations.rotate(
        img_u8,
        angle=20,
        method='bilinear'
    )

    return to_float32(rotated)


def aug_translate(img_f32):
    """
    Augmentation 2:
    Translate image to the right and downward
    """
    img_u8 = to_uint8(img_f32)

    translated = Transformations.translate(
        img_u8,
        tx=6,
        ty=6
    )

    return to_float32(translated)


def aug_flip(img_f32):
    """
    Augmentation 3:
    Horizontal flip
    """
    return np.fliplr(img_f32).copy()


def aug_brightness(img_f32):
    """
    Augmentation 4:
    Gamma correction for brightness enhancement
    gamma < 1 → brighter image
    """

    img_u8 = to_uint8(img_f32)

    result = np.zeros_like(img_u8)

    for c in range(3):

        result[:, :, c] = ImageProcessing.gamma_correction(
            img_u8[:, :, c],
            gamma=0.6
        )

    return to_float32(result)


def aug_blur(img_f32):
    """
    Augmentation 5:
    Gaussian blur applied per channel
    """

    img_u8 = to_uint8(img_f32)

    result = np.zeros_like(img_u8, dtype=np.float32)

    for c in range(3):

        blurred = ImageProcessing.gaussian_filter(
            img_u8[:, :, c].astype(np.float32),
            size=3,
            sigma=1.0
        )

        result[:, :, c] = Core.clip_pixel(blurred, 0, 255)

    return (result / 255.0).astype(np.float32)


# ============================================================
# List of augmentations
# ============================================================
AUGMENTATIONS = [
    ("Rotation",    aug_rotate),
    ("Translation", aug_translate),
    ("Flip",        aug_flip),
    ("Brightness",  aug_brightness),
    ("Blur",        aug_blur),
]

# ============================================================
# Load training data
# ============================================================
print("Loading training data...")

X_train = np.load(os.path.join(INPUT_DIR, 'X_train.npy'))
y_train = np.load(os.path.join(INPUT_DIR, 'y_train.npy'))

print(f"  X_train shape: {X_train.shape}")
print(f"  y_train shape: {y_train.shape}")

# ============================================================
# Apply augmentations
# ============================================================
print("\nApplying augmentations...")

# Start with original images
X_aug_list = [X_train]
y_aug_list = [y_train]

for aug_name, aug_fn in AUGMENTATIONS:

    print(f"  Applying {aug_name}...")

    augmented = np.array(
        [aug_fn(img) for img in X_train],
        dtype=np.float32
    )

    X_aug_list.append(augmented)
    y_aug_list.append(y_train)

# Concatenate all augmented datasets
X_train_aug = np.concatenate(X_aug_list, axis=0)
y_train_aug = np.concatenate(y_aug_list, axis=0)

print(f"\n  Original dataset : {X_train.shape[0]} images")

print(
    f"  Augmented dataset: "
    f"{X_train_aug.shape[0]} images "
    f"({len(AUGMENTATIONS)} augmentations + original)"
)

# ============================================================
# Save sample augmented images
# ============================================================
print("\nSaving sample augmented images...")

sample_indices = [0, 100, 300, 500, 800]

for idx in sample_indices:

    original = X_train[idx]
    label_str = LABEL_MAP_INV[y_train[idx]]

    # --------------------------------------------------------
    # Save original image
    # --------------------------------------------------------
    plt.imsave(
        os.path.join(
            IMAGES_DIR,
            f"img{idx}_{label_str}_original.jpg"
        ),
        Core.clip_pixel(original, 0, 1)
    )

    # --------------------------------------------------------
    # Save augmented versions
    # --------------------------------------------------------
    for aug_name, aug_fn in AUGMENTATIONS:

        aug_img = aug_fn(original)

        plt.imsave(
            os.path.join(
                IMAGES_DIR,
                f"img{idx}_{label_str}_{aug_name}.jpg"
            ),
            Core.clip_pixel(aug_img, 0, 1)
        )

print(f"  Sample images saved to: {IMAGES_DIR}/")

# ============================================================
# Generate Before/After Panel
# ============================================================
print("\nGenerating before/after panel...")

n_samples = 4

aug_names_all = ["Original"] + [
    name for name, _ in AUGMENTATIONS
]

aug_fns_all = [lambda x: x] + [
    fn for _, fn in AUGMENTATIONS
]

fig = plt.figure(figsize=(18, n_samples * 2.8))

fig.suptitle(
    "Augmentation — Before & After",
    fontsize=16,
    fontweight='bold',
    y=1.01
)

gs = gridspec.GridSpec(
    n_samples,
    6,
    figure=fig,
    hspace=0.4,
    wspace=0.15
)

sample_idxs = [0, 100, 300, 500]

for row, idx in enumerate(sample_idxs):

    img = X_train[idx]
    lbl = LABEL_MAP_INV[y_train[idx]]

    for col, (aug_name, aug_fn) in enumerate(
        zip(aug_names_all, aug_fns_all)
    ):

        ax = fig.add_subplot(gs[row, col])

        result = aug_fn(img)

        ax.imshow(Core.clip_pixel(result, 0, 1))
        ax.axis('off')

        # Column titles
        if row == 0:
            ax.set_title(
                aug_name,
                fontsize=10,
                fontweight='bold'
            )

        # Row labels
        if col == 0:
            ax.set_ylabel(
                lbl,
                fontsize=9,
                rotation=0,
                labelpad=45,
                va='center'
            )

# Save panel image
panel_path = os.path.join(
    OUTPUT_DIR,
    "before_after_panel.png"
)

plt.savefig(
    panel_path,
    dpi=150,
    bbox_inches='tight'
)

plt.close()

print(f"  Panel saved: {panel_path}")

# ============================================================
# Save augmented arrays
# ============================================================
np.save(
    os.path.join(OUTPUT_DIR, 'X_train_aug.npy'),
    X_train_aug
)

np.save(
    os.path.join(OUTPUT_DIR, 'y_train_aug.npy'),
    y_train_aug
)

# ============================================================
# Final Summary
# ============================================================
print(f"\n{'=' * 55}")
print("Augmentation Completed")
print(f"{'=' * 55}")

print("\nOutput Files:")

print(
    f"  {OUTPUT_DIR}/X_train_aug.npy "
    f"→ shape: {X_train_aug.shape}"
)

print(
    f"  {OUTPUT_DIR}/y_train_aug.npy "
    f"→ shape: {y_train_aug.shape}"
)

print(f"  {OUTPUT_DIR}/before_after_panel.png")

print(
    f"  {IMAGES_DIR}/ "
    f"(sample augmented images)"
)

print("\nApplied Augmentations:")

for aug_name, _ in AUGMENTATIONS:
    print(f"  - {aug_name}")