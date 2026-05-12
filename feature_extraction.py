import os
import sys
import numpy as np

# ============================================================
# Paths
# ============================================================
sys.path.append(os.path.abspath("lib"))

from feature_extractor import FeatureExtractor

INPUT_DIR = "augmentation_Output"       # X_train_aug.npy + y_train_aug.npy
PREP_DIR = "preprocessing_Output"      # Validation/Test arrays
OUTPUT_DIR = "feature_extraction_Output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Feature Index Map
# Used to identify the meaning of each feature range
# ============================================================
#
# color_histogram  → dims   0 :  95   (96 dims)
# basic_statistics → dims  96 :  99   (4 dims)
# hog_lite         → dims 100 : 108   (9 dims)
# edge_histogram   → dims 109 : 252   (144 dims)
# ------------------------------------------------------------
# Total            → 253 dims
# ============================================================

FEATURE_INFO = {

    "color_histogram": {
        "start": 0,
        "end": 95,
        "dims": 96,
        "desc": "RGB Color Histogram (3×32 bins)"
    },

    "basic_statistics": {
        "start": 96,
        "end": 99,
        "dims": 4,
        "desc": "Statistical Moments (mean, std, skew, kurtosis)"
    },

    "hog_lite": {
        "start": 100,
        "end": 108,
        "dims": 9,
        "desc": "Histogram of Oriented Gradients (9 bins)"
    },

    "edge_histogram": {
        "start": 109,
        "end": 252,
        "dims": 144,
        "desc": "Edge Histogram Descriptor (4×4 grid, 9 bins)"
    }
}

TOTAL_DIMS = 253

# ============================================================
# Extract features from a single image
# ============================================================
def extract_features(img_f32: np.ndarray) -> np.ndarray:
    """
    Extract a feature vector from one image.

    Input:
        img_f32 : np.ndarray
            Shape: (64, 64, 3)
            Type : float32
            Range: [0,1]

    Output:
        vector : np.ndarray
            Shape: (253,)
            Type : float32
    """

    # --------------------------------------------------------
    # Convert image to uint8 for FeatureExtractor compatibility
    # --------------------------------------------------------
    img_u8 = (
        np.clip(img_f32, 0, 1) * 255
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Feature Extraction
    # --------------------------------------------------------

    # RGB Color Histogram → 96 dims
    f1 = FeatureExtractor.color_histogram(
        img_u8,
        bins=32
    )

    # Statistical Features → 4 dims
    f2 = FeatureExtractor.basic_statistics(
        img_u8
    )

    # HOG Features → 9 dims
    f3 = FeatureExtractor.hog_lite(
        img_u8,
        n_bins=9
    )

    # Edge Histogram Descriptor → 144 dims
    f4 = FeatureExtractor.edge_histogram_descriptor(
        img_u8,
        grid=(4, 4),
        n_bins=9
    )

    # --------------------------------------------------------
    # Combine all features into one vector
    # --------------------------------------------------------
    return np.concatenate(
        [f1, f2, f3, f4]
    ).astype(np.float32)


# ============================================================
# Extract features for a full dataset
# ============================================================
def extract_all(X: np.ndarray, name: str) -> np.ndarray:
    """
    Input:
        X : np.ndarray
            Shape: (N, 64, 64, 3)

    Output:
        features : np.ndarray
            Shape: (N, 253)
    """

    N = X.shape[0]

    features = np.zeros(
        (N, TOTAL_DIMS),
        dtype=np.float32
    )

    print(f"\n{'=' * 55}")
    print(
        f"Extracting features: "
        f"{name} ({N} images × {TOTAL_DIMS} dims)"
    )
    print(f"{'=' * 55}")

    for i in range(N):

        features[i] = extract_features(X[i])

        # Progress update
        if (i + 1) % 500 == 0 or (i + 1) == N:
            print(f"  {i + 1}/{N} completed...", end='\r')

    print(f"\n  Done! Shape: {features.shape}")

    return features


# ============================================================
# Load datasets
# ============================================================
print("Loading datasets...")

X_train = np.load(
    os.path.join(INPUT_DIR, 'X_train_aug.npy')
)

y_train = np.load(
    os.path.join(INPUT_DIR, 'y_train_aug.npy')
)

X_val = np.load(
    os.path.join(PREP_DIR, 'X_validation.npy')
)

y_val = np.load(
    os.path.join(PREP_DIR, 'y_validation.npy')
)

X_test = np.load(
    os.path.join(PREP_DIR, 'X_test.npy')
)

y_test = np.load(
    os.path.join(PREP_DIR, 'y_test.npy')
)

print(f"  X_train : {X_train.shape}")
print(f"  X_val   : {X_val.shape}")
print(f"  X_test  : {X_test.shape}")

# ============================================================
# Feature Extraction
# ============================================================
F_train = extract_all(X_train, "Train")
F_val = extract_all(X_val, "Validation")
F_test = extract_all(X_test, "Test")

# ============================================================
# Save Features and Labels
# ============================================================
np.save(
    os.path.join(OUTPUT_DIR, 'F_train.npy'),
    F_train
)

np.save(
    os.path.join(OUTPUT_DIR, 'F_val.npy'),
    F_val
)

np.save(
    os.path.join(OUTPUT_DIR, 'F_test.npy'),
    F_test
)

np.save(
    os.path.join(OUTPUT_DIR, 'y_train.npy'),
    y_train
)

np.save(
    os.path.join(OUTPUT_DIR, 'y_val.npy'),
    y_val
)

np.save(
    os.path.join(OUTPUT_DIR, 'y_test.npy'),
    y_test
)

# ============================================================
# Final Summary
# ============================================================
print(f"\n{'=' * 55}")
print("Feature Extraction Completed")
print(f"{'=' * 55}")

print(f"\nOutput Files → {OUTPUT_DIR}/")

print(f"  F_train.npy → {F_train.shape} float32")
print(f"  F_val.npy   → {F_val.shape}")
print(f"  F_test.npy  → {F_test.shape}")

print("\nFeature Vector Layout (253 dimensions):")

print(
    f"  {'Feature':<20} "
    f"{'Dims':>5}  "
    f"{'Index Range':<15} "
    f"Description"
)

print(f"  {'-' * 70}")

for fname, info in FEATURE_INFO.items():

    print(
        f"  {fname:<20} "
        f"{info['dims']:>5}  "
        f"[{info['start']:>3} : {info['end']:>3}]   "
        f"{info['desc']}"
    )

print(f"  {'-' * 70}")
print(f"  {'TOTAL':<20} {TOTAL_DIMS:>5}")



# ============================================================
# Before / After Augmentation Panel — Feature Extraction
# Add this block at the END of feature_extraction.py
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

LABEL_MAP_INV = {0:'buildings', 1:'forest', 2:'glacier',
                 3:'mountain',  4:'sea',    5:'street'}

# Load original (pre-augmentation) train images
X_orig = np.load(os.path.join("preprocessing_Output", "X_train.npy"))
y_orig = np.load(os.path.join("preprocessing_Output", "y_train.npy"))
N_orig = X_orig.shape[0]

# Pick 3 sample indices
SAMPLES     = [0, 100, 300]
AUG_NAMES   = ["Original", "Rotation", "Translation", "Flip", "Brightness", "Blur"]

fig = plt.figure(figsize=(22, len(SAMPLES) * 4.5))
fig.suptitle("Feature Extraction — Before & After Augmentation\n"
             "(Each row = same image in different augmentation versions)",
             fontsize=13, fontweight='bold')

gs = gridspec.GridSpec(len(SAMPLES), len(AUG_NAMES),
                       figure=fig, hspace=0.5, wspace=0.25)

for row, sample_idx in enumerate(SAMPLES):

    label_str = LABEL_MAP_INV[y_orig[sample_idx]]

    # Collect all versions of this image:
    # version 0 = original, version k = augmentation k
    versions = [X_orig[sample_idx]] + \
               [X_train[N_orig * k + sample_idx] for k in range(1, 6)]

    for col, (aug_name, img) in enumerate(zip(AUG_NAMES, versions)):

        feat = extract_features(img)    # already defined above in feature_extraction.py

        ax = fig.add_subplot(gs[row, col])

        # Top half: image
        # We split the axes area manually using inset_axes
        ax_img = ax.inset_axes([0, 0.5, 1, 0.5])
        ax_img.imshow(np.clip(img, 0, 1))
        ax_img.axis('off')
        if row == 0:
            ax_img.set_title(aug_name, fontsize=9, fontweight='bold')

        # Bottom half: feature vector plot
        ax_feat = ax.inset_axes([0, 0, 1, 0.45])
        ax_feat.plot(feat, linewidth=0.6, color='#4E79A7')
        ax_feat.axvline(x=96,  color='red',    linewidth=0.8, linestyle='--', alpha=0.6)
        ax_feat.axvline(x=100, color='green',  linewidth=0.8, linestyle='--', alpha=0.6)
        ax_feat.axvline(x=109, color='orange', linewidth=0.8, linestyle='--', alpha=0.6)
        ax_feat.set_xlim(0, 253)
        ax_feat.set_ylim(0, None)
        ax_feat.tick_params(labelsize=5)
        ax_feat.grid(alpha=0.3)
        if col == 0:
            ax_feat.set_ylabel(f"{label_str}\n253 dims", fontsize=7)

        ax.axis('off')   # hide the outer axes frame

# Legend for vertical lines
legend_elements = [
    plt.Line2D([0], [0], color='red',    linestyle='--', linewidth=1, label='stats (idx 96)'),
    plt.Line2D([0], [0], color='green',  linestyle='--', linewidth=1, label='hog   (idx 100)'),
    plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=1, label='edge  (idx 109)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3,
           fontsize=9, framealpha=0.8, bbox_to_anchor=(0.5, -0.02))

panel_path = os.path.join(OUTPUT_DIR, "features_before_after_aug.png")
plt.savefig(panel_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved panel: {panel_path}")