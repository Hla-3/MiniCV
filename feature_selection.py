import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Install mrmr if not already installed:
#   pip install mrmr-selection
# ============================================================
try:
    from mrmr import mrmr_classif
    import pandas as pd
except ImportError:
    print("ERROR: Required packages not found.")
    print("Please run:  pip install mrmr-selection pandas")
    sys.exit(1)

# ============================================================
# Paths
# ============================================================
INPUT_DIR  = "feature_extraction_Output"
OUTPUT_DIR = "feature_selection_Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Number of top features to select
K = 100

# ============================================================
# Feature names for readability
# Index map (from feature_extraction.py):
#   [  0 :  95 ]  color_histogram  (96 dims)
#   [ 96 :  99 ]  basic_statistics  (4 dims)
#   [100 : 108 ]  hog_lite          (9 dims)
#   [109 : 252 ]  edge_histogram  (144 dims)
# ============================================================
def build_feature_names(total=253):
    names = []
    for i in range(96):
        names.append(f"color_hist_{i:03d}")
    for name in ["mean", "std", "skewness", "kurtosis"]:
        names.append(f"stats_{name}")
    for i in range(9):
        names.append(f"hog_{i:02d}")
    for i in range(144):
        names.append(f"edge_{i:03d}")
    return names

FEATURE_NAMES = build_feature_names()

# ============================================================
# Load Data
# ============================================================
print("Loading features...")

F_train = np.load(os.path.join(INPUT_DIR, 'F_train.npy'))
F_val   = np.load(os.path.join(INPUT_DIR, 'F_val.npy'))
F_test  = np.load(os.path.join(INPUT_DIR, 'F_test.npy'))
y_train = np.load(os.path.join(INPUT_DIR, 'y_train.npy'))
y_val   = np.load(os.path.join(INPUT_DIR, 'y_val.npy'))
y_test  = np.load(os.path.join(INPUT_DIR, 'y_test.npy'))

print(f"  F_train : {F_train.shape}")
print(f"  F_val   : {F_val.shape}")
print(f"  F_test  : {F_test.shape}")

# ============================================================
# Run MRMR
# (MRMR works with pandas DataFrames)
# ============================================================
print(f"\nRunning MRMR — selecting top {K} features from {F_train.shape[1]}...")

# Convert to DataFrame (required by mrmr library)
df_X = pd.DataFrame(F_train, columns=FEATURE_NAMES)
df_y = pd.Series(y_train, name='label')

# Run MRMR selection
selected_names = mrmr_classif(X=df_X, y=df_y, K=K)

# Convert selected names back to integer indices
selected_indices = np.array([FEATURE_NAMES.index(name) for name in selected_names])

print(f"\nSelected {len(selected_indices)} features.")
print(f"Selected indices (first 10): {selected_indices[:10]}")

# ============================================================
# Apply Selection to all splits
# ============================================================
F_train_sel = F_train[:, selected_indices]
F_val_sel   = F_val[:,   selected_indices]
F_test_sel  = F_test[:,  selected_indices]

print(f"\nAfter selection:")
print(f"  F_train : {F_train.shape} → {F_train_sel.shape}")
print(f"  F_val   : {F_val.shape}   → {F_val_sel.shape}")
print(f"  F_test  : {F_test.shape}  → {F_test_sel.shape}")

# ============================================================
# Save Outputs
# ============================================================
np.save(os.path.join(OUTPUT_DIR, 'F_train_selected.npy'), F_train_sel)
np.save(os.path.join(OUTPUT_DIR, 'F_val_selected.npy'),   F_val_sel)
np.save(os.path.join(OUTPUT_DIR, 'F_test_selected.npy'),  F_test_sel)
np.save(os.path.join(OUTPUT_DIR, 'selected_indices.npy'), selected_indices)
np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'),          y_train)
np.save(os.path.join(OUTPUT_DIR, 'y_val.npy'),            y_val)
np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'),           y_test)

# Save selected feature names as text file
with open(os.path.join(OUTPUT_DIR, 'selected_feature_names.txt'), 'w') as f:
    f.write(f"Top {K} features selected by MRMR\n")
    f.write("=" * 40 + "\n")
    for rank, name in enumerate(selected_names, 1):
        idx = FEATURE_NAMES.index(name)
        f.write(f"Rank {rank:3d}  |  index {idx:3d}  |  {name}\n")

print(f"\nSaved selected_feature_names.txt")

# ============================================================
# Plot: Feature Family Distribution in Selected Features
# ============================================================
families = {"color_histogram": 0, "basic_statistics": 0,
            "hog_lite": 0,        "edge_histogram": 0}

for idx in selected_indices:
    if   idx <= 95:  families["color_histogram"]  += 1
    elif idx <= 99:  families["basic_statistics"] += 1
    elif idx <= 108: families["hog_lite"]          += 1
    else:            families["edge_histogram"]    += 1

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"MRMR Feature Selection — Top {K} Features", fontsize=14, fontweight='bold')

colors = ['#4E79A7', '#F28E2B', '#59A14F', '#E15759']

# --- Bar chart: how many from each family ---
ax = axes[0]
bars = ax.bar(families.keys(), families.values(), color=colors, edgecolor='white', linewidth=1.5)
ax.set_title("Selected Features per Family")
ax.set_xlabel("Feature Family")
ax.set_ylabel("Count")
ax.tick_params(axis='x', rotation=15)
for bar, val in zip(bars, families.values()):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3, str(val),
            ha='center', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# --- Pie chart ---
ax2 = axes[1]
ax2.pie(families.values(), labels=families.keys(), colors=colors,
        autopct='%1.1f%%', startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2})
ax2.set_title("Distribution (%)")

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, 'mrmr_scores.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved plot: {plot_path}")

# ============================================================
# Final Summary
# ============================================================
print(f"\n{'='*55}")
print("MRMR Feature Selection Completed!")
print(f"{'='*55}")
print(f"\nDimensionality Reduction:")
print(f"  Before : 253 features")
print(f"  After  : {K}  features  ({K/253*100:.1f}% of original)")
print(f"\nSelected Features by Family:")
for fam, count in families.items():
    print(f"  {fam:<22} : {count:>3} / {K} selected")
print(f"\nOutput Files → {OUTPUT_DIR}/")
print(f"  F_train_selected.npy     → {F_train_sel.shape}")
print(f"  F_val_selected.npy       → {F_val_sel.shape}")
print(f"  F_test_selected.npy      → {F_test_sel.shape}")
print(f"  selected_indices.npy     → {selected_indices.shape}")
print(f"  selected_feature_names.txt")
print(f"  mrmr_scores.png")

# ============================================================
# Before / After Augmentation Panel — Feature Selection (MRMR)
# Add this block at the END of feature_selection.py
# ============================================================
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
sys.path.append("lib")
from feature_extractor import FeatureExtractor

LABEL_MAP_INV = {0:'buildings', 1:'forest', 2:'glacier',
                 3:'mountain',  4:'sea',    5:'street'}

def extract_features_raw(img_f32):
    """Extract full 253-dim vector from a float32 image."""
    img_u8 = (np.clip(img_f32, 0, 1) * 255).astype("uint8")
    f1 = FeatureExtractor.color_histogram(img_u8, bins=32)
    f2 = FeatureExtractor.basic_statistics(img_u8)
    f3 = FeatureExtractor.hog_lite(img_u8, n_bins=9)
    f4 = FeatureExtractor.edge_histogram_descriptor(img_u8, grid=(4,4), n_bins=9)
    return np.concatenate([f1, f2, f3, f4]).astype("float32")

# Load original images
X_orig = np.load(os.path.join("preprocessing_Output", "X_train.npy"))
X_aug  = np.load(os.path.join("augmentation_Output",  "X_train_aug.npy"))
y_orig = np.load(os.path.join("preprocessing_Output", "y_train.npy"))
N_orig = X_orig.shape[0]

SAMPLES   = [0, 100, 300]
AUG_NAMES = ["Original", "Rotation", "Translation", "Flip", "Brightness", "Blur"]

fig = plt.figure(figsize=(22, len(SAMPLES) * 5))
fig.suptitle(
    f"Feature Selection (MRMR) — Before & After Augmentation\n"
    f"Top row of each pair = Full 253 dims  |  Bottom row = Selected {K} dims",
    fontsize=12, fontweight='bold'
)

gs = gridspec.GridSpec(len(SAMPLES) * 3, len(AUG_NAMES),
                       figure=fig, hspace=0.6, wspace=0.25)

for s_row, sample_idx in enumerate(SAMPLES):

    label_str = LABEL_MAP_INV[y_orig[sample_idx]]
    versions  = [X_orig[sample_idx]] + \
                [X_aug[N_orig * k + sample_idx] for k in range(1, 6)]

    base_row = s_row * 3   # 3 sub-rows per sample: image / full feat / selected feat

    for col, (aug_name, img) in enumerate(zip(AUG_NAMES, versions)):

        full_feat     = extract_features_raw(img)           # 253 dims
        selected_feat = full_feat[selected_indices]         # 100 dims (from MRMR)

        # --- Row 0: Image ---
        ax_img = fig.add_subplot(gs[base_row, col])
        ax_img.imshow(np.clip(img, 0, 1))
        ax_img.axis('off')
        if s_row == 0:
            ax_img.set_title(aug_name, fontsize=9, fontweight='bold')
        if col == 0:
            ax_img.set_ylabel(label_str, fontsize=8, rotation=0,
                              labelpad=40, va='center')

        # --- Row 1: Full feature vector (253 dims) ---
        ax_full = fig.add_subplot(gs[base_row + 1, col])
        ax_full.plot(full_feat, linewidth=0.7, color='#4E79A7', alpha=0.85)
        ax_full.axvline(x=96,  color='red',    linewidth=0.8, linestyle='--', alpha=0.5)
        ax_full.axvline(x=100, color='green',  linewidth=0.8, linestyle='--', alpha=0.5)
        ax_full.axvline(x=109, color='orange', linewidth=0.8, linestyle='--', alpha=0.5)
        ax_full.set_xlim(0, 253)
        ax_full.tick_params(labelsize=5)
        ax_full.grid(alpha=0.3)
        if col == 0:
            ax_full.set_ylabel("Full\n253 dims", fontsize=7)

        # --- Row 2: Selected feature vector (K dims) ---
        ax_sel = fig.add_subplot(gs[base_row + 2, col])
        ax_sel.plot(selected_feat, linewidth=0.7, color='#E15759', alpha=0.85)
        ax_sel.set_xlim(0, K)
        ax_sel.tick_params(labelsize=5)
        ax_sel.grid(alpha=0.3)
        if col == 0:
            ax_sel.set_ylabel(f"MRMR\n{K} dims", fontsize=7)

# Legend
legend_elements = [
    plt.Line2D([0], [0], color='#4E79A7', linewidth=1.5, label=f'Full features (253 dims)'),
    plt.Line2D([0], [0], color='#E15759', linewidth=1.5, label=f'MRMR selected ({K} dims)'),
    plt.Line2D([0], [0], color='red',    linestyle='--', linewidth=1, label='stats boundary (96)'),
    plt.Line2D([0], [0], color='green',  linestyle='--', linewidth=1, label='hog boundary (100)'),
    plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=1, label='edge boundary (109)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=5,
           fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

panel_path = os.path.join(OUTPUT_DIR, "mrmr_before_after_aug.png")
plt.savefig(panel_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved panel: {panel_path}")