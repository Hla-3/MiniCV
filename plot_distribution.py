import os
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Dataset path
# ============================================================
DATASET_PATH = "Dataset"   # Main dataset folder

# ============================================================
# Read number of images in each class
# ============================================================
classes = sorted(os.listdir(DATASET_PATH))
counts = []

for cls in classes:
    folder = os.path.join(DATASET_PATH, cls)

    if os.path.isdir(folder):
        n = len([
            f for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        counts.append(n)

# ============================================================
# Plot the charts
# ============================================================
colors = [
    '#4E79A7',
    '#F28E2B',
    '#59A14F',
    '#E15759',
    '#76B7B2',
    '#EDC948'
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

fig.suptitle(
    'Dataset Class Distribution',
    fontsize=16,
    fontweight='bold',
    y=1.02
)

# ------------------------------------------------------------
# Bar Chart
# ------------------------------------------------------------
bars = axes[0].bar(
    classes,
    counts,
    color=colors,
    edgecolor='white',
    linewidth=1.5
)

axes[0].set_title('Images per Class', fontsize=13)
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Number of Images')

axes[0].set_ylim(0, max(counts) * 1.2)

# Add values above each bar
for bar, count in zip(bars, counts):

    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 10,
        str(count),
        ha='center',
        va='bottom',
        fontweight='bold',
        fontsize=11
    )

axes[0].tick_params(axis='x', rotation=15)
axes[0].grid(axis='y', alpha=0.3)

# ------------------------------------------------------------
# Pie Chart
# ------------------------------------------------------------
wedges, texts, autotexts = axes[1].pie(
    counts,
    labels=classes,
    colors=colors,
    autopct='%1.1f%%',
    startangle=90,
    wedgeprops={
        'edgecolor': 'white',
        'linewidth': 2
    }
)

for text in autotexts:
    text.set_fontsize(10)
    text.set_fontweight('bold')

axes[1].set_title('Class Distribution (%)', fontsize=13)

# ============================================================
# Print Summary
# ============================================================
total = sum(counts)

print("=" * 40)
print(f"{'Class':<12} {'Count':>6} {'%':>7}")
print("=" * 40)

for cls, count in zip(classes, counts):

    print(
        f"{cls:<12} "
        f"{count:>6} "
        f"{count / total * 100:>6.1f}%"
    )

print("=" * 40)
print(f"{'Total':<12} {total:>6} {'100.0%':>7}")

# ============================================================
# Save and Show Figure
# ============================================================
plt.tight_layout()

plt.savefig(
    'class_distribution.png',
    dpi=150,
    bbox_inches='tight'
)

plt.show()

print("\nImage saved as: class_distribution.png")