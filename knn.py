import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# Paths
# ============================================================
INPUT_DIR  = "feature_selection_Output"
OUTPUT_DIR = "knn_Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_MAP_INV = {0:'buildings', 1:'forest', 2:'glacier',
                 3:'mountain',  4:'sea',    5:'street'}
CLASS_NAMES   = [LABEL_MAP_INV[i] for i in range(6)]

# ============================================================
# Load Data
# ============================================================
print("Loading data...")

F_train = np.load(os.path.join(INPUT_DIR, 'F_train_selected.npy'))
F_val   = np.load(os.path.join(INPUT_DIR, 'F_val_selected.npy'))
F_test  = np.load(os.path.join(INPUT_DIR, 'F_test_selected.npy'))
y_train = np.load(os.path.join(INPUT_DIR, 'y_train.npy'))
y_val   = np.load(os.path.join(INPUT_DIR, 'y_val.npy'))
y_test  = np.load(os.path.join(INPUT_DIR, 'y_test.npy'))

print(f"  F_train : {F_train.shape}")
print(f"  F_val   : {F_val.shape}")
print(f"  F_test  : {F_test.shape}")
###Scaling
mean = F_train.mean(axis=0)
std  = F_train.std(axis=0) + 1e-8
F_train = (F_train - mean) / std
F_val   = (F_val   - mean) / std
F_test  = (F_test  - mean) / std


# ============================================================
# Distance Metric — from scratch
# ============================================================
def euclidean_distance(X_train: np.ndarray,
                       X_query: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between query points and training set.

    Uses the identity: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a @ b.T
    Fully vectorized — no Python loops over samples.

    Parameters
    ----------
    X_train : (N, D)  training feature matrix
    X_query : (M, D)  query feature matrix

    Returns
    -------
    distances : (M, N)  float32
    """
    # ||a||^2 for each query point  → (M, 1)
    query_sq  = np.sum(X_query ** 2, axis=1, keepdims=True)
    # ||b||^2 for each train point  → (1, N)
    train_sq  = np.sum(X_train ** 2, axis=1, keepdims=True).T
    # cross term                    → (M, N)
    cross     = X_query @ X_train.T

    dist_sq = query_sq + train_sq - 2.0 * cross
    # numerical safety: clip negatives caused by floating-point errors
    dist_sq = np.clip(dist_sq, 0.0, None)

    return np.sqrt(dist_sq).astype(np.float32)

# ============================================================
# KNN Predict — from scratch
# ============================================================
def knn_predict(X_train: np.ndarray,
                y_train: np.ndarray,
                X_query: np.ndarray,
                k: int) -> np.ndarray:
    """
    Predict class labels for query points using K-Nearest Neighbours.

    Parameters
    ----------
    X_train : (N, D)
    y_train : (N,)
    X_query : (M, D)
    k       : int  number of neighbours

    Returns
    -------
    predictions : (M,)  int
    """
    distances   = euclidean_distance(X_train, X_query)   # (M, N)
    nn_indices  = np.argsort(distances, axis=1)[:, :k]   # (M, k)
    nn_labels   = y_train[nn_indices]                     # (M, k)

    # Majority vote — count votes per class and pick the max
    n_classes   = len(np.unique(y_train))
    predictions = np.zeros(X_query.shape[0], dtype=np.int32)

    for i in range(X_query.shape[0]):
        votes           = np.bincount(nn_labels[i], minlength=n_classes)
        predictions[i]  = np.argmax(votes)

    return predictions

# ============================================================
# Accuracy Helper
# ============================================================
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# ============================================================
# K Sweep on Validation Set
# ============================================================
K_VALUES = [1, 3, 5, 7, 9, 11, 15, 21, 31]

print(f"\nK Sweep on Validation Set...")
print(f"{'k':>4}  {'Val Accuracy':>14}")
print("-" * 22)

val_accuracies = []

for k in K_VALUES:
    preds       = knn_predict(F_train, y_train, F_val, k)
    acc         = accuracy(y_val, preds)
    val_accuracies.append(acc)
    print(f"{k:>4}  {acc * 100:>13.2f}%")

best_k   = K_VALUES[np.argmax(val_accuracies)]
best_acc = max(val_accuracies)
print(f"\nBest k = {best_k}  (Val Accuracy = {best_acc * 100:.2f}%)")

# Save best k
with open(os.path.join(OUTPUT_DIR, 'best_k.txt'), 'w') as f:
    f.write(f"Best k : {best_k}\n")
    f.write(f"Val Acc: {best_acc * 100:.2f}%\n")
    for k, acc in zip(K_VALUES, val_accuracies):
        f.write(f"k={k:>2}  acc={acc * 100:.2f}%\n")

# ============================================================
# Final Evaluation on Test Set with Best K
# ============================================================
print(f"\nEvaluating on Test Set with k = {best_k}...")
test_preds = knn_predict(F_train, y_train, F_test, best_k)
test_acc   = accuracy(y_test, test_preds)
print(f"  Test Accuracy = {test_acc * 100:.2f}%")

# ============================================================
# Confusion Matrix — from scratch
# ============================================================
def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

# ============================================================
# Precision / Recall / F1 — from scratch
# ============================================================
def compute_metrics(y_true, y_pred, n_classes):
    cm        = confusion_matrix(y_true, y_pred, n_classes)
    precision = np.zeros(n_classes)
    recall    = np.zeros(n_classes)
    f1        = np.zeros(n_classes)

    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        precision[c] = tp / (tp + fp + 1e-8)
        recall[c]    = tp / (tp + fn + 1e-8)
        f1[c]        = (2 * precision[c] * recall[c] /
                        (precision[c] + recall[c] + 1e-8))

    support    = np.bincount(y_true, minlength=n_classes)
    macro_f1   = np.mean(f1)
    weighted_f1= np.sum(f1 * support) / support.sum()

    return cm, precision, recall, f1, macro_f1, weighted_f1

cm, precision, recall, f1, macro_f1, weighted_f1 = \
    compute_metrics(y_test, test_preds, 6)

# ============================================================
# Print Metrics
# ============================================================
print(f"\n{'='*55}")
print(f"KNN Results  (k = {best_k})")
print(f"{'='*55}")
print(f"Test Accuracy : {test_acc * 100:.2f}%")
print(f"Macro F1      : {macro_f1 * 100:.2f}%")
print(f"Weighted F1   : {weighted_f1 * 100:.2f}%")
print(f"\n{'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print("-" * 42)
for i, cls in enumerate(CLASS_NAMES):
    print(f"{cls:<12} {precision[i]*100:>9.2f}% "
          f"{recall[i]*100:>7.2f}% {f1[i]*100:>7.2f}%")

# ============================================================
# Plots
# ============================================================
fig = plt.figure(figsize=(16, 6))
fig.suptitle(f'KNN Results  (Best k = {best_k})', fontsize=14, fontweight='bold')
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

# --- Plot 1: Accuracy vs K ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(K_VALUES, [a * 100 for a in val_accuracies],
         marker='o', linewidth=2, color='#4E79A7', markersize=7)
ax1.axvline(x=best_k, color='red', linestyle='--', linewidth=1.5,
            label=f'Best k={best_k}')
ax1.set_title('Validation Accuracy vs K')
ax1.set_xlabel('k')
ax1.set_ylabel('Accuracy (%)')
ax1.legend()
ax1.grid(alpha=0.3)

# --- Plot 2: Confusion Matrix ---
ax2 = fig.add_subplot(gs[0, 1])
im  = ax2.imshow(cm, cmap='Blues')
ax2.set_xticks(range(6)); ax2.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=8)
ax2.set_yticks(range(6)); ax2.set_yticklabels(CLASS_NAMES, fontsize=8)
ax2.set_title('Confusion Matrix (Test)')
ax2.set_xlabel('Predicted'); ax2.set_ylabel('True')
for i in range(6):
    for j in range(6):
        ax2.text(j, i, str(cm[i, j]), ha='center', va='center',
                 fontsize=8, color='white' if cm[i,j] > cm.max()/2 else 'black')
plt.colorbar(im, ax=ax2, fraction=0.046)

# --- Plot 3: Per-Class F1 ---
ax3 = fig.add_subplot(gs[0, 2])
colors = ['#4E79A7','#F28E2B','#59A14F','#E15759','#76B7B2','#EDC948']
bars   = ax3.bar(CLASS_NAMES, f1 * 100, color=colors, edgecolor='white')
ax3.axhline(y=macro_f1 * 100, color='red', linestyle='--',
            linewidth=1.5, label=f'Macro F1 = {macro_f1*100:.1f}%')
ax3.set_title('Per-Class F1 Score')
ax3.set_ylabel('F1 (%)')
ax3.set_ylim(0, 110)
ax3.tick_params(axis='x', rotation=45)
ax3.legend(fontsize=8)
ax3.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, f1):
    ax3.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 1, f'{val*100:.1f}',
             ha='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'knn_results.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# Save Metrics CSV
# ============================================================
import csv
with open(os.path.join(OUTPUT_DIR, 'knn_metrics.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['class', 'precision', 'recall', 'f1'])
    for i, cls in enumerate(CLASS_NAMES):
        writer.writerow([cls,
                         f'{precision[i]:.4f}',
                         f'{recall[i]:.4f}',
                         f'{f1[i]:.4f}'])
    writer.writerow(['macro',    '', '', f'{macro_f1:.4f}'])
    writer.writerow(['weighted', '', '', f'{weighted_f1:.4f}'])

# ============================================================
# Final Summary
# ============================================================
print(f"\n{'='*55}")
print("KNN Training Completed!")
print(f"{'='*55}")
print(f"Output Files → {OUTPUT_DIR}/")
print(f"  best_k.txt")
print(f"  knn_results.png   (accuracy vs k + confusion matrix + F1)")
print(f"  knn_metrics.csv")

