import numpy as np
import csv
import json
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from cnn_model import Dense, ReLU, Flatten, softmax, categorical_crossentropy, crossentropy_backward, step_decay

sys.path.append(os.path.abspath("."))
from shared.optimizer import SGD, Adam
from shared.scheduler import ReduceOnPlateau

# ============================================================
# Config
# ============================================================
OUTPUT_DIR  = "."
CLASS_NAMES = ["Class0", "Class1", "Class2", "Class3", "Class4", "Class5"]  # ← update as needed


# ============================================================
# SGD Optimizer
# ============================================================
class SGDOptimizer:
    """SGD optimizer with momentum and optional Nesterov acceleration."""

    def __init__(self, params, lr=0.01, momentum=0.9, nesterov=False):
        self.lr         = lr
        self.momentum   = momentum
        self.nesterov   = nesterov
        self.velocities = [np.zeros_like(p) for p in params]

    def step(self, params, grads):
        for i, (p, g) in enumerate(zip(params, grads)):
            if self.nesterov:
                v_prev              = self.velocities[i].copy()
                self.velocities[i]  = self.momentum * self.velocities[i] - self.lr * g
                p += -self.momentum * v_prev + (1 + self.momentum) * self.velocities[i]
            else:
                self.velocities[i] = self.momentum * self.velocities[i] - self.lr * g
                p += self.velocities[i]


# ============================================================
# Helpers
# ============================================================
def to_categorical(y, num_classes=None):
    """One-hot encode integer labels."""
    y = np.array(y, dtype=int).flatten()
    if not num_classes:
        num_classes = np.max(y) + 1
    return np.eye(num_classes)[y], num_classes


# ============================================================
# Metrics — from scratch (no sklearn)
# ============================================================
def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


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

    support     = np.bincount(y_true, minlength=n_classes)
    macro_f1    = np.mean(f1)
    weighted_f1 = np.sum(f1 * support) / support.sum()
    test_acc    = np.mean(np.array(y_true) == np.array(y_pred))

    return cm, precision, recall, f1, macro_f1, weighted_f1, test_acc


# ============================================================
# Training loop
# ============================================================
def train_network(X_train, Y_train, X_val, Y_val, epochs, batch_size, run_config):
    """SGD training loop with history tracking for plotting."""
    input_dim   = X_train.shape[1]
    num_classes = Y_train.shape[1]
    network = [
        Flatten(),
        Dense(input_dim, 128, l2_lambda=run_config['l2_lambda']),
        ReLU(),
        Dense(128, 64, l2_lambda=run_config['l2_lambda']),
        ReLU(),
        Dense(64, num_classes, l2_lambda=run_config['l2_lambda'])
    ]

    trainable_params = []
    for layer in network:
        if isinstance(layer, Dense):
            trainable_params.extend([layer.weights, layer.biases])

    optimizer = SGDOptimizer(
        trainable_params,
        lr       = run_config['initial_lr'],
        momentum = run_config.get('momentum', 0.9),
        nesterov = run_config.get('nesterov', False)
    )

    '''scheduler = ReduceOnPlateau(
    initial_lr=run_config['initial_lr'],
    factor=0.5,       # Or whatever factor you prefer
    patience=3,       # How many epochs to wait for improvement
    min_lr=1e-6
    )'''

    best_val_loss    = float('inf')
    patience_counter = 0
    patience_limit   = 15

    history = {
        'epoch':      [],
        'train_loss': [],
        'val_loss':   [],
        'train_acc':  [],
        'val_acc':    [],
        'lr':         []
    }

    csv_file = 'SGD_training_logs.csv'
    with open(csv_file, 'w', newline='') as f:
        csv.writer(f).writerow(
            ['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'learning_rate'])

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        indices              = np.random.permutation(len(X_train))
        X_shuffled, Y_shuffled = X_train[indices], Y_train[indices]
        #optimizer.lr         = step_decay(epoch, initial_lr=run_config['initial_lr'])

        # ── Mini-batch loop ──────────────────────────────────
        for i in range(0, len(X_train), batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            Y_batch = Y_shuffled[i:i + batch_size]

            output = X_batch
            for layer in network:
                output = layer.forward(output)

            preds = softmax(output)
            grad  = crossentropy_backward(preds, Y_batch)

            layer_gradients = []
            for layer in reversed(network):
                if isinstance(layer, Dense):
                    grad, dW, db = layer.backward(grad)
                    layer_gradients.extend([db, dW])
                else:
                    grad = layer.backward(grad)

            clip_val        = run_config.get('clip_value', 1.0)
            layer_gradients = [np.clip(g, -clip_val, clip_val) for g in layer_gradients]
            layer_gradients.reverse()
            optimizer.step(trainable_params, layer_gradients)

        # ── Epoch metrics ────────────────────────────────────
        train_out = X_train
        for layer in network:
            train_out = layer.forward(train_out)
        train_preds = softmax(train_out)
        train_loss  = categorical_crossentropy(train_preds, Y_train)
        train_acc   = np.mean(np.argmax(train_preds, axis=1) == np.argmax(Y_train, axis=1))

        val_out = X_val
        for layer in network:
            val_out = layer.forward(val_out)
        val_preds = softmax(val_out)
        val_loss  = categorical_crossentropy(val_preds, Y_val)
        val_acc   = np.mean(np.argmax(val_preds, axis=1) == np.argmax(Y_val, axis=1))

        #new_lr = scheduler.step(val_loss=val_loss)
        #optimizer.lr = new_lr

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc * 100)
        history['val_acc'].append(val_acc * 100)
        history['lr'].append(optimizer.lr)

        with open(csv_file, 'a', newline='') as f:
            csv.writer(f).writerow(
                [epoch + 1, train_loss, val_loss, train_acc, val_acc, optimizer.lr])

        print(f"Epoch {epoch+1:02d}/{epochs}  "
              f"loss={train_loss:.4f}  acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"lr={optimizer.lr:.6f}")

        # ── Checkpoint & early stopping ──────────────────────
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            checkpoint = {
                'epoch':           epoch + 1,
                'optimizer_state': {'lr': optimizer.lr, 'momentum': optimizer.momentum},
                'run_config':      run_config,
                'weights':         [p.tolist() for p in trainable_params]
            }
            with open('SGD_best_checkpoint.json', 'w') as f:
                json.dump(checkpoint, f)
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping at epoch {epoch+1}.  "
                      f"Best val loss: {best_val_loss:.4f}")
                break

    return network, history


# ============================================================
# Plots
# ============================================================
def plot_results(history, cm, precision, recall, f1,
                 macro_f1, weighted_f1, test_acc, class_names, output_dir):

    n      = len(class_names)
    colors = ['#4E79A7', '#F28E2B', '#59A14F', '#E15759', '#76B7B2', '#EDC948',
              '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']

    fig = plt.figure(figsize=(20, 6))
    fig.suptitle('SGD Results', fontsize=14, fontweight='bold')
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.40)

    # ── Plot 1 : Accuracy per Epoch ───────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history['epoch'], history['train_acc'],
             marker='o', linewidth=2, color='#4E79A7', markersize=4, label='Train')
    ax1.plot(history['epoch'], history['val_acc'],
             marker='s', linewidth=2, color='#F28E2B', markersize=4, label='Val')
    ax1.set_title('Accuracy per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ── Plot 2 : Loss per Epoch ───────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history['epoch'], history['train_loss'],
             marker='o', linewidth=2, color='#59A14F', markersize=4, label='Train')
    ax2.plot(history['epoch'], history['val_loss'],
             marker='s', linewidth=2, color='#E15759', markersize=4, label='Val')
    ax2.set_title('Loss per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # ── Plot 3 : Confusion Matrix ─────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    im  = ax3.imshow(cm, cmap='Blues')
    ax3.set_xticks(range(n))
    ax3.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    ax3.set_yticks(range(n))
    ax3.set_yticklabels(class_names, fontsize=8)
    ax3.set_title('Confusion Matrix (Test)')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('True')
    for i in range(n):
        for j in range(n):
            ax3.text(j, i, str(cm[i, j]), ha='center', va='center',
                     fontsize=8,
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.colorbar(im, ax=ax3, fraction=0.046)

    # ── Plot 4 : Per-Class F1 ─────────────────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    bars = ax4.bar(class_names, f1 * 100, color=colors[:n], edgecolor='white')
    ax4.axhline(y=macro_f1 * 100, color='red', linestyle='--',
                linewidth=1.5, label=f'Macro F1 = {macro_f1*100:.1f}%')
    ax4.set_title('Per-Class F1 Score')
    ax4.set_ylabel('F1 (%)')
    ax4.set_ylim(0, 115)
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend(fontsize=8)
    ax4.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, f1):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1, f'{val*100:.1f}',
                 ha='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'sgd_results.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved → {out_path}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    # 1. Load data
    print("Loading data files...")
    X_train     = np.load('./feature_selection_Output/F_train_selected.npy')
    y_train_raw = np.load('./feature_selection_Output/y_train.npy')
    X_val       = np.load('./feature_selection_Output/F_val_selected.npy')
    y_val_raw   = np.load('./feature_selection_Output/y_val.npy')
    X_test      = np.load('./feature_selection_Output/F_test_selected.npy')
    y_test_raw  = np.load('./feature_selection_Output/y_test.npy')

    # 2. Labels
    Y_train, num_classes = to_categorical(y_train_raw)
    Y_val,   _           = to_categorical(y_val_raw, num_classes)

    # ============================================================
    # Feature Standardization
    # ============================================================
    mean = X_train.mean(axis=0, keepdims=True)
    std  = X_train.std(axis=0, keepdims=True) + 1e-8

    X_train = (X_train - mean) / std    
    X_val   = (X_val   - mean) / std
    X_test  = (X_test  - mean) / std


    # 3. Config
    config = {
        'epochs':       50,
        'batch_size':   32,
        'initial_lr':   0.01,       # SGD typically needs a higher LR than Adam
        'momentum':     0.9,
        'nesterov':     False,      # set True for Nesterov SGD
        'l2_lambda':    0.01,
        'clip_value':   1.0,
        'architecture': ['Flatten', 'Dense(128)', 'ReLU', f'Dense({num_classes})']
    }

    # 4. Train
    trained_network, history = train_network(
        X_train, Y_train, X_val, Y_val,
        epochs     = config['epochs'],
        batch_size = config['batch_size'],
        run_config = config
    )

    # 5. Test inference
    print("\nEvaluating on Test Set...")
    test_out = X_test
    for layer in trained_network:
        test_out = layer.forward(test_out)

    test_preds_probs = softmax(test_out)
    y_pred = np.argmax(test_preds_probs, axis=1)
    y_true = np.array(y_test_raw, dtype=int).flatten()

    # 6. Metrics
    cm, precision, recall, f1, macro_f1, weighted_f1, test_acc = \
        compute_metrics(y_true, y_pred, num_classes)

    # 7. Print report
    print(f"\n{'='*55}")
    print("SGD Results")
    print(f"{'='*55}")
    print(f"Test Accuracy : {test_acc * 100:.2f}%")
    print(f"Macro F1      : {macro_f1 * 100:.2f}%")
    print(f"Weighted F1   : {weighted_f1 * 100:.2f}%")
    print(f"\n{'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 42)
    for i, cls in enumerate(CLASS_NAMES[:num_classes]):
        print(f"{cls:<12} {precision[i]*100:>9.2f}%"
              f" {recall[i]*100:>7.2f}% {f1[i]*100:>7.2f}%")

    # 8. Save text report
    with open('SGD_evaluation_report.txt', 'w') as f:
        f.write(f"Test Accuracy : {test_acc * 100:.2f}%\n")
        f.write(f"Macro F1      : {macro_f1:.4f}\n")
        f.write(f"Weighted F1   : {weighted_f1:.4f}\n\n")
        f.write(f"{'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8}\n")
        f.write("-" * 42 + "\n")
        for i, cls in enumerate(CLASS_NAMES[:num_classes]):
            f.write(f"{cls:<12} {precision[i]*100:>9.2f}%"
                    f" {recall[i]*100:>7.2f}% {f1[i]*100:>7.2f}%\n")
        f.write("\nConfusion Matrix:\n" + str(cm) + "\n")

    # 9. Plots
    plot_results(history, cm, precision, recall, f1,
                 macro_f1, weighted_f1, test_acc,
                 CLASS_NAMES[:num_classes], OUTPUT_DIR)