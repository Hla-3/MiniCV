import os
import sys
import numpy as np

sys.path.append(os.path.abspath("."))
from shared.optimizer import SGD, Adam
from shared.scheduler import ReduceOnPlateau
from shared.logger    import Logger
from shared.metrics   import classification_report, plot_confusion_matrix, plot_f1_bars

# ============================================================
# Paths
# ============================================================
INPUT_DIR  = "feature_selection_Output"
OUTPUT_DIR = "softmax_Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Hyperparameters
# ============================================================
CONFIG = {
    'model_name'  : 'Softmax',
    'n_features'  : 256,
    'n_classes'   : 6,
    'lr'          : 5e-4,
    'optimizer'   : 'Adam',       # 'SGD' or 'Adam'
    'weight_decay': 5e-4,
    'grad_clip'   : 5.0,
    'batch_size'  : 32,
    'n_epochs'    : 150,
    'patience'    : 15,
    'scheduler'   : 'ReduceOnPlateau',
    'sched_factor': 0.5,
    'sched_patience': 5,
    'min_lr'      : 1e-6,
}

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

N_FEATURES = F_train.shape[1]
N_CLASSES  = 6

# ============================================================
# Feature Standardization
# ============================================================
mean = F_train.mean(axis=0, keepdims=True)
std  = F_train.std(axis=0, keepdims=True) + 1e-8

F_train = (F_train - mean) / std
F_val   = (F_val   - mean) / std
F_test  = (F_test  - mean) / std

# ============================================================
# Softmax Regression — from scratch
# ============================================================
class SoftmaxRegression:
    """
    Multiclass Softmax Regression trained with mini-batch gradient descent.

    Parameters
    ----------
    n_features : int   input dimensionality
    n_classes  : int   number of output classes

    Attributes
    ----------
    W : (n_features, n_classes)  weight matrix
    b : (1, n_classes)           bias vector
    """

    def __init__(self, n_features: int, n_classes: int):
        # Xavier initialization
        scale    = np.sqrt(1.0 / n_features)
        self.W   = np.random.randn(n_features, n_classes).astype(np.float32) * scale
        self.b   = np.zeros((1, n_classes), dtype=np.float32)

    # --------------------------------------------------------
    # Numerically Stable Softmax
    # --------------------------------------------------------
    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        """
        Numerically stable softmax.

        Subtract max per row before exp to prevent overflow:
            softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

        Parameters
        ----------
        logits : (N, C)  raw scores

        Returns
        -------
        probs : (N, C)  probabilities, each row sums to 1
        """
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_x   = np.exp(shifted)
        return exp_x / (exp_x.sum(axis=1, keepdims=True) + 1e-8)

    # --------------------------------------------------------
    # Forward Pass
    # --------------------------------------------------------
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Compute class probabilities.

        Parameters
        ----------
        X : (N, D)

        Returns
        -------
        probs : (N, C)
        """
        logits = X @ self.W + self.b       # (N, C)
        return self.softmax(logits)

    # --------------------------------------------------------
    # Cross-Entropy Loss
    # --------------------------------------------------------
    @staticmethod
    def cross_entropy_loss(probs: np.ndarray,
                           y: np.ndarray) -> float:
        """
        Cross-entropy loss with epsilon clipping to prevent log(0).

        L = -1/N * sum( log(probs[i, y[i]] + epsilon) )

        Parameters
        ----------
        probs : (N, C)  predicted probabilities
        y     : (N,)    integer true labels

        Returns
        -------
        float  scalar loss
        """
        N       = probs.shape[0]
        epsilon = 1e-8
        # Clip probabilities to avoid log(0)
        p_clipped = np.clip(probs[np.arange(N), y.astype(int)], epsilon, 1.0)
        return float(-np.mean(np.log(p_clipped)))

    # --------------------------------------------------------
    # Backward Pass — gradient of loss w.r.t. W and b
    # --------------------------------------------------------
    def backward(self, X: np.ndarray,
                 probs: np.ndarray,
                 y: np.ndarray):
        """
        Compute gradients via backpropagation.

        dL/dlogits = (probs - one_hot(y)) / N
        dL/dW      = X.T @ dL/dlogits
        dL/db      = sum(dL/dlogits, axis=0)

        Parameters
        ----------
        X     : (N, D)
        probs : (N, C)
        y     : (N,)

        Returns
        -------
        dW : (D, C)
        db : (1, C)
        """
        N = X.shape[0]

        # One-hot encode y
        one_hot        = np.zeros_like(probs)
        smooth = 0.1
        one_hot += smooth / probs.shape[1]
        one_hot[np.arange(N), y.astype(int)] += (1.0 - smooth)

        # Gradient of softmax + cross-entropy combined
        d_logits = (probs - one_hot) / N     # (N, C)

        dW = X.T @ d_logits                  # (D, C)
        db = d_logits.sum(axis=0, keepdims=True)  # (1, C)

        return dW, db

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices."""
        return np.argmax(self.forward(X), axis=1).astype(np.int32)

    # --------------------------------------------------------
    # Get / Set Parameters (for optimizer + checkpoint)
    # --------------------------------------------------------
    def get_params(self) -> list:
        return [self.W, self.b]

    def set_params(self, params: list) -> None:
        self.W = params[0].astype(np.float32)
        self.b = params[1].astype(np.float32)


# ============================================================
# Training Loop
# ============================================================
def train():
    np.random.seed(42)

    model = SoftmaxRegression(N_FEATURES, N_CLASSES)

    # Build optimizer
    if CONFIG['optimizer'] == 'Adam':
        optimizer = Adam(lr=CONFIG['lr'],
                         weight_decay=CONFIG['weight_decay'],
                         grad_clip=CONFIG['grad_clip'])
    else:
        optimizer = SGD(lr=CONFIG['lr'],
                        weight_decay=CONFIG['weight_decay'],
                        grad_clip=CONFIG['grad_clip'])

    # Build scheduler
    scheduler = ReduceOnPlateau(
        initial_lr=CONFIG['lr'],
        factor=CONFIG['sched_factor'],
        patience=CONFIG['sched_patience'],
        min_lr=CONFIG['min_lr']
    )

    # Logger + Early Stopping
    logger      = Logger(OUTPUT_DIR, 'Softmax', CONFIG)
    early_stop  = Logger.EarlyStopping(patience=CONFIG['patience'])

    print(f"\n{'='*55}")
    print(f"Training Softmax Regression")
    print(f"  Optimizer : {CONFIG['optimizer']}")
    print(f"  LR        : {CONFIG['lr']}")
    print(f"  Epochs    : {CONFIG['n_epochs']}")
    print(f"  Batch Size: {CONFIG['batch_size']}")
    print(f"{'='*55}\n")

    N = F_train.shape[0]

    for epoch in range(CONFIG['n_epochs']):

        # ---- Mini-batch Shuffling ----
        indices = np.random.permutation(N)
        X_shuf  = F_train[indices]
        y_shuf  = y_train[indices]

        # ---- Mini-batch Training ----
        batch_losses = []
        batch_size   = CONFIG['batch_size']

        for start in range(0, N, batch_size):
            X_batch = X_shuf[start:start + batch_size]
            y_batch = y_shuf[start:start + batch_size]

            # Forward
            probs = model.forward(X_batch)
            loss  = model.cross_entropy_loss(probs, y_batch)
            batch_losses.append(loss)

            # Backward
            dW, db = model.backward(X_batch, probs, y_batch)

            # Optimizer step
            params  = model.get_params()
            grads   = [dW, db]
            updated = optimizer.update(params, grads)
            model.set_params(updated)

        # ---- Epoch Metrics ----
        train_loss = float(np.mean(batch_losses))
        train_preds = model.predict(F_train)
        train_acc   = float(np.mean(train_preds == y_train))

        val_probs  = model.forward(F_val)
        val_loss   = model.cross_entropy_loss(val_probs, y_val)
        val_preds  = model.predict(F_val)
        val_acc    = float(np.mean(val_preds == y_val))

        # ---- Scheduler step ----
        new_lr = scheduler.step(val_loss=val_loss)
        optimizer.lr = new_lr

        # ---- Log ----
        logger.log_epoch(epoch + 1, train_loss, val_loss,
                         train_acc, val_acc, new_lr)

        # ---- Checkpoint ----
        logger.save_checkpoint(
            model.get_params(),
            optimizer.get_state(),
            scheduler.get_state(),
            epoch + 1,
            val_loss
        )

        # ---- Early Stopping ----
        if early_stop.step(val_loss):
            break

    # ---- Training Curves ----
    logger.plot_curves()

    return model


# ============================================================
# Evaluate on Test Set
# ============================================================
def evaluate(model):
    print(f"\n{'='*55}")
    print("Evaluating on Test Set...")

    # Load best checkpoint
    ckpt        = logger_ref.load_checkpoint()
    model.set_params(ckpt['params'])

    test_preds  = model.predict(F_test)

    report = classification_report(y_test, test_preds, "Softmax Regression")

    plot_confusion_matrix(
        y_test, test_preds,
        model_name="Softmax Regression",
        save_path=os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    )
    plot_f1_bars(
        y_test, test_preds,
        model_name="Softmax Regression",
        save_path=os.path.join(OUTPUT_DIR, 'f1_bars.png')
    )

    # Save predictions for evaluate.py
    np.save(os.path.join(OUTPUT_DIR, 'test_preds.npy'), test_preds)
    np.save(os.path.join(OUTPUT_DIR, 'test_true.npy'),  y_test)

    return report

# ============================================================
# Save Metrics CSV  (like knn_metrics.csv)
# ============================================================
import csv
from shared.metrics import precision_recall_f1, accuracy

CLASS_NAMES = ['buildings','forest','glacier','mountain','sea','street']

test_preds_loaded = np.load(os.path.join(OUTPUT_DIR, 'test_preds.npy'))
test_true_loaded  = np.load(os.path.join(OUTPUT_DIR, 'test_true.npy'))

precision, recall, f1, macro_f1, weighted_f1 = \
    precision_recall_f1(test_true_loaded, test_preds_loaded)
acc = accuracy(test_true_loaded, test_preds_loaded)

with open(os.path.join(OUTPUT_DIR, 'softmax_metrics.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['class', 'precision', 'recall', 'f1', 'support'])
    for i, cls in enumerate(CLASS_NAMES):
        support = int(np.sum(test_true_loaded == i))
        writer.writerow([cls,
                         f'{precision[i]:.4f}',
                         f'{recall[i]:.4f}',
                         f'{f1[i]:.4f}',
                         support])
    writer.writerow(['macro',    '', '', f'{macro_f1:.4f}',    ''])
    writer.writerow(['weighted', '', '', f'{weighted_f1:.4f}', ''])
    writer.writerow(['accuracy', f'{acc:.4f}', '', '', ''])

print(f"Saved: softmax_metrics.csv")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    model      = train()

    # Re-create logger just for loading checkpoint
    logger_ref = Logger(OUTPUT_DIR, 'Softmax', CONFIG, resume=True)

    report     = evaluate(model)

    print(f"\n{'='*55}")
    print("Softmax Training Complete!")
    print(f"{'='*55}")
    print(f"Output Files → {OUTPUT_DIR}/")
    print(f"  logs.csv")
    print(f"  best_checkpoint.npz")
    print(f"  best_checkpoint_meta.json")
    print(f"  config.json")
    print(f"  training_curves.png")
    print(f"  confusion_matrix.png")
    print(f"  f1_bars.png")
    print(f"  test_preds.npy")