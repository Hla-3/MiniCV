"""
ConvNeXt-Based CNN Model — PyTorch Reimplementation
Project: End-to-End Supervised Machine Vision Pipeline

Drop-in replacement for the original NumPy implementation.
Same public API: ConvNeXtCNN, ConvNeXtTrainer, TrainingConfig, ModelEvaluator.

Requirements:
    pip install torch torchvision matplotlib
"""

import os
import csv
import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ============================================================================
# DEVICE
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[ConvNeXt] Using device: {DEVICE}")


# ============================================================================
# 1. BUILDING BLOCKS
# ============================================================================

class LayerNorm2d(nn.Module):
    """
    Layer Normalization for (N, C, H, W) tensors.
    Normalizes over C, H, W for each sample — matches the original paper.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(num_features, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (N, C, H, W) → (N, H, W, C) → LayerNorm → (N, C, H, W)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block:
      1. Large depthwise conv  (7×7, groups=C)
      2. LayerNorm
      3. Pointwise expand      (1×1, C → 4C)
      4. GELU
      5. Pointwise compress    (1×1, 4C → C)
      6. Residual addition
    """
    def __init__(self, channels: int, kernel_size: int = 7, expansion_ratio: float = 4.0):
        super().__init__()
        expand_dim = int(channels * expansion_ratio)
        pad = kernel_size // 2

        self.dw_conv  = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                                   padding=pad, groups=channels, bias=True)
        self.norm     = LayerNorm2d(channels)
        self.pw_conv1 = nn.Conv2d(channels, expand_dim, kernel_size=1, bias=True)
        self.act      = nn.GELU()
        self.pw_conv2 = nn.Conv2d(expand_dim, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.dw_conv(x)
        out = self.norm(out)
        out = self.pw_conv1(out)
        out = self.act(out)
        out = self.pw_conv2(out)
        return out + residual


# ============================================================================
# 2. CONVNEXT CNN
# ============================================================================

class ConvNeXtCNN(nn.Module):
    """
    ConvNeXt-inspired CNN for image / feature-map classification.

    Architecture
    ────────────
    Stem      : Conv2d 4×4, stride 4   (patchify)
    Stage 1   : num_blocks × ConvNeXtBlock(C)
    Downsample: Conv2d 2×2, stride 2   (C  → 2C)
    Stage 2   : num_blocks × ConvNeXtBlock(2C)
    Downsample: Conv2d 2×2, stride 2   (2C → 4C)
    Stage 3   : num_blocks × ConvNeXtBlock(4C)
    GAP       : global average pool
    Head      : Linear(4C, num_classes)

    Args:
        num_classes     : number of output classes
        input_channels  : 3 for RGB / feature-map
        num_blocks      : blocks per stage  (default 2)
        initial_channels: base channel count (default 64)
    """
    def __init__(self, num_classes: int, input_channels: int = 3,
                 num_blocks: int = 2, initial_channels: int = 64):
        super().__init__()
        C = initial_channels

        # ── Stem ──────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, C, kernel_size=4, stride=4, padding=0, bias=True),
            LayerNorm2d(C),
        )

        # ── Stage 1 ───────────────────────────────────────────────────────
        self.stage1 = nn.Sequential(
            *[ConvNeXtBlock(C) for _ in range(num_blocks)]
        )

        # ── Downsample 1 → Stage 2 ────────────────────────────────────────
        self.down2 = nn.Sequential(
            LayerNorm2d(C),
            nn.Conv2d(C, C * 2, kernel_size=2, stride=2, bias=True),
        )
        self.stage2 = nn.Sequential(
            *[ConvNeXtBlock(C * 2) for _ in range(num_blocks)]
        )

        # ── Downsample 2 → Stage 3 ────────────────────────────────────────
        self.down3 = nn.Sequential(
            LayerNorm2d(C * 2),
            nn.Conv2d(C * 2, C * 4, kernel_size=2, stride=2, bias=True),
        )
        self.stage3 = nn.Sequential(
            *[ConvNeXtBlock(C * 4) for _ in range(num_blocks)]
        )

        # ── Head ──────────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # (N, 4C, 1, 1)
            nn.Flatten(),              # (N, 4C)
            nn.LayerNorm(C * 4),
            nn.Linear(C * 4, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm,)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down2(x)
        x = self.stage2(x)
        x = self.down3(x)
        x = self.stage3(x)
        return self.head(x)


# ============================================================================
# 3. TRAINING CONFIG
# ============================================================================

@dataclass
class TrainingConfig:
    """
    Training hyper-parameters — same fields as the original implementation,
    plus checkpoint_dir which is now used directly.
    """
    num_epochs              : int   = 100
    batch_size              : int   = 32
    learning_rate           : float = 0.001
    optimizer               : str   = 'adam'     # 'sgd' | 'adam' | 'rmsprop'
    lr_scheduler            : str   = 'cosine'   # 'step' | 'exponential' | 'cosine' | 'reduce_on_plateau'
    weight_decay            : float = 0.0001
    gradient_clip           : float = 1.0
    early_stopping_patience : int   = 15
    seed                    : int   = 42
    checkpoint_dir          : str   = './checkpoints'

    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# 4. PLOTTING UTILITIES
# ============================================================================

def plot_results(history: Dict, cm: np.ndarray, precision: np.ndarray,
                 recall: np.ndarray, f1: np.ndarray, macro_f1: float,
                 weighted_f1: float, test_acc: float, class_names: List[str],
                 output_dir: str = '.'):
    """
    FIXED: Complete implementation of results plotting.
    
    Args:
        history: dict with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        cm: confusion matrix (num_classes, num_classes)
        precision, recall, f1: per-class metrics
        macro_f1, weighted_f1: aggregated metrics
        test_acc: test accuracy
        class_names: list of class names
        output_dir: directory to save plot
    """
    n = len(class_names)
    fig = plt.figure(figsize=(20, 6))
    fig.suptitle('ConvNeXt Results', fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.40)

    colors = ['#4E79A7', '#F28E2B', '#59A14F', '#E15759', '#76B7B2', '#EDC948',
              '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC']

    # ── Plot 1: Training & Validation Accuracy ──────────────
    ax1 = fig.add_subplot(gs[0, 0])
    epochs = range(1, len(history['train_acc']) + 1)
    
    train_acc_pct = [acc * 100 for acc in history['train_acc']]
    val_acc_pct = [acc * 100 for acc in history['val_acc']]
    
    ax1.plot(epochs, train_acc_pct, marker='o', linewidth=2,
             color='#4E79A7', markersize=4, label='Train')
    ax1.plot(epochs, val_acc_pct, marker='s', linewidth=2,
             color='#F28E2B', markersize=4, label='Val')
    ax1.set_title('Accuracy per Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ── Plot 2: Training & Validation Loss ──────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, history['train_loss'], marker='o', linewidth=2,
             color='#59A14F', markersize=4, label='Train')
    ax2.plot(epochs, history['val_loss'], marker='s', linewidth=2,
             color='#E15759', markersize=4, label='Val')
    ax2.set_title('Loss per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # ── Plot 3: Confusion Matrix ────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    im = ax3.imshow(cm, cmap='Blues')
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

    # ── Plot 4: Per-Class F1 Score ──────────────────────────
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
    out_path = os.path.join(output_dir, 'convnext_results.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Plot saved → {out_path}")


# ============================================================================
# 5. TRAINER
# ============================================================================

class ConvNeXtTrainer:
    """
    Complete training pipeline — same public API as the original.

    Usage
    ─────
    trainer = ConvNeXtTrainer(model, config)
    history = trainer.train(X_train, y_train, X_val, y_val)
    trainer.load_checkpoint('checkpoints/best_model.pkl')
    """

    def __init__(self, model: ConvNeXtCNN, config: TrainingConfig):
        self.config = config
        self.model = model.to(DEVICE)

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # ── Optimizer ──────────────────────────────────────────────────────
        opt = config.optimizer.lower()
        if opt == 'sgd':
            self.optimizer = torch.optim.SGD(
                model.parameters(), lr=config.learning_rate,
                momentum=0.9, weight_decay=config.weight_decay)
        elif opt == 'adam':
            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=config.learning_rate,
                weight_decay=config.weight_decay)
        elif opt == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(
                model.parameters(), lr=config.learning_rate,
                weight_decay=config.weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")

        # ── LR Scheduler ───────────────────────────────────────────────────
        sched = config.lr_scheduler.lower()
        if sched == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.1)
        elif sched == 'exponential':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.95)
        elif sched == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.num_epochs, eta_min=0.0)
        elif sched == 'reduce_on_plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.1, patience=10)
        else:
            raise ValueError(f"Unknown scheduler: {config.lr_scheduler}")

        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.criterion = nn.CrossEntropyLoss()

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _to_tensor(X: np.ndarray, y: np.ndarray) -> TensorDataset:
        """Convert numpy arrays to a TensorDataset on CPU (DataLoader will batch)."""
        X_t = torch.from_numpy(X.astype(np.float32))
        # y can be one-hot (N, C) or integer (N,)
        if y.ndim == 2:
            y_t = torch.from_numpy(np.argmax(y, axis=1).astype(np.int64))
        else:
            y_t = torch.from_numpy(y.astype(np.int64))
        return TensorDataset(X_t, y_t)

    def _current_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    # ── one epoch ──────────────────────────────────────────────────────────

    def train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """FIXED: Complete epoch training with proper metric computation."""
        self.model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            loss.backward()

            if self.config.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.config.gradient_clip)
            self.optimizer.step()

            total_loss += loss.item() * len(y_batch)
            total_correct += (logits.argmax(1) == y_batch).sum().item()
            total_samples += len(y_batch)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return avg_loss, avg_acc

    def validate(self, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, float]:
        """FIXED: Complete validation with proper metric computation."""
        self.model.eval()
        dataset = self._to_tensor(X_val, y_val)
        loader = DataLoader(dataset, batch_size=self.config.batch_size * 2,
                            shuffle=False, num_workers=4,
                            pin_memory=(DEVICE.type == 'cuda'))
        total_loss, total_correct, total_samples = 0.0, 0, 0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)

                total_loss += loss.item() * len(y_batch)
                total_correct += (logits.argmax(1) == y_batch).sum().item()
                total_samples += len(y_batch)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return avg_loss, avg_acc

    # ── main train loop ────────────────────────────────────────────────────

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              log_file: str = 'training_logs.csv') -> Dict:
        """
        Train the model.

        Args:
            X_train / y_train : training split (numpy, y one-hot or integer)
            X_val / y_val     : validation split (numpy, y one-hot or integer)
            log_file          : CSV filename written inside checkpoint_dir

        Returns:
            history dict with keys train_loss, val_loss, train_acc, val_acc
        """
        log_path = Path(self.config.checkpoint_dir) / log_file
        train_ds = self._to_tensor(X_train, y_train)
        train_load = DataLoader(train_ds, batch_size=self.config.batch_size,
                                shuffle=True, drop_last=False, num_workers=0,
                                pin_memory=(DEVICE.type == 'cuda'))

        history = {'train_loss': [], 'val_loss': [],
                   'train_acc': [], 'val_acc': []}

        with open(log_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=[
                'epoch', 'train_loss', 'val_loss',
                'train_acc', 'val_acc', 'learning_rate'])
            writer.writeheader()

            print(f"Starting training for {self.config.num_epochs} epochs...")
            print(f"Checkpoint dir: {self.config.checkpoint_dir}\n")

            for epoch in range(self.config.num_epochs):
                train_loss, train_acc = self.train_epoch(train_load)
                val_loss, val_acc = self.validate(X_val, y_val)

                # scheduler step
                if isinstance(self.scheduler,
                              torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

                current_lr = self._current_lr()

                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['train_acc'].append(train_acc)
                history['val_acc'].append(val_acc)

                writer.writerow({
                    'epoch': epoch + 1,
                    'train_loss': f'{train_loss:.6f}',
                    'val_loss': f'{val_loss:.6f}',
                    'train_acc': f'{train_acc:.6f}',
                    'val_acc': f'{val_acc:.6f}',
                    'learning_rate': f'{current_lr:.6e}',
                })
                csv_file.flush()

                # progress print every 5 epochs
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f"Epoch {epoch+1:>4}/{self.config.num_epochs} | "
                          f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
                          f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
                          f"LR: {current_lr:.2e}")

                # checkpoint
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint(epoch, val_loss, history)
                    print(f"  → Best model saved (val_loss={val_loss:.4f})")
                else:
                    self.patience_counter += 1

                # early stopping
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch+1} "
                          f"(patience={self.config.early_stopping_patience})")
                    break

        print(f"\nTraining complete. Logs → {log_path}")
        return history

    # ── checkpoint helpers ─────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_loss: float, history: Dict):
        """FIXED: Complete checkpoint saving."""
        ckpt = {
            'epoch': epoch,
            'val_loss': val_loss,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'config': self.config.to_dict(),
            'history': history,
        }
        path = Path(self.config.checkpoint_dir) / 'best_model.pkl'
        with open(path, 'wb') as f:
            pickle.dump(ckpt, f)

    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load best checkpoint back into the model."""
        with open(checkpoint_path, 'rb') as f:
            ckpt = pickle.load(f)
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.scheduler.load_state_dict(ckpt['scheduler_state'])
        self.best_val_loss = ckpt['val_loss']
        print(f"Loaded checkpoint from epoch {ckpt['epoch']+1} "
              f"(val_loss={ckpt['val_loss']:.4f})")
        return ckpt['history']


# ============================================================================
# 6. MODEL EVALUATOR
# ============================================================================

class ModelEvaluator:
    """
    Comprehensive evaluation — same API as the original.
    evaluate() returns accuracy, confusion_matrix, per_class_metrics,
    macro_f1, weighted_f1.
    """

    @staticmethod
    def evaluate(model: ConvNeXtCNN,
                 X_test: np.ndarray,
                 y_test: np.ndarray,
                 batch_size: int = 64) -> Dict:
        """
        Args:
            model   : trained ConvNeXtCNN
            X_test  : numpy (N, C, H, W) float32
            y_test  : numpy one-hot (N, num_classes) or integer (N,)
            batch_size: inference batch size
        Returns:
            dict with accuracy, confusion_matrix, per_class_metrics,
                 macro_f1, weighted_f1
        """
        model.eval()
        model.to(DEVICE)

        X_t = torch.from_numpy(X_test.astype(np.float32))
        if y_test.ndim == 2:
            labels_np = np.argmax(y_test, axis=1)
        else:
            labels_np = y_test.astype(int)

        all_preds = []
        with torch.no_grad():
            for i in range(0, len(X_t), batch_size):
                batch = X_t[i:i+batch_size].to(DEVICE)
                logits = model(batch)
                preds = logits.argmax(1).cpu().numpy()
                all_preds.append(preds)

        predictions = np.concatenate(all_preds)
        num_classes = (y_test.shape[1] if y_test.ndim == 2
                       else int(labels_np.max()) + 1)

        # ── accuracy ───────────────────────────────────────────────────────
        accuracy = np.mean(predictions == labels_np)

        # ── confusion matrix ───────────────────────────────────────────────
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(labels_np, predictions):
            cm[t, p] += 1

        # ── per-class P / R / F1 ───────────────────────────────────────────
        per_class_metrics = {}
        for c in range(num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)

            per_class_metrics[f'class_{c}'] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
            }

        class_counts = cm.sum(axis=1)
        macro_f1 = np.mean([m['f1'] for m in per_class_metrics.values()])
        weighted_f1 = np.sum([
            per_class_metrics[f'class_{i}']['f1'] * class_counts[i]
            for i in range(num_classes)
        ]) / class_counts.sum()

        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'per_class_metrics': per_class_metrics,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
        }

    @staticmethod
    def print_report(eval_results: Dict, class_names: List[str]):
        """FIXED: Complete evaluation report printing."""
        print(f"\n{'='*60}")
        print("ConvNeXt Model Evaluation Report")
        print(f"{'='*60}")
        print(f"Test Accuracy : {eval_results['accuracy'] * 100:.2f}%")
        print(f"Macro F1      : {eval_results['macro_f1'] * 100:.2f}%")
        print(f"Weighted F1   : {eval_results['weighted_f1'] * 100:.2f}%")
        print(f"\n{'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8}")
        print("-" * 42)
        
        for i, cls_name in enumerate(class_names):
            metrics = eval_results['per_class_metrics'][f'class_{i}']
            print(f"{cls_name:<12} {metrics['precision']*100:>9.2f}%"
                  f" {metrics['recall']*100:>7.2f}% {metrics['f1']*100:>7.2f}%")
