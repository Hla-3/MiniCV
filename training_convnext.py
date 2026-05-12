"""
training_pipeline_example.py
────────────────────────────
Pipeline using MRMR feature-selection outputs + PyTorch ConvNeXt.

Expected files (update DATA_DIR if needed):
    F_train_selected.npy   – (N_train, num_selected_features)
    F_val_selected.npy     – (N_val,   num_selected_features)
    F_test_selected.npy    – (N_test,  num_selected_features)
    y_train.npy            – (N_train,)  integer class labels
    y_val.npy              – (N_val,)    integer class labels
    y_test.npy             – (N_test,)   integer class labels
    selected_indices.npy   – (num_selected_features,)
    selected_feature_names.txt

Output artefacts (written to OUT_DIR):
    checkpoints/training_logs.csv
    checkpoints/best_model.pkl
    evaluation_report.txt
"""

import os
import numpy as np
from convnext_cnn_model import ConvNeXtCNN, ConvNeXtTrainer, TrainingConfig, ModelEvaluator, plot_results
import time

# ═════════════════════════════════════════════════════════════════════════════
# 0.  PATHS
# ═════════════════════════════════════════════════════════════════════════════

DATA_DIR = "."
OUT_DIR  = "."

# ═════════════════════════════════════════════════════════════════════════════
# 1.  HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def load_npy(filename: str) -> np.ndarray:
    return np.load(os.path.join(DATA_DIR, filename)).astype(np.float32)


def to_onehot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    onehot = np.zeros((len(labels), num_classes), dtype=np.float32)
    onehot[np.arange(len(labels)), labels.astype(int)] = 1.0
    return onehot


def reshape_for_convnext(X: np.ndarray) -> np.ndarray:
    """
    Reshape 1-D feature vectors to (N, 3, H, W) for ConvNeXt.

    Minimum side = 64 so spatial dims survive:
        stem (÷4) -> 16  ->  down2 (÷2) -> 8  ->  down3 (÷2) -> 4
    which is safely above 0 for global-average-pooling.
    """
    N, F = X.shape
    MIN_SIDE = 16
    side     = max(int(np.ceil(np.sqrt(F))), MIN_SIDE)
    pad      = side * side - F

    X_padded = np.pad(X, ((0, 0), (0, pad)))       # (N, side^2)
    X_2d     = X_padded.reshape(N, 1, side, side)  # (N, 1, H, W)
    X_3ch    = np.repeat(X_2d, 3, axis=1)          # (N, 3, H, W)
    return X_3ch


def print_section(title: str) -> None:
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")

if __name__ == '__main__':

    # ═════════════════════════════════════════════════════════════════════════════
    # 2.  LOAD DATA
    # ═════════════════════════════════════════════════════════════════════════════

    print_section("Loading feature-selected data")

    F_train = load_npy("./feature_selection_Output/F_train_selected.npy")
    F_val   = load_npy("./feature_selection_Output/F_val_selected.npy")
    F_test  = load_npy("./feature_selection_Output/F_test_selected.npy")

    y_train_raw = load_npy("./feature_selection_Output/y_train.npy")
    y_val_raw   = load_npy("./feature_selection_Output/y_val.npy")
    y_test_raw  = load_npy("./feature_selection_Output/y_test.npy")

    selected_indices = np.load(os.path.join(DATA_DIR, "./feature_selection_Output/selected_indices.npy"))
    feat_names_path  = os.path.join(DATA_DIR, "./feature_selection_Output/selected_feature_names.txt")
    if os.path.exists(feat_names_path):
        with open(feat_names_path) as fh:
            feature_names = [ln.strip() for ln in fh if ln.strip()]
    else:
        feature_names = [f"feat_{i}" for i in selected_indices]

    NUM_FEATURES = F_train.shape[1]
    NUM_CLASSES  = int(y_train_raw.max()) + 1

    print(f"  Train samples : {len(F_train):,}")
    print(f"  Val   samples : {len(F_val):,}")
    print(f"  Test  samples : {len(F_test):,}")
    print(f"  Selected feats: {NUM_FEATURES}")
    print(f"  Classes       : {NUM_CLASSES}")

    # ═════════════════════════════════════════════════════════════════════════════
    # 3.  RESHAPE  (no re-normalisation — already done in preprocessing)
    # ═════════════════════════════════════════════════════════════════════════════

    print_section("Reshaping feature vectors -> (N, 3, H, W)")

    X_train = reshape_for_convnext(F_train)
    X_val   = reshape_for_convnext(F_val)
    X_test  = reshape_for_convnext(F_test)

    print(f"  X_train : {X_train.shape}")
    print(f"  X_val   : {X_val.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  NaN in X_train: {np.isnan(X_train).any()}")

    # ═════════════════════════════════════════════════════════════════════════════
    # 4.  ONE-HOT LABELS
    # ═════════════════════════════════════════════════════════════════════════════

    y_train = to_onehot(y_train_raw, NUM_CLASSES)
    y_val   = to_onehot(y_val_raw,   NUM_CLASSES)
    y_test  = to_onehot(y_test_raw,  NUM_CLASSES)

    # ═════════════════════════════════════════════════════════════════════════════
    # 5.  MODEL & CONFIG
    # ═════════════════════════════════════════════════════════════════════════════

    print_section("Building model & config")

    CHECKPOINT_DIR = os.path.join(OUT_DIR, "ConvNeXt_logs")

    config = TrainingConfig(
        num_epochs              = 50,
        batch_size              = 512,
        learning_rate           = 0.001,
        optimizer               = 'adam',
        lr_scheduler            = 'cosine',
        weight_decay            = 0.0001,
        early_stopping_patience = 15,
        gradient_clip           = 1.0,
        checkpoint_dir          = CHECKPOINT_DIR,
    )

    model = ConvNeXtCNN(
        num_classes      = NUM_CLASSES,
        input_channels   = 3,
        num_blocks       = 2,  # Reduced from 4 for faster training on tabular data
        initial_channels = 64,
    )

    trainer = ConvNeXtTrainer(model, config)

    print(f"  Optimizer  : {config.optimizer}")
    print(f"  Scheduler  : {config.lr_scheduler}")
    print(f"  Epochs     : {config.num_epochs}  (patience={config.early_stopping_patience})")
    print(f"  Batch size : {config.batch_size}")
    print(f"  Checkpoint : {CHECKPOINT_DIR}/best_model.pkl")

    # ═════════════════════════════════════════════════════════════════════════════
    # 6.  TRAIN
    # ═════════════════════════════════════════════════════════════════════════════

    print_section("Training")

    t0      = time.time()
    history = trainer.train(X_train, y_train, X_val, y_val)
    elapsed = time.time() - t0

    print(f"\n  Training finished in {elapsed/60:.1f} min")
    print(f"  Best val accuracy : {max(history['val_acc']):.4f}")
    print(f"  Best val loss     : {min(history['val_loss']):.6f}")

    # ═════════════════════════════════════════════════════════════════════════════
    # 7.  LOAD BEST CHECKPOINT & EVALUATE
    # ═════════════════════════════════════════════════════════════════════════════

    print_section("Evaluating on test set")

    best_ckpt = os.path.join(CHECKPOINT_DIR, "best_model.pkl")
    if os.path.exists(best_ckpt):
        trainer.load_checkpoint(best_ckpt)
    else:
        print("  [WARNING] No checkpoint found – using last epoch weights.")

    results = ModelEvaluator.evaluate(model, X_test, y_test)

    print(f"\n  Test Accuracy : {results['accuracy']:.4f}")
    print(f"  Macro   F1    : {results['macro_f1']:.4f}")
    print(f"  Weighted F1   : {results['weighted_f1']:.4f}")
    print(f"\n  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("  " + "-" * 54)
    for cls, m in results['per_class_metrics'].items():
        print(f"  {cls:<20} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f}")
    print("\n  Confusion matrix:")
    print(results['confusion_matrix'])

    # ═════════════════════════════════════════════════════════════════════════════
    # 8.  WRITE EVALUATION REPORT
    # ═════════════════════════════════════════════════════════════════════════════

    report_path = os.path.join(OUT_DIR, "ConvNeXt_evaluation_report.txt")
    with open(report_path, "w") as rpt:
        rpt.write("ConvNeXt (PyTorch) – Feature-Selected Pipeline | Evaluation Report\n")
        rpt.write("=" * 65 + "\n\n")
        rpt.write(f"Selected features : {NUM_FEATURES}\n")
        rpt.write(f"Spatial input     : {X_train.shape[2]}x{X_train.shape[3]}\n")
        rpt.write(f"Classes           : {NUM_CLASSES}\n")
        rpt.write(f"Train/Val/Test    : {len(X_train)} / {len(X_val)} / {len(X_test)}\n")
        rpt.write(f"Training time     : {elapsed/60:.1f} min\n\n")
        rpt.write(f"Test Accuracy  : {results['accuracy']:.4f}\n")
        rpt.write(f"Macro   F1     : {results['macro_f1']:.4f}\n")
        rpt.write(f"Weighted F1    : {results['weighted_f1']:.4f}\n\n")
        rpt.write("Per-class metrics:\n")
        rpt.write(f"  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}\n")
        rpt.write("  " + "-" * 54 + "\n")
        for cls, m in results['per_class_metrics'].items():
            rpt.write(f"  {cls:<20} {m['precision']:>10.3f}"
                    f" {m['recall']:>10.3f} {m['f1']:>10.3f}\n")
        rpt.write("\nConfusion matrix:\n")
        rpt.write(str(results['confusion_matrix']) + "\n\n")
        rpt.write("Training history (last 10 epochs):\n")
        rpt.write(f"  {'Epoch':>6} {'TrainLoss':>12} {'ValLoss':>12} "
                f"{'TrainAcc':>10} {'ValAcc':>10}\n")
        n = len(history['train_loss'])
        for i in range(max(0, n-10), n):
            rpt.write(f"  {i+1:>6}   {history['train_loss'][i]:>12.6f}"
                    f" {history['val_loss'][i]:>12.6f}"
                    f" {history['train_acc'][i]:>10.4f}"
                    f" {history['val_acc'][i]:>10.4f}\n")

    print(f"\n  Report saved -> {report_path}")

    # ═════════════════════════════════════════════════════════════════════════════
    # 7.5 PLOT RESULTS
    # ═════════════════════════════════════════════════════════════════════════════

    print_section("Plotting results")

    # Extract the per-class metrics into numpy arrays for the plotting function
    class_names = [str(c) for c in results['per_class_metrics'].keys()]
    precision = np.array([m['precision'] for m in results['per_class_metrics'].values()])
    recall = np.array([m['recall'] for m in results['per_class_metrics'].values()])
    f1 = np.array([m['f1'] for m in results['per_class_metrics'].values()])

    # Call the imported plotting function
    plot_results(
        history=history,
        cm=results['confusion_matrix'],
        precision=precision,
        recall=recall,
        f1=f1,
        macro_f1=results['macro_f1'],
        weighted_f1=results['weighted_f1'],
        test_acc=results['accuracy'],
        class_names=class_names,
        output_dir=OUT_DIR
    )

    # ═════════════════════════════════════════════════════════════════════════════
    # 9.  SUMMARY
    # ═════════════════════════════════════════════════════════════════════════════

    print_section("Done")
    print(f"  training_logs.csv     -> {CHECKPOINT_DIR}/ConvNeXt_training_logs.csv")
    print(f"  best_model.pkl        -> {CHECKPOINT_DIR}/ConvNeXt_best_model.pkl")
    print(f"  evaluation_report.txt -> {report_path}")
    print()
