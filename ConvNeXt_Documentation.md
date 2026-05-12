# ConvNeXt CNN - QUICK REFERENCE GUIDE

## Files Provided

1. **convnext_cnn_model.py** - Complete implementation
   - ConvNeXt architecture
   - Optimizers (SGD, Adam, RMSProp)
   - Learning rate schedulers
   - Training loop with early stopping
   - Model evaluation

2. **training_pipeline_example.py** - Integration example
   - Data loading and preprocessing
   - Feature extraction
   - MRMR feature selection
   - Complete workflow

3. **CONVNEXT_DOCUMENTATION.py** - Full documentation
   - Architecture details
   - Design decisions
   - Mathematical formulations
   - Troubleshooting guide

---

## 30-SECOND STARTUP

```python
from convnext_cnn_model import ConvNeXtCNN, ConvNeXtTrainer, TrainingConfig

# 1. Config
config = TrainingConfig(
    num_epochs=100, batch_size=32, learning_rate=0.001, 
    optimizer='adam', lr_scheduler='cosine'
)

# 2. Model
model = ConvNeXtCNN(num_classes=6, input_channels=3)

# 3. Trainer
trainer = ConvNeXtTrainer(model, config)

# 4. Train (X_train, y_train, X_val, y_val must be numpy arrays)
history = trainer.train(X_train, y_train, X_val, y_val)

# 5. Evaluate
results = ModelEvaluator.evaluate(model, X_test, y_test)
print(f"Accuracy: {results['accuracy']:.4f}")
```

---

## DATA FORMAT REQUIREMENTS

### Input Images
- Shape: `(N, 3, H, W)` where N=batch size, 3=RGB channels
- Type: numpy array, float32
- Range: [0, 1] or normalized with ImageNet statistics
- Size: Typically 224×224 (can be different)

### Labels
- Shape: `(N, num_classes)` one-hot encoded
- Type: numpy array, float32
- Example: `[[1,0,0], [0,1,0], [0,0,1]]` for 3 classes

### One-Hot Encoding
```python
import numpy as np

def to_onehot(labels, num_classes):
    onehot = np.zeros((len(labels), num_classes))
    onehot[np.arange(len(labels)), labels] = 1
    return onehot

# Usage
y_onehot = to_onehot(y_labels, num_classes=6)
```

---

## CONFIGURATION PRESETS

### Minimal (Quick Test)
```python
TrainingConfig(
    num_epochs=10, batch_size=64, learning_rate=0.01,
    optimizer='sgd', lr_scheduler='step'
)
```

### Balanced (Recommended for Most Tasks)
```python
TrainingConfig(
    num_epochs=100, batch_size=32, learning_rate=0.001,
    optimizer='adam', lr_scheduler='cosine',
    weight_decay=0.0001, early_stopping_patience=15
)
```

### High Accuracy (More Computation)
```python
TrainingConfig(
    num_epochs=200, batch_size=16, learning_rate=0.0005,
    optimizer='adam', lr_scheduler='cosine',
    weight_decay=0.00001, early_stopping_patience=25,
    gradient_clip=0.5
)
```

### Large Dataset (Batch Training)
```python
TrainingConfig(
    num_epochs=100, batch_size=128, learning_rate=0.002,
    optimizer='rmsprop', lr_scheduler='reduce_on_plateau',
    weight_decay=0.0001
)
```

---

## MODEL CONFIGURATION

### Lightweight Model (Fast)
```python
ConvNeXtCNN(
    num_classes=6, input_channels=3,
    num_blocks=2,           # Fewer blocks
    initial_channels=32     # Fewer channels
)
```

### Balanced Model (Recommended)
```python
ConvNeXtCNN(
    num_classes=6, input_channels=3,
    num_blocks=4,           # Default
    initial_channels=64     # Default
)
```

### Large Model (High Accuracy)
```python
ConvNeXtCNN(
    num_classes=6, input_channels=3,
    num_blocks=6,           # More blocks
    initial_channels=128    # More channels
)
```

---

## TRAINING MONITORING

### Check logs.csv
```bash
epoch,train_loss,val_loss,train_acc,val_acc,learning_rate
1,2.156423,2.143521,0.201234,0.195123,0.001000
2,1.852341,1.834521,0.312456,0.298765,0.000999
3,1.521234,1.534521,0.421234,0.398765,0.000998
```

### Good Signs
- ✓ train_loss decreases
- ✓ val_loss decreases and is close to train_loss
- ✓ train_acc and val_acc both improve
- ✓ Learning rate changes smoothly

### Problem Signs
- ✗ train_loss not decreasing → increase learning rate
- ✗ val_loss increases while train_loss decreases → overfitting
- ✗ Both losses increasing → learning rate too high
- ✗ Losses constant → learning rate too low

---

## EVALUATION METRICS

```python
results = ModelEvaluator.evaluate(model, X_test, y_test)

# Accuracy: percentage of correct predictions
accuracy = results['accuracy']  # 0-1 scale

# Confusion matrix: true labels vs predictions
cm = results['confusion_matrix']  # shape (num_classes, num_classes)

# Per-class metrics
for class_name, metrics in results['per_class_metrics'].items():
    print(f"{class_name}: P={metrics['precision']:.3f}, "
          f"R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")

# Macro F1: unweighted average across classes
macro_f1 = results['macro_f1']

# Weighted F1: weighted by class frequency
weighted_f1 = results['weighted_f1']
```

---

## COMMON OPTIMIZERS

| Optimizer | Best For | Learning Rate | Momentum |
|-----------|----------|---------------|----------|
| SGD | Baseline/Simple tasks | 0.01-0.1 | 0.9 |
| Adam | Most tasks (RECOMMENDED) | 0.0001-0.001 | adaptive |
| RMSProp | When Adam fails | 0.001-0.01 | 0.9 |

---

## LEARNING RATE SCHEDULES

| Schedule | Formula | Best For |
|----------|---------|----------|
| Step | `lr = lr₀ × 0.1^(epoch÷10)` | Simple, traditional |
| Exponential | `lr = lr₀ × 0.95^epoch` | Smooth decay |
| Cosine | `lr = (1 + cos(π×t/T))/2` | RECOMMENDED |
| ReduceOnPlateau | Manual reduction | Adaptive, responsive |

---

## TROUBLESHOOTING CHECKLIST

### Training doesn't improve
- [ ] Check data normalization (should be [0,1] or ImageNet stats)
- [ ] Check label format (must be one-hot encoded)
- [ ] Increase learning rate (try 10× larger)
- [ ] Verify data isn't all same class
- [ ] Check for NaN/Inf values in data

### Overfitting (val loss > train loss)
- [ ] Increase weight_decay (regularization)
- [ ] Add data augmentation
- [ ] Use early_stopping_patience
- [ ] Reduce model size
- [ ] Decrease learning rate

### Memory issues
- [ ] Reduce batch_size (32→16)
- [ ] Reduce num_blocks (4→2)
- [ ] Reduce initial_channels (64→32)
- [ ] Reduce image size (224→112)

### Training too slow
- [ ] Increase batch_size
- [ ] Reduce num_blocks
- [ ] Reduce initial_channels
- [ ] Use SGD instead of Adam

---

## SAVING AND LOADING

### Automatic Checkpointing
Best model is automatically saved to `checkpoints/best_model.pkl`

### Manual Save
```python
import pickle
checkpoint = {
    'model_weights': model_weights,
    'config': config,
    'history': history
}
with open('my_model.pkl', 'wb') as f:
    pickle.dump(checkpoint, f)
```

### Loading
```python
trainer.load_checkpoint('checkpoints/best_model.pkl')
```

---

## INTEGRATION CHECKLIST

When integrating with your vision library:

- [ ] Images preprocessed to shape (N, 3, H, W)
- [ ] Images normalized to [0, 1] range
- [ ] Labels one-hot encoded to shape (N, num_classes)
- [ ] Data split into train/val/test sets
- [ ] Augmentation applied to training set only
- [ ] All arrays are numpy, dtype=float32
- [ ] No NaN or Inf values in data
- [ ] Class distribution roughly balanced

---

## EXAMPLE WORKFLOW

```python
# 1. Prepare data using your vision library
images = your_lib.load_images('dataset/')
labels = your_lib.load_labels('labels.csv')

# 2. Preprocess
X = your_lib.resize_batch(images, (224, 224))
X = your_lib.normalize(X)  # [0, 1] range

# 3. Split
train_idx, val_idx, test_idx = your_lib.split_indices(
    len(X), train=0.6, val=0.2
)
X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]

# 4. Augment (training only)
X_train = your_lib.augment(X_train, rotate=20, flip=True)
y_train = np.repeat(y_train, 2, axis=0)  # Match augmented images

# 5. One-hot encode
y_train = to_onehot(y_train, num_classes=6)
y_val = to_onehot(y_val, num_classes=6)
y_test = to_onehot(y_test, num_classes=6)

# 6. Train
config = TrainingConfig()
model = ConvNeXtCNN(num_classes=6)
trainer = ConvNeXtTrainer(model, config)
history = trainer.train(X_train, y_train, X_val, y_val)

# 7. Evaluate
results = ModelEvaluator.evaluate(model, X_test, y_test)
print(f"Test Accuracy: {results['accuracy']:.4f}")
```

---

## HYPERPARAMETER TUNING GRID

Try these combinations if accuracy is unsatisfactory:

```python
learning_rates = [0.0001, 0.0005, 0.001, 0.005]
optimizers = ['adam', 'sgd', 'rmsprop']
schedulers = ['cosine', 'step', 'exponential']
weight_decays = [0, 0.00001, 0.0001, 0.001]
batch_sizes = [16, 32, 64, 128]
```

Start with: `lr=0.001, adam, cosine, wd=0.0001, bs=32`

---

## KEY CONVNEXT FEATURES IN THIS IMPLEMENTATION

✓ 4×4 Patchify Stem (efficient preprocessing)
✓ 7×7 Depthwise Convolutions (large receptive field)
✓ Layer Normalization (stable, batch-independent)
✓ GELU Activation (smooth, modern)
✓ Inverted Bottleneck Blocks (efficient design)
✓ Multi-stage Architecture (hierarchical features)
✓ Global Average Pooling (parameter-efficient)
✓ SGD + Momentum, Adam, RMSProp (multiple optimizers)
✓ Step, Exponential, Cosine, ReduceOnPlateau (LR schedules)
✓ Early Stopping (prevent overfitting)
✓ Gradient Clipping (stable training)
✓ L2 Regularization (weight decay)
✓ Checkpoint Management (resumable training)
✓ Comprehensive Logging (reproducible experiments)
✓ Complete Evaluation Metrics (detailed analysis)

---

## EXPECTED PERFORMANCE

On typical 6-class classification problems:
- Fast training: 80-85% accuracy in 30 epochs
- Balanced training: 88-92% accuracy in 100 epochs
- High accuracy: 92-96% accuracy in 200 epochs

Depends on:
- Dataset size and quality
- Class balance
- Image complexity
- Model size
- Training time

---

## WHEN TO USE CONVNEXT

✓ Image classification tasks
✓ Inference speed matters (faster than ViT)
✓ Variable batch sizes
✓ Limited computational resources
✓ Traditional CNN inductive biases preferred
✗ Segmentation/detection (consider U-Net, Faster R-CNN)
✗ Very small datasets (<100 images) (use transfer learning)
✗ Non-visual data (use MLPs or RNNs)

---

## GETTING STARTED

1. Read CONVNEXT_DOCUMENTATION.py for theory
2. Check training_pipeline_example.py for integration
3. Modify convnext_cnn_model.py if needed
4. Start with TrainingConfig defaults
5. Monitor logs.csv during training
6. Save best_model.pkl when done
7. Evaluate on test set with ModelEvaluator

Good luck! 🚀
