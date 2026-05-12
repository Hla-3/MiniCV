"""
CONVNEXT CNN MODEL - COMPREHENSIVE DOCUMENTATION

Implementation: "A ConvNet for the 2020s" (Liu et al., CVPR 2022)

This document covers:
1. Architecture overview
2. Key design choices
3. Usage instructions
4. Integration with your vision library
5. Advanced configurations
6. Troubleshooting

================================================================================
1. ARCHITECTURE OVERVIEW
================================================================================

ConvNeXt is a modernized CNN architecture that combines traditional CNN strengths
with design principles from Vision Transformers (ViTs).

KEY CHARACTERISTICS:
- Hybrid design: CNN inductive biases + Transformer design principles
- Efficient: Faster inference than ViTs, better than ResNets
- Stable: Layer Normalization instead of BatchNorm
- Smooth: GELU activation instead of ReLU
- Hierarchical: Multi-stage design with progressive downsampling

ARCHITECTURE STAGES:

    Input Images (N, 3, H, W)
           ↓
    [STEM: 4×4 Patchify with stride 4]
    Output: (N, 64, H/4, W/4)
           ↓
    [STAGE 1: 4 ConvNeXt Blocks, kernel=7×7]
    Features: 64 channels
           ↓
    [DOWNSAMPLE: 2×2 conv, stride 2]
           ↓
    [STAGE 2: 4 ConvNeXt Blocks, kernel=7×7]
    Features: 128 channels
           ↓
    [DOWNSAMPLE: 2×2 conv, stride 2]
           ↓
    [STAGE 3: 4 ConvNeXt Blocks, kernel=7×7]
    Features: 256 channels
           ↓
    [GLOBAL AVERAGE POOLING]
    Output: (N, 256)
           ↓
    [FINAL LAYER NORM]
           ↓
    [CLASSIFICATION HEAD: Dense layer]
    Output: (N, num_classes)


================================================================================
2. KEY DESIGN CHOICES (ConvNeXt vs Traditional CNNs)
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│ FEATURE 1: STEM LAYER (Patchify)                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ TRADITIONAL (ResNet):                                                       │
│   7×7 conv (stride 2) → MaxPool (3×3, stride 2)                           │
│   Result: 4× downsampling in 2 layers                                      │
│                                                                              │
│ CONVNEXT (Patchify):                                                       │
│   4×4 conv (stride 4) with non-overlapping patches                         │
│   Result: 4× downsampling in 1 layer                                       │
│   Benefit: Matches Vision Transformer patch approach, more efficient       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ FEATURE 2: LARGE KERNEL CONVOLUTIONS                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ TRADITIONAL: 3×3 kernels (small receptive field)                           │
│ CONVNEXT: 7×7 kernels (larger receptive field)                            │
│                                                                              │
│ Why 7×7?                                                                    │
│   - Matches receptive field of multi-head attention in ViTs                │
│   - Captures more context with fewer layers                                │
│   - Better for detecting large patterns and relationships                  │
│                                                                              │
│ Implementation Detail:                                                      │
│   - Used as DEPTHWISE convolutions (groups=channels)                       │
│   - More efficient: fewer parameters than standard conv                    │
│   - Followed by 1×1 convolutions for channel mixing                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ FEATURE 3: LAYER NORMALIZATION                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ TRADITIONAL (BatchNorm):                                                    │
│   - Normalizes across batch dimension                                      │
│   - Statistics depend on batch composition                                 │
│   - Different behavior at train vs inference time                          │
│   - Problems with small batches or batch size 1                            │
│                                                                              │
│ CONVNEXT (LayerNorm):                                                      │
│   - Normalizes within each sample independently                            │
│   - Same behavior at train and inference                                   │
│   - Works with any batch size                                              │
│   - Matches Transformer design                                             │
│                                                                              │
│ Mathematical Difference:                                                    │
│   BatchNorm: y = γ * (x - mean_batch) / std_batch + β                     │
│   LayerNorm: y = γ * (x - mean_sample) / std_sample + β                   │
│                                                                              │
│ Why LayerNorm is Better:                                                    │
│   ✓ More stable training                                                   │
│   ✓ Better inference (no batch statistics needed)                         │
│   ✓ Works with any batch size                                              │
│   ✓ Modern and proven in Transformers                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ FEATURE 4: GELU ACTIVATION                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ TRADITIONAL (ReLU):                                                         │
│   f(x) = max(0, x)                                                          │
│   ├─ Simple, efficient                                                      │
│   ├─ Sharp at 0 (hard non-linearity)                                       │
│   └─ Can cause gradient issues in deep networks                            │
│                                                                              │
│ CONVNEXT (GELU):                                                           │
│   f(x) = x * Φ(x)  where Φ is cumulative normal distribution              │
│   ├─ Smooth, probabilistic                                                 │
│   ├─ Gradients flow better through network                                │
│   ├─ Better for very deep architectures                                    │
│   └─ Standard in modern Transformers (BERT, GPT, ViT)                     │
│                                                                              │
│ Approximation Used:                                                         │
│   GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715*x³)))             │
│                                                                              │
│ Comparison:                                                                 │
│                  ReLU (sharp)          GELU (smooth)                       │
│         4 |      /                    ⋰                                     │
│         3 |     /                   ⋰                                       │
│         2 |    /                  ⋰                                         │
│         1 |   /                 ⋰                                           │
│         0 |__/________      __⋰_______________                             │
│        -1 |        ↓      ↓                                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ FEATURE 5: INVERTED BOTTLENECK                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ TRADITIONAL (ResNet Bottleneck):                                            │
│   1×1 conv (reduce channels)                                               │
│        ↓                                                                     │
│   3×3 conv (process)                                                        │
│        ↓                                                                     │
│   1×1 conv (expand channels)                                               │
│   Strategy: Compress → Process → Expand                                    │
│                                                                              │
│ CONVNEXT (Inverted Bottleneck):                                            │
│   1×1 conv (expand channels 4×)                                            │
│        ↓                                                                     │
│   7×7 depthwise conv (process)                                             │
│        ↓                                                                     │
│   1×1 conv (compress back)                                                 │
│   Strategy: Expand → Process → Compress                                    │
│                                                                              │
│ Why Inverted?                                                               │
│   - More efficient (depthwise conv on larger channels)                     │
│   - Better feature mixing (wider intermediate representation)              │
│   - Inspired by MobileNetV2 and modern architectures                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


================================================================================
3. COMPLETE USAGE GUIDE
================================================================================

BASIC USAGE:
────────────────────────────────────────────────────────────────────────────

    from convnext_cnn_model import ConvNeXtCNN, ConvNeXtTrainer, TrainingConfig
    
    # Step 1: Create configuration
    config = TrainingConfig(
        num_epochs=100,
        batch_size=32,
        learning_rate=0.001,
        optimizer='adam',
        lr_scheduler='cosine',
        weight_decay=0.0001,
        early_stopping_patience=15,
        seed=42
    )
    
    # Step 2: Initialize model
    model = ConvNeXtCNN(
        num_classes=6,           # Your number of classes
        input_channels=3,        # RGB images
        num_blocks=4,            # Blocks per stage
        initial_channels=64      # Initial filters
    )
    
    # Step 3: Create trainer
    trainer = ConvNeXtTrainer(model, config)
    
    # Step 4: Prepare data (X_train, y_train, X_val, y_val as numpy arrays)
    # X shape: (N, 3, 224, 224) - preprocessed images
    # y shape: (N, num_classes) - one-hot encoded labels
    
    # Step 5: Train
    history = trainer.train(X_train, y_train, X_val, y_val)
    
    # Step 6: Evaluate
    from convnext_cnn_model import ModelEvaluator
    results = ModelEvaluator.evaluate(model, X_test, y_test)


ADVANCED CONFIGURATION:
────────────────────────────────────────────────────────────────────────────

    # Custom training config for different scenarios
    
    # Scenario 1: Fast training (fewer epochs, simpler model)
    config_fast = TrainingConfig(
        num_epochs=30,
        batch_size=64,
        learning_rate=0.01,
        optimizer='sgd',
        lr_scheduler='step',
        num_blocks=2,
        initial_channels=32
    )
    
    # Scenario 2: Accurate training (more epochs, larger model)
    config_accurate = TrainingConfig(
        num_epochs=200,
        batch_size=16,
        learning_rate=0.0005,
        optimizer='adam',
        lr_scheduler='cosine',
        weight_decay=0.0001,
        early_stopping_patience=20,
        num_blocks=6,
        initial_channels=128
    )
    
    # Scenario 3: Large dataset training
    config_large = TrainingConfig(
        num_epochs=100,
        batch_size=128,
        learning_rate=0.002,
        optimizer='rmsprop',
        lr_scheduler='reduce_on_plateau',
        gradient_clip=0.5
    )


================================================================================
4. INTEGRATION WITH YOUR VISION LIBRARY
================================================================================

The implementation is designed to work seamlessly with custom vision libraries.

YOUR VISION LIBRARY RESPONSIBILITIES:
────────────────────────────────────────────────────────────────────────────

    1. Image Loading:
       - Load images from disk
       - Support various formats (JPG, PNG, etc.)
    
    2. Preprocessing:
       - Resizing to target size (224×224)
       - Normalization (mean/std)
       - Color space conversion if needed
    
    3. Augmentation (training only):
       - Random rotation
       - Random flipping
       - Random cropping
       - Color jittering
       - Gaussian blur
    
    4. Feature Extraction:
       - Color histograms
       - HOG features
       - Texture features
       - Edge features
    
    5. Output Format:
       - Images: (N, 3, H, W) numpy array, float32, [0, 1] range
       - Labels: (N, num_classes) one-hot encoded

EXAMPLE INTEGRATION:
────────────────────────────────────────────────────────────────────────────

    import your_vision_lib as vl
    from convnext_cnn_model import ConvNeXtCNN, ConvNeXtTrainer, TrainingConfig
    
    # Load data using your library
    images = vl.load_images('dataset/train')
    labels = vl.load_labels('dataset/labels.csv')
    
    # Preprocess
    X = vl.batch_resize(images, (224, 224))
    X = vl.normalize(X, mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    
    # Data split
    X_train, X_val, X_test, y_train, y_val, y_test = vl.train_val_test_split(
        X, labels, train_ratio=0.6, val_ratio=0.2
    )
    
    # Augmentation (train only)
    X_train = vl.augment(X_train,
                        rotate_range=20,
                        flip_horizontal=True,
                        color_jitter=0.2)
    
    # One-hot encode labels
    y_train_onehot = vl.to_onehot(y_train, num_classes=6)
    y_val_onehot = vl.to_onehot(y_val, num_classes=6)
    y_test_onehot = vl.to_onehot(y_test, num_classes=6)
    
    # Now use ConvNeXt
    config = TrainingConfig()
    model = ConvNeXtCNN(num_classes=6)
    trainer = ConvNeXtTrainer(model, config)
    history = trainer.train(X_train, y_train_onehot,
                           X_val, y_val_onehot)
    
    # Evaluate
    results = ModelEvaluator.evaluate(model, X_test, y_test_onehot)


================================================================================
5. OPTIMIZERS AND LEARNING RATE SCHEDULES
================================================================================

THREE OPTIMIZERS IMPLEMENTED:
────────────────────────────────────────────────────────────────────────────

    1. SGD (Stochastic Gradient Descent)
       - With optional Momentum
       - Best for: Simple tasks, baseline comparisons
       - Recommended learning rate: 0.01 - 0.1
       - Formula: w_t+1 = w_t - lr * grad + momentum * velocity_t
    
    2. Adam (Adaptive Moment Estimation)
       - Default recommended optimizer
       - Best for: Most tasks, good default choice
       - Recommended learning rate: 0.0001 - 0.001
       - Combines momentum with per-parameter adaptive learning rates
    
    3. RMSProp (Root Mean Square Propagation)
       - Good for recurrent networks and when Adam fails
       - Best for: Problematic training cases
       - Recommended learning rate: 0.001 - 0.01


FOUR LEARNING RATE SCHEDULES:
────────────────────────────────────────────────────────────────────────────

    1. Step Decay
       lr = initial_lr * (gamma ^ (epoch // step_size))
       Example: lr starts at 0.001, multiplied by 0.1 every 10 epochs
       When to use: Traditional choice, simple and effective
    
    2. Exponential Decay
       lr = initial_lr * (gamma ^ epoch)
       Example: lr decays exponentially throughout training
       When to use: Smooth decay, predictable schedule
    
    3. Cosine Annealing
       lr = lr_min + (initial_lr - lr_min) * (1 + cos(π * epoch / T_max)) / 2
       When to use: RECOMMENDED for deep learning, works well with SGD
    
    4. Reduce on Plateau
       Reduce lr by factor when validation loss stops improving
       When to use: Adaptive, responds to actual training progress


RECOMMENDED COMBINATIONS:
────────────────────────────────────────────────────────────────────────────

    For Fast Training:
        optimizer='sgd', lr=0.01, scheduler='step', decay=10 epochs
    
    For Accuracy (Recommended):
        optimizer='adam', lr=0.001, scheduler='cosine'
    
    For Large Datasets:
        optimizer='rmsprop', lr=0.005, scheduler='reduce_on_plateau'


================================================================================
6. ADVANCED FEATURES
================================================================================

EARLY STOPPING:
────────────────────────────────────────────────────────────────────────────

    Prevents overfitting by stopping training when validation loss stops improving.
    
    config = TrainingConfig(early_stopping_patience=15)
    
    Behavior:
    - Epoch 1: val_loss = 0.5 (best) - patience = 0
    - Epoch 2: val_loss = 0.49 (best) - patience = 0
    - Epoch 3: val_loss = 0.50 (worse) - patience = 1
    - Epoch 4: val_loss = 0.51 (worse) - patience = 2
    ...
    - Epoch 18: val_loss = 0.52 (worse) - patience = 15 → STOP

GRADIENT CLIPPING:
────────────────────────────────────────────────────────────────────────────

    Prevents exploding gradients in deep networks.
    
    config = TrainingConfig(gradient_clip=1.0)
    
    Effect: Large gradients are clipped to max 1.0
    Prevents sudden large parameter updates that destabilize training

L2 REGULARIZATION (Weight Decay):
────────────────────────────────────────────────────────────────────────────

    Penalizes large weights to prevent overfitting.
    
    config = TrainingConfig(weight_decay=0.0001)
    
    Formula: loss = cross_entropy + weight_decay * ||weights||²
    Effect: Encourages smaller, simpler weights

MINI-BATCH SHUFFLING:
────────────────────────────────────────────────────────────────────────────

    Data is shuffled each epoch for better convergence.
    Automatically handled by trainer.train_epoch()


================================================================================
7. TROUBLESHOOTING & TIPS
================================================================================

PROBLEM: Training loss not decreasing
────────────────────────────────────────────────────────────────────────────
Solutions:
  ✓ Increase learning rate (try 10× larger)
  ✓ Reduce weight decay
  ✓ Check data normalization
  ✓ Verify data is properly shuffled

PROBLEM: Training loss decreasing but validation loss increasing (overfitting)
────────────────────────────────────────────────────────────────────────────
Solutions:
  ✓ Increase weight decay (regularization)
  ✓ Add more data augmentation
  ✓ Reduce model size (fewer blocks or channels)
  ✓ Use early stopping
  ✓ Decrease learning rate

PROBLEM: Very slow training
────────────────────────────────────────────────────────────────────────────
Solutions:
  ✓ Increase batch size (if memory allows)
  ✓ Reduce num_blocks
  ✓ Reduce initial_channels
  ✓ Use smaller input images

PROBLEM: GPU out of memory
────────────────────────────────────────────────────────────────────────────
Solutions:
  ✓ Reduce batch_size
  ✓ Reduce image size
  ✓ Reduce num_blocks
  ✓ Reduce initial_channels

BEST PRACTICES:
────────────────────────────────────────────────────────────────────────────

  1. Always normalize inputs (images to [0,1] or use ImageNet normalization)
  2. One-hot encode classification labels
  3. Start with default hyperparameters, then tune
  4. Monitor both training and validation loss
  5. Save checkpoints frequently
  6. Use gradient clipping for very deep networks
  7. Apply data augmentation to training set only
  8. Use early stopping to prevent overfitting
  9. Try learning rate decay if loss plateaus
  10. Consider batch size: larger batches = faster training, smaller = better generalization


================================================================================
8. OUTPUT FILES AND LOGGING
================================================================================

LOGS.CSV:
────────────────────────────────────────────────────────────────────────────

Columns:
  - epoch: Training epoch number
  - train_loss: Cross-entropy loss on training set
  - val_loss: Cross-entropy loss on validation set
  - train_acc: Accuracy on training set (0-1)
  - val_acc: Accuracy on validation set (0-1)
  - learning_rate: Current learning rate (may change with schedule)

Example output:
    epoch,train_loss,val_loss,train_acc,val_acc,learning_rate
    1,2.156423,2.143521,0.201234,0.195123,0.001000
    2,1.852341,1.834521,0.312456,0.298765,0.000999
    3,1.521234,1.534521,0.421234,0.398765,0.000998
    ...

CHECKPOINT FILES:
────────────────────────────────────────────────────────────────────────────

best_model.pkl:
  Contains:
    - model weights and biases
    - optimizer state (for resuming training)
    - training configuration
    - training history
    - epoch and validation loss

Load checkpoint:
    trainer.load_checkpoint('checkpoints/best_model.pkl')

EVALUATION RESULTS:
────────────────────────────────────────────────────────────────────────────

Results dict contains:
  - accuracy: Overall test accuracy
  - confusion_matrix: (num_classes, num_classes) matrix
  - per_class_metrics: precision, recall, F1 for each class
  - macro_f1: Unweighted average F1 across classes
  - weighted_f1: Weighted average F1 (weighted by class frequency)

Example:
    {
        'accuracy': 0.876,
        'confusion_matrix': array([[45,  3,  2],
                                   [ 2, 48,  0],
                                   [ 1,  2, 47]]),
        'per_class_metrics': {
            'class_0': {'precision': 0.93, 'recall': 0.90, 'f1': 0.91},
            'class_1': {'precision': 0.92, 'recall': 0.96, 'f1': 0.94},
            'class_2': {'precision': 0.96, 'recall': 0.94, 'f1': 0.95}
        },
        'macro_f1': 0.933,
        'weighted_f1': 0.935
    }


================================================================================
9. MATHEMATICAL FORMULATIONS
================================================================================

SOFTMAX (Numerically Stable Version):
────────────────────────────────────────────────────────────────────────────

Standard (unstable):          softmax(x_i) = exp(x_i) / Σ exp(x_j)
Numerically stable:           softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))

Why subtract max? Prevents overflow for large x values.

CROSS-ENTROPY LOSS (with epsilon clipping):
────────────────────────────────────────────────────────────────────────────

Standard:           L = -Σ y_i * log(p_i)
With clipping:      p_i = clip(p_i, ε, 1-ε)  where ε ≈ 1e-7
Formula:            L = -Σ y_i * log(clip(p_i, ε, 1-ε))

Why clip? Prevents log(0) = -∞ when p_i = 0.

LAYER NORMALIZATION:
────────────────────────────────────────────────────────────────────────────

Given input X ∈ ℝ^(N × D):
  mean = (1/D) * Σ x_d
  var = (1/D) * Σ (x_d - mean)²
  x̂ = (x - mean) / √(var + ε)
  output = γ * x̂ + β

Where γ and β are learned parameters.

GELU ACTIVATION:
────────────────────────────────────────────────────────────────────────────

Exact:          GELU(x) = x * Φ(x)  where Φ is standard normal CDF
Approximation:  GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715*x³)))

Derivative (approximation):
  GELU'(x) = Φ(x) + x * φ(x) + derivative of tanh term

ADAM OPTIMIZER:
────────────────────────────────────────────────────────────────────────────

For each parameter w_t:
  m_t = β₁ * m_{t-1} + (1-β₁) * ∇L        (first moment estimate)
  v_t = β₂ * v_{t-1} + (1-β₂) * (∇L)²     (second moment estimate)
  
  m̂_t = m_t / (1 - β₁^t)                  (bias correction)
  v̂_t = v_t / (1 - β₂^t)                  (bias correction)
  
  w_{t+1} = w_t - α * m̂_t / (√v̂_t + ε)

Default: β₁=0.9, β₂=0.999, α=0.001, ε=1e-8


================================================================================
10. REFERENCES
================================================================================

Paper: A ConvNet for the 2020s
Authors: Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, 
         Trevor Darrell, Saining Xie
Venue: CVPR 2022
Paper Link: https://arxiv.org/abs/2201.03545

Related Work:
  - Vision Transformers (ViT): Dosovitskiy et al., 2021
  - Swin Transformers: Liu et al., 2021
  - ResNet: He et al., 2015
  - MobileNetV2: Sandler et al., 2018
  - EfficientNet: Tan & Le, 2019

Implementation Notes:
  - This is a simplified implementation for educational purposes
  - Production versions would use optimized CUDA kernels
  - For research, consider PyTorch/TensorFlow implementations
  - This code emphasizes clarity over performance

"""

# Print this documentation
if __name__ == "__main__":
    print(__doc__)
