import numpy as np
import csv
import json
import os

class Layer:
    def forward(self, input_data): raise NotImplementedError
    def backward(self, output_gradient, learning_rate): raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_size, output_size, l2_lambda=0.0):
        # He Initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))
        self.l2_lambda = l2_lambda

    def forward(self, input_data):
        self.input = input_data
        return np.dot(self.input, self.weights) + self.biases

    def backward(self, output_gradient):
        # Gradients with L2 Regularization (Weight Decay)
        self.weights_gradient = np.dot(self.input.T, output_gradient) + (self.l2_lambda * self.weights)
        self.biases_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        input_gradient = np.dot(output_gradient, self.weights.T)
        return input_gradient, self.weights_gradient, self.biases_gradient

class ReLU(Layer):
    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)

    def backward(self, output_gradient):
        return output_gradient * (self.input > 0)

class Flatten(Layer):
    def forward(self, input_data):
        self.input_shape = input_data.shape
        # Flatten everything except the batch dimension
        return input_data.reshape(input_data.shape[0], -1)

    def backward(self, output_gradient):
        return output_gradient.reshape(self.input_shape)

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # Initialize filters: (out_channels, in_channels, k_h, k_w)
        self.filters = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1
        self.biases = np.zeros((out_channels, 1))

    def forward(self, input_data):
        self.input = input_data
        # Note: A production MiniCV library would use im2col here for speed. 
        # This is the structural loop-based mathematical definition.
        batch_size, in_c, in_h, in_w = input_data.shape
        out_h = (in_h - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_w = (in_w - self.kernel_size + 2 * self.padding) // self.stride + 1
        
        output = np.zeros((batch_size, self.out_channels, out_h, out_w))
        
        # Padding input if necessary
        padded_input = np.pad(input_data, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        self.padded_input = padded_input

        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start, w_start = i * self.stride, j * self.stride
                        h_end, w_end = h_start + self.kernel_size, w_start + self.kernel_size
                        slice_val = padded_input[b, :, h_start:h_end, w_start:w_end]
                        output[b, c_out, i, j] = np.sum(slice_val * self.filters[c_out]) + self.biases[c_out]
        return output

    def backward(self, output_gradient):
        # Backward pass requires correlating the output_gradient with the padded input
        # (Implementation of dF, db, dX goes here using similar loops or im2col)
        pass # To be completed based on your spatial_filter Core library logic

class MaxPool2D(Layer):
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input_data):
        self.input = input_data
        batch_size, in_c, in_h, in_w = input_data.shape
        out_h = (in_h - self.pool_size) // self.stride + 1
        out_w = (in_w - self.pool_size) // self.stride + 1
        
        self.output = np.zeros((batch_size, in_c, out_h, out_w))
        
        for b in range(batch_size):
            for c in range(in_c):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start, w_start = i * self.stride, j * self.stride
                        h_end, w_end = h_start + self.pool_size, w_start + self.pool_size
                        slice_val = input_data[b, c, h_start:h_end, w_start:w_end]
                        self.output[b, c, i, j] = np.max(slice_val)
        return self.output
        
    def backward(self, output_gradient):
        # Routing the gradient only to the pixel that had the max value
        pass

def softmax(z):
    # Shift z for numerical stability to prevent overflow
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

def categorical_crossentropy(y_pred, y_true):
    # Add epsilon to prevent log(0)
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def crossentropy_backward(y_pred, y_true):
    # The combined derivative of Softmax + Cross Entropy simplifies beautifully to:
    return (y_pred - y_true) / y_true.shape[0]

class AdamOptimizer:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.m = [np.zeros_like(p) for p in parameters]
        self.v = [np.zeros_like(p) for p in parameters]
        self.t = 0

    def step(self, parameters, gradients, clip_value=1.0):
        self.t += 1
        for i, (param, grad) in enumerate(zip(parameters, gradients)):
            # Gradient Clipping (Safety Feature 2)
            grad = np.clip(grad, -clip_value, clip_value)
            
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

def step_decay(epoch, initial_lr, drop=0.5, epochs_drop=5):
    # Learning Rate Schedule (Section 6)
    return initial_lr * (drop ** np.floor((1 + epoch) / epochs_drop))

def train_network(X_train, Y_train, X_val, Y_val, epochs, batch_size):
    # Network Initialization
    network = [
        Flatten(),
        Dense(X_train.shape[1], 128, l2_lambda=0.01), # Added L2 Weight Decay
        ReLU(),
        Dense(128, Y_train.shape[1], l2_lambda=0.01)
    ]
    
    # Optimizer Setup
    trainable_params = []
    for layer in network:
        if isinstance(layer, Dense):
            trainable_params.extend([layer.weights, layer.biases])
            
    optimizer = AdamOptimizer(trainable_params, lr=0.001)
    
    # State tracking
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 5 # Early Stopping (Safety Feature 1)
    
    csv_file = 'log_cnn.csv'
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'learning_rate'])

    for epoch in range(epochs):
        # Mini-batch shuffling (Safety Feature 4)
        indices = np.random.permutation(len(X_train))
        X_shuffled, Y_shuffled = X_train[indices], Y_train[indices]
        
        optimizer.lr = step_decay(epoch, initial_lr=0.001)
        
        train_loss, val_loss = 0, 0
        
        # Training Phase
        for i in range(0, len(X_train), batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            Y_batch = Y_shuffled[i:i+batch_size]
            
            # Forward Pass
            output = X_batch
            for layer in network:
                output = layer.forward(output)
            
            preds = softmax(output)
            train_loss += categorical_crossentropy(preds, Y_batch)
            
            # Backward Pass
            grad = crossentropy_backward(preds, Y_batch)
            layer_gradients = []
            
            for layer in reversed(network):
                if isinstance(layer, Dense):
                    grad, dW, db = layer.backward(grad)
                    layer_gradients.extend([db, dW]) # Append reversed to match param extraction
                else:
                    grad = layer.backward(grad)
                    
            # Optimize step
            layer_gradients.reverse() # Align with trainable_params order
            optimizer.step(trainable_params, layer_gradients)

        # Validation & Logging Logic goes here (predict on X_val, calc accuracy)
        # Checkpoint Saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save checkpoint (Section 7)
            checkpoint = {
                'epoch': epoch,
                'optimizer_t': optimizer.t,
                'weights': [p.tolist() for p in trainable_params]
            }
            with open('best_checkpoint.json', 'w') as f:
                json.dump(checkpoint, f)
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print("Early stopping triggered.")
                break