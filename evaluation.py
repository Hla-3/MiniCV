import numpy as np

class Evaluator:
    @staticmethod
    def get_confusion_matrix(y_true, y_pred, num_classes):
        """Builds a confusion matrix from scratch."""
        matrix = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            matrix[t, p] += 1
        return matrix

    @staticmethod
    def calculate_metrics(conf_matrix):
        """Calculates Precision, Recall, and F1-score per class."""
        num_classes = conf_matrix.shape[0]
        precision = np.zeros(num_classes)
        recall = np.zeros(num_classes)
        f1 = np.zeros(num_classes)
        
        for i in range(num_classes):
            tp = conf_matrix[i, i]
            fp = np.sum(conf_matrix[:, i]) - tp
            fn = np.sum(conf_matrix[i, :]) - tp
            
            # Add epsilon to avoid division by zero
            precision[i] = tp / (tp + fp + 1e-9)
            recall[i] = tp / (tp + fn + 1e-9)
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-9)
            
        return precision, recall, f1

    @staticmethod
    def aggregate_f1(f1_scores, y_true, num_classes):
        """Calculates Macro-F1 and Weighted-F1."""
        macro_f1 = np.mean(f1_scores)
        
        # Calculate class weights based on support (number of true instances)
        class_counts = np.bincount(y_true, minlength=num_classes)
        total_samples = len(y_true)
        weights = class_counts / total_samples
        
        weighted_f1 = np.sum(f1_scores * weights)
        
        return macro_f1, weighted_f1

# Example Usage:
# y_true = np.array([0, 1, 2, 2, 0, 1])  # Actual indices (0=Birds, 1=Cars, etc.)
# y_pred = np.array([0, 2, 2, 2, 0, 1])  # Model predictions
# conf_mat = Evaluator.get_confusion_matrix(y_true, y_pred, num_classes=6)
# prec, rec, f1 = Evaluator.calculate_metrics(conf_mat)
# m_f1, w_f1 = Evaluator.aggregate_f1(f1, y_true, num_classes=6)