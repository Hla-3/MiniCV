import numpy as np
from core import Core
from processing import ImageProcessing

class FeatureExtractor:
    """
    Extracts high-level features from images for use in machine learning models.
    Includes Global and Gradient-based descriptors.
    """

    # --- 6.1 Global Descriptors ---

    @staticmethod
    def color_histogram(image, bins=32):
        """
        Global Descriptor 1: Color/Intensity Histogram.
        Captures the global distribution of intensities.
        """
        if len(image.shape) == 3:
            # For RGB, concatenate histograms of all channels
            hist_features = []
            for i in range(3):
                hist, _ = np.histogram(image[:, :, i], bins=bins, range=(0, 256))
                hist_features.append(hist)
            descriptor = np.concatenate(hist_features)
        else:
            # For Grayscale
            descriptor, _ = np.histogram(image, bins=bins, range=(0, 256))
        
        # Normalize to ensure the descriptor is scale-invariant
        return descriptor.astype(np.float32) / (image.size + 1e-8)

    @staticmethod
    def basic_statistics(image):
        """
        Global Descriptor 2: Statistical Moments.
        Captures mean, variance, skewness, and kurtosis of the pixel intensities.
        """
        pixels = image.flatten().astype(np.float32)
        mean = np.mean(pixels)
        std = np.std(pixels)
        
        # Adding a small epsilon to avoid division by zero
        skewness = np.mean((pixels - mean)**3) / (std**3 + 1e-8)
        kurtosis = np.mean((pixels - mean)**4) / (std**4 + 1e-8)
        
        return np.array([mean, std, skewness, kurtosis], dtype=np.float32)

    # --- 6.2 Gradient Descriptors ---

    @staticmethod
    def hog_lite(image, cell_size=8, n_bins=9):
        """
        Gradient Descriptor 1: Histogram of Oriented Gradients (Lite version).
        Captures local shape information by distributing gradient orientations into bins.
        """
        # 1. Compute Gradients using Sobel
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        
        gx = Core.spatial_filter(image, Kx)
        gy = Core.spatial_filter(image, Ky)
        
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180
        
        # 2. Binning (Simplified global orientation histogram)
        hist, _ = np.histogram(orientation, bins=n_bins, range=(0, 180), weights=magnitude)
        
        # Normalize
        return hist.astype(np.float32) / (np.sum(hist) + 1e-8)

    @staticmethod
    def local_binary_pattern(image):
        """
        Gradient Descriptor 2: Local Binary Pattern (LBP).
        Captures local texture by comparing a pixel with its 8 neighbors.
        """
        if len(image.shape) == 3:
            # Convert to grayscale first for texture analysis
            from io_utils import IO
            image = IO.rgb_to_grayscale(image)
            
        h, w = image.shape
        lbp_image = np.zeros((h-2, w-2), dtype=np.uint8)
        
        # Directions for 8-neighbors
        offsets = [( -1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        
        # Nested loops are justified here for the bitwise neighborhood comparison
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                center = image[i, j]
                bit_string = 0
                for idx, (dr, dc) in enumerate(offsets):
                    if image[i + dr, j + dc] >= center:
                        bit_string |= (1 << idx)
                lbp_image[i-1, j-1] = bit_string
        
        # Return the histogram of LBP codes as the descriptor
        hist, _ = np.histogram(lbp_image, bins=256, range=(0, 256))
        return hist.astype(np.float32) / (lbp_image.size + 1e-8)