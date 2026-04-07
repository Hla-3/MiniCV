import numpy as np
from core import Core  # Assuming Core class is in core.py

class ImageProcessing:
    """
    Advanced Image Processing techniques including spatial filters,
    thresholding, and histogram operations. [cite: 28]
    """

    @staticmethod
    def mean_filter(image, size=3):
        """
        4.1 Mean/Box filter:
        Smooths image using a normalized box kernel. [cite: 29]
        """
        kernel = np.ones((size, size), dtype=np.float32) / (size * size)
        return Core.spatial_filter(image, kernel)

    @staticmethod
    def gaussian_kernel(size, sigma):
        """
        4.2 Gaussian kernel generation:
        Creates a 2D Gaussian distribution kernel. [cite: 31]
        """
        ax = np.linspace(-(size // 2), size // 2, size)
        gauss = np.exp(-0.5 * (ax / sigma)**2)
        kernel = np.outer(gauss, gauss)
        return kernel / kernel.sum()

    @staticmethod
    def gaussian_filter(image, size=3, sigma=1.0):
        """
        4.2 Filtering using your convolution pipeline. 
        """
        kernel = ImageProcessing.gaussian_kernel(size, sigma)
        return Core.spatial_filter(image, kernel)

    @staticmethod
    def median_filter(image, size=3):
        """
        4.3 Median filter:
        Non-linear filter used to remove salt-and-pepper noise. 
        
        Justification for Loops:
        Loops are used here because the median operation is a rank-order statistic
        that cannot be expressed as a linear matrix convolution (dot product). 
        Each neighborhood must be sorted independently to find the middle value. 
        """
        pad_v = size // 2
        padded = Core.pad_image(image, pad_v, mode='edge')
        output = np.zeros_like(image)
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                window = padded[i:i+size, j:j+size]
                output[i, j] = np.median(window)
        return output

    @staticmethod
    def threshold_global(image, thresh=127):
        """4.4 Global thresholding: Simple binary segmentation. [cite: 34]"""
        return (image > thresh).astype(np.uint8) * 255

    @staticmethod
    def threshold_otsu(image):
        """4.4 Otsu thresholding: Automatically calculates optimal threshold. [cite: 34]"""
        pixel_counts, bin_edges = np.histogram(image, bins=256, range=(0, 256))
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
        
        weight1 = np.cumsum(pixel_counts)
        weight2 = np.cumsum(pixel_counts[::-1])[::-1]
        
        mean1 = np.cumsum(pixel_counts * bin_mids) / weight1
        mean2 = (np.cumsum((pixel_counts * bin_mids)[::-1]) / weight2[::-1])[::-1]
        
        inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:])**2
        index_of_max_var = np.argmax(inter_class_variance)
        return ImageProcessing.threshold_global(image, bin_mids[index_of_max_var])

    @staticmethod
    def sobel_gradients(image):
        """4.5 Sobel gradients: Extracts horizontal and vertical edges. [cite: 36]"""
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        
        Gx = Core.spatial_filter(image, Kx)
        Gy = Core.spatial_filter(image, Ky)
        
        magnitude = np.sqrt(Gx**2 + Gy**2)
        return Core.clip_pixels(magnitude, 0, 255).astype(np.uint8)

    @staticmethod
    def bit_plane_slicing(image, bit=7):
        """4.6 Bit-plane slicing: Extracts the contribution of a specific bit. [cite: 37]"""
        return ((image.astype(np.uint8) >> bit) & 1) * 255

    @staticmethod
    def histogram_equalization(image):
        """4.7 Histogram equalization: Enhances image contrast. [cite: 38]"""
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        return cdf_normalized[image.astype(np.uint8)]

    @staticmethod
    def laplacian_filter(image):
        """4.8 Additional Technique 1: Laplacian (Edge enhancement). [cite: 39]"""
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        return Core.spatial_filter(image, kernel)

    @staticmethod
    def gamma_correction(image, gamma=1.0):
        """4.8 Additional Technique 2: Power-Law Transformation. [cite: 39]"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)])
        return table[image.astype(np.uint8)].astype(np.uint8)