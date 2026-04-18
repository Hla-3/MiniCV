import numpy as np
from .core import Core  # Assuming Core class is in core.py
from matplotlib import pyplot as plt

class ImageProcessing:
    """
    Advanced Image Processing techniques including spatial filters,
    thresholding, and histogram operations.
    """

    @staticmethod
    def mean_filter(image, size=3):
        """
        4.1 Mean/Box filter:
        Smooths image using a normalized box kernel. 

        Parameters:
        -----------
        image : numpy.ndarray
            The input grayscale image (2D array).
        size : int, optional
            The dimensions of the square kernel (size x size). Default is 3.

        Returns:
        --------
        numpy.ndarray
            The smoothed image.

        Raises:
        -------
        TypeError: If 'image' is not a numpy array or 'size' is not an integer.
        ValueError: If 'image' is not a 2D array or 'size' is less than 1.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected image as numpy.ndarray, got {type(image)}.")
        if image.ndim != 2:
            raise ValueError(f"Expected 2D grayscale image, got shape {image.shape}.")
        if not isinstance(size, int):
            raise TypeError(f"Kernel size must be an integer, got {type(size)}.")
        if size < 1:
            raise ValueError(f"Kernel size must be at least 1, got {size}.")

        kernel = np.ones((size, size), dtype=np.float32) / (size * size)
        return Core.spatial_filter(image, kernel)

    @staticmethod
    def gaussian_kernel(size, sigma):
        """
        4.2 Gaussian kernel generation:
        Creates a 2D Gaussian distribution kernel. 

        Parameters:
        -----------
        size : int
            The dimensions of the square kernel (size x size).
        sigma : float
            Standard deviation of the Gaussian distribution.

        Returns:
        --------
        numpy.ndarray
            A normalized 2D Gaussian kernel.

        Raises:
        -------
        TypeError: If 'size' is not an integer or 'sigma' is not numeric.
        ValueError: If 'size' is not odd/positive or 'sigma' is non-positive.
        """
        if not isinstance(size, int):
            raise TypeError(f"Size must be an integer, got {type(size)}.")
        if size % 2 == 0 or size < 1:
            raise ValueError(f"Size must be a positive odd integer, got {size}.")
        if not isinstance(sigma, (int, float, np.float32)):
            raise TypeError(f"Sigma must be numeric, got {type(sigma)}.")
        if sigma <= 0:
            raise ValueError(f"Sigma must be greater than 0, got {sigma}.")

        ax = np.linspace(-(size // 2), size // 2, size)
        gauss = np.exp(-0.5 * (ax / sigma)**2) 
        kernel = np.outer(gauss, gauss)
        return kernel / kernel.sum()

    @staticmethod
    def gaussian_filter(image, size=3, sigma=1.0):
        """
        4.2 Filtering using a Gaussian kernel.

        Parameters:
        -----------
        image : numpy.ndarray
            Input grayscale image (2D array).
        size : int, optional
            Square kernel dimension. Must be odd. Default is 3.
        sigma : float, optional
            Spread of the Gaussian. Default is 1.0.

        Returns:
        --------
        numpy.ndarray
            The Gaussian-blurred image.

        Raises:
        -------
        TypeError: If 'image' is not a numpy array.
        ValueError: If 'image' is not 2D.
        """
        if not isinstance(image, np.ndarray) or image.ndim != 2:
            raise ValueError("Input image must be a 2D numpy array.")
            
        kernel = ImageProcessing.gaussian_kernel(size, sigma)
        return Core.spatial_filter(image, kernel)

    @staticmethod
    def median_filter(image, size=3):
        """
        4.3 Median filter:
        Non-linear filter used to remove salt-and-pepper noise. 

        Parameters:
        -----------
        image : numpy.ndarray
            Input 2D grayscale image (dtype uint8 or float).
        size : int, optional
            Dimensions of the sliding window. Default is 3.

        Returns:
        --------
        numpy.ndarray
            The filtered image of the same shape as input.

        Raises:
        -------
        TypeError: If types are incorrect.
        ValueError: If window size is invalid for the image dimensions.
        """
        if not isinstance(image, np.ndarray) or image.ndim != 2:
            raise ValueError("Image must be a 2D numpy array.")
        if not isinstance(size, int) or size < 1 or size % 2 == 0:
            raise ValueError(f"Size must be a positive odd integer, got {size}.")

        pad_v = size // 2
        padded = Core.pad_image(image, pad_v)
        output = np.zeros_like(image)
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                window = padded[i:i+size, j:j+size]
                output[i, j] = np.median(window)
        return output

    @staticmethod
    def threshold_global(image, thresh=127):
        """
        4.4 Global thresholding: Keeps values above threshold, sets others to 0.

        Parameters:
        -----------
        image : numpy.ndarray
            Input 2D array.
        thresh : int, optional
            The intensity value used as a cutoff (0-255). Default is 127.

        Returns:
        --------
        numpy.ndarray
            Binary image (uint8).
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a numpy array.")
        if not isinstance(thresh, (int, float)):
            raise TypeError("Threshold must be numeric.")

        return np.where(image > thresh, 255, 0).astype(np.uint8)

    @staticmethod
    def threshold_otsu(image):
        """
        Automatically calculates threshold using Otsu's Between-Class Variance.

        Parameters:
        -----------
        image : numpy.ndarray
            Input 2D grayscale image. Expected dtype uint8.

        Returns:
        --------
        numpy.ndarray
            Binary image (uint8).
        """
        if not isinstance(image, np.ndarray) or image.ndim != 2:
            raise ValueError("Image must be a 2D numpy array.")

        hist, bins = np.histogram(image.ravel(), 256, [0, 256])
        total_pixels = image.size
        current_max = -1
        threshold = 0
        
        for t in range(256):
            w0 = np.sum(hist[:t]) / total_pixels
            w1 = np.sum(hist[t:]) / total_pixels
            
            if w0 == 0 or w1 == 0: continue
            
            mu0 = np.sum(np.arange(t) * hist[:t]) / (np.sum(hist[:t]) + 1e-6)
            mu1 = np.sum(np.arange(t, 256) * hist[t:]) / (np.sum(hist[t:]) + 1e-6)
            
            variance = w0 * w1 * ((mu0 - mu1) ** 2)
            
            if variance > current_max:
                current_max = variance
                threshold = t
        return np.where(image > threshold, 255 , 0).astype(np.uint8)      

    @staticmethod
    def threshold_adaptive(image, block_size=11, C=2):
        """
        4.6 Adaptive Thresholding: Calculates thresholds for local neighborhoods.

        Parameters:
        -----------
        image : numpy.ndarray
            Input 2D grayscale image.
        block_size : int, optional
            The size of the local neighborhood. Default is 11.
        C : float, optional
            Constant subtracted from the mean. Default is 2.

        Returns:
        --------
        numpy.ndarray
            Binary image (uint8).
        """
        if not isinstance(image, np.ndarray) or image.ndim != 2:
            raise ValueError("Image must be a 2D numpy array.")
        if not isinstance(block_size, int):
            raise TypeError("block_size must be an integer.")

        if block_size % 2 == 0:
            block_size += 1
                
        kernel = np.ones((block_size, block_size), dtype=np.float32) / (block_size**2)
        local_mean = Core.convolve2d(image.astype(np.float32), kernel, padding_mode='constant')
        
        return np.where(image > (local_mean - C), 255, 0).astype(np.uint8)
    
    @staticmethod
    def sobel(image):
        """
        4.5 Sobel Visualization:
        Returns magnitude for edge detection and a color-coded direction map.

        Parameters:
        -----------
        image : numpy.ndarray
            Input 2D grayscale image.

        Returns:
        --------
        tuple (mag_viz, dir_viz)
            mag_viz: uint8 magnitude map.
            dir_viz: RGB direction map based on HSI conversion.
        """
        if not isinstance(image, np.ndarray) or image.ndim != 2:
            raise ValueError("Image must be a 2D numpy array.")

        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        
        Gx = Core.spatial_filter(image, Kx)
        Gy = Core.spatial_filter(image, Ky)
        
        mag = np.sqrt(Gx**2 + Gy**2)
        mag_viz = Core.clip_pixels(mag, 0, 255).astype(np.uint8)
        
        angle = (np.arctan2(Gy, Gx) * 180 / np.pi) % 180
        hsi = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        hsi[..., 0] = (angle * (255.0 / 180.0)).astype(np.uint8)
        hsi[..., 1] = 255
        hsi[..., 2] = mag_viz
        
        dir_viz = Core.hsi_to_rgb(hsi) 
        return mag_viz, dir_viz

    @staticmethod
    def bit_plane_slicing(image, bit):
        """
        4.6 Bit-plane slicing: Extracts the contribution of a specific bit.

        Parameters:
        -----------
        image : numpy.ndarray
            Input 2D grayscale image.
        bit : int
            The bit plane to extract (0-7).

        Returns:
        --------
        numpy.ndarray
            Image representing the specified bit plane (values 0 or 255).
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a numpy array.")
        if not isinstance(bit, int) or not (0 <= bit <= 7):
            raise ValueError(f"Bit must be an integer between 0 and 7, got {bit}.")

        return ((image.astype(np.uint8) >> bit) & 1) * 255

    @staticmethod
    def histogram(image):
        """
        Displays a histogram of intensity values.

        Parameters:
        -----------
        image : numpy.ndarray
            Input 2D grayscale image.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a numpy array.")

        pixels = image.ravel().astype(np.uint8) 
        hist_data = np.bincount(pixels, minlength=256)
        plt.bar(range(256), hist_data, color='blue', width=1.0)
        plt.title("Intensity Histogram")
        plt.show()

    @staticmethod
    def histogram_equalization(image):
        """
        4.7 Histogram equalization: Enhances image contrast. 

        Parameters:
        -----------
        image : numpy.ndarray
            Input 2D grayscale image (uint8).

        Returns:
        --------
        numpy.ndarray
            Equalized image (uint8).
        """
        if not isinstance(image, np.ndarray) or image.ndim != 2:
            raise ValueError("Input must be a 2D numpy array.")

        L = 256
        total_pixels = image.size
        hist, bins = np.histogram(image.flatten(), L, [0, L])
        cdf = hist.cumsum()
        s = cdf / total_pixels
        z = np.round(L * s - 1)
        z = np.clip(z, 0, L - 1).astype(np.uint8)
        
        return z[image.astype(np.uint8)]

    @staticmethod
    def laplacian_filter(image):
        """
        4.8 Additional Technique 1: Laplacian (Edge enhancement).

        Parameters:
        -----------
        image : numpy.ndarray
            Input 2D grayscale image.

        Returns:
        --------
        numpy.ndarray
            Edge-detected image result.
        """
        if not isinstance(image, np.ndarray) or image.ndim != 2:
            raise ValueError("Input must be a 2D numpy array.")

        kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
        return Core.spatial_filter(image, kernel)

    @staticmethod
    def gamma_correction(image, gamma=1.0, s_min=0, s_max=255):
        """
        4.8 Gamma Correction using normalized power-law.

        Parameters:
        -----------
        image : numpy.ndarray
            Input grayscale image.
        gamma : float, optional
            Exponent value. γ < 1 brightens, γ > 1 darkens. Default is 1.0.
        s_min : int, optional
            Target range minimum. Default is 0.
        s_max : int, optional
            Target range maximum. Default is 255.

        Returns:
        --------
        numpy.ndarray
            Gamma-corrected image (uint8).
        """
        if not isinstance(image, np.ndarray) or image.ndim != 2:
            raise ValueError("Input must be a 2D numpy array.")
        if not isinstance(gamma, (int, float)):
            raise TypeError("Gamma must be a numeric value.")

        img_float = image.astype(np.float32)
        r_min = np.min(img_float)
        r_max = np.max(img_float)
        
        if r_max == r_min:
            return image
        
        r_gamma = np.power(img_float, gamma)
        r_min_gamma = np.power(r_min, gamma)
        r_max_gamma = np.power(r_max, gamma)
        
        numerator = r_gamma - r_min_gamma
        denominator = r_max_gamma - r_min_gamma
        
        S = s_min + (numerator / denominator) * (s_max - s_min)
        return np.clip(S, s_min, s_max).astype(np.uint8)