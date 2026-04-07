import numpy as np

class Core:
    """
    Foundation functions for image processing including normalization, 
    padding, and convolution. Part of Milestone 1[cite: 6, 17].
    """

    @staticmethod
    def clip_pixels(image, min_val=0, max_val=255):
        """
        3.2 Pixel clipping[cite: 19]:
        Ensures all pixel values stay within a defined range.
        """
        return np.clip(image, min_val, max_val)

    @staticmethod
    def normalize_image(image, mode='minmax'):
        """
        3.1 Image normalization (3 Modes)[cite: 18]:
        - 'minmax': Scales to [0, 1].
        - 'standard': Z-score normalization (mean 0, std 1).
        - 'uint8': Scales/Clips back to [0, 255] range.
        """
        img = image.astype(np.float32)
        if mode == 'minmax':
            return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        elif mode == 'standard':
            return (img - np.mean(img)) / (np.std(img) + 1e-8)
        elif mode == 'uint8':
            return Core.clip_pixels(img, 0, 255).astype(np.uint8)
        else:
            raise ValueError("Mode must be 'minmax', 'standard', or 'uint8'[cite: 70].")

    @staticmethod
    def pad_image(image, pad_width, mode='constant', constant_values=0):
        """
        3.3 Padding (3 Modes)[cite: 20]:
        - 'constant': Pads with a fixed value (e.g., zero padding).
        - 'edge': Pads with the last known edge pixel values.
        - 'reflect': Pads by reflecting the image pixels.
        """
        if mode not in ['constant', 'edge', 'reflect']:
            raise ValueError("Padding mode must be 'constant', 'edge', or 'reflect'[cite: 70].")
        
        # Check if RGB or Grayscale to apply padding correctly to spatial dims only
        if len(image.shape) == 3:
            pad_param = ((pad_width, pad_width), (pad_width, pad_width), (0, 0))
        else:
            pad_param = (pad_width, pad_width)
            
        return np.pad(image, pad_param, mode=mode, constant_values=constant_values)

    @staticmethod
    def convolve2d(image, kernel, padding_mode='constant'):
        """
        3.4 2D convolution[cite: 21]:
        True 2D convolution for grayscale images.
        
        Validation:
        - Kernel must be non-empty.
        - Kernel dimensions must be odd.
        - Numeric type check.
        """
        if not isinstance(kernel, np.ndarray) or not np.issubdtype(kernel.dtype, np.number):
            raise TypeError("Kernel must be a numeric NumPy array[cite: 69].")
        if kernel.size == 0:
            raise ValueError("Kernel cannot be empty[cite: 25, 70].")
        if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
            raise ValueError("Kernel dimensions must be odd[cite: 25, 70].")

        # True convolution involves flipping the kernel 
        kernel = np.flipud(np.fliplr(kernel))
        
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        
        # Boundary handling using internal padding function 
        padded_img = Core.pad_image(image, pad_h, mode=padding_mode)
        
        output = np.zeros_like(image, dtype=np.float32)
        
        # Sliding window (Looping justified for window operations) [cite: 75]
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                window = padded_img[i:i+k_h, j:j+k_w]
                output[i, j] = np.sum(window * kernel)
                
        return output

    @staticmethod
    def spatial_filter(image, kernel, padding_mode='constant'):
        """
        3.5 2D spatial filtering (grayscale + RGB)[cite: 26]:
        Applies convolution per-channel for RGB images[cite: 27].
        """
        if len(image.shape) == 2:
            return Core.convolve2d(image, kernel, padding_mode)
        
        elif len(image.shape) == 3:
            # Documented Strategy: Per-channel convolution [cite: 27]
            channels = [Core.convolve2d(image[:, :, c], kernel, padding_mode) 
                        for c in range(image.shape[2])]
            return np.stack(channels, axis=-1)
        
        else:
            raise ValueError("Unsupported image shape[cite: 70].")