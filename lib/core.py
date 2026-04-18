import numpy as np

class Core:
    """
    Core image processing utilities for the MiniCV library.

    Provides foundational operations used internally by higher-level
    modules (filtering, convolution) and available directly to users.

    All methods operate on NumPy arrays. Supported image layouts:
        - Grayscale : shape (H, W),    dtype uint8 or float32
        - RGB       : shape (H, W, 3), dtype uint8 or float32

    Section map
    -----------
    3.1  normalize  — intensity normalization (3 modes)
    3.2  clip       — pixel value clamping
    3.3  pad        — spatial border padding  (3 modes)
    """

    # Converting RGB image into HSI 
    @staticmethod
    def rgb_to_hsi(image_rgb):
        """
        Manually converts an RGB image to HSI color space.

        Parameters:
        -----------
        image_rgb : numpy.ndarray
            Input RGB image of shape (H, W, 3). 
            Expected dtype: uint8 [0, 255] or float [0.0, 255.0].

        Returns:
        --------
        numpy.ndarray
            Converted image in HSI format with shape (H, W, 3) and dtype float32.
            Values for H, S, and I are normalized to the range [0.0, 1.0].

        Raises:
        -------
        TypeError: If image_rgb is not a numpy.ndarray.
        ValueError: If image_rgb does not have 3 channels (shape (H, W, 3)).
        """
        if not isinstance(image_rgb, np.ndarray):
            raise TypeError(f"rgb_to_hsi: image must be numpy.ndarray, got {type(image_rgb).__name__}.")
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"rgb_to_hsi: Expected shape (H, W, 3), got {image_rgb.shape}.")

        # Normalize pixels to [0, 1]
        img = image_rgb.astype(np.float32) / 255.0
        R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
        
        # Calculate Intensity (I)
        I = (R + G + B) / 3.0
        
        # Calculate Saturation (S)
        min_rgb = np.minimum(np.minimum(R, G), B)
        # Avoid division by zero where I=0
        S = 1 - (min_rgb / (I + 1e-6))
        S[I == 0] = 0
        
        # Calculate Hue (H)
        num = 0.5 * ((R - G) + (R - B))
        den = np.sqrt((R - G)**2 + (R - B)*(G - B))
        theta = np.arccos(num / (den + 1e-6))
        
        H = theta.copy()
        H[B > G] = 2 * np.pi - H[B > G]
        H = H / (2 * np.pi) # Normalize H to [0, 1]
        
        return np.stack([H, S, I], axis=-1)
    
    @staticmethod
    def hsi_to_rgb(image_hsi):
        """
        Converts HSI image back to RGB color space.

        Parameters:
        -----------
        image_hsi : numpy.ndarray
            Input HSI image of shape (H, W, 3). 
            Expected range: [0, 255] for all channels (mapped internally to [0, 1]).

        Returns:
        --------
        numpy.ndarray
            Converted RGB image of shape (H, W, 3) and dtype uint8 [0, 255].

        Raises:
        -------
        TypeError: If image_hsi is not a numpy.ndarray.
        ValueError: If image_hsi does not have 3 channels.
        """
        if not isinstance(image_hsi, np.ndarray):
            raise TypeError(f"hsi_to_rgb: image must be numpy.ndarray, got {type(image_hsi).__name__}.")
        if image_hsi.ndim != 3 or image_hsi.shape[2] != 3:
            raise ValueError(f"hsi_to_rgb: Expected shape (H, W, 3), got {image_hsi.shape}.")

        # 1. Internal Normalization
        # Map H: 0-255 -> 0-2pi
        H = (image_hsi[:,:,0] / 255.0) * 2 * np.pi
        # Map S: 0-255 -> 0-1
        S = image_hsi[:,:,1] / 255.0
        # Map I: 0-255 -> 0-1 (for the ratio math)
        I = image_hsi[:,:,2] / 255.0
        
        R, G, B = np.zeros_like(H), np.zeros_like(H), np.zeros_like(H)
        
        # Sector 1: 0 <= H < 120 deg
        idx = (H >= 0) & (H < 2*np.pi/3)
        B[idx] = I[idx] * (1 - S[idx])
        R[idx] = I[idx] * (1 + (S[idx] * np.cos(H[idx])) / np.cos(np.pi/3 - H[idx]))
        G[idx] = 3*I[idx] - (R[idx] + B[idx])
        
        # Sector 2: 120 <= H < 240 deg
        idx = (H >= 2*np.pi/3) & (H < 4*np.pi/3)
        H_shift = H[idx] - 2*np.pi/3
        R[idx] = I[idx] * (1 - S[idx])
        G[idx] = I[idx] * (1 + (S[idx] * np.cos(H_shift)) / np.cos(np.pi/3 - H_shift))
        B[idx] = 3*I[idx] - (R[idx] + G[idx])
        
        # Sector 3: 240 <= H < 360 deg
        idx = (H >= 4*np.pi/3) & (H < 2*np.pi)
        H_shift = H[idx] - 4*np.pi/3
        G[idx] = I[idx] * (1 - S[idx])
        B[idx] = I[idx] * (1 + (S[idx] * np.cos(H_shift)) / np.cos(np.pi/3 - H_shift))
        R[idx] = 3*I[idx] - (G[idx] + B[idx])
        
        # 2. Rescale back to 0-255
        rgb = np.stack([R, G, B], axis=-1)
        return (np.clip(rgb * 255, 0, 255)).astype(np.uint8)

    @staticmethod
    def normalize(image: np.ndarray, mode: str = 'min_max') -> np.ndarray:
        """
        Normalize image pixel intensities using one of four strategies.

        Parameters
        ----------
        image : numpy.ndarray
            Input image of shape (H, W) or (H, W, 3).
            Accepted dtypes: uint8 or float32/float64.
        mode : str, optional
            Normalization strategy. One of:
            'min_max'  : Scales pixel values to [0.0, 1.0].
            'z_score'  : Standardises to zero mean and unit variance.
            'mean_norm': Centers around zero, bounded by image range.
            'uint8'    : Scales the full intensity range to [0, 255].
            Default is 'min_max'.

        Returns
        -------
        numpy.ndarray
            Normalised image, same shape as input.
            dtype is float32 for all modes except 'uint8'.

        Raises
        ------
        TypeError
            If image is not a numpy.ndarray, or mode is not a str.
        ValueError
            If image is not 2-D or 3-D, or mode is not recognised.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"normalize: image must be a numpy.ndarray, got {type(image).__name__}.")
        if image.ndim not in (2, 3):
            raise ValueError(f"normalize: image must be 2-D or 3-D, got shape {image.shape}.")
        if not isinstance(mode, str):
            raise TypeError(f"normalize: mode must be a str, got {type(mode).__name__}.")
        
        valid_modes = ('min_max', 'z_score', 'mean_norm', 'uint8')
        if mode not in valid_modes:
            raise ValueError(f"normalize: mode must be one of {valid_modes}, got '{mode}'.")

        img = image.astype(np.float32)

        if mode == 'min_max':
            lo, hi = img.min(), img.max()
            if hi == lo: return np.zeros_like(img, dtype=np.float32)
            return ((img - lo) / (hi - lo)).astype(np.float32)

        elif mode == 'z_score':
            mean, std = img.mean(), img.std()
            if std == 0.0: return np.zeros_like(img, dtype=np.float32)
            return ((img - mean) / std).astype(np.float32)

        elif mode == 'mean_norm':
            mean, lo, hi = img.mean(), img.min(), img.max()
            if hi == lo: return np.zeros_like(img, dtype=np.float32)
            return ((img - mean) / (hi - lo)).astype(np.float32)
        
        elif mode == 'uint8':
            lo, hi = img.min(), img.max()
            if hi == lo: return np.zeros_like(img, dtype=np.uint8)
            norm = (img - lo) / (hi - lo)
            return (norm * 255).astype(np.uint8)

    @staticmethod
    def clip_pixels(image: np.ndarray, low: float = 0.0, high: float = 1.0) -> np.ndarray:
        """
        Clamp pixel values to the closed interval [low, high].

        Parameters
        ----------
        image : numpy.ndarray
            Input image of shape (H, W) or (H, W, 3), any numeric dtype.
        low : float or int, optional
            Lower bound (inclusive). Default is 0.0.
        high : float or int, optional
            Upper bound (inclusive). Default is 1.0.

        Returns
        -------
        numpy.ndarray
            Clipped image, same shape and dtype as input.

        Raises
        ------
        TypeError
            If image is not a numpy.ndarray, or low/high are not numeric.
        ValueError
            If image is not 2-D or 3-D, or low >= high.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"clip: image must be a numpy.ndarray, got {type(image).__name__}.")
        if image.ndim not in (2, 3):
            raise ValueError(f"clip: image must be 2-D or 3-D, got shape {image.shape}.")
        if not isinstance(low, (int, float, np.number)):
            raise TypeError(f"clip: low must be numeric, got {type(low).__name__}.")
        if not isinstance(high, (int, float, np.number)):
            raise TypeError(f"clip: high must be numeric, got {type(high).__name__}.")
        if low >= high:
            raise ValueError(f"clip: low must be < high, got low={low}, high={high}.")

        return np.clip(image, low, high).astype(image.dtype)

    @staticmethod
    def pad_image(image, pad_width, mode='constant', constant_values=0):
        """
        Add a border of pixels around an image.

        Parameters
        ----------
        image : numpy.ndarray
            Input image of shape (H, W) or (H, W, 3).
        pad_width : int
            Number of pixels to add to all sides. (Non-negative).
        mode : str, optional
            Strategy: 'constant', 'reflect', or 'edge'. Default 'constant'.
        constant_values : float or int, optional
            Fill value for mode='constant'. Default is 0.

        Returns
        -------
        numpy.ndarray
            Padded image with increased dimensions. Same dtype as input.

        Raises
        ------
        TypeError
            If image is not a numpy.ndarray or constant_values is not numeric.
        ValueError
            If pad_width is negative or mode is not recognized.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"pad: image must be numpy.ndarray, got {type(image).__name__}.")
        if not isinstance(pad_width, int) or pad_width < 0:
            raise ValueError(f"pad: pad_width must be non-negative int, got {pad_width}.")
        if not isinstance(mode, str):
            raise TypeError("pad: mode must be a string.")
        
        valid_modes = ['constant', 'edge', 'reflect']
        if mode not in valid_modes:
            raise ValueError(f"pad: mode must be one of {valid_modes}, got '{mode}'.")
        
        # Set spatial padding
        if image.ndim == 3:
            pad_param = ((pad_width, pad_width), (pad_width, pad_width), (0, 0))
        elif image.ndim == 2:
            pad_param = ((pad_width, pad_width), (pad_width, pad_width))
        else:
            raise ValueError(f"pad: Expected 2D or 3D image, got {image.ndim}D.")

        kwargs = {}
        if mode == 'constant':
            kwargs['constant_values'] = constant_values
            
        return np.pad(image, pad_param, mode=mode, **kwargs)

    @staticmethod
    def convolve2d(image, kernel, padding_mode='constant'):
        """
        Performs true 2D convolution on a grayscale image.

        Parameters:
        -----------
        image : numpy.ndarray
            Input grayscale image (H, W).
        kernel : numpy.ndarray
            Filter kernel. Must be numeric, non-empty, and have odd dimensions.
        padding_mode : str, optional
            Border handling ('constant', 'reflect', 'edge'). Default 'constant'.

        Returns:
        --------
        numpy.ndarray
            Convolved image with dtype float32.

        Raises:
        -------
        TypeError: If image/kernel are not numpy arrays.
        ValueError: If kernel dimensions are even or image is not 2D.
        """
        if not isinstance(image, np.ndarray) or image.ndim != 2:
            raise ValueError(f"convolve2d: image must be a 2D array, got {getattr(image, 'shape', type(image))}.")
        if not isinstance(kernel, np.ndarray) or not np.issubdtype(kernel.dtype, np.number):
            raise TypeError("convolve2d: Kernel must be a numeric NumPy array.")
        if kernel.size == 0:
            raise ValueError("convolve2d: Kernel cannot be empty.")
        if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
            raise ValueError(f"convolve2d: Kernel dims must be odd, got {kernel.shape}.")

        # True convolution flips the kernel
        kernel = np.flipud(np.fliplr(kernel))
        
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2 
        
        padded_img = Core.pad_image(image, pad_h, mode=padding_mode)
        output = np.zeros_like(image, dtype=np.float32)
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                window = padded_img[i:i+k_h, j:j+k_w]
                output[i, j] = np.sum(window * kernel)
                
        return output

    @staticmethod
    def spatial_filter(image, kernel, padding_mode='constant', mode='per_channel'):
        """
        Higher-level interface for 2D spatial filtering on Grayscale and RGB images.

        Parameters:
        -----------
        image : numpy.ndarray
            Input image (H, W) or (H, W, 3).
        kernel : numpy.ndarray
            Filter kernel (must have odd dimensions).
        padding_mode : str, optional
            Padding strategy. Default is 'constant'.
        mode : str, optional
            For RGB: 'per_channel' (RGB) or 'hsi' (filter Intensity only).

        Returns:
        --------
        numpy.ndarray
            Filtered image (float32 if per-channel, uint8 if HSI mode).

        Raises:
        -------
        ValueError: If image shape is unsupported or mode is invalid.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("spatial_filter: Input must be a numpy array.")

        if image.ndim == 2:
            return Core.convolve2d(image, kernel, padding_mode)
        
        elif image.ndim == 3:
            if mode == 'per_channel':
                channels = [Core.convolve2d(image[:, :, c], kernel, padding_mode) 
                            for c in range(image.shape[2])]
                return np.stack(channels, axis=-1)
             
            elif mode == 'hsi':
                hsi = Core.rgb_to_hsi(image) # Normalized 0-1
                intensity = hsi[:, :, 2]
                filtered_i = Core.convolve2d(intensity, kernel, padding_mode)
                hsi[:, :, 2] = np.clip(filtered_i, 0, 1)
                
                # Convert normalized HSI [0,1] to [0,255] for the hsi_to_rgb function
                return Core.hsi_to_rgb(hsi * 255.0)
            
            else:
                raise ValueError(f"spatial_filter: Invalid mode '{mode}'. Use 'per_channel' or 'hsi'.")
        else:
            raise ValueError(f"spatial_filter: Unsupported image shape {image.shape}.")