import numpy as np


class FeatureExtractor:
    """
    Image feature extraction for the MiniCV library.

    Provides global descriptors (capture whole-image statistics) and
    gradient descriptors (capture local shape and texture).

    All descriptors return a 1-D float32 NumPy array ready to be
    concatenated into a feature vector for machine-learning models.

    Expected input
    --------------
    - Grayscale : (H, W)    uint8 [0-255] or float32 [0-1]
    - RGB       : (H, W, 3) uint8 [0-255] or float32 [0-1]

    Feature index summary (default parameters)
    -------------------------------------------
    color_histogram  → 32  dims  (grayscale) / 96 dims (RGB, 3×32)
    basic_statistics →  4  dims  [mean, std, skewness, kurtosis]
    hog_lite         →  9  dims  (n_bins=9)
    Edge Histogram Descriptor (EHD)     →  144  dims  (n_bins=9)
    """

    # ------------------------------------------------------------------ #
    #  Shared helper                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_grayscale_float(image: np.ndarray) -> np.ndarray:
        """
        Internal helper: ensure image is a 2-D float32 array in [0, 1].

        Converts RGB → grayscale if needed, and normalises uint8 → float32.

        Parameters
        ----------
        image : numpy.ndarray
            Shape (H, W) or (H, W, 3), dtype uint8 or float32.

        Returns
        -------
        numpy.ndarray
            Shape (H, W), dtype float32, values in [0, 1].
        """
        img = image.astype(np.float32)
        if img.ndim == 3:
            img = (0.299 * img[:, :, 0] +
                   0.587 * img[:, :, 1] +
                   0.114 * img[:, :, 2])
        if img.max() > 1.0:
            img = img / 255.0
        return img

    @staticmethod
    def _validate_image(image: np.ndarray, func_name: str) -> None:
        """
        Internal helper: shared input validation for all descriptors.

        Parameters
        ----------
        image : numpy.ndarray
            Image to validate.
        func_name : str
            Caller name for error messages.

        Raises
        ------
        TypeError
            If ``image`` is not a numpy.ndarray.
        ValueError
            If ``image`` is not 2-D or 3-D, or has wrong channel count.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"{func_name}: image must be a numpy.ndarray, "
                f"got {type(image).__name__}."
            )
        if image.ndim == 3 and image.shape[2] not in (3, 4):
            raise ValueError(
                f"{func_name}: 3-D image must have 3 or 4 channels, "
                f"got shape {image.shape}."
            )
        if image.ndim not in (2, 3):
            raise ValueError(
                f"{func_name}: image must be 2-D (H,W) or 3-D (H,W,3), "
                f"got shape {image.shape}."
            )

    # ================================================================== #
    #  6.1  Global Descriptors                                            #
    # ================================================================== #

    @staticmethod
    def color_histogram(image: np.ndarray, bins: int = 32) -> np.ndarray:
        """
        Global Descriptor 1: Color / Intensity Histogram.

        Captures the global distribution of pixel intensities.
        For RGB images, histograms of all three channels are computed
        independently and concatenated → richer colour information.

        Parameters
        ----------
        image : numpy.ndarray
            Grayscale (H, W) or RGB (H, W, 3).
            dtype: uint8 [0-255] or float32 [0-1].
        bins : int, optional
            Number of histogram bins per channel. Default is 32.
            Must be a positive integer.

        Returns
        -------
        numpy.ndarray
            1-D float32 descriptor, L2-normalised:
            - Grayscale → length ``bins``
            - RGB       → length ``3 × bins``

        Raises
        ------
        TypeError
            If ``image`` is not a numpy.ndarray, or ``bins`` is not int.
        ValueError
            If ``image`` has an unsupported shape, or ``bins`` ≤ 0.

        Notes
        -----
        Pixel values are mapped to [0, 256) before binning so uint8
        and float32 images are handled consistently.
        Normalisation: descriptor / (sum + ε) → scale-invariant.
        """
        FeatureExtractor._validate_image(image, "color_histogram")
        if not isinstance(bins, (int, np.integer)):
            raise TypeError(
                f"color_histogram: bins must be an int, "
                f"got {type(bins).__name__}."
            )
        if bins <= 0:
            raise ValueError(
                f"color_histogram: bins must be positive, got {bins}."
            )

        # normalise to [0, 256) for consistent binning
        img = image.astype(np.float32)
        if img.max() <= 1.0:
            img = img * 255.0

        if img.ndim == 2:
            hist, _ = np.histogram(img, bins=bins, range=(0.0, 256.0))
            descriptor = hist.astype(np.float32)
        else:
            # per-channel histograms concatenated
            hists = []
            for c in range(3):
                h, _ = np.histogram(img[:, :, c], bins=bins, range=(0.0, 256.0))
                hists.append(h)
            descriptor = np.concatenate(hists).astype(np.float32)

        # L1 normalise → sum = 1
        return descriptor / (descriptor.sum() + 1e-8)

    @staticmethod
    def basic_statistics(image: np.ndarray) -> np.ndarray:
        """
        Global Descriptor 2: Statistical Moments.

        Captures four statistical moments of the pixel intensity
        distribution: mean, standard deviation, skewness, kurtosis.

        Parameters
        ----------
        image : numpy.ndarray
            Grayscale (H, W) or RGB (H, W, 3).
            dtype: uint8 [0-255] or float32 [0-1].

        Returns
        -------
        numpy.ndarray
            1-D float32 array of length 4:
            [mean, std, skewness, kurtosis].
            Values computed on the flattened grayscale version of the image.

        Raises
        ------
        TypeError
            If ``image`` is not a numpy.ndarray.
        ValueError
            If ``image`` has an unsupported shape.

        Notes
        -----
        For RGB input, grayscale conversion (luminosity weights) is applied
        before computing moments so the descriptor is always 4-D.

        Formulas
        --------
        μ  = mean(x)
        σ  = std(x)
        skewness = E[(x - μ)³] / (σ³ + ε)
        kurtosis = E[(x - μ)⁴] / (σ⁴ + ε)

        ε = 1e-8 prevents division by zero for constant images.
        """
        FeatureExtractor._validate_image(image, "basic_statistics")

        gray = FeatureExtractor._to_grayscale_float(image)
        pixels = gray.flatten()

        mean = np.mean(pixels)
        std  = np.std(pixels)

        skewness = np.mean((pixels - mean) ** 3) / (std ** 3 + 1e-8)
        kurtosis = np.mean((pixels - mean) ** 4) / (std ** 4 + 1e-8)

        return np.array([mean, std, skewness, kurtosis], dtype=np.float32)

    # ================================================================== #
    #  6.2  Gradient Descriptors                                          #
    # ================================================================== #

    @staticmethod
    def hog_lite(image: np.ndarray,
                 n_bins: int = 9) -> np.ndarray:
        """
        Gradient Descriptor 1: Histogram of Oriented Gradients (lite).

        Captures the global distribution of gradient orientations,
        weighted by gradient magnitude — encodes dominant edge directions.

        Parameters
        ----------
        image : numpy.ndarray
            Grayscale (H, W) or RGB (H, W, 3).
        n_bins : int, optional
            Number of orientation bins in [0°, 180°). Default is 9.
            Must be a positive integer.

        Returns
        -------
        numpy.ndarray
            1-D float32 descriptor of length ``n_bins``, L1-normalised.

        Raises
        ------
        TypeError
            If ``image`` is not a numpy.ndarray, or ``n_bins`` is not int.
        ValueError
            If ``image`` has an unsupported shape, or ``n_bins`` ≤ 0.

        Notes
        -----
        Gradient computation (no external deps — pure NumPy):
            Gx = image[:, 2:] - image[:, :-2]   (central difference, x)
            Gy = image[2:, :] - image[:-2, :]   (central difference, y)
        Magnitude : M = √(Gx² + Gy²)
        Orientation: θ = arctan2(Gy, Gx) mapped to [0°, 180°)
        Unsigned orientations (0-180°) are used — standard for HOG.
        No spatial cells/blocks (lite version): single global histogram.
        """
        FeatureExtractor._validate_image(image, "hog_lite")
        if not isinstance(n_bins, (int, np.integer)):
            raise TypeError(
                f"hog_lite: n_bins must be an int, got {type(n_bins).__name__}."
            )
        if n_bins <= 0:
            raise ValueError(
                f"hog_lite: n_bins must be positive, got {n_bins}."
            )

        gray = FeatureExtractor._to_grayscale_float(image)

        # central-difference gradients — fully vectorised, no convolution dep
        # Gx: horizontal gradient (trim 1 row top/bottom to match shape)
        gx = gray[:, 2:] - gray[:, :-2]        # shape (H, W-2)
        gx = gx[1:-1, :]                        # shape (H-2, W-2)

        # Gy: vertical gradient
        gy = gray[2:, :] - gray[:-2, :]         # shape (H-2, W)
        gy = gy[:, 1:-1]                        # shape (H-2, W-2)

        magnitude   = np.sqrt(gx ** 2 + gy ** 2)
        orientation = np.arctan2(gy, gx) * (180.0 / np.pi) % 180.0

        hist, _ = np.histogram(
            orientation, bins=n_bins, range=(0.0, 180.0), weights=magnitude
        )

        descriptor = hist.astype(np.float32)
        return descriptor / (descriptor.sum() + 1e-8)

    @staticmethod
    def edge_histogram_descriptor(image: np.ndarray,
                                grid: tuple = (4, 4),
                                n_bins: int = 9) -> np.ndarray:
        """
        Gradient Descriptor 2: Edge Histogram Descriptor (EHD).

        Divides the image into a spatial grid of cells and computes a
        weighted orientation histogram per cell using Sobel gradients.
        Captures BOTH spatial layout AND dominant edge directions —
        complementary to the global hog_lite descriptor.

        Parameters
        ----------
        image : numpy.ndarray
            Grayscale (H, W) or RGB (H, W, 3).
        grid : tuple of int, optional
            (rows, cols) — number of cells to divide the image into.
            Default is (4, 4) → 16 cells.
            Both values must be positive integers.
        n_bins : int, optional
            Orientation bins per cell in [0°, 180°). Default is 9.
            Must be a positive integer.

        Returns
        -------
        numpy.ndarray
            1-D float32 descriptor of length ``grid[0] × grid[1] × n_bins``.
            Default: 4×4×9 = 144 dims. Each cell's histogram is L1-normalised.

        Raises
        ------
        TypeError
            If ``image`` is not a numpy.ndarray, ``grid`` is not a tuple/list,
            or ``n_bins`` is not an int.
        ValueError
            If ``image`` has unsupported shape, ``grid`` values are not positive,
            ``n_bins`` ≤ 0, or image is too small for the requested grid.

        Notes
        -----
        Gradient computation (central difference, vectorised):
            Gx = image[:, 2:] - image[:, :-2]
            Gy = image[2:, :] - image[:-2, :]

        Per-cell histogram:
            magnitude-weighted orientation histogram in [0°, 180°).
            L1-normalised per cell → invariant to local illumination changes.

        Spatial grid → spatial locality:
            Unlike hog_lite (global histogram), EHD preserves WHERE edges
            occur in the image, making it more discriminative for classification.

        Descriptor layout:
            [cell(0,0) hist | cell(0,1) hist | ... | cell(R-1,C-1) hist]
            Row-major order, each block is ``n_bins`` floats.
        """
        # --- input validation ---
        FeatureExtractor._validate_image(image, "edge_histogram_descriptor")

        if not isinstance(grid, (tuple, list)) or len(grid) != 2:
            raise ValueError(
                f"edge_histogram_descriptor: grid must be a tuple of 2 ints "
                f"(rows, cols), got {grid}."
            )
        grid_r, grid_c = grid
        if not (isinstance(grid_r, (int, np.integer)) and
                isinstance(grid_c, (int, np.integer))):
            raise TypeError(
                "edge_histogram_descriptor: grid values must be integers, "
                f"got ({type(grid_r).__name__}, {type(grid_c).__name__})."
            )
        if grid_r <= 0 or grid_c <= 0:
            raise ValueError(
                f"edge_histogram_descriptor: grid values must be positive, "
                f"got {grid}."
            )
        if not isinstance(n_bins, (int, np.integer)):
            raise TypeError(
                f"edge_histogram_descriptor: n_bins must be an int, "
                f"got {type(n_bins).__name__}."
            )
        if n_bins <= 0:
            raise ValueError(
                f"edge_histogram_descriptor: n_bins must be positive, "
                f"got {n_bins}."
            )

        gray = FeatureExtractor._to_grayscale_float(image)

        # --- compute gradients (central difference, vectorised) ---
        gx = gray[:, 2:] - gray[:, :-2]   # (H,   W-2)
        gx = gx[1:-1, :]                  # (H-2, W-2)
        gy = gray[2:, :] - gray[:-2, :]   # (H-2, W  )
        gy = gy[:, 1:-1]                  # (H-2, W-2)

        magnitude   = np.sqrt(gx ** 2 + gy ** 2)          # (H-2, W-2)
        orientation = np.arctan2(gy, gx) * (180.0 / np.pi) % 180.0

        h, w = magnitude.shape
        if h < grid_r or w < grid_c:
            raise ValueError(
                f"edge_histogram_descriptor: image ({h}×{w}) too small "
                f"for grid {grid}. Reduce grid size or use a larger image."
            )

        # --- spatial grid: split into cells ---
        # Use array_split → handles non-divisible sizes gracefully
        row_splits = np.array_split(np.arange(h), grid_r)
        col_splits = np.array_split(np.arange(w), grid_c)

        cell_histograms = []

        # Loop over grid cells only (grid_r × grid_c iterations, not pixels)
        # Justified: spatial decomposition requires per-cell indexing
        for row_idx in row_splits:
            for col_idx in col_splits:
                # Extract cell slices
                cell_mag = magnitude[np.ix_(row_idx, col_idx)]
                cell_ori = orientation[np.ix_(row_idx, col_idx)]

                # Weighted orientation histogram for this cell
                hist, _ = np.histogram(
                    cell_ori, bins=n_bins,
                    range=(0.0, 180.0),
                    weights=cell_mag
                )
                hist = hist.astype(np.float32)

                # L1 normalise per cell
                cell_histograms.append(hist / (hist.sum() + 1e-8))

        return np.concatenate(cell_histograms)