import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

class IO:
    """
    Image I/O and color space conversion utilities for the MiniCV library.

    Handles loading images from disk, exporting images to disk,
    and converting between RGB and Grayscale color spaces.

    All images are represented as NumPy arrays:
        - Grayscale : shape (H, W),        dtype float32 or uint8
        - RGB       : shape (H, W, 3),     dtype float32 or uint8
        - RGBA      : shape (H, W, 4),     dtype float32 or uint8
    
    Notes
    -----
    Images are read in RGB channel order (not BGR like OpenCV).
    This is consistent with Matplotlib's display functions.
    """

    @staticmethod
    def read_image(file_path: str, normalize: bool = False) -> np.ndarray:
        """
        Load an image from disk into a NumPy array.

        Parameters
        ----------
        file_path : str
            Absolute or relative path to the image file.
        normalize : bool, optional
            If True, converts output to float32 in [0, 1] regardless
            of source format (PNG or JPEG). Default is False.

        Returns
        -------
        numpy.ndarray
            Image array with shape (H, W), (H, W, 3), or (H, W, 4).
            - PNG  → float32 in [0, 1]  (or normalized if normalize=True)
            - JPEG → uint8  in [0, 255] (or float32 if normalize=True)

        Raises
        ------
        TypeError
            If ``file_path`` is not a string, or ``normalize`` is not bool.
        FileNotFoundError
            If no file exists at the given path.
        ValueError
            If the file cannot be decoded as an image.

        Notes
        -----
        Use ``normalize=True`` for consistent float32 output across
        PNG and JPEG — recommended before any arithmetic operations.
        """
        if not isinstance(file_path, str):
            raise TypeError(
                f"file_path must be a string, got {type(file_path).__name__}."
            )
        if not isinstance(normalize, bool):
            raise TypeError(
                f"normalize must be a bool, got {type(normalize).__name__}."
            )
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"No file found at '{file_path}'. Check the path and try again."
            )

        try:
            image_array = mpimg.imread(file_path)
        except Exception as e:
            raise ValueError(f"Could not decode image at '{file_path}': {e}")

        if normalize:
            image_array = image_array.astype(np.float32)
            if image_array.max() > 1.0:   # كانت uint8 → نقسم على 255
                image_array /= 255.0

        return image_array

    @staticmethod
    def export_image(image_array: np.ndarray, output_path: str) -> None:
        """
        Save a NumPy image array to disk.

        Parameters
        ----------
        image_array : numpy.ndarray
            Image to save. Shape: (H, W) or (H, W, 3).
            dtype: uint8 [0-255] or float32/float64 [0.0-1.0].
        output_path : str
            Destination file path including extension (e.g. 'out.png').

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If ``image_array`` is not a NumPy ndarray, or
            ``output_path`` is not a string.
        ValueError
            If array shape is unsupported, or file cannot be written.

        Notes
        -----
        Float arrays are clipped to [0, 1] before saving.
        uint8 arrays are saved as-is.
        """
        if not isinstance(image_array, np.ndarray):
            raise TypeError(
                f"image_array must be a numpy.ndarray, "
                f"got {type(image_array).__name__}."
            )
        if not isinstance(output_path, str):
            raise TypeError(
                f"output_path must be a string, got {type(output_path).__name__}."
            )
        if image_array.ndim not in (2, 3) or (
            image_array.ndim == 3 and image_array.shape[2] not in (3, 4)
        ):
            raise ValueError(
                f"image_array must have shape (H, W) or (H, W, 3), "
                f"got {image_array.shape}."
            )

        if np.issubdtype(image_array.dtype, np.floating):
            save_array = np.clip(image_array, 0.0, 1.0)
        else:
            save_array = image_array.astype(np.uint8)

        try:
            if save_array.ndim == 2:
                plt.imsave(output_path, save_array, cmap='gray')
            else:
                plt.imsave(output_path, save_array)
        except Exception as e:
            raise ValueError(f"Failed to write image to '{output_path}': {e}")
        
    @staticmethod
    def rgb_to_grayscale(image_array: np.ndarray) -> np.ndarray:
        """
        Convert an RGB or RGBA image to grayscale using luminosity weights.

        Formula: Y = 0.299·R + 0.587·G + 0.114·B

        Parameters
        ----------
        image_array : numpy.ndarray
            Array of shape (H, W, 3) for RGB or (H, W, 4) for RGBA.

        Returns
        -------
        numpy.ndarray
            Grayscale array of shape (H, W), same dtype as input.

        Raises
        ------
        TypeError
            If ``image_array`` is not a NumPy ndarray.
        ValueError
            If the array is not 3-D or does not have 3 or 4 channels.

        Notes
        -----
        RGBA images: alpha channel is dropped before conversion.
        uint8 input: result is rounded before casting to preserve accuracy.
        Uses NumPy vectorisation; no Python pixel loops.
        """
        if not isinstance(image_array, np.ndarray):
            raise TypeError(
                f"image_array must be a numpy.ndarray, "
                f"got {type(image_array).__name__}."
            )
        if image_array.ndim != 3 or image_array.shape[2] not in (3, 4):
            raise ValueError(
                f"Expected shape (H, W, 3) or (H, W, 4), "
                f"got {image_array.shape}."
            )

        # Drop alpha if RGBA, cast to float32 for accurate computation
        rgb = image_array[:, :, :3].astype(np.float32)

        grayscale = (
            0.299 * rgb[:, :, 0]
            + 0.587 * rgb[:, :, 1]
            + 0.114 * rgb[:, :, 2]
        )

        # uint8: round first to avoid truncation error
        if image_array.dtype == np.uint8:
            return np.round(grayscale).astype(np.uint8)

        return grayscale.astype(image_array.dtype)

        
    @staticmethod
    def grayscale_to_rgb(image_array: np.ndarray) -> np.ndarray:
        """
        Convert a grayscale image to a 3-channel RGB image.

        Parameters
        ----------
        image_array : numpy.ndarray
            Grayscale array of shape (H, W).

        Returns
        -------
        numpy.ndarray
            RGB array of shape (H, W, 3), same dtype as input.

        Raises
        ------
        TypeError
            If ``image_array`` is not a NumPy ndarray.
        ValueError
            If the array is not 2-D.

        Notes
        -----
        Each output channel is identical to the input.
        No colour information is added.
        """
        if not isinstance(image_array, np.ndarray):
            raise TypeError(
                f"image_array must be a numpy.ndarray, "
                f"got {type(image_array).__name__}."
            )
        if image_array.ndim != 2:
            raise ValueError(
                f"Expected a 2-D grayscale array (H, W), "
                f"got shape {image_array.shape}."
            )

        return np.stack([image_array] * 3, axis=-1)