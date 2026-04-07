import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class IO:
    """
    A core utility class for Image I/O and color space conversions.
    Part of the MiniCV library for CSE480[cite: 5, 7].
    """

    @staticmethod
    def read_image(file_path):
        """
        2.1 Read image: Load images from disk into NumPy arrays[cite: 13].
        Supports formats allowed by Matplotlib backends (PNG/JPG)[cite: 14].

        Parameters:
        -----------
        file_path : str
            The path to the image file.

        Returns:
        --------
        image_array : numpy.ndarray
            The image data as a NumPy array.
        """
        try:
            # mpimg.imread returns a NumPy array [cite: 13]
            image_array = mpimg.imread(file_path)
            return image_array
        except FileNotFoundError:
            raise FileNotFoundError(f"The file at {file_path} was not found.")
        except Exception as e:
            raise ValueError(f"Error reading image: {e}")

    @staticmethod
    def export_image(image_array, output_path):
        """
        2.2 Export image: Save in-memory NumPy arrays to disk[cite: 15].
        Supports both grayscale and RGB formats[cite: 15].

        Parameters:
        -----------
        image_array : numpy.ndarray
            The image array to save.
        output_path : str
            Destination path (e.g., 'output.png' or 'output.jpg').
        """
        if not isinstance(image_array, np.ndarray):
            raise TypeError("Input must be a NumPy array[cite: 69].")

        try:
            # Determine if grayscale or RGB based on array shape [cite: 15]
            if len(image_array.shape) == 2:
                # Save as grayscale using a gray colormap
                plt.imsave(output_path, image_array, cmap='gray')
            elif len(image_array.shape) == 3:
                # Save as RGB [cite: 15]
                plt.imsave(output_path, image_array)
            else:
                raise ValueError(f"Invalid image shape: {image_array.shape}[cite: 70].")
        except Exception as e:
            raise ValueError(f"Failed to export image: {e}")

    @staticmethod
    def rgb_to_grayscale(image_array):
        """
        2.3 Color conversion: RGB to Grayscale[cite: 16].
        Ensures consistent output shape (H, W)[cite: 16].

        Parameters:
        -----------
        image_array : numpy.ndarray
            A 3D array of shape (H, W, 3).

        Returns:
        --------
        grayscale : numpy.ndarray
            A 2D array of shape (H, W).
        """
        if len(image_array.shape) != 3 or image_array.shape[2] != 3:
            raise ValueError("Input must be a 3D RGB array[cite: 70, 71].")

        # Applying luminosity weights: Y = 0.299R + 0.587G + 0.114B
        # Uses NumPy vectorization to avoid loops 
        grayscale = (0.299 * image_array[:, :, 0] + 
                     0.587 * image_array[:, :, 1] + 
                     0.114 * image_array[:, :, 2])
        
        return grayscale.astype(image_array.dtype)

    @staticmethod
    def grayscale_to_rgb(image_array):
        """
        2.3 Color conversion: Grayscale to RGB[cite: 16].
        Ensures consistent output shape (H, W, 3)[cite: 16].

        Parameters:
        -----------
        image_array : numpy.ndarray
            A 2D array of shape (H, W).

        Returns:
        --------
        rgb : numpy.ndarray
            A 3D array of shape (H, W, 3).
        """
        if len(image_array.shape) != 2:
            raise ValueError("Input must be a 2D grayscale array[cite: 70, 71].")

        # Stack the single channel 3 times to create RGB [cite: 16]
        rgb = np.stack([image_array] * 3, axis=-1)
        return rgb