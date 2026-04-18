# MiniCV
A reusable Python library that emulates a subset of OpenCV's core functionality. Built entirely from scratch using only NumPy, Pandas, and Matplotlib, this library handles everything from low-level 2D convolutions and geometric transformations to advanced feature extraction and canvas drawing primitives.

# System Architecture 
<img width="1392" height="897" alt="image" src="https://github.com/user-attachments/assets/520a1430-357e-4942-a298-1da9f0395ba9" />


# MiniCV Function Descriptions
<ins>1. Packaging & Project Structure
__init__.py
Initializes the MiniCV package and exposes public APIs for easy importing of core modules (I/O, filtering, transforms, features, drawing, utils).

<ins>2. Image I/O & Core Utilities
read_image(filepath)
Loads an image from disk into a NumPy array using Matplotlib backends. Supports common formats like PNG and JPG, returning the image as a multi-dimensional array for processing.

export_image(image, filepath)
Saves a NumPy array as an image file to disk. Handles both grayscale and RGB formats, ensuring proper conversion and file format compatibility (PNG/JPG).

rgb_to_gray(image)
Converts an RGB image to grayscale using weighted luminance coefficients (typically 0.299R + 0.587G + 0.114B), returning a 2D array.

gray_to_rgb(image)
Converts a grayscale image to RGB format by replicating the single channel across three channels, maintaining consistent shape conventions.

<ins>3. Core Operations (Foundation Functions)
normalize(image, mode)
Normalizes pixel values using one of at least three modes (e.g., min-max normalization to [0, 1], standardization, or scale to [0, 255]). Ensures consistent intensity ranges for processing.

clip_pixels(image, min_val, max_val)
Clips pixel values to a specified range [min_val, max_val], preventing overflow or underflow in image operations.

pad_image(image, pad_width, mode)
Adds padding around image borders using at least three modes (e.g., constant/zero padding, edge replication, or reflection). Essential for boundary handling in convolution operations.

convolve2d(image, kernel)
Performs 2D convolution on grayscale images by applying a kernel/filter. Validates kernel properties (odd dimensions, numeric type) and uses padding for boundary handling. Core function for all filtering operations.

filter2d(image, kernel)
Applies convolution-based filtering to both grayscale and RGB images. For RGB, processes each channel independently and recombines results.

<ins>4. Image Processing Techniques
mean_filter(image, kernel_size)
Applies a mean (box) filter for smoothing by averaging pixel values within a local neighborhood. Reduces noise while blurring edges.

gaussian_kernel(size, sigma)
Generates a Gaussian kernel of specified size using the given sigma (standard deviation). Returns a normalized 2D array representing the Gaussian distribution.

gaussian_filter(image, kernel_size, sigma)
Smooths an image using Gaussian filtering. Generates a Gaussian kernel and applies it via the convolution pipeline, providing edge-preserving noise reduction.

median_filter(image, kernel_size)
Applies median filtering for noise reduction, particularly effective against salt-and-pepper noise. Uses local neighborhood sorting (with justified loops) to compute median values.

threshold_global(image, threshold_value)
Binarizes an image using a global threshold value. Pixels above the threshold become white (255), others become black (0).

threshold_otsu(image)
Automatically determines optimal threshold using Otsu's method by maximizing inter-class variance. Returns both the threshold value and the binarized image.

threshold_adaptive(image, block_size, method, C)
Performs adaptive thresholding by calculating local thresholds for different regions. Supports mean and Gaussian methods with offset constant C for varying illumination conditions.

sobel_gradients(image)
Computes image gradients using Sobel operators in both x and y directions. Returns gradient magnitude and direction, useful for edge detection.

bitplane_slice(image, bit_plane)
Extracts a specific bit plane (0-7) from the image, isolating contribution of that bit position to pixel values. Useful for analyzing image compression and significance of bits.

compute_histogram(image)
Calculates the intensity histogram of a grayscale image, returning frequency distribution of pixel values (0-255).

histogram_equalization(image)
Enhances image contrast by redistributing pixel intensities to span the full range uniformly using cumulative distribution function (CDF) transformation.

laplacian_filter(image)
Applies Laplacian operator for edge detection and image sharpening. Detects regions of rapid intensity change by computing the second derivative of the image, highlighting edges and fine details.

gamma_correction(image, gamma)
Performs non-linear gamma correction to adjust image brightness and contrast. Values of gamma < 1 brighten the image, while gamma > 1 darkens it. Uses power-law transformation: output = input^gamma.

<ins>5. Geometric Transformations
resize(image, new_width, new_height, interpolation)
Resizes an image to specified dimensions using interpolation methods. Supports at least nearest-neighbor (required minimum) and bilinear interpolation for quality scaling.

rotate(image, angle, interpolation)
Rotates an image around its center by a specified angle (in degrees). Uses defined interpolation method to handle sub-pixel positioning and minimize artifacts.

translate(image, tx, ty)
Shifts an image by specified offsets (tx, ty) along x and y axes. Handles boundary regions appropriately.

<ins>6. Feature Extractors
color_histogram(image, bins)
Extracts a global color histogram descriptor by computing the distribution of color values across channels. For RGB images, computes per-channel histograms and concatenates them into a single feature vector characterizing overall color distribution.

basic_statistics(image)
Computes basic statistical descriptors for the entire image including mean, standard deviation, variance, min, max, and other moments. Returns a feature vector summarizing global intensity characteristics.

hog_lite(image, cell_size, bins)
Implements a lightweight Histogram of Oriented Gradients (HOG) descriptor. Divides the image into cells, computes gradient orientations, and builds histograms of gradient directions. Captures local shape and texture information through edge orientation patterns.

edge_histogram_descriptor(image, bins)
Extracts a gradient-based edge histogram descriptor by analyzing the distribution of edge orientations and magnitudes across the image. Computes edge information using gradient operators and creates a histogram representing edge patterns and their spatial distribution.

<ins>6. Drawing Primitives (Canvas Operations)
draw_point(image, x, y, color, thickness)
Draws a single point at coordinates (x, y) with specified color and thickness. Handles grayscale (scalar) and RGB (tuple) color formats.

draw_line(image, x1, y1, x2, y2, color, thickness)
Draws a line between two points using Bresenham's algorithm or equivalent. Supports color specification and thickness control with boundary clipping.

draw_rectangle(image, x, y, width, height, color, thickness, filled)
Draws a rectangle with top-left corner at (x, y). Supports both outline and filled modes with specified color and thickness.

draw_polygon(image, points, color, thickness, filled)
Draws a polygon defined by a list of vertices. Supports outline mode (required) and optionally filled mode, with proper edge rendering and boundary clipping.

<ins>7. Text Placement
put_text(image, text, x, y, font_scale, color, thickness)
Renders text string on the image at position (x, y). Supports font scaling for size control, color specification (grayscale/RGB), and thickness for text weight.
