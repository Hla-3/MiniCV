import numpy as np


class Transformations:
    """
    Geometric transformations for image arrays using inverse mapping.

    All functions operate on NumPy arrays and support both grayscale
    (H, W) and RGB (H, W, 3) images unless stated otherwise.

    Inverse Mapping Strategy
    ------------------------
    For each destination pixel, we compute where it came from in the
    source image, then sample that location. This prevents holes in
    the output that forward mapping would cause.
    """

    # ------------------------------------------------------------------ #
    #  5.1  Resize                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def resize(image: np.ndarray,
               new_shape: tuple,
               method: str = 'nearest') -> np.ndarray:
        """
        Resize an image to a new spatial size.

        Uses inverse mapping: for each destination pixel (x', y'),
        compute the corresponding source coordinate, then interpolate.

        Parameters
        ----------
        image : numpy.ndarray
            Input image of shape (H, W) or (H, W, 3).
        new_shape : tuple of int
            Target size as (new_height, new_width). Both must be > 0.
        method : str, optional
            Interpolation method:
            - 'nearest'  : Nearest-neighbour (fast, blocky).
            - 'bilinear' : Bilinear interpolation (smoother).
            Default is 'nearest'.

        Returns
        -------
        numpy.ndarray
            Resized image of shape (new_H, new_W) or (new_H, new_W, 3),
            same dtype as input.

        Raises
        ------
        TypeError
            If ``image`` is not a numpy.ndarray, ``new_shape`` is not a
            tuple/list, or ``method`` is not a string.
        ValueError
            If ``image`` is not 2-D or 3-D, ``new_shape`` does not have
            exactly 2 positive integers, or ``method`` is not recognised.

        Notes
        -----
        Nearest-neighbour: src = round(dest * scale) → fastest, aliasing.
        Bilinear: weighted average of 4 surrounding pixels → smoother.
        Scale factors: scale_y = H / new_H, scale_x = W / new_W.
        """
        # --- input validation ---
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"image must be a numpy.ndarray, got {type(image).__name__}."
            )
        if image.ndim not in (2, 3):
            raise ValueError(
                f"image must be 2-D (H,W) or 3-D (H,W,3), "
                f"got shape {image.shape}."
            )
        if not isinstance(new_shape, (tuple, list)) or len(new_shape) != 2:
            raise ValueError(
                f"new_shape must be a tuple of 2 ints (new_H, new_W), "
                f"got {new_shape}."
            )
        new_h, new_w = new_shape
        if not (isinstance(new_h, (int, np.integer)) and
                isinstance(new_w, (int, np.integer))):
            raise TypeError(
                "new_shape values must be integers, "
                f"got ({type(new_h).__name__}, {type(new_w).__name__})."
            )
        if new_h <= 0 or new_w <= 0:
            raise ValueError(
                f"new_shape values must be positive, got ({new_h}, {new_w})."
            )
        if not isinstance(method, str):
            raise TypeError(
                f"method must be a string, got {type(method).__name__}."
            )
        if method not in ('nearest', 'bilinear'):
            raise ValueError(
                f"method must be 'nearest' or 'bilinear', got '{method}'."
            )

        h, w = image.shape[:2]

        # destination grid → all (row, col) pairs in output
        y_dest, x_dest = np.indices((new_h, new_w))

        # inverse map: destination → source coordinates
        y_src = y_dest * (h / new_h)
        x_src = x_dest * (w / new_w)

        # ---- Nearest-Neighbour ----
        if method == 'nearest':
            y_idx = np.clip(np.round(y_src).astype(int), 0, h - 1)
            x_idx = np.clip(np.round(x_src).astype(int), 0, w - 1)
            return image[y_idx, x_idx]

        # ---- Bilinear ----
        # four surrounding integer coordinates
        y0 = np.floor(y_src).astype(int)
        y1 = np.clip(y0 + 1, 0, h - 1)
        x0 = np.floor(x_src).astype(int)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y0 = np.clip(y0, 0, h - 1)
        x0 = np.clip(x0, 0, w - 1)

        # fractional distances
        dy = (y_src - np.floor(y_src))
        dx = (x_src - np.floor(x_src))

        # expand dims for RGB broadcasting
        if image.ndim == 3:
            dy = dy[..., None]
            dx = dx[..., None]

        # weighted average of 4 neighbours
        top    = image[y0, x0] * (1 - dx) + image[y0, x1] * dx
        bottom = image[y1, x0] * (1 - dx) + image[y1, x1] * dx
        result = top * (1 - dy) + bottom * dy

        return result.astype(image.dtype)

    # ------------------------------------------------------------------ #
    #  5.2  Rotate                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def rotate(image: np.ndarray,
               angle: float,
               method: str = 'nearest') -> np.ndarray:
        """
        Rotate an image about its centre by a given angle.

        Uses inverse mapping: for each destination pixel, compute the
        corresponding source pixel using the inverse rotation matrix,
        then sample with the chosen interpolation.

        Parameters
        ----------
        image : numpy.ndarray
            Input image of shape (H, W) or (H, W, 3).
        angle : float or int
            Counter-clockwise rotation angle in degrees.
        method : str, optional
            Interpolation method: 'nearest' or 'bilinear'.
            Default is 'nearest'.

        Returns
        -------
        numpy.ndarray
            Rotated image of same shape and dtype as input.
            Pixels outside the source boundary are filled with 0 (black).

        Raises
        ------
        TypeError
            If ``image`` is not a numpy.ndarray, ``angle`` is not numeric,
            or ``method`` is not a string.
        ValueError
            If ``image`` is not 2-D or 3-D, or ``method`` is not recognised.

        Notes
        -----
        Inverse rotation formula (rotate destination back to source):
            x_src = (x - cx)·cos(θ) + (y - cy)·sin(θ) + cx
            y_src = -(x - cx)·sin(θ) + (y - cy)·cos(θ) + cy
        where (cx, cy) is the image centre.

        Pixels that map outside the source canvas are set to zero.
        Output canvas size equals input canvas size.
        """
        # --- input validation ---
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"image must be a numpy.ndarray, got {type(image).__name__}."
            )
        if image.ndim not in (2, 3):
            raise ValueError(
                f"image must be 2-D (H,W) or 3-D (H,W,3), "
                f"got shape {image.shape}."
            )
        if not isinstance(angle, (int, float, np.floating, np.integer)):
            raise TypeError(
                f"angle must be a numeric type, got {type(angle).__name__}."
            )
        if not isinstance(method, str):
            raise TypeError(
                f"method must be a string, got {type(method).__name__}."
            )
        if method not in ('nearest', 'bilinear'):
            raise ValueError(
                f"method must be 'nearest' or 'bilinear', got '{method}'."
            )

        h, w = image.shape[:2]
        cy, cx = h / 2.0, w / 2.0
        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)

        output = np.zeros_like(image)

        # destination grid
        y_dest, x_dest = np.indices((h, w))

        # shift to centre, apply inverse rotation, shift back
        xc = x_dest - cx
        yc = y_dest - cy

        x_src = xc * cos_a + yc * sin_a + cx
        y_src = -xc * sin_a + yc * cos_a + cy

        # mask: only pixels whose source falls inside the image
        valid = (x_src >= 0) & (x_src <= w - 1) & \
                (y_src >= 0) & (y_src <= h - 1)

        # ---- Nearest-Neighbour ----
        if method == 'nearest':
            x_idx = np.clip(np.round(x_src).astype(int), 0, w - 1)
            y_idx = np.clip(np.round(y_src).astype(int), 0, h - 1)
            output[valid] = image[y_idx[valid], x_idx[valid]]

        # ---- Bilinear ----
        else:
            x0 = np.floor(x_src).astype(int)
            y0 = np.floor(y_src).astype(int)
            x1 = np.clip(x0 + 1, 0, w - 1)
            y1 = np.clip(y0 + 1, 0, h - 1)
            x0 = np.clip(x0, 0, w - 1)
            y0 = np.clip(y0, 0, h - 1)

            dx = (x_src - np.floor(x_src))[valid]
            dy = (y_src - np.floor(y_src))[valid]

            if image.ndim == 3:
                dx = dx[:, None]
                dy = dy[:, None]

            top    = (image[y0[valid], x0[valid]] * (1 - dx) +
                      image[y0[valid], x1[valid]] * dx)
            bottom = (image[y1[valid], x0[valid]] * (1 - dx) +
                      image[y1[valid], x1[valid]] * dx)
            output[valid] = (top * (1 - dy) + bottom * dy).astype(image.dtype)

        return output

    # ------------------------------------------------------------------ #
    #  5.3  Translate                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def translate(image: np.ndarray, tx: int, ty: int) -> np.ndarray:
        """
        Shift an image by (tx, ty) pixels.

        Parameters
        ----------
        image : numpy.ndarray
            Input image of shape (H, W) or (H, W, 3).
        tx : int
            Horizontal shift in pixels.
            Positive → shift right, Negative → shift left.
        ty : int
            Vertical shift in pixels.
            Positive → shift down, Negative → shift up.

        Returns
        -------
        numpy.ndarray
            Translated image of same shape and dtype as input.
            Areas outside the original boundary are filled with 0.

        Raises
        ------
        TypeError
            If ``image`` is not a numpy.ndarray, or ``tx``/``ty`` are
            not integers.
        ValueError
            If ``image`` is not 2-D or 3-D.

        Notes
        -----
        No interpolation needed: pixels move to integer positions.
        Regions that shift outside the canvas are clipped (filled with 0).
        Equivalent to an affine transform with identity rotation/scale.
        """
        # --- input validation ---
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"image must be a numpy.ndarray, got {type(image).__name__}."
            )
        if image.ndim not in (2, 3):
            raise ValueError(
                f"image must be 2-D (H,W) or 3-D (H,W,3), "
                f"got shape {image.shape}."
            )
        if not isinstance(tx, (int, np.integer)):
            raise TypeError(
                f"tx must be an integer, got {type(tx).__name__}."
            )
        if not isinstance(ty, (int, np.integer)):
            raise TypeError(
                f"ty must be an integer, got {type(ty).__name__}."
            )

        h, w = image.shape[:2]
        output = np.zeros_like(image)

        # source and destination slice boundaries
        src_y0  = max(0, -ty);   src_y1  = min(h, h - ty)
        src_x0  = max(0, -tx);   src_x1  = min(w, w - tx)
        dest_y0 = max(0,  ty);   dest_y1 = min(h, h + ty)
        dest_x0 = max(0,  tx);   dest_x1 = min(w, w + tx)

        # guard: if shift is larger than image → output stays all zeros
        if src_y0 < src_y1 and src_x0 < src_x1:
            output[dest_y0:dest_y1, dest_x0:dest_x1] = \
                image[src_y0:src_y1, src_x0:src_x1]

        return output