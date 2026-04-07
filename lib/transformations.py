import numpy as np

class Transformations:
    """
    Geometric transformations including resizing, rotation, and translation.
    Implements Inverse Mapping and Interpolation.
    """

    @staticmethod
    def resize(image, new_shape, method='nearest'):
        """
        5.1 Resizing: Supports Nearest-Neighbor and Bilinear interpolation.
        new_shape: tuple (new_height, new_width)
        """
        h, w = image.shape[:2]
        new_h, new_w = new_shape
        
        # Create destination grid
        y_new, x_new = np.indices((new_h, new_w))
        
        # Map back to source coordinates (Inverse Mapping)
        y_src = y_new * (h / new_h)
        x_src = x_new * (w / new_w)

        if method == 'nearest':
            # Round to nearest integer coordinates
            y_src = np.clip(np.round(y_src).astype(int), 0, h - 1)
            x_src = np.clip(np.round(x_src).astype(int), 0, w - 1)
            return image[y_src, x_src]

        elif method == 'bilinear':
            # Get integer boundaries
            y0 = np.floor(y_src).astype(int)
            y1 = np.clip(y0 + 1, 0, h - 1)
            x0 = np.floor(x_src).astype(int)
            x1 = np.clip(x0 + 1, 0, w - 1)

            # Distances from integer boundaries
            dy = y_src - y0
            dx = x_src - x0

            # Interpolation weights
            # Ensure weights are 3D for RGB broadcasting
            if len(image.shape) == 3:
                dy = dy[..., None]
                dx = dx[..., None]

            # Bilinear formula: 4-pixel weighted average
            top = image[y0, x0] * (1 - dx) + image[y0, x1] * dx
            bottom = image[y1, x0] * (1 - dx) + image[y1, x1] * dx
            return (top * (1 - dy) + bottom * dy).astype(image.dtype)
        
        else:
            raise ValueError("Method must be 'nearest' or 'bilinear'.")

    @staticmethod
    def rotate(image, angle, method='nearest'):
        """
        5.2 Rotation: Rotate about center by angle (degrees).
        Uses inverse mapping to prevent holes in the output.
        """
        h, w = image.shape[:2]
        cy, cx = h // 2, w // 2
        rad = np.deg2rad(angle)
        
        # Output canvas
        output = np.zeros_like(image)
        
        # Destination coordinates
        y_dest, x_dest = np.indices((h, w))
        
        # Shift origin to center, rotate, and shift back (Inverse)
        # x' = (x-cx)cos + (y-cy)sin + cx
        # y' = -(x-cx)sin + (y-cy)cos + cy
        x_src = (x_dest - cx) * np.cos(rad) + (y_dest - cy) * np.sin(rad) + cx
        y_src = -(x_dest - cx) * np.sin(rad) + (y_dest - cy) * np.cos(rad) + cy

        # Mask for pixels that fall within source boundaries
        mask = (x_src >= 0) & (x_src < w - 1) & (y_src >= 0) & (y_src < h - 1)

        if method == 'nearest':
            x_src = np.round(x_src).astype(int)
            y_src = np.round(y_src).astype(int)
            output[mask] = image[y_src[mask], x_src[mask]]
        else:
            # Bilinear can be implemented similarly to resize for smoother rotation
            pass
            
        return output

    @staticmethod
    def translate(image, tx, ty):
        """
        5.3 Translation: Shifts image by tx (horizontal) and ty (vertical).
        """
        h, w = image.shape[:2]
        output = np.zeros_like(image)
        
        # Define the valid destination range
        # Destination: max(0, ty) to min(h, h+ty)
        # Source: max(0, -ty) to min(h, h-ty)
        start_y_dest, end_y_dest = max(0, ty), min(h, h + ty)
        start_x_dest, end_x_dest = max(0, tx), min(w, w + tx)
        
        start_y_src, end_y_src = max(0, -ty), min(h, h - ty)
        start_x_src, end_x_src = max(0, -tx), min(w, w - tx)

        # Check if shift is completely out of bounds
        if start_y_dest < end_y_dest and start_x_dest < end_x_dest:
            output[start_y_dest:end_y_dest, start_x_dest:end_x_dest] = \
                image[start_y_src:end_y_src, start_x_src:end_x_src]
                
        return output