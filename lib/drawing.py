import numpy as np
import math

class Drawing:
    """
    Canvas operations for drawing primitives directly on NumPy arrays.
    Supports Grayscale and RGB color formats with boundary clipping.
    """

    @staticmethod
    def _get_color(canvas, color):
        """
        Internal helper to ensure color format matches canvas depth.
        
        Parameters:
        -----------
        canvas : np.ndarray
            The target image array.
        color : int | list | tuple | np.ndarray
            The color value to be processed.
            
        Returns:
        --------
        np.ndarray | int
            A color value formatted to match the canvas channels.
        """
        if len(canvas.shape) == 3:  # RGB Canvas
            return np.array(color) if isinstance(color, (list, tuple, np.ndarray)) else np.array([color]*3)
        return color  # Grayscale scalar

    @staticmethod
    def _is_inside(canvas, x, y):
        """Checks if a point (x, y) is within the canvas boundaries."""
        h, w = canvas.shape[:2]
        return 0 <= x < w and 0 <= y < h

    @staticmethod
    def draw_point(canvas, x, y, color, thickness=1):
        """
        Draws a point (square) centered at (x, y).
        Includes clipping to canvas boundaries.

        Parameters:
        -----------
        canvas : numpy.ndarray
            The image to draw on. Shape (H, W) or (H, W, 3).
        x : int
            Horizontal center coordinate of the point.
        y : int
            Vertical center coordinate of the point.
        color : int | tuple | list | numpy.ndarray
            Color value. Scaler for grayscale, 3-element for RGB.
        thickness : int, optional
            Side length of the square point. Default is 1.

        Returns:
        --------
        None : Modifies the canvas in-place.

        Raises:
        -------
        TypeError: If canvas is not a numpy array, or x, y, thickness are not integers.
        ValueError: If thickness < 1 or canvas is not 2D/3D.

        Notes on expected input:
        ------------------------
        - Coordinates (x, y) can be outside canvas bounds; clipping is handled.
        - thickness should be an odd integer for perfect centering, but all positive ints are accepted.
        """
        # Input Validation
        if not isinstance(canvas, np.ndarray):
            raise TypeError(f"draw_point: canvas must be numpy.ndarray, got {type(canvas).__name__}.")
        if not all(isinstance(v, (int, np.integer)) for v in [x, y, thickness]):
            raise TypeError("draw_point: x, y, and thickness must be integers.")
        if thickness < 1:
            raise ValueError(f"draw_point: thickness must be >= 1, got {thickness}.")
        if canvas.ndim not in [2, 3]:
            raise ValueError(f"draw_point: canvas must be 2D or 3D, got {canvas.ndim}D.")

        draw_color = Drawing._get_color(canvas, color)
        h, w = canvas.shape[:2]
        
        offset = thickness // 2 # To have the line thickness centered around the point
        for i in range(y - offset, y - offset + thickness):
            for j in range(x - offset, x - offset + thickness):
                if 0 <= i < h and 0 <= j < w: # Check if we are within the bounds of the canvas
                    canvas[i, j] = draw_color

    @staticmethod
    def draw_line(canvas, x1, y1, x2, y2, color, thickness=1):
        """
        Draws a line using Bresenham's Line Algorithm.

        Description:
        ------------
        Determines which points in a grid-based canvas should be colored to form a 
        close approximation of a straight line between (x1, y1) and (x2, y2).

        Parameters:
        -----------
        canvas : numpy.ndarray
            Target image array.
        x1, y1 : int
            Start point coordinates.
        x2, y2 : int
            End point coordinates.
        color : int | tuple | list
            Color value matching canvas format.
        thickness : int, optional
            Line thickness in pixels. Default is 1.

        Returns:
        --------
        None : Modifies canvas in-place.

        Raises:
        -------
        TypeError: If coordinates or thickness are not integers.
        ValueError: If thickness < 1.
        """
        # Input Validation
        coords = [x1, y1, x2, y2, thickness]
        if not all(isinstance(c, (int, np.integer)) for c in coords):
            raise TypeError("draw_line: All coordinates and thickness must be integers.")
        if thickness < 1:
            raise ValueError(f"draw_line: thickness must be >= 1, got {thickness}.")

        dx = abs(x2 - x1)
        dy = -abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx + dy

        while True:
            # Draw the line with multiple points given a thickness and a color
            Drawing.draw_point(canvas, x1, y1, color, thickness)
            if x1 == x2 and y1 == y2: # If we have reached the end point, we break out of the loop.
                break

            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x1 += sx
            if e2 <= dx:
                err += dx
                y1 += sy

    @staticmethod
    def draw_rectangle(canvas, x1, y1, x2, y2, color, thickness=1, fill=False):
        """
        Draws a rectangle from top-left (x1, y1) to bottom-right (x2, y2).

        Parameters:
        -----------
        canvas : numpy.ndarray
            Target image array.
        x1, y1 : int
            Coordinates of the first corner.
        x2, y2 : int
            Coordinates of the opposite corner.
        color : int | tuple | list
            Color value.
        thickness : int, optional
            Thickness of the border. Only used if fill=False. Default 1.
        fill : bool, optional
            If True, fills the entire rectangle. Default is False.

        Returns:
        --------
        None : Modifies canvas in-place.

        Raises:
        -------
        TypeError: If coordinates are not ints or fill is not a boolean.
        ValueError: If thickness < 1.
        """
        # Input Validation
        if not all(isinstance(c, (int, np.integer)) for c in [x1, y1, x2, y2, thickness]):
            raise TypeError("draw_rectangle: Coordinates and thickness must be integers.")
        if not isinstance(fill, bool):
            raise TypeError(f"draw_rectangle: fill must be bool, got {type(fill).__name__}.")
        if thickness < 1:
            raise ValueError(f"draw_rectangle: thickness must be >= 1, got {thickness}.")

        if fill: 
            draw_color = Drawing._get_color(canvas, color)
            h, w = canvas.shape[:2]
            # Clip coordinates to canvas
            start_x, end_x = max(0, min(x1, x2)), min(w, max(x1, x2))
            start_y, end_y = max(0, min(y1, y2)), min(h, max(y1, y2))
            canvas[start_y:end_y, start_x:end_x] = draw_color
        else:
            # Draw four lines
            Drawing.draw_line(canvas, x1, y1, x2, y1, color, thickness) # Top
            Drawing.draw_line(canvas, x1, y2, x2, y2, color, thickness) # Bottom
            Drawing.draw_line(canvas, x1, y1, x1, y2, color, thickness) # Left
            Drawing.draw_line(canvas, x2, y1, x2, y2, color, thickness) # Right

    @staticmethod
    def draw_polygon(canvas, vertices, center=(0,0), color=255, thickness=1):
        """
        Draws a polygon defined by a list of vertices relative to a center point.

        Parameters:
        -----------
        canvas : numpy.ndarray
            Target image array.
        vertices : list of tuples/lists
            List of (x, y) coordinates. Must contain at least 2 vertices.
        center : tuple | list
            The (x, y) center point to shift all vertices. Default (0,0).
        color : int | tuple | list
            Color value. Default is 255.
        thickness : int
            Thickness of the polygon edges. Default is 1.

        Returns:
        --------
        None : Modifies canvas in-place.

        Raises:
        -------
        TypeError: If vertices is not a list/tuple, or thickness is not an int.
        ValueError: If vertices list contains fewer than 2 points.
        """
        # Input Validation
        if not isinstance(vertices, (list, tuple, np.ndarray)):
            raise TypeError(f"draw_polygon: vertices must be a list/tuple, got {type(vertices).__name__}.")
        if not isinstance(thickness, (int, np.integer)):
            raise TypeError("draw_polygon: thickness must be an integer.")
        if len(vertices) < 2:
            raise ValueError(f"draw_polygon: At least 2 vertices required, got {len(vertices)}.")

        num_vertices = len(vertices)
        cx, cy = center

        for i in range(num_vertices):
            p1 = (int(vertices[i][0] + cx), int(vertices[i][1] + cy))
            p2 = (int(vertices[(i + 1) % num_vertices][0] + cx), int(vertices[(i + 1) % num_vertices][1] + cy))
            Drawing.draw_line(canvas, p1[0], p1[1], p2[0], p2[1], color, thickness)

    @staticmethod        
    def draw_regular_polygon(canvas, center, sides, radius, color=255, thickness=1):
        """
        Generates and draws a regular polygon (all sides/angles equal).

        Parameters:
        -----------
        canvas : numpy.ndarray
            Target image array.
        center : tuple | list
            (x, y) coordinates for the center of the polygon.
        sides : int
            Number of sides. Must be >= 3.
        radius : int | float
            Distance from center to vertices.
        color : int | tuple | list
            Color value. Default is 255.
        thickness : int
            Line thickness. Default is 1.

        Returns:
        --------
        None : Modifies canvas in-place.

        Raises:
        -------
        TypeError: If sides is not an int or radius is not numeric.
        ValueError: If sides < 3 or radius <= 0.
        """
        # Input Validation
        if not isinstance(sides, (int, np.integer)):
            raise TypeError(f"draw_regular_polygon: sides must be an integer, got {type(sides).__name__}.")
        if not isinstance(radius, (int, float, np.number)):
            raise TypeError("draw_regular_polygon: radius must be numeric.")
        if sides < 3:
            raise ValueError(f"draw_regular_polygon: sides must be >= 3 to form a polygon, got {sides}.")
        if radius <= 0:
            raise ValueError(f"draw_regular_polygon: radius must be positive, got {radius}.")

        vertices = []
        for i in range(sides):
            angle = 2 * math.pi * i / sides
            vx = radius * math.cos(angle)
            vy = radius * math.sin(angle)
            vertices.append((vx, vy))
        
        Drawing.draw_polygon(canvas, vertices, center, color, thickness)