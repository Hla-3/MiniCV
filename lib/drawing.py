import numpy as np

class Drawing:
    """
    Canvas operations for drawing primitives directly on NumPy arrays.
    Supports Grayscale and RGB color formats with boundary clipping.
    """

    @staticmethod
    def _get_color(canvas, color):
        """
        Internal helper to ensure color format matches canvas depth.
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
        """
        draw_color = Drawing._get_color(canvas, color)
        h, w = canvas.shape[:2]
        
        offset = thickness // 2
        for i in range(y - offset, y - offset + thickness):
            for j in range(x - offset, x - offset + thickness):
                if 0 <= i < h and 0 <= j < w:
                    canvas[i, j] = draw_color

    @staticmethod
    def draw_line(canvas, x1, y1, x2, y2, color, thickness=1):
        """
        Draws a line using Bresenham's Line Algorithm.
        """
        dx = abs(x2 - x1)
        dy = -abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx + dy

        while True:
            Drawing.draw_point(canvas, x1, y1, color, thickness)
            if x1 == x2 and y1 == y2:
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
        Supports both filled and outline modes.
        """
        if fill:
            draw_color = Drawing._get_color(canvas, color)
            # Clip coordinates to canvas
            h, w = canvas.shape[:2]
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
    def draw_polygon(canvas, vertices, color, thickness=1, fill=False):
        """
        Draws a polygon from a list of (x, y) vertices.
        Connects the last vertex back to the first.
        """
        num_vertices = len(vertices)
        if num_vertices < 2:
            return

        for i in range(num_vertices):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % num_vertices]
            Drawing.draw_line(canvas, p1[0], p1[1], p2[0], p2[1], color, thickness)
        
        # Note: Filled polygon logic usually requires a Scan-line Fill algorithm
        # which is significantly more complex. Outline is the baseline requirement.