import numpy as np

class Text:
    """
    Renders text onto NumPy arrays using a custom bitmap font.
    Supports position, scaling, and color.
    """

    # Basic 5x7 Bitmap Font for characters A-Z, 0-9, and space
    # 1 represents a pixel, 0 represents empty space
    _FONT = {
        'A': [[0,1,1,1,0], [1,0,0,0,1], [1,1,1,1,1], [1,0,0,0,1], [1,0,0,0,1]],
        'B': [[1,1,1,1,0], [1,0,0,0,1], [1,1,1,1,0], [1,0,0,0,1], [1,1,1,1,0]],
        'C': [[0,1,1,1,1], [1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0], [0,1,1,1,1]],
        'H': [[1,0,0,0,1], [1,0,0,0,1], [1,1,1,1,1], [1,0,0,0,1], [1,0,0,0,1]],
        'L': [[1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0], [1,1,1,1,1]],
        '0': [[0,1,1,1,0], [1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [0,1,1,1,0]],
        '1': [[0,0,1,0,0], [0,1,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,1,1,1,0]],
        ' ': [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],
        # ... You can expand this dictionary with more characters as needed
    }

    @staticmethod
    def _get_color(canvas, color):
        """Internal helper to match color format to canvas depth."""
        if len(canvas.shape) == 3:
            return np.array(color) if isinstance(color, (list, tuple, np.ndarray)) else np.array([color]*3)
        return color

    @staticmethod
    def put_text(canvas, text, x, y, font_scale=1, color=255):
        """
        Places text on the canvas at position (x, y).
        
        Parameters:
        -----------
        canvas : numpy.ndarray
            The image to draw on.
        text : str
            The string to be rendered (Uppercase).
        x, y : int
            Top-left starting coordinate of the first character.
        font_scale : int
            Multiplier for the size of the font (default 1).
        color : int or tuple
            Color value matching the canvas format.
        """
        draw_color = Text._get_color(canvas, color)
        h, w = canvas.shape[:2]
        
        current_x = x
        for char in text.upper():
            if char not in Text._FONT:
                char = ' ' # Default to space if character is missing
            
            bitmap = np.array(Text._FONT[char])
            rows, cols = bitmap.shape
            
            # Draw each "pixel" of the bitmap character
            for r in range(rows):
                for c in range(cols):
                    if bitmap[r, c] == 1:
                        # Scale the "pixel" based on font_scale
                        start_r = y + r * font_scale
                        start_c = current_x + c * font_scale
                        
                        # Drawing a square for each bitmap pixel to handle scaling
                        for i in range(start_r, start_r + font_scale):
                            for j in range(start_c, start_c + font_scale):
                                if 0 <= i < h and 0 <= j < w:
                                    canvas[i, j] = draw_color
            
            # Move cursor for next character (5 pixels + 1 pixel spacing) * scale
            current_x += (cols + 1) * font_scale