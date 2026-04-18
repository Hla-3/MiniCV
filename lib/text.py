import numpy as np

class Text:
    """
    Renders text onto NumPy arrays using a custom bitmap font.
    Supports position, scaling, and color.
    """

    # Basic 5x7 Bitmap Font for characters A-Z, 0-9, and space
    # 1 represents a pixel, 0 represents empty space
    _FONT = {
        # Letters
        'A': [[0,1,1,1,0], [1,0,0,0,1], [1,1,1,1,1], [1,0,0,0,1], [1,0,0,0,1]],
        'B': [[1,1,1,1,0], [1,0,0,0,1], [1,1,1,1,0], [1,0,0,0,1], [1,1,1,1,0]],
        'C': [[0,1,1,1,1], [1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0], [0,1,1,1,1]],
        'D': [[1,1,1,1,0], [1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [1,1,1,1,0]],
        'E': [[1,1,1,1,1], [1,0,0,0,0], [1,1,1,1,0], [1,0,0,0,0], [1,1,1,1,1]],
        'F': [[1,1,1,1,1], [1,0,0,0,0], [1,1,1,1,0], [1,0,0,0,0], [1,0,0,0,0]],
        'G': [[0,1,1,1,1], [1,0,0,0,0], [1,0,1,1,1], [1,0,0,0,1], [0,1,1,1,1]],
        'H': [[1,0,0,0,1], [1,0,0,0,1], [1,1,1,1,1], [1,0,0,0,1], [1,0,0,0,1]],
        'I': [[0,1,1,1,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,1,1,1,0]],
        'J': [[0,0,0,0,1], [0,0,0,0,1], [0,0,0,0,1], [1,0,0,0,1], [0,1,1,1,0]],
        'K': [[1,0,0,0,1], [1,0,0,1,0], [1,1,1,0,0], [1,0,0,1,0], [1,0,0,0,1]],
        'L': [[1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0], [1,0,0,0,0], [1,1,1,1,1]],
        'M': [[1,0,0,0,1], [1,1,0,1,1], [1,0,1,0,1], [1,0,0,0,1], [1,0,0,0,1]],
        'N': [[1,0,0,0,1], [1,1,0,0,1], [1,0,1,0,1], [1,0,0,1,1], [1,0,0,0,1]],
        'O': [[0,1,1,1,0], [1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [0,1,1,1,0]],
        'P': [[1,1,1,1,0], [1,0,0,0,1], [1,1,1,1,0], [1,0,0,0,0], [1,0,0,0,0]],
        'Q': [[0,1,1,1,0], [1,0,0,0,1], [1,0,1,0,1], [1,0,0,1,0], [0,1,1,0,1]],
        'R': [[1,1,1,1,0], [1,0,0,0,1], [1,1,1,1,0], [1,0,0,1,0], [1,0,0,0,1]],
        'S': [[0,1,1,1,1], [1,0,0,0,0], [0,1,1,1,0], [0,0,0,0,1], [1,1,1,1,0]],
        'T': [[1,1,1,1,1], [0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0]],
        'U': [[1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [0,1,1,1,0]],
        'V': [[1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [0,1,0,1,0], [0,0,1,0,0]],
        'W': [[1,0,0,0,1], [1,0,0,0,1], [1,0,1,0,1], [1,1,0,1,1], [1,0,0,0,1]],
        'X': [[1,0,0,0,1], [0,1,0,1,0], [0,0,1,0,0], [0,1,0,1,0], [1,0,0,0,1]],
        'Y': [[1,0,0,0,1], [0,1,0,1,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0]],
        'Z': [[1,1,1,1,1], [0,0,0,1,0], [0,0,1,0,0], [0,1,0,0,0], [1,1,1,1,1]],
        
        # Numbers
        '0': [[0,1,1,1,0], [1,0,0,1,1], [1,0,1,0,1], [1,1,0,0,1], [0,1,1,1,0]],
        '1': [[0,0,1,0,0], [0,1,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,1,1,1,0]],
        '2': [[1,1,1,1,0], [0,0,0,0,1], [0,1,1,1,0], [1,0,0,0,0], [1,1,1,1,1]],
        '3': [[1,1,1,1,0], [0,0,0,0,1], [0,1,1,1,0], [0,0,0,0,1], [1,1,1,1,0]],
        '4': [[1,0,0,0,1], [1,0,0,0,1], [1,1,1,1,1], [0,0,0,0,1], [0,0,0,0,1]],
        '5': [[1,1,1,1,1], [1,0,0,0,0], [1,1,1,1,0], [0,0,0,0,1], [1,1,1,1,0]],
        '6': [[0,1,1,1,0], [1,0,0,0,0], [1,1,1,1,0], [1,0,0,0,1], [0,1,1,1,0]],
        '7': [[1,1,1,1,1], [0,0,0,0,1], [0,0,0,1,0], [0,0,1,0,0], [0,1,0,0,0]],
        '8': [[0,1,1,1,0], [1,0,0,0,1], [0,1,1,1,0], [1,0,0,0,1], [0,1,1,1,0]],
        '9': [[0,1,1,1,0], [1,0,0,0,1], [0,1,1,1,1], [0,0,0,0,1], [0,1,1,1,0]],
        
        # Special
        ' ': [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]],
        '.': [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,1,0,0]],
        '-': [[0,0,0,0,0], [0,0,0,0,0], [1,1,1,1,1], [0,0,0,0,0], [0,0,0,0,0]],
    }

    @staticmethod
    def _get_color(canvas, color):
        """
        Internal helper to match color format to canvas depth.

        Description:
        ------------
        Determines the appropriate color representation based on whether 
        the canvas is grayscale (2D) or RGB (3D).

        Parameters:
        -----------
        canvas : numpy.ndarray
            The image array to check for dimensionality.
        color : int, list, tuple, or numpy.ndarray
            The desired color value.

        Return Value:
        -------------
        numpy.ndarray or int
            A numpy array for RGB canvases or a scalar for grayscale canvases.

        Raised Exceptions:
        ------------------
        TypeError: If canvas is not a numpy.ndarray.
        ValueError: If canvas is not 2D or 3D.
        """
        if not isinstance(canvas, np.ndarray):
            raise TypeError(f"Canvas must be a numpy.ndarray, got {type(canvas)}.")
        
        if canvas.ndim == 3:
            if isinstance(color, (list, tuple, np.ndarray)):
                if len(color) != canvas.shape[2]:
                    raise ValueError(f"Color length {len(color)} must match canvas channels {canvas.shape[2]}.")
                return np.array(color)
            return np.array([color] * canvas.shape[2])
        elif canvas.ndim == 2:
            if isinstance(color, (list, tuple, np.ndarray)):
                return color[0] # Fallback for scalar canvas
            return color
        else:
            raise ValueError(f"Canvas must be 2D or 3D, got {canvas.ndim}D.")

    @staticmethod
    def put_text(canvas, text, x, y, font_scale=1, color=255):
        """
        Places text on the canvas at position (x, y).
        
        Description:
        ------------
        Iterates through the characters of the input string, retrieves their 
        bitmap representation from the internal font dictionary, scales them, 
        and draws them onto the provided canvas.

        Parameters:
        -----------
        canvas : numpy.ndarray
            The image array to draw on. Expected 2D (Grayscale) or 3D (RGB).
        text : str
            The string to be rendered. Non-supported characters default to space.
        x : int
            The horizontal (column) starting coordinate for the text.
        y : int
            The vertical (row) starting coordinate for the text.
        font_scale : int
            Multiplier for the font size (1 = 5x7 pixels). Must be >= 1.
        color : int or tuple or list or numpy.ndarray
            The color to use for the text pixels.

        Return Value:
        -------------
        None (The canvas is modified in-place).

        Raised Exceptions:
        ------------------
        TypeError: If canvas is not ndarray, text is not str, or x/y/scale are not int.
        ValueError: If font_scale < 1 or canvas dimensions are invalid.

        Notes on expected input ranges/dtypes:
        --------------------------------------
        - canvas: numpy.ndarray (uint8 recommended).
        - x, y: Can be outside bounds (clipping handled internally).
        - font_scale: Positive integers.
        """
        # Input Validation
        if not isinstance(canvas, np.ndarray):
            raise TypeError(f"Canvas must be a numpy.ndarray, got {type(canvas)}.")
        if not isinstance(text, str):
            raise TypeError(f"Text must be a string, got {type(text)}.")
        if not all(isinstance(param, int) for param in [x, y, font_scale]):
            raise TypeError("Coordinates (x, y) and font_scale must be integers.")
        if font_scale < 1:
            raise ValueError(f"font_scale must be >= 1, got {font_scale}.")
        if canvas.ndim not in [2, 3]:
            raise ValueError(f"Expected 2D or 3D canvas, got {canvas.ndim}D.")

        # Check if image is grayscale or rgb
        draw_color = Text._get_color(canvas, color)

        # Get the size of the image
        h, w = canvas.shape[:2]
        
        # Initialization
        current_x = x
        
        # Convert all characters to uppercase to match the letters at the bitmap _Font
        for char in text.upper():

            # if character is not in bitmap, add space
            if char not in Text._FONT:
                char = ' ' 
            
            # Check the array of the character in the bitmap _Font
            bitmap = np.array(Text._FONT[char])

            # The rows = 5 and columns = 7 as character size in bitmap _Font is 5*7
            rows, cols = bitmap.shape
            
            # Draw each "pixel" of the character
            for r in range(rows):
                for c in range(cols):
                    if bitmap[r, c] == 1: # This mean we need to color this pixel

                        # Scale the "pixel" based on font_scale
                        start_r = y + r * font_scale
                        start_c = current_x + c * font_scale
                        
                        # Drawing a square for each bitmap pixel to handle scaling
                        for i in range(start_r, start_r + font_scale): 
                            for j in range(start_c, start_c + font_scale):

                                if 0 <= i < h and 0 <= j < w: # checking if we are within the bounds of the canvas
                                    canvas[i, j] = draw_color # give the pixel a color
            
            # Move cursor for next character (5 pixels + 1 pixel spacing) * scale
            current_x += (cols + 1) * font_scale