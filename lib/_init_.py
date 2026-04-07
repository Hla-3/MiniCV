"""
MiniCV: A custom Image Processing Library for "CSE480s: Machine Vision" Course .
Faculty of Engineering, Ain Shams University - Spring 2026.
"""

# Import classes from their respective modules
from .io import IO
from .core import Core
from .image_processing import ImageProcessing
from .drawing import Drawing
from .text import Text
from .transformations import Transformations
from .feature_extractor import FeatureExtractor

# Define what is accessible when a user writes "from minicv import *"
__all__ = [
    'IO',
    'Core',
    'ImageProcessing',
    'Drawing',
    'Text',
    'Transformations',
    'FeatureExtractor'
]

# Package metadata
__version__ = "1.0.0"
__author__ = ["Hla Ehab" , "Arwa Ramadan"]