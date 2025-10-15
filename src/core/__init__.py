"""核心模块初始化"""

from .config import Config
from .preprocessor import ImagePreprocessor
from .detector import TextDetector
from .recognizer import WheelRecognizer
from .postprocessor import PostProcessor

__all__ = [
    "Config",
    "ImagePreprocessor",
    "TextDetector",
    "WheelRecognizer",
    "PostProcessor",
]
