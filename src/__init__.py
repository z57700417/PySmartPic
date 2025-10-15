"""
汽车轮毂字母识别系统
Wheel Hub Letter Recognition System

一个专门用于识别手机拍摄的汽车轮毂照片中的字母和文字信息的系统。
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "汽车轮毂字母识别系统"

from .core.recognizer import WheelRecognizer
from .core.config import Config

__all__ = [
    "WheelRecognizer",
    "Config",
]
