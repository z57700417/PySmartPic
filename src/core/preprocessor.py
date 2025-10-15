"""
图像预处理模块
实现图像增强、去噪、二值化等预处理功能
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from loguru import logger


class ImagePreprocessor:
    """图像预处理器"""
    
    def __init__(self, config: dict):
        """
        初始化预处理器
        
        Args:
            config: 预处理配置字典
        """
        self.config = config
        self.enable = config.get("enable", True)
        
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        执行完整的预处理流程
        
        Args:
            image: 输入图像(BGR格式)
            
        Returns:
            预处理后的图像
        """
        if not self.enable:
            return image
            
        result = image.copy()
        
        # 亮度对比度调整
        if self.config.get("brightness_contrast", {}).get("enable", True):
            result = self._adjust_brightness_contrast(result)
            
        # 去噪处理
        if self.config.get("denoise", {}).get("enable", True):
            result = self._denoise(result)
            
        # 边缘增强
        if self.config.get("edge_enhancement", {}).get("enable", False):
            result = self._enhance_edges(result)
            
        # 二值化处理
        if self.config.get("binarization", {}).get("enable", False):
            result = self._binarize(result)
            
        # 形态学处理
        if self.config.get("morphology", {}).get("enable", False):
            result = self._morphology(result)
            
        return result
        
    def _adjust_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        调整亮度和对比度(使用CLAHE)
        
        Args:
            image: 输入图像
            
        Returns:
            调整后的图像
        """
        config = self.config.get("brightness_contrast", {})
        clip_limit = config.get("clip_limit", 2.0)
        tile_size = tuple(config.get("tile_size", [8, 8]))
        
        # 转换到LAB色彩空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 对L通道应用CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        l = clahe.apply(l)
        
        # 合并通道
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        logger.debug("应用亮度对比度调整(CLAHE)")
        return result
        
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """
        去噪处理
        
        Args:
            image: 输入图像
            
        Returns:
            去噪后的图像
        """
        config = self.config.get("denoise", {})
        method = config.get("method", "bilateral")
        kernel_size = config.get("kernel_size", 5)
        sigma = config.get("sigma", 1.5)
        
        if method == "gaussian":
            result = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        elif method == "median":
            result = cv2.medianBlur(image, kernel_size)
        elif method == "bilateral":
            result = cv2.bilateralFilter(image, kernel_size, sigma * 20, sigma * 20)
        else:
            logger.warning(f"未知的去噪方法: {method}, 跳过去噪")
            result = image
            
        logger.debug(f"应用去噪处理: {method}")
        return result
        
    def _enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """
        边缘增强
        
        Args:
            image: 输入图像
            
        Returns:
            边缘增强后的图像
        """
        config = self.config.get("edge_enhancement", {})
        method = config.get("method", "sobel")
        kernel_size = config.get("kernel_size", 3)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if method == "sobel":
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)
            edges = np.sqrt(sobelx**2 + sobely**2)
        elif method == "laplacian":
            edges = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size)
        else:
            logger.warning(f"未知的边缘检测方法: {method}, 跳过边缘增强")
            return image
            
        # 归一化并叠加到原图
        edges = np.uint8(np.absolute(edges))
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)
        result = cv2.addWeighted(image, 0.7, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.3, 0)
        
        logger.debug(f"应用边缘增强: {method}")
        return result
        
    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """
        二值化处理
        
        Args:
            image: 输入图像
            
        Returns:
            二值化后的图像
        """
        config = self.config.get("binarization", {})
        method = config.get("method", "adaptive")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if method == "adaptive":
            block_size = config.get("block_size", 11)
            c_value = config.get("c_value", 2)
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, block_size, c_value
            )
        elif method == "otsu":
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            logger.warning(f"未知的二值化方法: {method}, 跳过二值化")
            return image
            
        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        logger.debug(f"应用二值化处理: {method}")
        return result
        
    def _morphology(self, image: np.ndarray) -> np.ndarray:
        """
        形态学处理
        
        Args:
            image: 输入图像
            
        Returns:
            形态学处理后的图像
        """
        config = self.config.get("morphology", {})
        operation = config.get("operation", "close")
        kernel_size = tuple(config.get("kernel_size", [3, 3]))
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        
        if operation == "dilate":
            result = cv2.dilate(image, kernel, iterations=1)
        elif operation == "erode":
            result = cv2.erode(image, kernel, iterations=1)
        elif operation == "open":
            result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operation == "close":
            result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        else:
            logger.warning(f"未知的形态学操作: {operation}, 跳过形态学处理")
            result = image
            
        logger.debug(f"应用形态学处理: {operation}")
        return result
