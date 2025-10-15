"""
文字检测模块
集成PaddleOCR文字检测功能
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from loguru import logger


class TextDetector:
    """文字检测器"""
    
    def __init__(self, config: dict):
        """
        初始化文字检测器
        
        Args:
            config: 检测配置字典
        """
        self.config = config
        self.engine = config.get("engine", "paddleocr")
        self.detector = None
        self._init_detector()
        
    def _init_detector(self):
        """初始化检测引擎"""
        if self.engine == "paddleocr":
            self._init_paddleocr()
        else:
            logger.error(f"不支持的检测引擎: {self.engine}")
            
    def _init_paddleocr(self):
        """初始化PaddleOCR检测器"""
        try:
            from paddleocr import PaddleOCR
            
            paddleocr_config = self.config.get("paddleocr", {})
            
            self.detector = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=self.config.get("use_gpu", False),
                det_db_thresh=paddleocr_config.get("det_db_thresh", 0.3),
                det_db_box_thresh=paddleocr_config.get("det_db_box_thresh", 0.5),
                det_db_unclip_ratio=paddleocr_config.get("det_db_unclip_ratio", 1.6),
                max_side_len=paddleocr_config.get("max_side_len", 960),
                use_dilation=paddleocr_config.get("use_dilation", True),
                show_log=False
            )
            logger.info("PaddleOCR检测器初始化成功")
        except Exception as e:
            logger.error(f"PaddleOCR检测器初始化失败: {e}")
            self.detector = None
            
    def detect(self, image: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """
        检测图像中的文字区域
        
        Args:
            image: 输入图像(BGR格式)
            
        Returns:
            检测结果列表,每个元素为(边界框坐标, 置信度)
        """
        if self.detector is None:
            logger.error("检测器未初始化")
            return []
            
        if self.engine == "paddleocr":
            return self._detect_paddleocr(image)
        else:
            return []
            
    def _detect_paddleocr(self, image: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """
        使用PaddleOCR检测文字区域
        
        Args:
            image: 输入图像
            
        Returns:
            检测结果列表
        """
        try:
            # PaddleOCR检测
            result = self.detector.ocr(image, rec=False)
            
            if result is None or len(result) == 0 or result[0] is None:
                logger.warning("未检测到文字区域")
                return []
                
            # 提取检测框
            detections = []
            for line in result[0]:
                bbox = np.array(line).astype(np.float32)
                # PaddleOCR返回的是4个点的坐标
                # 计算置信度(这里简单设为1.0,因为PaddleOCR检测结果已经过滤)
                confidence = 1.0
                detections.append((bbox, confidence))
                
            logger.info(f"检测到 {len(detections)} 个文字区域")
            return detections
            
        except Exception as e:
            logger.error(f"文字检测失败: {e}")
            return []
            
    def visualize_detections(self, image: np.ndarray, detections: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image: 原始图像
            detections: 检测结果列表
            
        Returns:
            标注后的图像
        """
        result = image.copy()
        
        for bbox, confidence in detections:
            # 绘制边界框
            pts = bbox.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(result, [pts], True, (0, 255, 0), 2)
            
            # 添加置信度标签
            x, y = int(bbox[0][0]), int(bbox[0][1])
            cv2.putText(result, f"{confidence:.2f}", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                       
        return result
