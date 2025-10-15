"""
主识别器模块
整合预处理、检测、识别、后处理等功能
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from loguru import logger

from .config import Config
from .preprocessor import ImagePreprocessor
from .detector import TextDetector
from .postprocessor import PostProcessor


class WheelRecognizer:
    """轮毂字母识别器"""
    
    def __init__(self, config: Optional[Union[str, Config]] = None):
        """
        初始化识别器
        
        Args:
            config: 配置对象或配置文件路径
        """
        # 加载配置
        if isinstance(config, Config):
            self.config = config
        else:
            self.config = Config(config)
            
        # 初始化各个模块
        self.preprocessor = ImagePreprocessor(self.config["preprocessing"])
        self.detector = None
        self.ocr_engine = None  # legacy placeholder
        self.paddle_engine = None
        self.easy_engine = None
        self.engine_name = None
        self.postprocessor = PostProcessor(self.config["postprocessing"])
        
        # 初始化识别引擎
        self._init_recognition_engine()
        
    def _init_recognition_engine(self):
        """初始化识别引擎"""
        engine_type = self.config.get("recognition.engine", "auto")
        
        if engine_type == "auto":
            # 同时尝试初始化两个引擎
            self._init_paddleocr()
            self._init_easyocr()
            self.engine_name = "auto"
        elif engine_type == "paddleocr":
            self._init_paddleocr()
            self.engine_name = "paddleocr"
        elif engine_type == "easyocr":
            self._init_easyocr()
            self.engine_name = "easyocr"
        else:
            logger.error(f"不支持的识别引擎: {engine_type}")
            
    def _init_paddleocr(self):
        """初始化PaddleOCR引擎"""
        try:
            from paddleocr import PaddleOCR
            
            use_gpu = self.config.get("system.use_gpu", False)
            lang = self.config.get("recognition.paddleocr.lang", "en")
            
            self.paddle_engine = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                use_gpu=use_gpu,
                show_log=False
            )
            self.engine_name = "paddleocr"
            logger.info("PaddleOCR引擎初始化成功")
        except Exception as e:
            logger.error(f"PaddleOCR引擎初始化失败: {e}")
            self._init_easyocr()
            
    def _init_easyocr(self):
        """初始化EasyOCR引擎"""
        try:
            import easyocr
            
            use_gpu = self.config.get("system.use_gpu", False)
            lang_list = self.config.get("recognition.easyocr.lang_list", ["en"])
            
            self.easy_engine = easyocr.Reader(lang_list, gpu=use_gpu)
            self.engine_name = "easyocr"
            logger.info("EasyOCR引擎初始化成功")
        except Exception as e:
            logger.error(f"EasyOCR引擎初始化失败: {e}")
            self.easy_engine = None
            
    def recognize(self, image_path: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
        """
        识别单张图片
        
        Args:
            image_path: 图片路径或图片数组
            
        Returns:
            识别结果字典
        """
        start_time = time.time()
        
        # 读取图片
        if isinstance(image_path, np.ndarray):
            image = image_path
            image_path_str = "numpy_array"
        else:
            image = self._load_image(image_path)
            image_path_str = str(image_path)
            
        if image is None:
            return {
                "success": False,
                "error": "图片加载失败",
                "image_path": image_path_str
            }
            
        # 预处理
        preprocessed = self.preprocessor.process(image)
        
        # OCR识别
        raw_results = self._recognize_with_engine(preprocessed)

        # 如果引擎未正确初始化，直接返回失败
        no_engine = (
            (self.engine_name == "paddleocr" and self.paddle_engine is None) or
            (self.engine_name == "easyocr" and self.easy_engine is None) or
            (self.engine_name == "auto" and self.paddle_engine is None and self.easy_engine is None)
        )
        if no_engine:
            processing_time = time.time() - start_time
            return {
                "success": False,
                "error": "OCR引擎未初始化或模型不可用",
                "image_path": image_path_str,
                "processing_time": processing_time,
                "engine_used": self.engine_name or "none"
            }
        
        # 后处理
        final_results = self.postprocessor.process(raw_results)
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        # 构建返回结果
        return {
            "success": True,
            "image_path": image_path_str,
            "total_texts": len(final_results),
            "results": final_results,
            "processing_time": processing_time,
            "engine_used": self.engine_name or "unknown"
        }
        
    def _load_image(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        加载图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            图片数组,如果失败返回None
        """
        try:
            path = Path(image_path)
            if not path.exists():
                logger.error(f"图片文件不存在: {image_path}")
                return None
                
            image = cv2.imread(str(path))
            if image is None:
                logger.error(f"图片读取失败: {image_path}")
                return None
                
            logger.info(f"成功加载图片: {image_path} (尺寸: {image.shape})")
            return image
        except Exception as e:
            logger.error(f"加载图片时出错: {e}")
            return None
            
    def _recognize_with_engine(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        使用OCR引擎识别
        
        Args:
            image: 输入图像
            
        Returns:
            识别结果列表
        """
        if self.engine_name == "paddleocr":
            if self.paddle_engine is None:
                logger.error("PaddleOCR引擎未初始化")
                return []
            return self._recognize_paddleocr(image)
        elif self.engine_name == "easyocr":
            if self.easy_engine is None:
                logger.error("EasyOCR引擎未初始化")
                return []
            return self._recognize_easyocr(image)
        elif self.engine_name == "auto":
            combined = []
            if self.paddle_engine is not None:
                combined.extend(self._recognize_paddleocr(image))
            if self.easy_engine is not None:
                combined.extend(self._recognize_easyocr(image))
            return combined
        else:
            return []
            
    def _recognize_paddleocr(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        使用PaddleOCR识别
        
        Args:
            image: 输入图像
            
        Returns:
            识别结果列表
        """
        try:
            result = self.paddle_engine.ocr(image, cls=True)
            
            if result is None or len(result) == 0 or result[0] is None:
                logger.warning("PaddleOCR未识别到文字")
                return []
                
            results = []
            lines = result if isinstance(result, list) else []
            for line in lines:
                try:
                    bbox = line[0]
                    # PaddleOCR: line[1] == (text, confidence)
                    if isinstance(line[1], (list, tuple)) and len(line[1]) >= 2:
                        text = line[1][0]
                        confidence = float(line[1][1])
                    elif len(line) >= 3:
                        text = str(line[1])
                        confidence = float(line[2])
                    else:
                        text = str(line[1])
                        confidence = 0.0
                    # 置信度归一化到[0,1]
                    if confidence < 0:
                        confidence = 0.0
                    if confidence > 1:
                        confidence = 1.0
                    results.append({
                        "text": text,
                        "confidence": confidence,
                        "bbox": bbox,
                        "orientation": "horizontal"
                    })
                except Exception as ex:
                    logger.debug(f"解析PaddleOCR结果行失败: {ex}")
                
            logger.info(f"PaddleOCR识别到 {len(results)} 个文字区域")
            return results
            
        except Exception as e:
            logger.error(f"PaddleOCR识别失败: {e}")
            return []
            
    def _recognize_easyocr(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        使用EasyOCR识别
        
        Args:
            image: 输入图像
            
        Returns:
            识别结果列表
        """
        try:
            allowlist = self.config.get("recognition.easyocr.allowlist", None)
            
            result = self.easy_engine.readtext(
                image,
                allowlist=allowlist,
                paragraph=False
            )
            
            if not result:
                logger.warning("EasyOCR未识别到文字")
                return []
                
            results = []
            for bbox, text, confidence in result:
                results.append({
                    "text": text,
                    "confidence": confidence,
                    "bbox": bbox,
                    "orientation": "horizontal"
                })
                
            logger.info(f"EasyOCR识别到 {len(results)} 个文字区域")
            return results
            
        except Exception as e:
            logger.error(f"EasyOCR识别失败: {e}")
            return []
            
    def recognize_batch(self, image_paths: List[Union[str, Path]], 
                       parallel: bool = True) -> List[Dict[str, Any]]:
        """
        批量识别图片
        
        Args:
            image_paths: 图片路径列表
            parallel: 是否并行处理
            
        Returns:
            识别结果列表
        """
        results = []
        
        if parallel and len(image_paths) > 1:
            # 并行处理
            from concurrent.futures import ThreadPoolExecutor
            num_workers = self.config.get("system.num_workers", 4)
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(self.recognize, image_paths))
        else:
            # 串行处理
            for image_path in image_paths:
                result = self.recognize(image_path)
                results.append(result)
                
        logger.info(f"批量识别完成: 共 {len(image_paths)} 张图片")
        return results
        
    def visualize(self, image_path: Union[str, Path, np.ndarray], 
                 recognition_result: Optional[Dict[str, Any]] = None,
                 output_path: Optional[Union[str, Path]] = None) -> np.ndarray:
        """
        可视化识别结果
        
        Args:
            image_path: 图片路径或图片数组
            recognition_result: 识别结果,如果为None则重新识别
            output_path: 输出路径,如果指定则保存图片
            
        Returns:
            标注后的图像
        """
        # 读取图片
        if isinstance(image_path, np.ndarray):
            image = image_path
        else:
            image = self._load_image(image_path)
            
        if image is None:
            logger.error("无法加载图片进行可视化")
            return None
            
        # 如果未提供识别结果,执行识别
        if recognition_result is None:
            recognition_result = self.recognize(image)
            
        # 绘制结果
        vis_config = self.config.get("visualization", {})
        result_image = image.copy()
        
        for item in recognition_result.get("results", []):
            bbox = item.get("bbox", [])
            text = item.get("text", "")
            confidence = item.get("confidence", 0)
            
            if not bbox:
                continue
                
            # 绘制边界框
            if vis_config.get("draw_bbox", True):
                pts = np.array(bbox, dtype=np.int32).reshape((-1, 1, 2))
                color = tuple(vis_config.get("bbox_color", [0, 255, 0]))
                thickness = vis_config.get("thickness", 2)
                cv2.polylines(result_image, [pts], True, color, thickness)
                
            # 绘制文字
            if vis_config.get("draw_text", True):
                x = int(bbox[0][0])
                y = int(bbox[0][1]) - 10
                font_scale = vis_config.get("font_scale", 0.6)
                text_color = tuple(vis_config.get("text_color", [255, 0, 0]))
                thickness = vis_config.get("thickness", 2)
                
                display_text = text
                if vis_config.get("draw_confidence", True):
                    display_text = f"{text} ({confidence:.2f})"
                    
                cv2.putText(result_image, display_text, (x, y),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
                           
        # 保存图片
        if output_path is not None:
            cv2.imwrite(str(output_path), result_image)
            logger.info(f"可视化结果已保存到: {output_path}")
            
        return result_image
