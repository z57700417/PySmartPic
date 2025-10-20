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
from .cloud_ocr import CloudOCREngine


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
        
        # 初始化云OCR引擎
        self.cloud_ocr = None
        self._init_cloud_ocr()
        
        # 初始化识别引擎
        self._init_recognition_engine()
        
    def _init_cloud_ocr(self):
        """初始化云OCR引擎"""
        try:
            cloud_config = self.config.get("recognition.cloud_ocr", {})
            if cloud_config.get("enabled", False):
                self.cloud_ocr = CloudOCREngine(cloud_config)
                logger.info(f"云OCR引擎初始化成功: {cloud_config.get('provider', 'aliyun')}")
            else:
                logger.info("云OCR引擎未启用")
        except Exception as e:
            logger.warning(f"云OCR引擎初始化失败(将继续使用本地OCR): {e}")
            self.cloud_ocr = None
    
    def _init_recognition_engine(self):
        """初始化识别引擎"""
        engine_type = self.config.get("recognition.engine", "auto")
        
        if engine_type == "auto":
            # 先尝试初始化PaddleOCR，如果成功则不再尝试EasyOCR
            self._init_paddleocr()
            
            # 如果PaddleOCR初始化失败，再尝试EasyOCR
            if self.paddle_engine is None:
                self._init_easyocr()
                
            # 如果两个引擎都初始化失败，给出警告
            if self.paddle_engine is None and self.easy_engine is None:
                logger.warning("所有OCR引擎初始化失败，请检查依赖安装")
                
            self.engine_name = "auto"
        elif engine_type == "paddleocr":
            self._init_paddleocr()
            if self.paddle_engine is None:
                logger.warning("PaddleOCR引擎初始化失败，请检查依赖安装")
            self.engine_name = "paddleocr"
        elif engine_type == "easyocr":
            self._init_easyocr()
            if self.easy_engine is None:
                logger.warning("EasyOCR引擎初始化失败，请检查依赖安装")
            self.engine_name = "easyocr"
        else:
            logger.error(f"不支持的识别引擎: {engine_type}")
            
    def _init_paddleocr(self):
        """初始化PaddleOCR引擎"""
        try:
            from paddleocr import PaddleOCR
            
            use_gpu = self.config.get("system.use_gpu", False)
            lang = self.config.get("recognition.paddleocr.lang", "en")
            
            # 获取检测参数
            det_db_thresh = self.config.get("detection.paddleocr.det_db_thresh", 0.3)
            det_db_box_thresh = self.config.get("detection.paddleocr.det_db_box_thresh", 0.5)
            det_db_unclip_ratio = self.config.get("detection.paddleocr.det_db_unclip_ratio", 1.6)
            
            self.paddle_engine = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                use_gpu=use_gpu,
                show_log=False,
                det_db_thresh=det_db_thresh,
                det_db_box_thresh=det_db_box_thresh,
                det_db_unclip_ratio=det_db_unclip_ratio
            )
            self.engine_name = "paddleocr"
            logger.info(f"PaddleOCR引擎初始化成功 (检测阈值: {det_db_thresh}, 框阈值: {det_db_box_thresh}, 展开比例: {det_db_unclip_ratio})")
        except Exception as e:
            logger.error(f"PaddleOCR引擎初始化失败: {e}")
            self._init_easyocr()
            
    def _init_easyocr(self):
        """初始化EasyOCR引擎"""
        try:
            # 先检查是否存在shm.dll文件，这是Windows上常见的问题
            import os
            import sys
            import torch
            
            # 获取torch库路径
            torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
            shm_path = os.path.join(torch_lib_path, 'shm.dll')
            
            if sys.platform == 'win32' and not os.path.exists(shm_path):
                logger.warning(f"Windows环境下缺少shm.dll文件，EasyOCR可能无法正常工作")
                logger.warning(f"请考虑重新安装PyTorch或使用PaddleOCR引擎")
                self.easy_engine = None
                return
                
            import easyocr
            
            use_gpu = self.config.get("system.use_gpu", False)
            lang_list = self.config.get("recognition.easyocr.lang_list", ["en"])
            
            self.easy_engine = easyocr.Reader(lang_list, gpu=use_gpu)
            self.engine_name = "easyocr"
            logger.info("EasyOCR引擎初始化成功")
        except ImportError as ie:
            logger.error(f"EasyOCR库导入失败: {ie}")
            logger.warning("请安装EasyOCR: pip install easyocr")
            self.easy_engine = None
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
        
        # 智能切换云OCR
        raw_results = self._apply_cloud_ocr_fallback(raw_results, preprocessed)

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
        
        # 按行分组
        grouped_lines = self._group_by_lines(final_results)
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        # 构建返回结果
        return {
            "success": True,
            "image_path": image_path_str,
            "total_texts": len(final_results),
            "total_lines": len(grouped_lines),
            "results": final_results,  # 保留原始结果
            "lines": grouped_lines,     # 按行分组的结果
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
    
    def _group_by_lines(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将识别结果按行分组
        
        Args:
            results: 识别结果列表
            
        Returns:
            按行分组的结果列表
        """
        if not results:
            return []
        
        # 按y坐标排序
        sorted_results = sorted(results, key=lambda x: self._get_y_center(x.get('bbox', [])))
        
        # 分组阈值:y坐标差距小于此值认为是同一行
        y_threshold = 50  # 可以根据实际情况调整
        
        lines = []
        current_line = []
        current_y = None
        
        for item in sorted_results:
            y_center = self._get_y_center(item.get('bbox', []))
            
            if current_y is None:
                # 第一个元素
                current_line = [item]
                current_y = y_center
            elif abs(y_center - current_y) <= y_threshold:
                # 同一行
                current_line.append(item)
            else:
                # 新的一行
                if current_line:
                    # 保存当前行
                    lines.append(self._merge_line_results(current_line))
                # 开始新行
                current_line = [item]
                current_y = y_center
        
        # 保存最后一行
        if current_line:
            lines.append(self._merge_line_results(current_line))
        
        logger.info(f"识别结果分为 {len(lines)} 行")
        return lines
    
    def _get_y_center(self, bbox: List) -> float:
        """
        获取边界框的y中心坐标
        
        Args:
            bbox: 边界框坐标
            
        Returns:
            y中心坐标
        """
        if not bbox or len(bbox) == 0:
            return 0.0
        
        try:
            # bbox格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            y_coords = [point[1] for point in bbox]
            return sum(y_coords) / len(y_coords)
        except (IndexError, TypeError):
            return 0.0
    
    def _get_x_center(self, bbox: List) -> float:
        """
        获取边界框的x中心坐标
        
        Args:
            bbox: 边界框坐标
            
        Returns:
            x中心坐标
        """
        if not bbox or len(bbox) == 0:
            return 0.0
        
        try:
            # bbox格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            x_coords = [point[0] for point in bbox]
            return sum(x_coords) / len(x_coords)
        except (IndexError, TypeError):
            return 0.0
    
    def _merge_line_results(self, line_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        合并同一行的识别结果
        
        Args:
            line_items: 同一行的识别结果列表
            
        Returns:
            合并后的行结果
        """
        if not line_items:
            return {
                "text": "",
                "confidence": 0.0,
                "item_count": 0,
                "items": []
            }
        
        # 按x坐标排序(从左到右)
        sorted_items = sorted(line_items, key=lambda x: self._get_x_center(x.get('bbox', [])))
        
        # 合并文本
        merged_text = ' '.join([item.get('text', '') for item in sorted_items])
        
        # 计算平均置信度
        confidences = [item.get('confidence', 0) for item in sorted_items]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            "text": merged_text,
            "confidence": avg_confidence,
            "item_count": len(sorted_items),
            "items": sorted_items
        }
    
    def _apply_cloud_ocr_fallback(self, local_results: List[Dict[str, Any]], 
                                   image: np.ndarray) -> List[Dict[str, Any]]:
        """
        智能云OCR备用策略
        
        Args:
            local_results: 本地OCR识别结果
            image: 预处理后的图像
            
        Returns:
            优化后的识别结果
        """
        # 如果云OCR未启用,直接返回本地结果
        if self.cloud_ocr is None:
            return local_results
        
        cloud_config = self.config.get("recognition.cloud_ocr", {})
        
        # 检查是否需要使用云OCR
        use_cloud = False
        reason = ""
        
        if cloud_config.get("use_as_fallback", True):
            # 策略1: 识别结果数量少
            min_results = cloud_config.get("fallback_min_results", 2)
            if len(local_results) < min_results:
                use_cloud = True
                reason = f"本地识别结果少于{min_results}个"
            
            # 策略2: 置信度过低
            if not use_cloud and local_results:
                threshold = cloud_config.get("fallback_confidence_threshold", 0.7)
                avg_confidence = sum(r.get('confidence', 0) for r in local_results) / len(local_results)
                if avg_confidence < threshold:
                    use_cloud = True
                    reason = f"平均置信度{avg_confidence:.2f}低于阈值{threshold}"
        
        # 如果需要使用云OCR
        if use_cloud:
            logger.info(f"触发云OCR备用策略: {reason}")
            
            try:
                # 保存临时图片
                import tempfile
                import os
                temp_path = os.path.join(tempfile.gettempdir(), f"cloud_ocr_{int(time.time())}.jpg")
                cv2.imwrite(temp_path, image)
                
                # 调用云OCR
                cloud_results = self.cloud_ocr.recognize(temp_path)
                
                # 清理临时文件
                try:
                    os.remove(temp_path)
                except:
                    pass
                
                # 对比云OCR和本地OCR结果
                if cloud_results and len(cloud_results) > len(local_results):
                    logger.info(f"云OCR识别效果更好: 本地{len(local_results)}个 vs 云端{len(cloud_results)}个")
                    # 标记结果来源
                    for result in cloud_results:
                        result['source'] = 'cloud_ocr'
                    return cloud_results
                else:
                    logger.info(f"本地OCR结果更优或相当,继续使用本地结果")
                    for result in local_results:
                        result['source'] = 'local_ocr'
                    return local_results
                    
            except Exception as e:
                logger.error(f"云OCR调用失败,使用本地结果: {e}")
                for result in local_results:
                    result['source'] = 'local_ocr'
                return local_results
        
        # 不需要云OCR,返回本地结果
        for result in local_results:
            result['source'] = 'local_ocr'
        return local_results
