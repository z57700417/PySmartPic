"""
增强识别脚本 - 专门处理难以识别的轮毂图片
针对小文字、低对比度、光照不均等问题进行优化
"""

import cv2
import numpy as np
from pathlib import Path
from src.core.config import Config
from src.core.recognizer import WheelRecognizer
from loguru import logger
import sys


class EnhancedImagePreprocessor:
    """增强版图像预处理器"""
    
    @staticmethod
    def preprocess_for_ocr(image_path, scale_factor=3.0):
        """
        针对OCR优化的预处理
        
        Args:
            image_path: 图片路径
            scale_factor: 放大倍数 (默认3倍,针对小文字)
            
        Returns:
            预处理后的图像
        """
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"无法读取图片: {image_path}")
            return None
            
        logger.info(f"原始图片尺寸: {image.shape[1]}x{image.shape[0]}")
        
        # 1. 图像放大 (针对小文字)
        if scale_factor > 1.0:
            new_width = int(image.shape[1] * scale_factor)
            new_height = int(image.shape[0] * scale_factor)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            logger.info(f"放大后尺寸: {image.shape[1]}x{image.shape[0]} (x{scale_factor})")
        
        # 2. 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 3. 去除光照不均 - 使用形态学顶帽变换
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        gray = cv2.add(gray, tophat)
        gray = cv2.subtract(gray, blackhat)
        logger.info("应用形态学顶帽变换去除光照不均")
        
        # 4. 对比度增强 - 使用CLAHE
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)
        logger.info("应用CLAHE增强对比度")
        
        # 5. 双边滤波去噪 (保留边缘)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        logger.info("应用双边滤波去噪")
        
        # 6. 锐化处理
        kernel_sharpen = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        gray = cv2.filter2D(gray, -1, kernel_sharpen)
        logger.info("应用锐化处理")
        
        # 7. 自适应阈值二值化
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 15, 3
        )
        logger.info("应用自适应阈值二值化")
        
        # 8. 形态学闭运算 (连接断裂的文字)
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_morph)
        logger.info("应用形态学闭运算")
        
        # 转回BGR格式供OCR使用
        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        return result


def recognize_with_enhancement(image_path, output_dir="./enhanced_output"):
    """
    使用增强预处理进行识别
    
    Args:
        image_path: 图片路径
        output_dir: 输出目录
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"开始处理图片: {image_path}")
    logger.info("=" * 60)
    
    # 增强预处理
    preprocessor = EnhancedImagePreprocessor()
    
    # 尝试不同的放大倍数
    scale_factors = [2.0, 3.0, 4.0]
    all_results = []
    
    for scale in scale_factors:
        logger.info(f"\n尝试放大倍数: {scale}x")
        logger.info("-" * 60)
        
        enhanced_image = preprocessor.preprocess_for_ocr(image_path, scale_factor=scale)
        
        if enhanced_image is None:
            continue
        
        # 保存预处理后的图片
        enhanced_path = output_dir / f"{image_path.stem}_enhanced_{scale}x.jpg"
        cv2.imwrite(str(enhanced_path), enhanced_image)
        logger.info(f"保存预处理图片: {enhanced_path}")
        
        # 创建配置 - 使用更激进的检测参数
        config = Config()
        config.set("preprocessing.enable", False)  # 已经手动预处理了
        config.set("detection.paddleocr.det_db_thresh", 0.1)  # 降低检测阈值
        config.set("detection.paddleocr.det_db_box_thresh", 0.2)  # 降低框阈值
        config.set("detection.paddleocr.det_db_unclip_ratio", 2.0)  # 增大扩展比例
        config.set("detection.paddleocr.max_side_len", 1920)  # 增大最大边长
        config.set("postprocessing.min_confidence", 0.3)  # 降低置信度阈值
        config.set("recognition.engine", "paddleocr")
        
        # 创建识别器
        recognizer = WheelRecognizer(config)
        
        # 识别
        result = recognizer.recognize(str(enhanced_path))
        
        if result.get("success") and result.get("total_texts", 0) > 0:
            logger.info(f"✅ 识别成功! 检测到 {result['total_texts']} 个文字")
            for item in result["results"]:
                logger.info(f"   文字: {item['text']}, 置信度: {item['confidence']:.2%}")
            
            # 保存可视化结果
            vis_path = output_dir / f"{image_path.stem}_result_{scale}x.jpg"
            recognizer.visualize(str(enhanced_path), result, str(vis_path))
            logger.info(f"保存可视化结果: {vis_path}")
            
            all_results.append({
                "scale": scale,
                "result": result,
                "enhanced_path": enhanced_path,
                "vis_path": vis_path
            })
        else:
            logger.warning(f"❌ 放大 {scale}x 未识别到文字")
    
    # 汇总结果
    logger.info("\n" + "=" * 60)
    logger.info("汇总所有识别结果:")
    logger.info("=" * 60)
    
    if all_results:
        # 找出识别文字最多的结果
        best_result = max(all_results, key=lambda x: x["result"]["total_texts"])
        
        logger.info(f"\n最佳结果 (放大 {best_result['scale']}x):")
        for item in best_result["result"]["results"]:
            logger.info(f"  ✓ {item['text']} (置信度: {item['confidence']:.2%})")
        
        # 收集所有不重复的文字
        all_texts = set()
        for r in all_results:
            for item in r["result"]["results"]:
                all_texts.add(item["text"])
        
        logger.info(f"\n所有识别到的文字 (去重): {' '.join(sorted(all_texts))}")
        
        return best_result
    else:
        logger.error("\n所有尝试都未能识别到文字!")
        logger.info("\n建议:")
        logger.info("  1. 检查图片质量是否过低")
        logger.info("  2. 尝试拍摄更清晰的照片")
        logger.info("  3. 确保光照充足且均匀")
        logger.info("  4. 尽量拍摄正面照片,减少角度")
        
        return None


def recognize_with_multiple_engines(image_path, output_dir="./enhanced_output"):
    """
    使用多个OCR引擎进行识别并融合结果
    
    Args:
        image_path: 图片路径
        output_dir: 输出目录
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 先进行增强预处理
    preprocessor = EnhancedImagePreprocessor()
    enhanced_image = preprocessor.preprocess_for_ocr(image_path, scale_factor=3.0)
    
    if enhanced_image is None:
        return None
    
    enhanced_path = output_dir / f"{image_path.stem}_enhanced.jpg"
    cv2.imwrite(str(enhanced_path), enhanced_image)
    
    # 尝试不同的OCR引擎
    engines = ["paddleocr", "easyocr"]
    all_results = {}
    
    for engine in engines:
        logger.info(f"\n使用 {engine} 引擎识别...")
        logger.info("-" * 60)
        
        try:
            config = Config()
            config.set("preprocessing.enable", False)
            config.set("recognition.engine", engine)
            config.set("postprocessing.min_confidence", 0.3)
            config.set("detection.paddleocr.det_db_thresh", 0.1)
            config.set("detection.paddleocr.det_db_box_thresh", 0.2)
            
            recognizer = WheelRecognizer(config)
            result = recognizer.recognize(str(enhanced_path))
            
            if result.get("success"):
                all_results[engine] = result
                logger.info(f"✅ {engine} 识别到 {result['total_texts']} 个文字")
                for item in result["results"]:
                    logger.info(f"   {item['text']} ({item['confidence']:.2%})")
            else:
                logger.warning(f"❌ {engine} 未识别到文字")
                
        except Exception as e:
            logger.error(f"❌ {engine} 识别失败: {e}")
    
    # 融合结果
    logger.info("\n" + "=" * 60)
    logger.info("融合所有引擎的识别结果:")
    
    all_texts = {}
    for engine, result in all_results.items():
        for item in result["results"]:
            text = item["text"]
            if text not in all_texts:
                all_texts[text] = []
            all_texts[text].append({
                "engine": engine,
                "confidence": item["confidence"]
            })
    
    # 按出现次数和置信度排序
    sorted_texts = sorted(
        all_texts.items(),
        key=lambda x: (len(x[1]), max(r["confidence"] for r in x[1])),
        reverse=True
    )
    
    logger.info("=" * 60)
    for text, sources in sorted_texts:
        engines_str = ", ".join([f"{s['engine']}({s['confidence']:.2%})" for s in sources])
        logger.info(f"  ✓ {text} - 来源: {engines_str}")
    
    return sorted_texts


if __name__ == "__main__":
    # 设置日志级别
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # 使用示例
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # 默认测试图片路径
        print("使用方法:")
        print("  python enhance_recognition.py <图片路径>")
        print("\n示例:")
        print("  python enhance_recognition.py wheel.jpg")
        print("  python enhance_recognition.py e:\\images\\wheel.jpg")
        sys.exit(1)
    
    # 方法1: 尝试不同的放大倍数
    logger.info("方法1: 尝试不同的图像放大倍数")
    logger.info("=" * 60)
    result1 = recognize_with_enhancement(image_path)
    
    # 方法2: 尝试多个OCR引擎
    logger.info("\n\n方法2: 尝试多个OCR引擎")
    logger.info("=" * 60)
    result2 = recognize_with_multiple_engines(image_path)
    
    logger.info("\n" + "=" * 60)
    logger.info("识别完成! 请查看 enhanced_output 目录中的结果")
    logger.info("=" * 60)
