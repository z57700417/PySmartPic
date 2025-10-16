"""
识别结果纠正脚本
专门处理OCR常见的字符混淆问题
"""

import re
from typing import List, Dict, Tuple
from loguru import logger


class RecognitionCorrector:
    """识别结果纠正器"""
    
    # 常见的字符混淆规则 (基于轮毂文字特征)
    CONFUSION_RULES = {
        # 数字混淆
        '0': ['O', 'Q', 'D'],
        'O': ['0', 'Q', 'D'],
        '1': ['I', 'l', '|', 'i'],
        'I': ['1', 'l', '|'],
        '2': ['Z', '3'],
        '3': ['8', '2'],
        '4': ['A', '6'],
        '5': ['S', '6'],
        '6': ['G', '8', '5', '4', '9'],
        '7': ['T', '1', '6'],
        '8': ['B', '3', '6'],
        '9': ['g', 'q', '6'],
        
        # 字母混淆
        'B': ['8', 'R'],
        'G': ['6', 'C'],
        'S': ['5', '8'],
        'Z': ['2', '7'],
        'T': ['7', '1'],
        'A': ['4'],
    }
    
    # 轮毂编号的常见模式
    WHEEL_PATTERNS = [
        r'^[A-Z]{2}\d{5}$',  # AT66202 格式: 2个字母+5个数字
        r'^[A-Z]{3}\d{4}$',  # 3个字母+4个数字
        r'^\d{4}[A-Z]\d$',   # 4个数字+1个字母+1个数字 (0909W1D)
        r'^\d{4}[A-Z]{2}\d$', # 4个数字+2个字母+1个数字
    ]
    
    def __init__(self):
        self.correction_history = []
    
    def correct_text(self, text: str, confidence: float = 1.0) -> List[Dict]:
        """
        纠正识别文本
        
        Args:
            text: 原始识别文本
            confidence: 原始置信度
            
        Returns:
            可能的纠正结果列表,每个包含 text, confidence, corrections
        """
        results = []
        
        # 原始结果
        results.append({
            'text': text,
            'confidence': confidence,
            'corrections': [],
            'pattern_match': self._check_pattern(text)
        })
        
        # 如果原始结果已经匹配模式,优先级最高
        if results[0]['pattern_match']:
            results[0]['confidence'] *= 1.2  # 提高置信度
            logger.info(f"✓ '{text}' 匹配轮毂编号模式")
            return results
        
        # 生成纠正候选
        candidates = self._generate_candidates(text)
        
        for candidate, corrections in candidates:
            pattern_match = self._check_pattern(candidate)
            
            # 计算新的置信度
            new_confidence = confidence
            
            # 如果匹配模式,大幅提高置信度
            if pattern_match:
                new_confidence *= 1.5
                logger.info(f"✓ 纠正建议: '{text}' → '{candidate}' (匹配模式)")
            else:
                # 根据修改数量降低置信度
                new_confidence *= (0.9 ** len(corrections))
            
            results.append({
                'text': candidate,
                'confidence': new_confidence,
                'corrections': corrections,
                'pattern_match': pattern_match
            })
        
        # 按置信度和模式匹配排序
        results.sort(key=lambda x: (x['pattern_match'], x['confidence']), reverse=True)
        
        return results[:5]  # 返回前5个最佳候选
    
    def _check_pattern(self, text: str) -> bool:
        """检查文本是否匹配轮毂编号模式"""
        for pattern in self.WHEEL_PATTERNS:
            if re.match(pattern, text):
                return True
        return False
    
    def _generate_candidates(self, text: str, max_changes: int = 3) -> List[Tuple[str, List[str]]]:
        """
        生成纠正候选
        
        Args:
            text: 原始文本
            max_changes: 最大修改字符数
            
        Returns:
            (候选文本, 修改列表) 的列表
        """
        candidates = []
        
        # 单字符替换
        for i, char in enumerate(text):
            if char in self.CONFUSION_RULES:
                for replacement in self.CONFUSION_RULES[char]:
                    candidate = text[:i] + replacement + text[i+1:]
                    correction = f"位置{i}: '{char}' → '{replacement}'"
                    candidates.append((candidate, [correction]))
        
        # 双字符替换 (如果原文本较长)
        if len(text) >= 5 and max_changes >= 2:
            for i in range(len(text)):
                if text[i] not in self.CONFUSION_RULES:
                    continue
                for replacement1 in self.CONFUSION_RULES[text[i]]:
                    temp = text[:i] + replacement1 + text[i+1:]
                    for j in range(i+1, len(text)):
                        if text[j] not in self.CONFUSION_RULES:
                            continue
                        for replacement2 in self.CONFUSION_RULES[text[j]]:
                            candidate = temp[:j] + replacement2 + temp[j+1:]
                            corrections = [
                                f"位置{i}: '{text[i]}' → '{replacement1}'",
                                f"位置{j}: '{text[j]}' → '{replacement2}'"
                            ]
                            candidates.append((candidate, corrections))
        
        return candidates
    
    def batch_correct(self, results: List[Dict]) -> List[Dict]:
        """
        批量纠正识别结果
        
        Args:
            results: 原始识别结果列表,每个包含 text 和 confidence
            
        Returns:
            纠正后的结果列表
        """
        corrected_results = []
        
        for result in results:
            text = result.get('text', '')
            confidence = result.get('confidence', 0.0)
            
            # 获取纠正候选
            candidates = self.correct_text(text, confidence)
            
            # 使用最佳候选
            best = candidates[0]
            
            corrected_result = result.copy()
            corrected_result['original_text'] = text
            corrected_result['text'] = best['text']
            corrected_result['confidence'] = best['confidence']
            corrected_result['pattern_match'] = best['pattern_match']
            
            if best['corrections']:
                corrected_result['corrections'] = best['corrections']
                corrected_result['alternatives'] = [c['text'] for c in candidates[1:3]]
            
            corrected_results.append(corrected_result)
        
        return corrected_results


def correct_recognition_result(image_path: str, use_enhancement: bool = True) -> Dict:
    """
    识别并纠正结果
    
    Args:
        image_path: 图片路径
        use_enhancement: 是否使用图像增强
        
    Returns:
        纠正后的识别结果
    """
    from src.core.config import Config
    from src.core.recognizer import WheelRecognizer
    from pathlib import Path
    
    logger.info(f"开始识别: {image_path}")
    
    # 如果启用增强
    if use_enhancement:
        from enhance_recognition import EnhancedImagePreprocessor
        import cv2
        import tempfile
        
        preprocessor = EnhancedImagePreprocessor()
        enhanced_image = preprocessor.preprocess_for_ocr(image_path, scale_factor=3.0)
        
        if enhanced_image is not None:
            # 保存临时文件
            temp_path = Path(tempfile.gettempdir()) / f"enhanced_{Path(image_path).name}"
            cv2.imwrite(str(temp_path), enhanced_image)
            image_path = str(temp_path)
            logger.info("已应用图像增强")
    
    # 创建识别器
    config = Config()
    config.set("preprocessing.enable", False)
    config.set("detection.paddleocr.det_db_thresh", 0.1)
    config.set("detection.paddleocr.det_db_box_thresh", 0.2)
    config.set("postprocessing.min_confidence", 0.3)
    
    recognizer = WheelRecognizer(config)
    
    # 识别
    result = recognizer.recognize(image_path)
    
    if not result.get("success") or not result.get("results"):
        logger.warning("识别失败或无结果")
        return result
    
    # 纠正结果
    corrector = RecognitionCorrector()
    corrected = corrector.batch_correct(result["results"])
    
    # 更新结果
    result["results"] = corrected
    result["corrected"] = True
    
    # 输出对比
    logger.info("\n" + "=" * 60)
    logger.info("识别结果对比:")
    logger.info("=" * 60)
    
    for item in corrected:
        original = item.get('original_text', item['text'])
        current = item['text']
        confidence = item['confidence']
        pattern_match = item.get('pattern_match', False)
        
        if original != current:
            logger.info(f"❌ 原始: {original}")
            logger.info(f"✅ 纠正: {current} (置信度: {confidence:.2%}) {'✓ 匹配模式' if pattern_match else ''}")
            if 'corrections' in item:
                for correction in item['corrections']:
                    logger.info(f"   - {correction}")
            if 'alternatives' in item:
                logger.info(f"   备选: {', '.join(item['alternatives'])}")
        else:
            logger.info(f"✓ {current} (置信度: {confidence:.2%}) {'✓ 匹配模式' if pattern_match else ''}")
        logger.info("-" * 60)
    
    return result


def interactive_correction():
    """交互式纠正"""
    print("=" * 60)
    print("轮毂识别结果纠正工具")
    print("=" * 60)
    
    while True:
        print("\n请选择操作:")
        print("1. 输入图片路径进行识别和纠正")
        print("2. 直接输入识别结果进行纠正")
        print("3. 退出")
        
        choice = input("\n选择 (1/2/3): ").strip()
        
        if choice == '1':
            image_path = input("请输入图片路径: ").strip()
            use_enhancement = input("是否使用图像增强? (y/n, 默认y): ").strip().lower() != 'n'
            
            result = correct_recognition_result(image_path, use_enhancement)
            
            print("\n最终结果:")
            for item in result.get("results", []):
                print(f"  {item['text']} (置信度: {item['confidence']:.2%})")
        
        elif choice == '2':
            text = input("请输入识别的文字: ").strip()
            confidence = float(input("置信度 (0-1, 默认0.8): ").strip() or "0.8")
            
            corrector = RecognitionCorrector()
            candidates = corrector.correct_text(text, confidence)
            
            print("\n纠正建议:")
            for i, candidate in enumerate(candidates, 1):
                match_str = "✓ 匹配模式" if candidate['pattern_match'] else ""
                print(f"{i}. {candidate['text']} (置信度: {candidate['confidence']:.2%}) {match_str}")
                if candidate['corrections']:
                    for correction in candidate['corrections']:
                        print(f"   - {correction}")
        
        elif choice == '3':
            print("再见!")
            break
        else:
            print("无效选择,请重试")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 命令行模式
        image_path = sys.argv[1]
        use_enhancement = '--no-enhance' not in sys.argv
        
        result = correct_recognition_result(image_path, use_enhancement)
        
        print("\n" + "=" * 60)
        print("最终识别结果:")
        print("=" * 60)
        for item in result.get("results", []):
            print(f"✓ {item['text']} (置信度: {item['confidence']:.2%})")
            if item.get('pattern_match'):
                print(f"  ✓ 匹配轮毂编号模式")
    else:
        # 交互式模式
        interactive_correction()
