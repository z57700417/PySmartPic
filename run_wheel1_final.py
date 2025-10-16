"""
wheel1.png 最终识别结果
"""

from paddleocr import PaddleOCR
import cv2
import numpy as np
from pathlib import Path

def run_wheel1_final():
    """运行 wheel1.png 识别并显示最终结果"""
    print("=" * 70)
    print("wheel1.png 识别结果")
    print("=" * 70)
    
    image_path = "test_images/wheel1.png"
    
    # 读取原图
    img = cv2.imread(image_path)
    print(f"\n📷 原图信息: 尺寸 {img.shape}, 平均亮度 {img.mean():.2f}")
    
    # 初始化 OCR
    print("\n正在初始化 PaddleOCR...")
    ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
    
    # 创建输出目录
    Path("output").mkdir(exist_ok=True)
    
    print("\n" + "=" * 70)
    print("开始识别...")
    print("=" * 70)
    
    # 方法1: 原图识别
    print("\n【方法1】原图直接识别:")
    result1 = ocr.ocr(image_path, cls=True)
    print_ocr_result(result1)
    
    # 方法2: 放大2倍
    print("\n【方法2】放大2倍后识别:")
    h, w = img.shape[:2]
    enlarged = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    enlarged_path = "output/wheel1_enlarged.png"
    cv2.imwrite(enlarged_path, enlarged)
    result2 = ocr.ocr(enlarged_path, cls=True)
    print_ocr_result(result2)
    
    # 方法3: 综合增强（推荐）
    print("\n【方法3】综合增强识别（推荐）:")
    h, w = img.shape[:2]
    enhanced = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.3, beta=40)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    enhanced_path = "output/wheel1_best.png"
    cv2.imwrite(enhanced_path, enhanced)
    result3 = ocr.ocr(enhanced_path, cls=True)
    print_ocr_result(result3)
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("✓ 识别完成！")
    print("=" * 70)
    
    all_texts = []
    for result in [result1, result2, result3]:
        texts = extract_texts(result)
        all_texts.extend(texts)
    
    if all_texts:
        # 去重并显示所有识别到的文字
        unique_texts = {}
        for text, conf in all_texts:
            if text not in unique_texts or conf > unique_texts[text]:
                unique_texts[text] = conf
        
        print("\n📝 识别到的所有文字:")
        for idx, (text, conf) in enumerate(sorted(unique_texts.items(), key=lambda x: -x[1]), 1):
            print(f"  {idx}. {str(text):20s} (置信度: {conf:.2%})")
        
        print(f"\n💾 处理后的图片已保存到:")
        print(f"  - output/wheel1_enlarged.png (放大2倍)")
        print(f"  - output/wheel1_best.png (综合增强 - 推荐查看)")
    else:
        print("\n⚠️  所有方法都未识别到文字")
        print("可能原因:")
        print("  1. 图片中没有清晰可识别的文字")
        print("  2. 文字太小或被遮挡")
        print("  3. 需要手动标注或使用专门的模型")

def print_ocr_result(result):
    """打印OCR结果"""
    if not result or not result[0]:
        print("  ❌ 未检测到文字")
        return
    
    print(f"  ✓ 检测到 {len(result[0])} 个文字区域:")
    for idx, item in enumerate(result[0], 1):
        try:
            # 尝试不同的格式解析
            if isinstance(item, (list, tuple)):
                if len(item) >= 2:
                    # 格式1: [bbox, (text, confidence)]
                    if isinstance(item[1], (list, tuple)) and len(item[1]) >= 2:
                        text, conf = item[1]
                        print(f"    {idx}. {str(text):20s} (置信度: {conf:.2%})")
                    # 格式2: [bbox, text]
                    elif isinstance(item[1], str):
                        print(f"    {idx}. {item[1]}")
                    # 格式3: (text, confidence)
                    elif isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                        text, conf = item
                        print(f"    {idx}. {str(text):20s} (置信度: {conf:.2%})")
                    else:
                        print(f"    {idx}. {item}")
        except Exception as e:
            print(f"    {idx}. [解析错误: {e}]")

def extract_texts(result):
    """提取所有识别到的文字"""
    texts = []
    if not result or not result[0]:
        return texts
    
    for item in result[0]:
        try:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                if isinstance(item[1], (list, tuple)) and len(item[1]) >= 2:
                    texts.append((item[1][0], item[1][1]))
                elif isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
                    texts.append((item[0], item[1]))
        except:
            pass
    
    return texts

if __name__ == "__main__":
    run_wheel1_final()
