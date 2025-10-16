"""
诊断 wheel1.png 识别问题
"""

from paddleocr import PaddleOCR
import cv2
import numpy as np
from pathlib import Path

def diagnose_wheel1():
    """诊断 wheel1.png"""
    print("=" * 70)
    print("诊断 wheel1.png 识别问题")
    print("=" * 70)
    
    image_path = "test_images/wheel1.png"
    
    # 读取图片
    img = cv2.imread(image_path)
    print(f"\n📷 原始图片信息:")
    print(f"   形状: {img.shape}")
    print(f"   亮度: {img.mean():.2f}")
    
    # 测试1: 直接使用 PaddleOCR
    print("\n" + "-" * 70)
    print("测试 1: 直接使用 PaddleOCR (默认参数)")
    print("-" * 70)
    
    ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
    result = ocr.ocr(image_path, cls=True)
    
    print(f"原始返回: {result}")
    
    if result and result[0]:
        print(f"\n✓ 检测到 {len(result[0])} 个文字区域:")
        for idx, item in enumerate(result[0], 1):
            try:
                # PaddleOCR 返回格式: [[bbox], (text, confidence)] 或其他格式
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    if isinstance(item[1], (list, tuple)) and len(item[1]) >= 2:
                        text = item[1][0]
                        confidence = item[1][1]
                        print(f"  {idx}. {text} (置信度: {confidence:.2%})")
                    elif isinstance(item[1], str):
                        print(f"  {idx}. {item[1]}")
                    else:
                        print(f"  {idx}. 数据: {item}")
                else:
                    print(f"  {idx}. 原始数据: {item}")
            except Exception as e:
                print(f"  {idx}. 解析错误 ({e}): {item}")
            print(f"     位置: {bbox}")
    else:
        print("❌ 未检测到任何文字")
    
    # 测试2: 调整图片亮度后识别
    print("\n" + "-" * 70)
    print("测试 2: 增亮图片后识别")
    print("-" * 70)
    
    # 增加亮度
    bright_img = cv2.convertScaleAbs(img, alpha=1.5, beta=50)
    bright_path = "output/wheel1_bright.png"
    Path("output").mkdir(exist_ok=True)
    cv2.imwrite(bright_path, bright_img)
    print(f"增亮后图片已保存: {bright_path}")
    print(f"增亮后亮度: {bright_img.mean():.2f}")
    
    result = ocr.ocr(bright_path, cls=True)
    
    if result and result[0]:
        print(f"\n✓ 检测到 {len(result[0])} 个文字区域:")
        for idx, item in enumerate(result[0], 1):
            try:
                # PaddleOCR 返回格式: [[bbox], (text, confidence)] 或其他格式
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    if isinstance(item[1], (list, tuple)) and len(item[1]) >= 2:
                        text = item[1][0]
                        confidence = item[1][1]
                        print(f"  {idx}. {text} (置信度: {confidence:.2%})")
                    elif isinstance(item[1], str):
                        print(f"  {idx}. {item[1]}")
                    else:
                        print(f"  {idx}. 数据: {item}")
                else:
                    print(f"  {idx}. 原始数据: {item}")
            except Exception as e:
                print(f"  {idx}. 解析错误 ({e}): {item}")
    else:
        print("❌ 仍未检测到任何文字")
    
    # 测试3: 二值化后识别
    print("\n" + "-" * 70)
    print("测试 3: 二值化处理后识别")
    print("-" * 70)
    
    # 转灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 应用 CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    # 二值化
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    binary_path = "output/wheel1_binary.png"
    cv2.imwrite(binary_path, binary)
    print(f"二值化图片已保存: {binary_path}")
    
    result = ocr.ocr(binary_path, cls=True)
    
    if result and result[0]:
        print(f"\n✓ 检测到 {len(result[0])} 个文字区域:")
        for idx, item in enumerate(result[0], 1):
            try:
                # PaddleOCR 返回格式: [[bbox], (text, confidence)] 或其他格式
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    if isinstance(item[1], (list, tuple)) and len(item[1]) >= 2:
                        text = item[1][0]
                        confidence = item[1][1]
                        print(f"  {idx}. {text} (置信度: {confidence:.2%})")
                    elif isinstance(item[1], str):
                        print(f"  {idx}. {item[1]}")
                    else:
                        print(f"  {idx}. 数据: {item}")
                else:
                    print(f"  {idx}. 原始数据: {item}")
            except Exception as e:
                print(f"  {idx}. 解析错误 ({e}): {item}")
    else:
        print("❌ 仍未检测到任何文字")
    
    # 测试4: 放大图片
    print("\n" + "-" * 70)
    print("测试 4: 放大图片2倍后识别")
    print("-" * 70)
    
    h, w = img.shape[:2]
    enlarged = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    enlarged_path = "output/wheel1_enlarged.png"
    cv2.imwrite(enlarged_path, enlarged)
    print(f"放大后图片已保存: {enlarged_path}")
    print(f"新尺寸: {enlarged.shape}")
    
    result = ocr.ocr(enlarged_path, cls=True)
    
    if result and result[0]:
        print(f"\n✓ 检测到 {len(result[0])} 个文字区域:")
        for idx, item in enumerate(result[0], 1):
            try:
                # PaddleOCR 返回格式: [[bbox], (text, confidence)] 或其他格式
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    if isinstance(item[1], (list, tuple)) and len(item[1]) >= 2:
                        text = item[1][0]
                        confidence = item[1][1]
                        print(f"  {idx}. {text} (置信度: {confidence:.2%})")
                    elif isinstance(item[1], str):
                        print(f"  {idx}. {item[1]}")
                    else:
                        print(f"  {idx}. 数据: {item}")
                else:
                    print(f"  {idx}. 原始数据: {item}")
            except Exception as e:
                print(f"  {idx}. 解析错误 ({e}): {item}")
    else:
        print("❌ 仍未检测到任何文字")
    
    # 测试5: 综合增强
    print("\n" + "-" * 70)
    print("测试 5: 综合增强 (放大+增亮+锐化)")
    print("-" * 70)
    
    # 放大
    h, w = img.shape[:2]
    enhanced_img = cv2.resize(img, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
    # 增亮
    enhanced_img = cv2.convertScaleAbs(enhanced_img, alpha=1.3, beta=40)
    # 锐化
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    enhanced_img = cv2.filter2D(enhanced_img, -1, kernel)
    
    enhanced_path = "output/wheel1_enhanced_all.png"
    cv2.imwrite(enhanced_path, enhanced_img)
    print(f"综合增强图片已保存: {enhanced_path}")
    
    result = ocr.ocr(enhanced_path, cls=True)
    
    if result and result[0]:
        print(f"\n✓ 检测到 {len(result[0])} 个文字区域:")
        for idx, item in enumerate(result[0], 1):
            try:
                # PaddleOCR 返回格式: [[bbox], (text, confidence)] 或其他格式
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    if isinstance(item[1], (list, tuple)) and len(item[1]) >= 2:
                        text = item[1][0]
                        confidence = item[1][1]
                        print(f"  {idx}. {text} (置信度: {confidence:.2%})")
                    elif isinstance(item[1], str):
                        print(f"  {idx}. {item[1]}")
                    else:
                        print(f"  {idx}. 数据: {item}")
                else:
                    print(f"  {idx}. 原始数据: {item}")
            except Exception as e:
                print(f"  {idx}. 解析错误 ({e}): {item}")
    else:
        print("❌ 仍未检测到任何文字")
    
    print("\n" + "=" * 70)
    print("诊断完成")
    print("=" * 70)
    print("\n💡 建议:")
    print("  - 检查 output 目录下生成的处理后图片")
    print("  - 如果所有方法都无法识别，可能原因:")
    print("    1. 图片中没有文字或文字不清晰")
    print("    2. 文字是特殊字体或艺术字")
    print("    3. 文字被严重遮挡或变形")
    print("    4. 需要使用其他OCR引擎或模型")

if __name__ == "__main__":
    diagnose_wheel1()
