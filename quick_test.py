"""
快速测试脚本 - 一键测试图片识别
"""

import sys
from pathlib import Path

# 添加图片路径到这里
TEST_IMAGE = r"e:\web\PySmartPic\test_wheel.jpg"  # 修改为你的图片路径

def main():
    print("=" * 80)
    print("轮毂字母识别 - 快速测试")
    print("=" * 80)
    
    # 检查图片是否存在
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = TEST_IMAGE
    
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"\n❌ 错误: 图片不存在 - {image_path}")
        print("\n使用方法:")
        print(f"  python {Path(__file__).name} <图片路径>")
        print("\n示例:")
        print(f"  python {Path(__file__).name} wheel.jpg")
        return
    
    print(f"\n📁 图片路径: {image_path}")
    print(f"📊 文件大小: {image_path.stat().st_size / 1024:.2f} KB")
    
    # 导入增强识别模块
    try:
        from enhance_recognition import recognize_with_enhancement, recognize_with_multiple_engines
    except ImportError:
        print("\n❌ 错误: 无法导入 enhance_recognition 模块")
        print("请确保 enhance_recognition.py 文件存在")
        return
    
    print("\n" + "=" * 80)
    print("🚀 开始识别...")
    print("=" * 80)
    
    # 方法1: 尝试不同的放大倍数
    print("\n【方法1】尝试不同的图像放大倍数 (2x, 3x, 4x)")
    print("-" * 80)
    result1 = recognize_with_enhancement(str(image_path))
    
    # 方法2: 尝试多个OCR引擎
    print("\n\n【方法2】尝试多个OCR引擎 (PaddleOCR + EasyOCR)")
    print("-" * 80)
    try:
        result2 = recognize_with_multiple_engines(str(image_path))
    except Exception as e:
        print(f"⚠️ 多引擎识别失败: {e}")
        result2 = None
    
    # 汇总结果
    print("\n\n" + "=" * 80)
    print("📊 最终识别结果汇总")
    print("=" * 80)
    
    if result1:
        print("\n✅ 成功识别!")
        print(f"\n最佳结果 (放大 {result1['scale']}x):")
        for item in result1['result']['results']:
            print(f"  ✓ {item['text']} (置信度: {item['confidence']:.2%})")
        
        print(f"\n📂 结果已保存到: enhanced_output/")
        print(f"  - 预处理图片: {result1['enhanced_path'].name}")
        print(f"  - 可视化结果: {result1['vis_path'].name}")
    else:
        print("\n❌ 识别失败")
        print("\n💡 建议:")
        print("  1. 确保图片中的文字清晰可见")
        print("  2. 尝试调整拍摄角度和光照")
        print("  3. 确保图片分辨率足够高")
        print("  4. 检查文字是否被遮挡")
    
    print("\n" + "=" * 80)
    print("✨ 测试完成!")
    print("=" * 80)

if __name__ == "__main__":
    main()
