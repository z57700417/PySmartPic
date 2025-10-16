"""
测试 wheel1.png 识别
"""

from pathlib import Path
from src.core.recognizer import WheelRecognizer
from src.core.config import Config
from PIL import Image

def test_wheel1():
    """测试识别 wheel1.png"""
    print("=" * 70)
    print("测试识别 wheel1.png")
    print("=" * 70)
    
    image_path = "test_images/wheel1.png"
    
    # 检查图片
    if not Path(image_path).exists():
        print(f"❌ 图片不存在: {image_path}")
        return
    
    # 显示图片信息
    img = Image.open(image_path)
    print(f"\n📷 图片信息:")
    print(f"   路径: {image_path}")
    print(f"   尺寸: {img.size} (宽x高)")
    print(f"   模式: {img.mode}")
    print(f"   文件大小: {Path(image_path).stat().st_size / 1024:.2f} KB")
    
    # 测试1: 默认配置
    print("\n" + "-" * 70)
    print("测试 1: 使用默认配置")
    print("-" * 70)
    
    try:
        recognizer = WheelRecognizer()
        result = recognizer.recognize(image_path)
        
        print(f"✓ 识别完成 (耗时: {result['processing_time']:.2f}秒)")
        print(f"  引擎: {result['engine_used']}")
        print(f"  成功: {result['success']}")
        print(f"  检测到文字数: {result['total_texts']}")
        
        if result['total_texts'] > 0:
            print("\n  识别结果:")
            for idx, item in enumerate(result["results"], 1):
                print(f"    {idx}. {item['text']} (置信度: {item['confidence']:.2%})")
        else:
            print("  ⚠️  未识别到任何文字")
            
    except Exception as e:
        print(f"❌ 错误: {e}")
    
    # 测试2: 调整配置 - 降低置信度阈值
    print("\n" + "-" * 70)
    print("测试 2: 降低置信度阈值 (0.3)")
    print("-" * 70)
    
    try:
        config = Config()
        config.set("postprocessing.min_confidence", 0.3)
        config.set("preprocessing.denoise.enabled", True)
        config.set("preprocessing.sharpen.enabled", True)
        
        recognizer = WheelRecognizer(config)
        result = recognizer.recognize(image_path)
        
        print(f"✓ 识别完成 (耗时: {result['processing_time']:.2f}秒)")
        print(f"  检测到文字数: {result['total_texts']}")
        
        if result['total_texts'] > 0:
            print("\n  识别结果:")
            for idx, item in enumerate(result["results"], 1):
                print(f"    {idx}. {item['text']} (置信度: {item['confidence']:.2%})")
        else:
            print("  ⚠️  未识别到任何文字")
            
    except Exception as e:
        print(f"❌ 错误: {e}")
    
    # 测试3: 增强预处理
    print("\n" + "-" * 70)
    print("测试 3: 增强预处理")
    print("-" * 70)
    
    try:
        config = Config()
        config.set("postprocessing.min_confidence", 0.2)
        config.set("preprocessing.denoise.enabled", True)
        config.set("preprocessing.denoise.strength", 3)
        config.set("preprocessing.sharpen.enabled", True)
        config.set("preprocessing.sharpen.strength", 2)
        config.set("preprocessing.contrast.enabled", True)
        config.set("preprocessing.contrast.factor", 1.5)
        
        recognizer = WheelRecognizer(config)
        result = recognizer.recognize(image_path)
        
        print(f"✓ 识别完成 (耗时: {result['processing_time']:.2f}秒)")
        print(f"  检测到文字数: {result['total_texts']}")
        
        if result['total_texts'] > 0:
            print("\n  识别结果:")
            for idx, item in enumerate(result["results"], 1):
                print(f"    {idx}. {item['text']} (置信度: {item['confidence']:.2%})")
                if 'bbox' in item:
                    print(f"        位置: {item['bbox']}")
        else:
            print("  ⚠️  未识别到任何文字")
        
        # 生成可视化
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "wheel1_enhanced_result.jpg"
        recognizer.visualize(image_path, result, str(output_path))
        print(f"\n  💾 可视化结果已保存: {output_path}")
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)
    print("\n💡 提示:")
    print("  - 如果未识别到文字，可能需要:")
    print("    1. 检查图片中是否有清晰的文字")
    print("    2. 图片角度是否合适")
    print("    3. 文字是否被遮挡或模糊")
    print("  - 查看生成的可视化图片了解检测情况")
    print(f"  - 可视化结果保存在: output/wheel1_enhanced_result.jpg")

if __name__ == "__main__":
    test_wheel1()
