"""
简单示例脚本
演示如何使用识别器
"""

from pathlib import Path
from src.core.recognizer import WheelRecognizer
from src.core.config import Config
from src.core.multi_angle_fusion import MultiAngleFusion


def example_single_recognition():
    """单图识别示例"""
    print("=" * 50)
    print("示例1: 单图识别")
    print("=" * 50)
    
    # 创建识别器
    recognizer = WheelRecognizer()
    
    # 识别图片(替换为实际图片路径)
    image_path = "test_images/wheel1.jpg"
    
    if Path(image_path).exists():
        result = recognizer.recognize(image_path)
        
        if result["success"]:
            print(f"\n识别成功! 耗时: {result['processing_time']:.2f}秒")
            print(f"识别引擎: {result['engine_used']}")
            print(f"检测到 {result['total_texts']} 个文字区域:\n")
            
            for idx, item in enumerate(result["results"], 1):
                text = item["text"]
                confidence = item["confidence"]
                print(f"{idx}. {text:20s} (置信度: {confidence:.2%})")
        else:
            print(f"识别失败: {result.get('error', '未知错误')}")
    else:
        print(f"图片不存在: {image_path}")
        print("请将测试图片放在 test_images/ 目录下")


def example_batch_recognition():
    """批量识别示例"""
    print("\n" + "=" * 50)
    print("示例2: 批量识别")
    print("=" * 50)
    
    # 创建识别器
    recognizer = WheelRecognizer()
    
    # 批量识别(替换为实际图片目录)
    image_dir = Path("test_images")
    
    if image_dir.exists():
        image_files = list(image_dir.glob("*.jpg"))
        
        if image_files:
            print(f"\n找到 {len(image_files)} 张图片")
            
            # 批量识别
            results = recognizer.recognize_batch(image_files, parallel=True)
            
            # 打印结果
            print("\n批量识别结果:")
            for result in results:
                image_name = Path(result["image_path"]).name
                status = "成功" if result["success"] else "失败"
                total = result.get("total_texts", 0)
                time = result.get("processing_time", 0)
                print(f"  {image_name:30s} {status:4s} 文字数: {total:2d} 耗时: {time:.2f}秒")
        else:
            print(f"目录下没有图片: {image_dir}")
    else:
        print(f"目录不存在: {image_dir}")


def example_multi_angle_fusion():
    """多角度融合识别示例"""
    print("\n" + "=" * 50)
    print("示例3: 多角度融合识别")
    print("=" * 50)
    
    # 创建配置
    config = Config()
    config.set("multi_angle.fusion_method", "voting")
    config.set("multi_angle.return_alternatives", True)
    
    # 创建识别器
    recognizer = WheelRecognizer(config)
    
    # 多角度图片(替换为实际图片路径)
    image_paths = [
        "test_images/wheel_angle1.jpg",
        "test_images/wheel_angle2.jpg",
        "test_images/wheel_angle3.jpg"
    ]
    
    # 检查图片是否存在
    existing_images = [p for p in image_paths if Path(p).exists()]
    
    if len(existing_images) >= 2:
        print(f"\n使用 {len(existing_images)} 张图片进行融合识别")
        
        # 逐张识别
        individual_results = []
        for image_path in existing_images:
            result = recognizer.recognize(image_path)
            individual_results.append(result)
            print(f"  {Path(image_path).name}: 识别到 {result.get('total_texts', 0)} 个文字")
        
        # 创建融合器
        fusion = MultiAngleFusion(config["multi_angle"])
        
        # 融合结果
        fused_result = fusion.fuse_results(individual_results)
        
        if fused_result["success"]:
            print("\n融合结果:")
            print(f"  文字: {fused_result['merged_text']}")
            print(f"  置信度: {fused_result['confidence']:.2%}")
            print(f"  融合方法: {fused_result['fusion_method']}")
            
            # 备选结果
            alternatives = fused_result.get("alternatives", [])
            if alternatives:
                print("\n  备选结果:")
                for idx, alt in enumerate(alternatives, 1):
                    print(f"    {idx}. {alt['text']} (得分: {alt.get('score', alt.get('confidence', 0)):.2f})")
        else:
            print(f"融合失败: {fused_result.get('error', '未知错误')}")
    else:
        print("需要至少2张图片进行多角度融合")
        print("请在 test_images/ 目录下准备多张不同角度的轮毂照片")


def example_visualization():
    """可视化示例"""
    print("\n" + "=" * 50)
    print("示例4: 结果可视化")
    print("=" * 50)
    
    # 创建识别器
    recognizer = WheelRecognizer()
    
    # 识别图片
    image_path = "test_images/wheel1.jpg"
    output_path = "output/wheel1_result.jpg"
    
    if Path(image_path).exists():
        # 执行识别
        result = recognizer.recognize(image_path)
        
        if result["success"]:
            # 生成可视化图片
            Path("output").mkdir(exist_ok=True)
            recognizer.visualize(image_path, result, output_path)
            print(f"\n可视化结果已保存到: {output_path}")
        else:
            print(f"识别失败: {result.get('error', '未知错误')}")
    else:
        print(f"图片不存在: {image_path}")


def example_custom_config():
    """自定义配置示例"""
    print("\n" + "=" * 50)
    print("示例5: 自定义配置")
    print("=" * 50)
    
    # 创建自定义配置
    config = Config()
    
    # 修改配置参数
    config.set("recognition.engine", "paddleocr")
    config.set("postprocessing.min_confidence", 0.7)
    config.set("system.use_gpu", False)
    config.set("preprocessing.denoise.method", "bilateral")
    
    # 使用自定义配置创建识别器
    recognizer = WheelRecognizer(config)
    
    print("\n自定义配置:")
    print(f"  识别引擎: {config.get('recognition.engine')}")
    print(f"  最小置信度: {config.get('postprocessing.min_confidence')}")
    print(f"  使用GPU: {config.get('system.use_gpu')}")
    print(f"  去噪方法: {config.get('preprocessing.denoise.method')}")
    
    # 识别图片
    image_path = "test_images/wheel1.jpg"
    
    if Path(image_path).exists():
        result = recognizer.recognize(image_path)
        
        if result["success"]:
            print(f"\n识别成功! (使用 {result['engine_used']} 引擎)")
            print(f"识别到 {result['total_texts']} 个文字区域")
        else:
            print(f"识别失败: {result.get('error', '未知错误')}")
    else:
        print(f"\n图片不存在: {image_path}")


if __name__ == "__main__":
    # 创建测试目录
    Path("test_images").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    
    print("\n汽车轮毂字母识别系统 - 示例程序")
    print("=" * 50)
    print("\n请将测试图片放在 test_images/ 目录下")
    print("支持的格式: jpg, png, bmp, tiff, webp\n")
    
    # 运行示例
    try:
        example_single_recognition()
        example_batch_recognition()
        example_multi_angle_fusion()
        example_visualization()
        example_custom_config()
        
        print("\n" + "=" * 50)
        print("所有示例运行完成!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
