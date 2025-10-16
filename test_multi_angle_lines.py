"""
测试多角度融合的行分组功能
演示如何融合多张图片的行识别结果
"""

from src.core.recognizer import WheelRecognizer
from src.core.multi_angle_fusion import MultiAngleFusion
from src.core.config import Config
import json


def test_multi_angle_line_grouping(image_paths):
    """
    测试多角度融合的行分组
    
    Args:
        image_paths: 图片路径列表
    """
    print("=" * 60)
    print("测试多角度融合行分组功能")
    print("=" * 60)
    
    if len(image_paths) < 2:
        print("\n❌ 至少需要2张图片进行多角度融合")
        return
    
    # 创建识别器和融合器
    config = Config()
    recognizer = WheelRecognizer(config)
    fusion = MultiAngleFusion(config["multi_angle"])
    
    # 逐张识别
    print(f"\n正在识别 {len(image_paths)} 张图片...")
    individual_results = []
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"  [{i}/{len(image_paths)}] {image_path}")
        result = recognizer.recognize(image_path)
        individual_results.append(result)
        
        if result.get("success"):
            print(f"    ✓ 识别到 {result.get('total_lines', 0)} 行文字")
            for line in result.get('lines', []):
                print(f"      - {line['text']}")
        else:
            print(f"    ✗ 识别失败: {result.get('error')}")
    
    # 融合结果
    print("\n" + "=" * 60)
    print("融合识别结果")
    print("=" * 60)
    
    fused_result = fusion.fuse_results(individual_results)
    
    if not fused_result.get("success"):
        print(f"\n❌ 融合失败: {fused_result.get('error')}")
        return
    
    print(f"\n✅ 融合成功!")
    print(f"融合方法: {fused_result['fusion_method']}")
    print(f"源图片数: {fused_result['source_count']}")
    
    # 显示按行融合的结果
    print("\n" + "=" * 60)
    print(f"按行融合结果 (共 {fused_result['total_lines']} 行)")
    print("=" * 60)
    
    for i, line in enumerate(fused_result.get('lines', []), 1):
        print(f"\n第 {i} 行:")
        print(f"  文字: {line['text']}")
        print(f"  平均置信度: {line['confidence']:.2%}")
        print(f"  出现次数: 在 {line['occurrence_count']} 张图片中识别到")
    
    # 显示传统的合并文本
    print("\n" + "=" * 60)
    print("传统合并结果")
    print("=" * 60)
    print(f"合并文字: {fused_result.get('merged_text', '')}")
    print(f"置信度: {fused_result.get('confidence', 0):.2%}")
    
    # 显示备选结果
    if fused_result.get('alternatives'):
        print("\n备选结果:")
        for alt in fused_result['alternatives']:
            print(f"  • {alt['text']} (置信度: {alt.get('confidence', 0):.2%})")
    
    # 保存结果
    output_file = "multi_angle_fusion_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(fused_result, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n详细结果已保存到: {output_file}")
    
    return fused_result


def demo_api_response():
    """演示多角度融合的API响应格式"""
    print("\n" + "=" * 60)
    print("多角度融合 API 响应格式示例")
    print("=" * 60)
    
    example_response = {
        "success": True,
        "source_count": 3,
        "fusion_method": "voting",
        "total_lines": 2,
        "lines": [
            {
                "text": "AT64202",
                "confidence": 0.93,
                "occurrence_count": 3,
                "item_count": 1
            },
            {
                "text": "0909 W1D",
                "confidence": 0.89,
                "occurrence_count": 2,
                "item_count": 2
            }
        ],
        "merged_text": "AT64202",
        "confidence": 0.93,
        "alternatives": [
            {
                "text": "AT64203",
                "confidence": 0.85,
                "occurrence_count": 1
            }
        ]
    }
    
    print(json.dumps(example_response, ensure_ascii=False, indent=2))
    
    print("\n说明:")
    print("  • total_lines: 融合后的总行数")
    print("  • lines: 按行融合的结果 (推荐使用)")
    print("    - text: 该行的文字")
    print("    - confidence: 该行的平均置信度")
    print("    - occurrence_count: 该行在多少张图片中被识别到")
    print("  • merged_text: 传统的合并文本")
    print("  • alternatives: 备选结果")


if __name__ == "__main__":
    import sys
    
    print("多角度融合行分组功能测试\n")
    
    # 演示API格式
    demo_api_response()
    
    # 如果提供了图片路径,进行实际测试
    if len(sys.argv) > 1:
        image_paths = sys.argv[1:]
        print(f"\n收到 {len(image_paths)} 张图片:")
        for path in image_paths:
            print(f"  - {path}")
        
        result = test_multi_angle_line_grouping(image_paths)
    else:
        print("\n" + "=" * 60)
        print("使用方法:")
        print("  python test_multi_angle_lines.py <图片1> <图片2> [图片3...]")
        print("\n示例:")
        print("  python test_multi_angle_lines.py angle1.jpg angle2.jpg angle3.jpg")
        print("=" * 60)
