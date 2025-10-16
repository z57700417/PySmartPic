"""
测试行分组功能
演示如何返回按行分组的识别结果
"""

from src.core.recognizer import WheelRecognizer
from src.core.config import Config
import json


def test_line_grouping(image_path):
    """
    测试行分组识别
    
    Args:
        image_path: 图片路径
    """
    print("=" * 60)
    print("测试行分组识别功能")
    print("=" * 60)
    
    # 创建识别器
    config = Config()
    recognizer = WheelRecognizer(config)
    
    # 识别
    print(f"\n正在识别: {image_path}")
    result = recognizer.recognize(image_path)
    
    if not result.get("success"):
        print(f"\n❌ 识别失败: {result.get('error')}")
        return
    
    print(f"\n✅ 识别成功!")
    print(f"处理时间: {result['processing_time']:.2f}秒")
    print(f"使用引擎: {result['engine_used']}")
    
    # 显示行分组结果
    print("\n" + "=" * 60)
    print(f"按行分组结果 (共 {result['total_lines']} 行)")
    print("=" * 60)
    
    for i, line in enumerate(result.get('lines', []), 1):
        print(f"\n第 {i} 行:")
        print(f"  文字: {line['text']}")
        print(f"  置信度: {line['confidence']:.2%}")
        print(f"  包含 {line['item_count']} 个文字块")
        
        # 显示详细的文字块
        for j, item in enumerate(line.get('items', []), 1):
            print(f"    块{j}: {item['text']} ({item['confidence']:.2%})")
    
    # 显示原始结果(所有文字)
    print("\n" + "=" * 60)
    print(f"原始识别结果 (共 {result['total_texts']} 个文字)")
    print("=" * 60)
    
    for i, item in enumerate(result.get('results', []), 1):
        print(f"{i}. {item['text']} ({item['confidence']:.2%})")
    
    # 保存JSON结果
    output_file = "line_grouping_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n详细结果已保存到: {output_file}")
    
    return result


def demo_api_response():
    """演示API响应格式"""
    print("\n" + "=" * 60)
    print("API 响应格式示例")
    print("=" * 60)
    
    example_response = {
        "success": True,
        "image_path": "wheel.jpg",
        "total_texts": 3,
        "total_lines": 2,
        "lines": [
            {
                "text": "AT64202",
                "confidence": 0.92,
                "item_count": 1,
                "items": [
                    {
                        "text": "AT64202",
                        "confidence": 0.92,
                        "bbox": [[10, 20], [100, 20], [100, 40], [10, 40]]
                    }
                ]
            },
            {
                "text": "0909 W1D",
                "confidence": 0.88,
                "item_count": 2,
                "items": [
                    {
                        "text": "0909",
                        "confidence": 0.90,
                        "bbox": [[10, 60], [60, 60], [60, 80], [10, 80]]
                    },
                    {
                        "text": "W1D",
                        "confidence": 0.86,
                        "bbox": [[70, 60], [110, 60], [110, 80], [70, 80]]
                    }
                ]
            }
        ],
        "results": [
            {"text": "AT64202", "confidence": 0.92},
            {"text": "0909", "confidence": 0.90},
            {"text": "W1D", "confidence": 0.86}
        ],
        "processing_time": 1.23,
        "engine_used": "paddleocr"
    }
    
    print(json.dumps(example_response, ensure_ascii=False, indent=2))
    
    print("\n说明:")
    print("  • total_lines: 识别出的总行数")
    print("  • lines: 按行分组的结果(推荐使用)")
    print("  • results: 原始识别结果(所有文字块)")
    print("  • 每行包含:")
    print("    - text: 该行合并后的文字")
    print("    - confidence: 该行的平均置信度")
    print("    - item_count: 该行包含的文字块数量")
    print("    - items: 该行的详细文字块列表")


if __name__ == "__main__":
    import sys
    
    print("轮毂识别 - 行分组功能测试")
    print()
    
    # 演示API格式
    demo_api_response()
    
    # 如果提供了图片路径,进行实际测试
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = test_line_grouping(image_path)
    else:
        print("\n" + "=" * 60)
        print("使用方法:")
        print("  python test_line_grouping.py <图片路径>")
        print("\n示例:")
        print("  python test_line_grouping.py wheel.jpg")
        print("=" * 60)
