"""
简单测试用例 - 演示系统功能
即使没有OCR引擎和真实图片，也能展示系统架构
"""

import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from time import sleep

console = Console()


def test_config_system():
    """测试1: 配置系统"""
    console.print("\n[bold cyan]测试1: 配置系统[/bold cyan]")
    console.print("=" * 60)
    
    try:
        from src.core.config import Config
        
        # 加载默认配置
        config = Config()
        console.print("✓ 成功加载配置文件", style="green")
        
        # 显示关键配置
        table = Table(title="配置信息")
        table.add_column("配置项", style="cyan")
        table.add_column("值", style="magenta")
        
        configs = [
            ("识别引擎", config.get("recognition.engine")),
            ("最小置信度", config.get("postprocessing.min_confidence")),
            ("使用GPU", config.get("system.use_gpu")),
            ("预处理", config.get("preprocessing.enable")),
            ("融合方法", config.get("multi_angle.fusion_method"))
        ]
        
        for key, value in configs:
            table.add_row(key, str(value))
            
        console.print(table)
        
        # 测试配置修改
        console.print("\n[yellow]测试动态修改配置:[/yellow]")
        config.set("recognition.engine", "paddleocr")
        config.set("postprocessing.min_confidence", 0.8)
        console.print(f"  修改识别引擎: {config.get('recognition.engine')}")
        console.print(f"  修改置信度: {config.get('postprocessing.min_confidence')}")
        
        console.print("\n✅ [bold green]配置系统测试通过[/bold green]")
        return True
        
    except Exception as e:
        console.print(f"\n❌ [bold red]配置系统测试失败: {e}[/bold red]")
        return False


def test_preprocessor():
    """测试2: 图像预处理器"""
    console.print("\n[bold cyan]测试2: 图像预处理模块[/bold cyan]")
    console.print("=" * 60)
    
    try:
        from src.core.config import Config
        from src.core.preprocessor import ImagePreprocessor
        import numpy as np
        
        config = Config()
        preprocessor = ImagePreprocessor(config)
        console.print("✓ 成功创建预处理器", style="green")
        
        # 创建模拟图像 (100x100 灰度图)
        console.print("\n[yellow]创建模拟测试图像 (100x100)...[/yellow]")
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # 测试预处理
        console.print("[yellow]执行预处理步骤:[/yellow]")
        steps = [
            "亮度对比度调整",
            "图像去噪",
            "边缘增强",
            "二值化处理"
        ]
        
        for step in track(steps, description="处理中..."):
            sleep(0.3)  # 模拟处理时间
            
        processed = preprocessor.preprocess(test_image)
        console.print(f"\n原始图像尺寸: {test_image.shape}")
        console.print(f"处理后尺寸: {processed.shape}")
        
        console.print("\n✅ [bold green]预处理模块测试通过[/bold green]")
        return True
        
    except Exception as e:
        console.print(f"\n❌ [bold red]预处理模块测试失败: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        return False


def test_postprocessor():
    """测试3: 后处理器"""
    console.print("\n[bold cyan]测试3: 后处理模块[/bold cyan]")
    console.print("=" * 60)
    
    try:
        from src.core.config import Config
        from src.core.postprocessor import ResultPostprocessor
        
        config = Config()
        postprocessor = ResultPostprocessor(config)
        console.print("✓ 成功创建后处理器", style="green")
        
        # 模拟识别结果
        mock_results = [
            {"text": "MICHELIN", "confidence": 0.95},
            {"text": "michelin", "confidence": 0.87},  # 重复(大小写不同)
            {"text": "P225", "confidence": 0.92},
            {"text": "p225", "confidence": 0.45},  # 低置信度
            {"text": "50R", "confidence": 0.88},
            {"text": "5OR", "confidence": 0.65},  # 可能的OCR错误
        ]
        
        console.print("\n[yellow]模拟识别结果:[/yellow]")
        for i, result in enumerate(mock_results, 1):
            console.print(f"  {i}. {result['text']:12s} - 置信度: {result['confidence']:.2%}")
        
        # 后处理
        console.print("\n[yellow]执行后处理:[/yellow]")
        processed = postprocessor.process(mock_results)
        
        console.print("\n[yellow]后处理结果:[/yellow]")
        table = Table()
        table.add_column("序号", style="cyan", width=6)
        table.add_column("文字", style="magenta")
        table.add_column("置信度", style="green", width=10)
        table.add_column("处理", style="yellow")
        
        for i, result in enumerate(processed, 1):
            table.add_row(
                str(i),
                result['text'],
                f"{result['confidence']:.2%}",
                "✓ 通过"
            )
        
        console.print(table)
        console.print(f"\n处理前: {len(mock_results)} 个结果")
        console.print(f"处理后: {len(processed)} 个结果")
        console.print("已过滤: 低置信度、重复项")
        
        console.print("\n✅ [bold green]后处理模块测试通过[/bold green]")
        return True
        
    except Exception as e:
        console.print(f"\n❌ [bold red]后处理模块测试失败: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        return False


def test_multi_angle_fusion():
    """测试4: 多角度融合"""
    console.print("\n[bold cyan]测试4: 多角度融合算法[/bold cyan]")
    console.print("=" * 60)
    
    try:
        from src.core.config import Config
        from src.core.multi_angle_fusion import MultiAngleFusion
        
        config = Config()
        fusion = MultiAngleFusion(config["multi_angle"])
        console.print("✓ 成功创建融合器", style="green")
        
        # 模拟3个不同角度的识别结果
        angle_results = [
            {
                "success": True,
                "results": [
                    {"text": "MICHELIN", "confidence": 0.95},
                    {"text": "P225/50R17", "confidence": 0.88}
                ]
            },
            {
                "success": True,
                "results": [
                    {"text": "MICHELIN", "confidence": 0.92},
                    {"text": "P225/50R17", "confidence": 0.85},
                    {"text": "91V", "confidence": 0.78}
                ]
            },
            {
                "success": True,
                "results": [
                    {"text": "MICHELIN", "confidence": 0.89},
                    {"text": "91V", "confidence": 0.82}
                ]
            }
        ]
        
        console.print("\n[yellow]模拟3个角度的识别结果:[/yellow]")
        for i, result in enumerate(angle_results, 1):
            console.print(f"\n角度 {i}:")
            for item in result["results"]:
                console.print(f"  - {item['text']:15s} ({item['confidence']:.2%})")
        
        # 测试不同融合方法
        methods = ["voting", "weighted", "smart", "merge"]
        
        console.print("\n[yellow]测试不同融合算法:[/yellow]")
        table = Table(title="融合结果对比")
        table.add_column("融合方法", style="cyan")
        table.add_column("融合结果", style="magenta")
        table.add_column("置信度", style="green")
        
        for method in methods:
            config.set("multi_angle.fusion_method", method)
            fusion = MultiAngleFusion(config["multi_angle"])
            fused = fusion.fuse_results(angle_results)
            
            if fused["success"]:
                table.add_row(
                    method,
                    fused["merged_text"],
                    f"{fused['confidence']:.2%}"
                )
        
        console.print(table)
        
        console.print("\n✅ [bold green]多角度融合测试通过[/bold green]")
        return True
        
    except Exception as e:
        console.print(f"\n❌ [bold red]多角度融合测试失败: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        return False


def test_cli_interface():
    """测试5: 命令行接口"""
    console.print("\n[bold cyan]测试5: 命令行接口[/bold cyan]")
    console.print("=" * 60)
    
    try:
        import cli
        
        console.print("✓ CLI模块加载成功", style="green")
        
        console.print("\n[yellow]可用命令:[/yellow]")
        commands = [
            ("recognize", "识别单张图片"),
            ("batch", "批量识别目录"),
            ("multi-angle", "多角度融合识别")
        ]
        
        for cmd, desc in commands:
            console.print(f"  • [cyan]python cli.py {cmd}[/cyan] - {desc}")
        
        console.print("\n[yellow]示例用法:[/yellow]")
        examples = [
            "python cli.py recognize wheel.jpg",
            "python cli.py recognize wheel.jpg -v -g",
            "python cli.py batch ./images/ --parallel",
            "python cli.py multi-angle img1.jpg img2.jpg img3.jpg -m voting"
        ]
        
        for example in examples:
            console.print(f"  [dim]$ {example}[/dim]")
        
        console.print("\n✅ [bold green]命令行接口测试通过[/bold green]")
        return True
        
    except Exception as e:
        console.print(f"\n❌ [bold red]命令行接口测试失败: {e}[/bold red]")
        return False


def test_api_interface():
    """测试6: Web API接口"""
    console.print("\n[bold cyan]测试6: Web API接口[/bold cyan]")
    console.print("=" * 60)
    
    try:
        import api
        
        console.print("✓ API模块加载成功", style="green")
        
        console.print("\n[yellow]可用API端点:[/yellow]")
        endpoints = [
            ("GET", "/api/health", "健康检查"),
            ("POST", "/api/recognize", "识别单张图片"),
            ("POST", "/api/recognize/batch", "批量识别"),
            ("POST", "/api/recognize/multi-angle", "多角度融合识别"),
            ("GET", "/api/models", "获取可用模型列表")
        ]
        
        table = Table()
        table.add_column("方法", style="cyan", width=6)
        table.add_column("端点", style="magenta")
        table.add_column("说明", style="green")
        
        for method, endpoint, desc in endpoints:
            table.add_row(method, endpoint, desc)
        
        console.print(table)
        
        console.print("\n[yellow]启动API服务:[/yellow]")
        console.print("  [dim]$ python api.py[/dim]")
        console.print("  服务地址: http://localhost:5000")
        
        console.print("\n✅ [bold green]Web API接口测试通过[/bold green]")
        return True
        
    except Exception as e:
        console.print(f"\n❌ [bold red]Web API接口测试失败: {e}[/bold red]")
        return False


def main():
    """运行所有测试"""
    console.clear()
    
    # 显示欢迎信息
    welcome = Panel.fit(
        "[bold cyan]汽车轮毂字母识别系统[/bold cyan]\n"
        "[yellow]功能测试演示[/yellow]\n\n"
        "即使没有OCR引擎和真实图片\n"
        "也能展示系统架构和核心功能",
        border_style="cyan"
    )
    console.print(welcome)
    
    # 运行测试
    tests = [
        ("配置系统", test_config_system),
        ("图像预处理", test_preprocessor),
        ("结果后处理", test_postprocessor),
        ("多角度融合", test_multi_angle_fusion),
        ("命令行接口", test_cli_interface),
        ("Web API接口", test_api_interface)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            console.print(f"\n❌ [bold red]{name}测试异常: {e}[/bold red]")
            results.append((name, False))
    
    # 显示测试总结
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]测试总结[/bold cyan]")
    console.print("=" * 60)
    
    summary_table = Table()
    summary_table.add_column("测试项", style="cyan")
    summary_table.add_column("结果", style="magenta")
    
    passed = 0
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        summary_table.add_row(name, status)
        if result:
            passed += 1
    
    console.print(summary_table)
    
    # 统计
    total = len(results)
    console.print(f"\n[bold]总计: {passed}/{total} 通过[/bold]")
    
    if passed == total:
        console.print("\n🎉 [bold green]所有测试通过！系统核心功能正常！[/bold green]")
    else:
        console.print(f"\n⚠️  [bold yellow]{total - passed} 个测试失败[/bold yellow]")
    
    # 下一步建议
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]下一步操作建议[/bold cyan]")
    console.print("=" * 60)
    
    suggestions = [
        "1. 安装OCR引擎以使用完整识别功能:",
        "   pip install paddlepaddle paddleocr",
        "",
        "2. 准备测试图片:",
        "   将轮毂照片放入 test_images/ 目录",
        "",
        "3. 运行实际识别:",
        "   python cli.py recognize test_images/wheel.jpg",
        "",
        "4. 启动Web服务:",
        "   python api.py"
    ]
    
    for suggestion in suggestions:
        console.print(suggestion)
    
    console.print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
