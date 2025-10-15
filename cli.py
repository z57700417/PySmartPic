"""
命令行工具
提供便捷的命令行接口
"""

import click
import json
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from loguru import logger

from src.core.config import Config
from src.core.recognizer import WheelRecognizer
from src.core.multi_angle_fusion import MultiAngleFusion


console = Console()


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """汽车轮毂字母识别系统"""
    pass


@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--config', '-c', type=click.Path(exists=True), help='配置文件路径')
@click.option('--engine', '-e', type=click.Choice(['auto', 'paddleocr', 'easyocr']), 
              default='auto', help='识别引擎')
@click.option('--output', '-o', type=click.Path(), help='结果输出路径')
@click.option('--visualize', '-v', is_flag=True, help='生成可视化图片')
@click.option('--confidence', type=float, default=0.6, help='置信度阈值')
@click.option('--format', '-f', type=click.Choice(['json', 'text', 'table']), 
              default='table', help='输出格式')
@click.option('--gpu', '-g', is_flag=True, help='使用GPU加速')
@click.option('--verbose', is_flag=True, help='详细输出')
def recognize(image_path, config, engine, output, visualize, confidence, format, gpu, verbose):
    """识别单张图片中的文字"""
    
    # 配置日志
    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        
    console.print(f"[bold blue]正在识别图片:[/bold blue] {image_path}")
    
    try:
        # 加载配置
        cfg = Config(config)
        
        # 设置参数
        cfg.set("recognition.engine", engine)
        cfg.set("postprocessing.min_confidence", confidence)
        cfg.set("system.use_gpu", gpu)
        
        # 创建识别器
        recognizer = WheelRecognizer(cfg)
        
        # 执行识别
        result = recognizer.recognize(image_path)
        
        # 输出结果
        if format == 'json':
            output_json(result, output)
        elif format == 'text':
            output_text(result, output)
        else:
            output_table(result)
            
        # 可视化
        if visualize:
            vis_path = Path(image_path).parent / f"{Path(image_path).stem}_result.jpg"
            recognizer.visualize(image_path, result, vis_path)
            console.print(f"[green]可视化结果已保存到:[/green] {vis_path}")
            
        if result.get("success", False):
            console.print(f"[green]✓ 识别完成! 耗时: {result.get('processing_time', 0):.2f}秒[/green]")
        else:
            console.print(f"[red]✗ 识别失败: {result.get('error', '未知错误')}[/red]")
            
    except Exception as e:
        console.print(f"[red]✗ 错误: {e}[/red]")
        if verbose:
            raise


@cli.command()
@click.argument('image_dir', type=click.Path(exists=True))
@click.option('--config', '-c', type=click.Path(exists=True), help='配置文件路径')
@click.option('--engine', '-e', type=click.Choice(['auto', 'paddleocr', 'easyocr']), 
              default='auto', help='识别引擎')
@click.option('--output', '-o', type=click.Path(), help='结果输出路径')
@click.option('--pattern', '-p', default='*.jpg', help='文件匹配模式')
@click.option('--parallel', is_flag=True, help='并行处理')
@click.option('--gpu', '-g', is_flag=True, help='使用GPU加速')
def batch(image_dir, config, engine, output, pattern, parallel, gpu):
    """批量识别目录下的图片"""
    
    console.print(f"[bold blue]批量识别目录:[/bold blue] {image_dir}")
    
    try:
        # 查找图片文件
        image_dir_path = Path(image_dir)
        image_files = list(image_dir_path.glob(pattern))
        
        if not image_files:
            console.print(f"[yellow]未找到匹配的图片文件 (模式: {pattern})[/yellow]")
            return
            
        console.print(f"找到 {len(image_files)} 张图片")
        
        # 加载配置
        cfg = Config(config)
        cfg.set("recognition.engine", engine)
        cfg.set("system.use_gpu", gpu)
        
        # 创建识别器
        recognizer = WheelRecognizer(cfg)
        
        # 批量识别
        with Progress() as progress:
            task = progress.add_task("[cyan]处理中...", total=len(image_files))
            
            results = []
            for image_file in image_files:
                result = recognizer.recognize(image_file)
                results.append(result)
                progress.update(task, advance=1)
                
        # 输出结果
        if output:
            output_path = Path(output)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            console.print(f"[green]结果已保存到:[/green] {output_path}")
        else:
            # 显示汇总表格
            show_batch_summary(results)
            
        console.print(f"[green]✓ 批量识别完成![/green]")
        
    except Exception as e:
        console.print(f"[red]✗ 错误: {e}[/red]")
        raise


@cli.command()
@click.argument('images', nargs=-1, type=click.Path(exists=True), required=True)
@click.option('--config', '-c', type=click.Path(exists=True), help='配置文件路径')
@click.option('--fusion-method', '-m', type=click.Choice(['voting', 'weighted', 'smart', 'merge']),
              default='voting', help='融合方法')
@click.option('--output', '-o', type=click.Path(), help='结果输出路径')
@click.option('--show-alternatives', '-a', is_flag=True, help='显示备选结果')
@click.option('--gpu', '-g', is_flag=True, help='使用GPU加速')
def multi_angle(images, config, fusion_method, output, show_alternatives, gpu):
    """多角度融合识别"""
    
    console.print(f"[bold blue]多角度融合识别:[/bold blue] {len(images)} 张图片")
    
    try:
        # 加载配置
        cfg = Config(config)
        cfg.set("multi_angle.fusion_method", fusion_method)
        cfg.set("multi_angle.return_alternatives", show_alternatives)
        cfg.set("system.use_gpu", gpu)
        
        # 创建识别器
        recognizer = WheelRecognizer(cfg)
        
        # 逐张识别
        console.print("[cyan]正在识别各张图片...[/cyan]")
        individual_results = []
        with Progress() as progress:
            task = progress.add_task("[cyan]识别中...", total=len(images))
            
            for image_path in images:
                result = recognizer.recognize(image_path)
                individual_results.append(result)
                progress.update(task, advance=1)
                
        # 创建融合器
        fusion = MultiAngleFusion(cfg["multi_angle"])
        
        # 融合结果
        console.print("[cyan]正在融合结果...[/cyan]")
        fused_result = fusion.fuse_results(individual_results)
        
        # 输出结果
        if output:
            output_path = Path(output)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(fused_result, f, ensure_ascii=False, indent=2)
            console.print(f"[green]结果已保存到:[/green] {output_path}")
        else:
            output_fusion_result(fused_result)
            
        if fused_result.get("success", False):
            console.print(f"[green]✓ 融合识别完成![/green]")
        else:
            console.print(f"[red]✗ 融合失败: {fused_result.get('error', '未知错误')}[/red]")
            
    except Exception as e:
        console.print(f"[red]✗ 错误: {e}[/red]")
        raise


def output_json(result, output_path=None):
    """输出JSON格式"""
    json_str = json.dumps(result, ensure_ascii=False, indent=2)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
        console.print(f"结果已保存到: {output_path}")
    else:
        console.print(json_str)


def output_text(result, output_path=None):
    """输出纯文本格式"""
    lines = []
    
    for item in result.get("results", []):
        text = item.get("text", "")
        confidence = item.get("confidence", 0)
        lines.append(f"{text} ({confidence:.2f})")
        
    output = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output)
        console.print(f"结果已保存到: {output_path}")
    else:
        console.print(output)


def output_table(result):
    """输出表格格式"""
    table = Table(title="识别结果")
    table.add_column("序号", style="cyan", width=6)
    table.add_column("文字", style="magenta")
    table.add_column("置信度", style="green", width=10)
    
    for idx, item in enumerate(result.get("results", []), 1):
        text = item.get("text", "")
        confidence = item.get("confidence", 0)
        table.add_row(str(idx), text, f"{confidence:.2%}")
        
    console.print(table)


def show_batch_summary(results):
    """显示批量识别汇总"""
    table = Table(title="批量识别汇总")
    table.add_column("图片", style="cyan")
    table.add_column("状态", style="green", width=8)
    table.add_column("文字数", style="magenta", width=8)
    table.add_column("耗时(秒)", style="yellow", width=10)
    
    for result in results:
        image_path = Path(result.get("image_path", "")).name
        success = "成功" if result.get("success", False) else "失败"
        total_texts = result.get("total_texts", 0)
        processing_time = result.get("processing_time", 0)
        
        table.add_row(image_path, success, str(total_texts), f"{processing_time:.2f}")
        
    console.print(table)


def output_fusion_result(result):
    """输出融合结果"""
    console.print(f"\n[bold green]融合结果:[/bold green] {result.get('merged_text', '')}")
    console.print(f"[bold]置信度:[/bold] {result.get('confidence', 0):.2%}")
    console.print(f"[bold]来源图片数:[/bold] {result.get('source_count', 0)}")
    console.print(f"[bold]融合方法:[/bold] {result.get('fusion_method', '')}")
    
    # 显示备选结果
    alternatives = result.get("alternatives", [])
    if alternatives:
        console.print("\n[bold yellow]备选结果:[/bold yellow]")
        for idx, alt in enumerate(alternatives, 1):
            console.print(f"  {idx}. {alt.get('text', '')} (得分: {alt.get('score', alt.get('confidence', 0)):.2f})")


if __name__ == '__main__':
    cli()
