# 汽车轮毂字母识别系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-2.7.0-green.svg)](https://github.com/PaddlePaddle/PaddleOCR)

一个基于深度学习的汽车轮毂字母识别系统,专门用于识别手机拍摄的汽车轮毂照片中的字母和文字信息。

> 📅 **最后更新**: 2025-10-15

## ✨ 特性

- 🎯 **高准确率** - 基于PaddleOCR和EasyOCR双引擎,识别准确率高
- 🚀 **高性能** - 支持GPU加速,批量处理效率高
- 🔄 **多角度融合** - 支持多张图片融合识别,提升准确率
- 🎨 **结果可视化** - 自动标注识别结果,直观展示
- 🛠️ **灵活配置** - YAML配置文件,参数可调
- 📦 **多种部署** - 命令行工具、Web API服务

### 📊 项目完成度

| 模块 | 状态 | 说明 |
|------|------|------|
| 配置管理 | ✅ | YAML配置、嵌套键访问 |
| 图像预处理 | ✅ | 亮度调整、去噪、边缘增强等 |
| 文字检测 | ✅ | PaddleOCR检测器 |
| 文字识别 | ✅ | PaddleOCR + EasyOCR双引擎 |
| 后处理 | ✅ | 过滤、纠正、去重 |
| 单图识别 | ✅ | 完整的识别流程 |
| 批量识别 | ✅ | 并行处理支持 |
| 多角度融合 | ✅ | 4种融合算法 |
| 结果可视化 | ✅ | 边界框标注 |
| 命令行工具 | ✅ | CLI完整实现 |
| Web API | ✅ | RESTful API |
| 文档 | ✅ | 完整的使用指南 |
| 测试 | ✅ | 单元测试 |

## 📋 系统要求

- Python 3.8+
- Windows / Linux / macOS
- (可选) NVIDIA GPU + CUDA 用于GPU加速

## 🚀 快速开始

### 1. 安装依赖

```bash
# 克隆项目
git clone <repository_url>
cd pyPic

# 安装依赖
pip install -r requirements.txt
```

### 2. 基本使用

#### 命令行识别

```bash
# 识别单张图片
python cli.py recognize wheel.jpg

# 批量识别
python cli.py batch ./images/

# 多角度融合识别
python cli.py multi-angle img1.jpg img2.jpg img3.jpg --fusion-method voting
```

#### Python API

```python
from src.core.recognizer import WheelRecognizer

# 创建识别器
recognizer = WheelRecognizer()

# 识别图片
result = recognizer.recognize("wheel.jpg")

# 打印结果
for item in result["results"]:
    print(f"文字: {item['text']}, 置信度: {item['confidence']:.2%}")
```

#### Web API服务

```bash
# 启动服务
python api.py

# 访问 http://localhost:5000
# API文档: http://localhost:5000/api/health
```

## 📖 详细文档

### 命令行工具

#### 单图识别

```bash
python cli.py recognize <图片路径> [选项]

选项:
  -c, --config PATH       配置文件路径
  -e, --engine TEXT       识别引擎 (auto/paddleocr/easyocr)
  -o, --output PATH       结果输出路径
  -v, --visualize         生成可视化图片
  --confidence FLOAT      置信度阈值 (默认: 0.6)
  -f, --format TEXT       输出格式 (json/text/table)
  -g, --gpu              使用GPU加速
  --verbose              详细输出
```

#### 批量识别

```bash
python cli.py batch <图片目录> [选项]

选项:
  -c, --config PATH       配置文件路径
  -e, --engine TEXT       识别引擎
  -o, --output PATH       结果输出路径
  -p, --pattern TEXT      文件匹配模式 (默认: *.jpg)
  --parallel             并行处理
  -g, --gpu              使用GPU加速
```

#### 多角度融合识别

```bash
python cli.py multi-angle <图片1> <图片2> ... [选项]

选项:
  -c, --config PATH       配置文件路径
  -m, --fusion-method     融合方法 (voting/weighted/smart/merge)
  -o, --output PATH       结果输出路径
  -a, --show-alternatives 显示备选结果
  -g, --gpu              使用GPU加速
```

### Web API接口

#### 1. 健康检查

```http
GET /api/health
```

#### 2. 识别单张图片

```http
POST /api/recognize
Content-Type: multipart/form-data

参数:
- image: 图片文件 (必需)
- engine: 识别引擎 (可选, 默认: auto)
- visualize: 是否返回可视化图片 (可选, 默认: false)
- confidence_threshold: 置信度阈值 (可选, 默认: 0.6)
```

#### 3. 批量识别

```http
POST /api/recognize/batch
Content-Type: multipart/form-data

参数:
- images: 多个图片文件 (必需)
- engine: 识别引擎 (可选)
- parallel: 是否并行处理 (可选, 默认: true)
- confidence_threshold: 置信度阈值 (可选)
```

#### 4. 多角度融合识别

```http
POST /api/recognize/multi-angle
Content-Type: multipart/form-data

参数:
- images: 多个图片文件 (必需, 至少2张)
- engine: 识别引擎 (可选)
- fusion_method: 融合方法 (可选, 默认: voting)
- visualize: 是否返回可视化图片 (可选)
- confidence_threshold: 置信度阈值 (可选)
- return_alternatives: 是否返回备选结果 (可选, 默认: true)
```

### 配置文件

配置文件位于 `config/default_config.yaml`,可以自定义以下参数:

```yaml
# 预处理配置
preprocessing:
  enable: true
  brightness_contrast:
    enable: true
    clip_limit: 2.0
  denoise:
    enable: true
    method: "bilateral"
  # ...更多配置

# 识别配置
recognition:
  engine: "auto"
  paddleocr:
    lang: "en"
  # ...更多配置

# 后处理配置
postprocessing:
  min_confidence: 0.6
  enable_correction: true
  # ...更多配置
```

## 🎯 使用场景

- 汽车维修店快速识别轮毂型号
- 二手车交易平台自动提取轮毂信息
- 汽车配件电商平台商品信息录入
- 保险公司车辆配件核验

## 🔧 技术栈

- **深度学习框架**: PaddlePaddle, PyTorch
- **OCR引擎**: PaddleOCR, EasyOCR
- **图像处理**: OpenCV, Pillow
- **Web框架**: Flask
- **命令行工具**: Click, Rich

## 📊 性能指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 字符识别准确率 | ≥ 90% | 正确识别字符数 / 总字符数 |
| 端到端识别准确率 | ≥ 85% | 完全正确的识别结果 / 总样本数 |
| 平均处理时间 (GPU) | ≤ 1秒/张 | 单张图片处理时间 |
| 平均处理时间 (CPU) | ≤ 3秒/张 | 单张图片处理时间 |

## 🤝 贡献

欢迎提交Issue和Pull Request!

## 📝 许可证

MIT License

## 📧 联系方式

如有问题或建议,请联系: [your-email@example.com]

## 📚 文档导航

- 🚀 [**快速开始**](QUICKSTART.md) - 5分钟上手指南
- 📝 [**项目结构**](PROJECT_STRUCTURE.md) - 详细的项目结构说明
- 🛠️ [**安装指南**](docs/installation.md) - 完整的安装教程
- 📊 [**项目总结**](PROJECT_SUMMARY.md) - 项目完成情况总结
- 📖 [**更新日志**](CHANGELOG.md) - 版本更新记录
- 💻 [**示例代码**](examples.py) - 丰富的使用示例

---

**注意**: 首次运行时会自动下载OCR模型,请确保网络连接正常。
