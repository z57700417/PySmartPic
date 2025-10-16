# 汽车轮毂字母识别系统 / Wheel Hub OCR System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-2.7.0-green.svg)](https://github.com/PaddlePaddle/PaddleOCR)

[English](#english) | [中文](#chinese)

---

<a name="chinese"></a>
# 中文文档

## 📖 简介

一个基于深度学习的汽车轮毂字母识别系统,专门用于识别手机拍摄的汽车轮毂照片中的字母和文字信息。


## ✨ 特性

- 🎯 **高准确率** - 基于 PaddleOCR 和 EasyOCR 双引擎,识别准确率高
- 🚀 **高性能** - 支持 GPU 加速,批量处理效率高
- 🔄 **多角度融合** - 支持多张图片融合识别,提升准确率
- 📊 **按行分组** - 自动将识别结果按行分组返回,结构化输出
- 🎨 **结果可视化** - 自动标注识别结果,直观展示
- 🔧 **智能纠错** - 自动纠正常见字符混淆(如 2↔3, 6↔4)
- 🛠️ **灵活配置** - YAML 配置文件,参数可调
- 🌐 **多种部署** - 命令行工具、Web API、网页界面

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 使用方法

#### 方法 1: Web 界面 (推荐) 🌐

```bash
# 启动 API 服务
python api.py

# 在浏览器中打开 web_demo.html
```

**功能:**
- 上传 1 张图片 → 自动单图识别
- 上传多张图片 → 自动多角度融合识别
- 支持拖拽上传、图像增强、结果可视化

#### 方法 2: 命令行工具 💻

```bash
# 单图识别
python cli.py recognize wheel.jpg -v

# 批量识别
python cli.py batch ./images/ -v

# 多角度融合
python cli.py multi-angle img1.jpg img2.jpg img3.jpg -m voting
```

#### 方法 3: Python API 🐍

```python
from src.core.recognizer import WheelRecognizer

# 创建识别器
recognizer = WheelRecognizer()

# 识别图片
result = recognizer.recognize("wheel.jpg")

# 查看按行分组的结果
for i, line in enumerate(result['lines'], 1):
    print(f"第{i}行: {line['text']} (置信度: {line['confidence']:.2%})")
```

## 📊 识别结果格式

### 单图识别

```json
{
  "success": true,
  "total_texts": 3,
  "total_lines": 2,
  "lines": [
    {
      "text": "AT64202",
      "confidence": 0.92,
      "item_count": 1
    },
    {
      "text": "0909 W1D",
      "confidence": 0.88,
      "item_count": 2
    }
  ],
  "processing_time": 1.23
}
```

### 多角度融合

```json
{
  "success": true,
  "total_lines": 2,
  "lines": [
    {
      "text": "AT64202",
      "confidence": 0.93,
      "occurrence_count": 3
    },
    {
      "text": "0909 W1D",
      "confidence": 0.89,
      "occurrence_count": 2
    }
  ],
  "fusion_method": "voting"
}
```

## 🔧 常见问题解决

### 问题 1: 图片识别不出来

**解决方案:**

1. **使用图像增强** (推荐)
```bash
python enhance_recognition.py your_image.jpg
```

2. **启用 Web 界面的图像增强选项**
- 勾选 "启用图像增强"
- 选择放大倍数: 3倍或4倍

3. **调整配置参数**
```bash
python cli.py recognize image.jpg -c config/enhanced_config.yaml
```

### 问题 2: 字符识别错误 (如 AT64703 → AT64202)

**解决方案:**

```bash
# 使用纠正脚本
python correct_recognition.py your_image.jpg
```

系统会自动:
- 纠正常见混淆 (2↔3, 6↔4, 0↔O)
- 基于轮毂编号模式验证
- 提供多个候选结果

## 🌐 API 接口

### 启动服务

```bash
python api.py
# 服务运行在 http://localhost:5000
```

### 接口列表

#### 1. 单图识别
```bash
POST /api/recognize
参数:
- image: 图片文件
- enhance: 启用图像增强 (true/false)
- scale_factor: 放大倍数 (2.0-4.0)
- confidence_threshold: 置信度阈值 (0-1)
```

#### 2. 多角度融合
```bash
POST /api/recognize/multi-angle
参数:
- images: 多个图片文件
- fusion_method: 融合方法 (voting/weighted/smart/merge)
```

### JavaScript 调用示例

```javascript
const formData = new FormData();
formData.append('image', fileInput.files[0]);
formData.append('enhance', 'true');
formData.append('scale_factor', '3.0');

fetch('http://localhost:5000/api/recognize', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    // 处理按行分组的结果
    data.lines.forEach(line => {
        console.log(`${line.text} - ${line.confidence}`);
    });
});
```

## 📸 拍摄建议

### ✅ 推荐做法
- 充足光线,避免阴影
- 正面拍摄,减少角度倾斜
- 靠近拍摄,让文字占据较大画面
- 多拍几张不同角度(用于融合识别)

### ❌ 避免做法
- 逆光拍摄
- 距离太远
- 手抖模糊
- 强烈反光

## 📚 工具脚本

| 脚本 | 功能 | 使用场景 |
|------|------|---------|
| `enhance_recognition.py` | 图像增强识别 | 模糊、小文字、低对比度图片 |
| `correct_recognition.py` | 智能纠错识别 | 字符识别错误纠正 |
| `test_line_grouping.py` | 测试行分组 | 验证行分组功能 |
| `test_multi_angle_lines.py` | 测试多角度融合 | 验证融合识别功能 |

## 🛠️ 配置文件

- `config/default_config.yaml` - 默认配置
- `config/enhanced_config.yaml` - 增强识别配置(针对困难图片)

## 📝 许可证

MIT License

---

<a name="english"></a>
# English Documentation

## 📖 Introduction

A deep learning-based wheel hub character recognition system designed for recognizing text and characters in mobile-captured wheel hub photos.

### ✨ Key Features

- 🎯 **High Accuracy** - Dual-engine (PaddleOCR + EasyOCR) for high recognition rates
- 🚀 **High Performance** - GPU acceleration, efficient batch processing
- 🔄 **Multi-Angle Fusion** - Fuse multiple images for improved accuracy
- 📊 **Line Grouping** - Automatically group results by lines, structured output
- 🎨 **Result Visualization** - Automatic annotation and display
- 🔧 **Smart Correction** - Auto-correct common character confusions (2↔3, 6↔4)
- 🛠️ **Flexible Configuration** - YAML config files, adjustable parameters
- 🌐 **Multiple Deployment** - CLI, Web API, Web UI

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Usage

#### Method 1: Web Interface (Recommended) 🌐

```bash
# Start API service
python api.py

# Open web_demo.html in browser
```

**Features:**
- Upload 1 image → Auto single image recognition
- Upload multiple images → Auto multi-angle fusion
- Drag & drop, image enhancement, visualization

#### Method 2: Command Line 💻

```bash
# Single image
python cli.py recognize wheel.jpg -v

# Batch processing
python cli.py batch ./images/ -v

# Multi-angle fusion
python cli.py multi-angle img1.jpg img2.jpg img3.jpg -m voting
```

#### Method 3: Python API 🐍

```python
from src.core.recognizer import WheelRecognizer

# Create recognizer
recognizer = WheelRecognizer()

# Recognize image
result = recognizer.recognize("wheel.jpg")

# View line-grouped results
for i, line in enumerate(result['lines'], 1):
    print(f"Line {i}: {line['text']} (Confidence: {line['confidence']:.2%})")
```

## 📊 Response Format

### Single Image Recognition

```json
{
  "success": true,
  "total_texts": 3,
  "total_lines": 2,
  "lines": [
    {
      "text": "AT64202",
      "confidence": 0.92,
      "item_count": 1
    },
    {
      "text": "0909 W1D",
      "confidence": 0.88,
      "item_count": 2
    }
  ],
  "processing_time": 1.23
}
```

### Multi-Angle Fusion

```json
{
  "success": true,
  "total_lines": 2,
  "lines": [
    {
      "text": "AT64202",
      "confidence": 0.93,
      "occurrence_count": 3
    },
    {
      "text": "0909 W1D",
      "confidence": 0.89,
      "occurrence_count": 2
    }
  ],
  "fusion_method": "voting"
}
```

## 🔧 Troubleshooting

### Issue 1: Cannot Recognize Image

**Solutions:**

1. **Use Image Enhancement** (Recommended)
```bash
python enhance_recognition.py your_image.jpg
```

2. **Enable Enhancement in Web UI**
- Check "Enable Image Enhancement"
- Select scale factor: 3x or 4x

3. **Use Enhanced Config**
```bash
python cli.py recognize image.jpg -c config/enhanced_config.yaml
```

### Issue 2: Character Misrecognition (e.g., AT64703 → AT64202)

**Solution:**

```bash
# Use correction script
python correct_recognition.py your_image.jpg
```

System will automatically:
- Correct common confusions (2↔3, 6↔4, 0↔O)
- Validate with wheel code patterns
- Provide alternative candidates

## 🌐 API Reference

### Start Service

```bash
python api.py
# Service runs on http://localhost:5000
```

### Endpoints

#### 1. Single Image Recognition
```bash
POST /api/recognize
Parameters:
- image: Image file
- enhance: Enable enhancement (true/false)
- scale_factor: Scale factor (2.0-4.0)
- confidence_threshold: Confidence threshold (0-1)
```

#### 2. Multi-Angle Fusion
```bash
POST /api/recognize/multi-angle
Parameters:
- images: Multiple image files
- fusion_method: Fusion method (voting/weighted/smart/merge)
```

### JavaScript Example

```javascript
const formData = new FormData();
formData.append('image', fileInput.files[0]);
formData.append('enhance', 'true');
formData.append('scale_factor', '3.0');

fetch('http://localhost:5000/api/recognize', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    // Process line-grouped results
    data.lines.forEach(line => {
        console.log(`${line.text} - ${line.confidence}`);
    });
});
```

## 📸 Photography Tips

### ✅ Best Practices
- Adequate lighting, avoid shadows
- Front-facing angle, minimize distortion
- Close-up shots, make text fill frame
- Multiple angles (for fusion recognition)

### ❌ Avoid
- Backlit photos
- Too far distance
- Motion blur
- Strong reflections

## 📚 Utility Scripts

| Script | Function | Use Case |
|--------|----------|----------|
| `enhance_recognition.py` | Enhanced recognition | Blurry, small text, low contrast |
| `correct_recognition.py` | Smart correction | Character misrecognition |
| `test_line_grouping.py` | Test line grouping | Validate grouping feature |
| `test_multi_angle_lines.py` | Test fusion | Validate fusion feature |

## 🛠️ Configuration

- `config/default_config.yaml` - Default config
- `config/enhanced_config.yaml` - Enhanced config (for difficult images)

## 📝 License

MIT License

---

## 🔗 Quick Links

- 📖 [Full Documentation](DIFFICULT_IMAGE_GUIDE.md)
- 🐛 [Report Issues](https://github.com/your-repo/issues)
- 💬 [Discussions](https://github.com/your-repo/discussions)

