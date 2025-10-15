# 项目结构说明

## 目录结构

```
pyPic/
├── src/                          # 源代码目录
│   ├── __init__.py              # 包初始化
│   └── core/                    # 核心模块
│       ├── __init__.py          # 核心模块初始化
│       ├── config.py            # 配置管理模块
│       ├── preprocessor.py      # 图像预处理模块
│       ├── detector.py          # 文字检测模块
│       ├── recognizer.py        # 主识别器模块
│       ├── postprocessor.py     # 后处理模块
│       └── multi_angle_fusion.py # 多角度融合模块
│
├── config/                       # 配置文件目录
│   └── default_config.yaml      # 默认配置文件
│
├── tests/                        # 测试目录
│   ├── __init__.py              # 测试包初始化
│   ├── test_config.py           # 配置模块测试
│   └── test_preprocessor.py    # 预处理模块测试
│
├── docs/                         # 文档目录
│   └── installation.md          # 安装指南
│
├── cli.py                        # 命令行工具入口
├── api.py                        # Web API服务入口
├── examples.py                   # 示例程序
├── setup.py                      # 安装配置文件
├── requirements.txt              # 依赖列表
├── README.md                     # 项目说明
├── QUICKSTART.md                 # 快速启动指南
└── .gitignore                    # Git忽略文件

运行时生成的目录:
├── models/                       # 模型缓存目录(自动创建)
├── temp/                         # 临时文件目录(自动创建)
├── output/                       # 输出目录(自动创建)
└── test_images/                  # 测试图片目录(用户创建)
```

## 核心模块说明

### 1. config.py - 配置管理模块

**功能**: 负责加载和管理系统配置参数

**主要类**:
- `Config`: 配置管理器,支持YAML配置文件加载、嵌套键访问

**使用示例**:
```python
from src.core.config import Config

config = Config("config/default_config.yaml")
min_confidence = config.get("postprocessing.min_confidence")
config.set("system.use_gpu", True)
```

### 2. preprocessor.py - 图像预处理模块

**功能**: 提升图像质量,消除干扰因素

**主要类**:
- `ImagePreprocessor`: 图像预处理器

**处理功能**:
- 亮度对比度调整(CLAHE)
- 去噪处理(高斯、中值、双边滤波)
- 边缘增强(Sobel、Laplacian)
- 二值化(自适应、Otsu)
- 形态学处理(膨胀、腐蚀、开运算、闭运算)

**使用示例**:
```python
from src.core.preprocessor import ImagePreprocessor

preprocessor = ImagePreprocessor(config["preprocessing"])
processed_image = preprocessor.process(original_image)
```

### 3. detector.py - 文字检测模块

**功能**: 定位图像中的文字区域

**主要类**:
- `TextDetector`: 文字检测器,集成PaddleOCR检测功能

**使用示例**:
```python
from src.core.detector import TextDetector

detector = TextDetector(config["detection"])
detections = detector.detect(image)
```

### 4. recognizer.py - 主识别器模块

**功能**: 整合预处理、检测、识别、后处理等功能

**主要类**:
- `WheelRecognizer`: 轮毂字母识别器

**核心方法**:
- `recognize()`: 识别单张图片
- `recognize_batch()`: 批量识别
- `visualize()`: 结果可视化

**使用示例**:
```python
from src.core.recognizer import WheelRecognizer

recognizer = WheelRecognizer()
result = recognizer.recognize("wheel.jpg")
```

### 5. postprocessor.py - 后处理模块

**功能**: 过滤无效结果,提升准确性

**主要类**:
- `PostProcessor`: 后处理器

**处理功能**:
- 置信度过滤
- 长度过滤
- 字符类型过滤
- 相似字符纠正(O/0, I/1/l, S/5等)
- 结果去重

**使用示例**:
```python
from src.core.postprocessor import PostProcessor

postprocessor = PostProcessor(config["postprocessing"])
filtered_results = postprocessor.process(raw_results)
```

### 6. multi_angle_fusion.py - 多角度融合模块

**功能**: 融合多张图片的识别结果

**主要类**:
- `MultiAngleFusion`: 多角度融合器

**融合方法**:
- `voting`: 投票融合 - 选择出现频率最高的结果
- `weighted`: 加权融合 - 根据置信度加权
- `smart`: 智能融合 - 选择最高置信度的结果
- `merge`: 合并融合 - 合并所有不重复的文字

**使用示例**:
```python
from src.core.multi_angle_fusion import MultiAngleFusion

fusion = MultiAngleFusion(config["multi_angle"])
fused_result = fusion.fuse_results(individual_results)
```

## 入口文件说明

### cli.py - 命令行工具

**功能**: 提供命令行接口

**主要命令**:
- `recognize`: 识别单张图片
- `batch`: 批量识别
- `multi-angle`: 多角度融合识别

**技术栈**: Click, Rich

### api.py - Web API服务

**功能**: 提供RESTful API接口

**主要端点**:
- `GET /api/health`: 健康检查
- `POST /api/recognize`: 识别单张图片
- `POST /api/recognize/batch`: 批量识别
- `POST /api/recognize/multi-angle`: 多角度融合识别
- `GET /api/models`: 获取可用模型列表

**技术栈**: Flask, Flask-CORS

### examples.py - 示例程序

**功能**: 演示系统各种功能的使用方法

**示例内容**:
- 单图识别
- 批量识别
- 多角度融合识别
- 结果可视化
- 自定义配置

## 配置文件说明

### default_config.yaml

**配置节**:
- `preprocessing`: 预处理配置
- `detection`: 文字检测配置
- `recognition`: 文字识别配置
- `postprocessing`: 后处理配置
- `multi_angle`: 多角度融合配置
- `system`: 系统配置
- `visualization`: 可视化配置

**自定义配置**:
复制默认配置文件并修改参数:
```bash
cp config/default_config.yaml config/my_config.yaml
# 编辑 my_config.yaml
# 使用: python cli.py recognize wheel.jpg --config config/my_config.yaml
```

## 测试文件说明

### test_config.py

测试配置管理模块的功能:
- 配置加载
- 配置读取
- 配置设置
- 字典访问

### test_preprocessor.py

测试图像预处理模块的功能:
- 预处理器初始化
- 预处理流程
- 各种预处理算法
- 禁用预处理

**运行测试**:
```bash
pytest tests/ -v
```

## 扩展开发指南

### 添加新的预处理算法

1. 在 `preprocessor.py` 中添加方法
2. 在配置文件中添加相应配置
3. 在 `process()` 方法中调用

### 添加新的OCR引擎

1. 在 `recognizer.py` 中添加初始化方法
2. 添加识别方法
3. 在配置中添加引擎选项

### 添加新的融合算法

1. 在 `multi_angle_fusion.py` 中添加融合方法
2. 在 `fuse_results()` 中调用
3. 在配置中添加算法选项

## 性能优化建议

### 1. 使用GPU加速

```yaml
system:
  use_gpu: true
```

### 2. 调整批处理大小

```yaml
recognition:
  paddleocr:
    rec_batch_num: 6  # 根据GPU内存调整
```

### 3. 调整图片尺寸

```yaml
detection:
  paddleocr:
    max_side_len: 960  # 减小尺寸可提升速度
```

### 4. 禁用不必要的预处理

```yaml
preprocessing:
  enable: false  # 或只启用必要的处理
```

## 常见问题

### Q: 如何添加对中文的支持?

修改配置文件:
```yaml
recognition:
  paddleocr:
    lang: "ch"  # 从"en"改为"ch"
```

### Q: 如何保存识别历史?

可以扩展 `recognizer.py`,在识别后保存结果到数据库或文件。

### Q: 如何集成到现有项目?

作为Python包导入:
```python
from src.core.recognizer import WheelRecognizer
recognizer = WheelRecognizer()
```

或通过Web API调用:
```python
import requests
response = requests.post('http://localhost:5000/api/recognize', ...)
```

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交代码
4. 编写测试
5. 提交Pull Request

## 许可证

MIT License - 详见LICENSE文件
