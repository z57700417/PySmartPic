# 快速启动指南

## 5分钟快速上手

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行示例

```bash
# 准备测试图片
mkdir test_images
# 将轮毂图片复制到 test_images/ 目录

# 运行示例程序
python examples.py
```

### 3. 命令行识别

```bash
# 识别单张图片
python cli.py recognize test_images/wheel.jpg

# 批量识别
python cli.py batch test_images/

# 多角度融合识别
python cli.py multi-angle test_images/img1.jpg test_images/img2.jpg test_images/img3.jpg
```

### 4. 启动Web API

```bash
# 启动服务
python api.py

# 服务运行在 http://localhost:5000
```

### 5. 使用Web API

```bash
# 使用curl测试API
curl -X POST -F "image=@test_images/wheel.jpg" http://localhost:5000/api/recognize
```

## Python代码示例

### 最简单的使用方式

```python
from src.core.recognizer import WheelRecognizer

# 创建识别器
recognizer = WheelRecognizer()

# 识别图片
result = recognizer.recognize("wheel.jpg")

# 打印结果
if result["success"]:
    for item in result["results"]:
        print(f"{item['text']} - 置信度: {item['confidence']:.2%}")
```

### 自定义配置

```python
from src.core.config import Config
from src.core.recognizer import WheelRecognizer

# 创建自定义配置
config = Config()
config.set("recognition.engine", "paddleocr")
config.set("postprocessing.min_confidence", 0.7)
config.set("system.use_gpu", True)

# 使用自定义配置
recognizer = WheelRecognizer(config)
result = recognizer.recognize("wheel.jpg")
```

### 批量处理

```python
from pathlib import Path
from src.core.recognizer import WheelRecognizer

recognizer = WheelRecognizer()

# 获取所有图片
images = list(Path("test_images").glob("*.jpg"))

# 批量识别
results = recognizer.recognize_batch(images, parallel=True)

# 处理结果
for result in results:
    print(f"{result['image_path']}: {result['total_texts']} 个文字")
```

### 多角度融合

```python
from src.core.config import Config
from src.core.recognizer import WheelRecognizer
from src.core.multi_angle_fusion import MultiAngleFusion

# 配置
config = Config()
config.set("multi_angle.fusion_method", "voting")

# 识别器和融合器
recognizer = WheelRecognizer(config)
fusion = MultiAngleFusion(config["multi_angle"])

# 识别多张图片
images = ["angle1.jpg", "angle2.jpg", "angle3.jpg"]
individual_results = [recognizer.recognize(img) for img in images]

# 融合结果
fused = fusion.fuse_results(individual_results)
print(f"融合结果: {fused['merged_text']}")
```

## Web API示例

### JavaScript (Fetch API)

```javascript
// 识别单张图片
const formData = new FormData();
formData.append('image', fileInput.files[0]);
formData.append('engine', 'paddleocr');
formData.append('visualize', 'true');

fetch('http://localhost:5000/api/recognize', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('识别结果:', data);
});
```

### Python (requests)

```python
import requests

# 识别单张图片
url = 'http://localhost:5000/api/recognize'
files = {'image': open('wheel.jpg', 'rb')}
data = {
    'engine': 'paddleocr',
    'visualize': 'true',
    'confidence_threshold': '0.6'
}

response = requests.post(url, files=files, data=data)
result = response.json()

print(result)
```

### cURL

```bash
# 识别单张图片
curl -X POST \
  -F "image=@wheel.jpg" \
  -F "engine=paddleocr" \
  -F "visualize=true" \
  http://localhost:5000/api/recognize

# 批量识别
curl -X POST \
  -F "images=@wheel1.jpg" \
  -F "images=@wheel2.jpg" \
  -F "images=@wheel3.jpg" \
  http://localhost:5000/api/recognize/batch

# 多角度融合
curl -X POST \
  -F "images=@angle1.jpg" \
  -F "images=@angle2.jpg" \
  -F "images=@angle3.jpg" \
  -F "fusion_method=voting" \
  http://localhost:5000/api/recognize/multi-angle
```

## 常用配置

### 提高识别准确率

```yaml
# config/high_accuracy.yaml
recognition:
  engine: "paddleocr"
  
postprocessing:
  min_confidence: 0.7  # 提高置信度阈值
  enable_correction: true  # 启用字符纠正
  
preprocessing:
  enable: true
  brightness_contrast:
    enable: true
  denoise:
    enable: true
```

### 提高处理速度

```yaml
# config/fast_processing.yaml
system:
  use_gpu: true  # 使用GPU加速
  num_workers: 8  # 增加工作线程
  
detection:
  paddleocr:
    max_side_len: 640  # 减小图片尺寸
    
preprocessing:
  enable: false  # 禁用预处理
```

### 低资源环境

```yaml
# config/low_resource.yaml
system:
  use_gpu: false
  num_workers: 2
  
recognition:
  paddleocr:
    rec_batch_num: 2
    
detection:
  paddleocr:
    max_side_len: 480
```

## 获取帮助

- 查看完整文档: [README.md](README.md)
- 安装指南: [docs/installation.md](docs/installation.md)
- API文档: 访问 http://localhost:5000 查看在线文档
- 问题反馈: 提交Issue到GitHub仓库

## 下一步

- 尝试不同的配置参数,找到最适合你的设置
- 准备自己的测试数据,评估识别效果
- 如需要,可以基于系统进行二次开发
- 考虑模型微调以提升特定场景的识别率
