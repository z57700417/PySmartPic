# 安装指南

本文档详细介绍如何安装和配置汽车轮毂字母识别系统。

## 系统要求

### 硬件要求

- **CPU**: 建议4核心以上
- **内存**: 建议8GB以上
- **GPU** (可选): NVIDIA GPU,支持CUDA 10.2+,用于加速推理

### 软件要求

- **操作系统**: Windows 10/11, Linux (Ubuntu 18.04+), macOS 10.15+
- **Python**: 3.8, 3.9, 3.10, 3.11
- **其他**: Git (用于克隆项目)

## 安装步骤

### 1. 克隆项目

```bash
git clone <repository_url>
cd pyPic
```

### 2. 创建虚拟环境(推荐)

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

#### 方式1: 安装所有依赖(推荐)

```bash
pip install -r requirements.txt
```

#### 方式2: 分步安装

```bash
# 核心依赖
pip install paddlepaddle paddleocr easyocr opencv-python pillow numpy

# Web API依赖
pip install flask flask-cors

# 命令行工具依赖
pip install click rich

# 其他依赖
pip install pyyaml loguru
```

### 4. 验证安装

运行测试脚本验证安装:

```bash
python -c "from src.core.config import Config; print('安装成功!')"
```

## GPU加速配置(可选)

如果你有NVIDIA GPU,可以安装GPU版本的PaddlePaddle以获得更快的推理速度。

### 1. 检查CUDA版本

```bash
nvidia-smi
```

### 2. 安装GPU版本PaddlePaddle

根据你的CUDA版本选择对应的安装命令:

#### CUDA 11.2

```bash
pip uninstall paddlepaddle
pip install paddlepaddle-gpu==2.5.0.post112 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
```

#### CUDA 11.6

```bash
pip uninstall paddlepaddle
pip install paddlepaddle-gpu==2.5.0.post116 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
```

### 3. 验证GPU

```python
import paddle
print(paddle.device.cuda.device_count())  # 应该输出GPU数量
```

## 模型下载

首次运行时,系统会自动下载所需的OCR模型。请确保:

1. **网络连接正常**
2. **有足够的磁盘空间** (约500MB)

模型会下载到以下目录:
- Windows: `C:\Users\<用户名>\.paddleocr\`
- Linux/macOS: `~/.paddleocr/`

如果下载失败,可以手动下载并解压到上述目录。

## 配置文件

复制默认配置文件并根据需要修改:

```bash
# 默认配置在 config/default_config.yaml
# 可以创建自定义配置
cp config/default_config.yaml config/my_config.yaml
```

编辑 `config/my_config.yaml` 并修改参数:

```yaml
system:
  use_gpu: true  # 如果有GPU,设置为true
  num_workers: 4  # 根据CPU核心数调整
  
recognition:
  engine: "paddleocr"  # 或 "easyocr"
  
postprocessing:
  min_confidence: 0.6  # 根据实际情况调整
```

## 常见问题

### Q1: 安装paddleocr时报错

**解决方法**: 升级pip并重新安装

```bash
pip install --upgrade pip
pip install paddleocr
```

### Q2: OpenCV无法读取图片

**解决方法**: 安装opencv-python-headless

```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

### Q3: EasyOCR下载模型失败

**解决方法**: 设置代理或手动下载模型

```bash
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
```

### Q4: 内存不足

**解决方法**: 调整批处理大小

在配置文件中减小 `rec_batch_num` 和 `num_workers`

```yaml
recognition:
  paddleocr:
    rec_batch_num: 2  # 从6减少到2

system:
  num_workers: 2  # 从4减少到2
```

### Q5: GPU内存不足

**解决方法**: 减小输入图片尺寸或使用CPU

```yaml
detection:
  paddleocr:
    max_side_len: 640  # 从960减小到640

system:
  use_gpu: false  # 改用CPU
```

## 下一步

安装完成后,请参考以下文档:

- [使用指南](usage.md) - 了解如何使用系统
- [API文档](api.md) - 了解编程接口
- [README.md](../README.md) - 快速开始

## 获取帮助

如果遇到其他问题:

1. 查看[常见问题](faq.md)
2. 提交[Issue](https://github.com/your-repo/issues)
3. 联系开发团队
