# 项目索引 - 快速导航

## 📂 项目文件完整列表

### 📋 主要文档

1. **[README.md](README.md)** - 项目主页,快速开始
2. **[QUICKSTART.md](QUICKSTART.md)** - 5分钟快速上手指南
3. **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - 详细的项目结构说明
4. **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - 项目完成总结
5. **[CHANGELOG.md](CHANGELOG.md)** - 版本更新日志
6. **[docs/installation.md](docs/installation.md)** - 安装指南

### 💻 核心代码

#### 配置管理
- **[src/core/config.py](src/core/config.py)** - 配置管理模块 (179行)

#### 图像处理
- **[src/core/preprocessor.py](src/core/preprocessor.py)** - 图像预处理模块 (215行)

#### OCR识别
- **[src/core/detector.py](src/core/detector.py)** - 文字检测模块 (134行)
- **[src/core/recognizer.py](src/core/recognizer.py)** - 主识别器模块 (362行)

#### 结果处理
- **[src/core/postprocessor.py](src/core/postprocessor.py)** - 后处理模块 (295行)
- **[src/core/multi_angle_fusion.py](src/core/multi_angle_fusion.py)** - 多角度融合模块 (289行)

### 🚀 应用入口

1. **[cli.py](cli.py)** - 命令行工具 (297行)
2. **[api.py](api.py)** - Web API服务 (343行)
3. **[examples.py](examples.py)** - 示例程序 (230行)

### ⚙️ 配置文件

- **[config/default_config.yaml](config/default_config.yaml)** - 默认配置 (112行)

### 🧪 测试文件

1. **[tests/test_config.py](tests/test_config.py)** - 配置模块测试 (65行)
2. **[tests/test_preprocessor.py](tests/test_preprocessor.py)** - 预处理模块测试 (99行)

### 📦 项目配置

1. **[requirements.txt](requirements.txt)** - Python依赖列表
2. **[setup.py](setup.py)** - 安装配置文件
3. **[.gitignore](.gitignore)** - Git忽略文件
4. **[LICENSE](LICENSE)** - MIT许可证

## 🗺️ 学习路径推荐

### 新手入门 (30分钟)

1. 阅读 **[README.md](README.md)** (5分钟)
2. 按照 **[QUICKSTART.md](QUICKSTART.md)** 安装并运行 (15分钟)
3. 运行 **[examples.py](examples.py)** 查看示例 (10分钟)

### 深入了解 (2小时)

4. 阅读 **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** 理解架构 (30分钟)
5. 阅读 **[docs/installation.md](docs/installation.md)** 了解安装细节 (15分钟)
6. 查看核心代码 **[src/core/*.py](src/core/)** (60分钟)
7. 研究配置文件 **[config/default_config.yaml](config/default_config.yaml)** (15分钟)

### 高级开发 (1天)

8. 阅读 **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** 了解完整功能 (30分钟)
9. 研究 **[cli.py](cli.py)** 和 **[api.py](api.py)** 的实现 (2小时)
10. 查看测试代码 **[tests/*.py](tests/)** (1小时)
11. 尝试修改配置和扩展功能 (4小时)

## 📊 代码统计

| 类别 | 文件数 | 代码行数 |
|------|--------|----------|
| 核心模块 | 6 | ~1,474 |
| 应用入口 | 3 | ~870 |
| 测试 | 2 | ~164 |
| 配置 | 1 | 112 |
| 文档 | 7 | ~1,800 |
| **总计** | **19** | **~4,420** |

## 🎯 快速操作指南

### 安装

```bash
pip install -r requirements.txt
```

### 使用命令行

```bash
# 单图识别
python cli.py recognize wheel.jpg

# 批量识别
python cli.py batch ./images/

# 多角度融合
python cli.py multi-angle img1.jpg img2.jpg img3.jpg
```

### 启动Web服务

```bash
python api.py
```

### 运行示例

```bash
python examples.py
```

### 运行测试

```bash
pytest tests/ -v
```

## 🔍 按功能查找

### 配置相关
- 配置管理: [src/core/config.py](src/core/config.py)
- 默认配置: [config/default_config.yaml](config/default_config.yaml)
- 配置说明: [PROJECT_STRUCTURE.md#配置文件说明](PROJECT_STRUCTURE.md)

### 图像处理
- 预处理: [src/core/preprocessor.py](src/core/preprocessor.py)
- 检测: [src/core/detector.py](src/core/detector.py)
- 识别: [src/core/recognizer.py](src/core/recognizer.py)

### 结果处理
- 后处理: [src/core/postprocessor.py](src/core/postprocessor.py)
- 多角度融合: [src/core/multi_angle_fusion.py](src/core/multi_angle_fusion.py)

### 应用接口
- 命令行: [cli.py](cli.py)
- Web API: [api.py](api.py)
- Python API: [src/core/recognizer.py](src/core/recognizer.py)

### 测试
- 配置测试: [tests/test_config.py](tests/test_config.py)
- 预处理测试: [tests/test_preprocessor.py](tests/test_preprocessor.py)

## 📞 获取帮助

- **快速问题**: 查看 [README.md#常见问题](README.md)
- **安装问题**: 查看 [docs/installation.md](docs/installation.md)
- **使用问题**: 查看 [QUICKSTART.md](QUICKSTART.md)
- **架构问题**: 查看 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- **其他问题**: 提交 [GitHub Issue](https://github.com/your-repo/issues)

## 🔗 外部资源

- [PaddleOCR 官方文档](https://github.com/PaddlePaddle/PaddleOCR)
- [EasyOCR 官方文档](https://github.com/JaidedAI/EasyOCR)
- [OpenCV 文档](https://docs.opencv.org/)
- [Flask 文档](https://flask.palletsprojects.com/)
- [Click 文档](https://click.palletsprojects.com/)

---

**最后更新**: 2025-10-15
