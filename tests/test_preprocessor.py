"""
单元测试 - 图像预处理模块
"""

import pytest
import numpy as np
import cv2
from src.core.preprocessor import ImagePreprocessor


@pytest.fixture
def sample_image():
    """创建测试图像"""
    # 创建一个简单的测试图像
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # 添加一些内容
    cv2.rectangle(image, (20, 20), (80, 80), (255, 255, 255), -1)
    cv2.putText(image, "TEST", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return image


@pytest.fixture
def preprocessor_config():
    """预处理器配置"""
    return {
        "enable": True,
        "brightness_contrast": {
            "enable": True,
            "clip_limit": 2.0,
            "tile_size": [8, 8]
        },
        "denoise": {
            "enable": True,
            "method": "bilateral",
            "kernel_size": 5,
            "sigma": 1.5
        },
        "edge_enhancement": {
            "enable": False
        },
        "binarization": {
            "enable": False
        },
        "morphology": {
            "enable": False
        }
    }


def test_preprocessor_init(preprocessor_config):
    """测试预处理器初始化"""
    preprocessor = ImagePreprocessor(preprocessor_config)
    assert preprocessor.enable == True
    assert preprocessor.config == preprocessor_config


def test_preprocessor_process(sample_image, preprocessor_config):
    """测试预处理流程"""
    preprocessor = ImagePreprocessor(preprocessor_config)
    result = preprocessor.process(sample_image)
    
    # 检查输出是否为图像
    assert isinstance(result, np.ndarray)
    assert result.shape == sample_image.shape


def test_brightness_contrast_adjustment(sample_image, preprocessor_config):
    """测试亮度对比度调整"""
    preprocessor = ImagePreprocessor(preprocessor_config)
    result = preprocessor._adjust_brightness_contrast(sample_image)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == sample_image.shape


def test_denoise(sample_image, preprocessor_config):
    """测试去噪"""
    preprocessor = ImagePreprocessor(preprocessor_config)
    
    # 测试不同去噪方法
    for method in ["gaussian", "median", "bilateral"]:
        preprocessor.config["denoise"]["method"] = method
        result = preprocessor._denoise(sample_image)
        assert isinstance(result, np.ndarray)


def test_preprocessor_disabled(sample_image):
    """测试禁用预处理"""
    config = {"enable": False}
    preprocessor = ImagePreprocessor(config)
    result = preprocessor.process(sample_image)
    
    # 禁用时应该返回原图
    assert np.array_equal(result, sample_image)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
