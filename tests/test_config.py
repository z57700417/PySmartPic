"""
单元测试 - 配置管理模块
"""

import pytest
from pathlib import Path
from src.core.config import Config


def test_config_load_default():
    """测试加载默认配置"""
    config = Config()
    
    # 检查基本配置节是否存在
    assert "preprocessing" in config.config
    assert "detection" in config.config
    assert "recognition" in config.config
    assert "postprocessing" in config.config
    assert "system" in config.config


def test_config_get():
    """测试获取配置值"""
    config = Config()
    
    # 测试单层键
    assert config.get("system") is not None
    
    # 测试嵌套键
    min_confidence = config.get("postprocessing.min_confidence")
    assert min_confidence is not None
    assert isinstance(min_confidence, (int, float))
    
    # 测试不存在的键
    assert config.get("nonexistent.key", "default") == "default"


def test_config_set():
    """测试设置配置值"""
    config = Config()
    
    # 设置单层键
    config.set("test_key", "test_value")
    assert config.get("test_key") == "test_value"
    
    # 设置嵌套键
    config.set("test.nested.key", 123)
    assert config.get("test.nested.key") == 123


def test_config_dict_access():
    """测试字典访问方式"""
    config = Config()
    
    # 读取
    assert config["system"] is not None
    
    # 写入
    config["test_key"] = "test_value"
    assert config["test_key"] == "test_value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
