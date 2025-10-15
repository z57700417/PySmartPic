"""
配置管理模块
负责加载和管理系统配置参数
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger


class Config:
    """配置管理类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径,如果为None则使用默认配置
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            配置字典
        """
        # 默认配置文件路径
        default_config_path = Path(__file__).parent.parent.parent / "config" / "default_config.yaml"
        
        # 如果未指定配置文件,使用默认配置
        if self.config_path is None:
            config_path = default_config_path
        else:
            config_path = Path(self.config_path)
            
        # 检查配置文件是否存在
        if not config_path.exists():
            logger.warning(f"配置文件不存在: {config_path}, 使用默认配置")
            config_path = default_config_path
            
        # 加载配置文件
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"成功加载配置文件: {config_path}")
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return self._get_default_config()
            
    def _get_default_config(self) -> Dict[str, Any]:
        """
        获取默认配置
        
        Returns:
            默认配置字典
        """
        return {
            "preprocessing": {
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
                }
            },
            "detection": {
                "engine": "paddleocr"
            },
            "recognition": {
                "engine": "auto"
            },
            "postprocessing": {
                "min_confidence": 0.6,
                "min_length": 1,
                "max_length": 30
            },
            "system": {
                "use_gpu": False,
                "num_workers": 4,
                "log_level": "INFO"
            }
        }
        
    def _validate_config(self):
        """验证配置参数"""
        # 验证必要的配置项
        required_sections = ["preprocessing", "detection", "recognition", "postprocessing", "system"]
        for section in required_sections:
            if section not in self.config:
                logger.warning(f"缺少配置节: {section}, 使用默认值")
                self.config[section] = self._get_default_config().get(section, {})
                
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值(支持嵌套路径)
        
        Args:
            key_path: 配置键路径,用.分隔,如 "preprocessing.denoise.method"
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
        
    def set(self, key_path: str, value: Any):
        """
        设置配置值(支持嵌套路径)
        
        Args:
            key_path: 配置键路径,用.分隔
            value: 配置值
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
            
        config[keys[-1]] = value
        logger.debug(f"设置配置: {key_path} = {value}")
        
    def save(self, output_path: Optional[str] = None):
        """
        保存配置到文件
        
        Args:
            output_path: 输出路径,如果为None则覆盖原配置文件
        """
        if output_path is None:
            output_path = self.config_path
            
        if output_path is None:
            logger.error("未指定输出路径")
            return
            
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)
            logger.info(f"配置已保存到: {output_path}")
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            
    def __getitem__(self, key: str) -> Any:
        """支持字典访问方式"""
        return self.config.get(key)
        
    def __setitem__(self, key: str, value: Any):
        """支持字典赋值方式"""
        self.config[key] = value
        
    def __repr__(self) -> str:
        return f"Config(path={self.config_path})"
