"""
云端OCR识别引擎
支持阿里云、腾讯云、百度云等多个云服务提供商
"""

import base64
import json
import time
from typing import List, Dict, Any
from loguru import logger
import cv2
import numpy as np
import os


class CloudOCREngine:
    """云端OCR引擎基类"""
    
    def __init__(self, config: dict):
        self.config = config
        self.enabled = config.get("enabled", False)
        self.provider = config.get("provider", "aliyun")
        
    def recognize(self, image_path: str) -> List[Dict[str, Any]]:
        """识别图片中的文字"""
        if not self.enabled:
            return []
            
        if self.provider == "aliyun":
            return self._recognize_aliyun(image_path)
        elif self.provider == "tencent":
            return self._recognize_tencent(image_path)
        elif self.provider == "baidu":
            return self._recognize_baidu(image_path)
        else:
            logger.error(f"不支持的云OCR提供商: {self.provider}")
            return []
    
    def _recognize_aliyun(self, image_path: str) -> List[Dict[str, Any]]:
        """阿里云文字识别"""
        try:
            from aliyunsdkcore.client import AcsClient
            from aliyunsdkocr.request.v20191230 import RecognizeCharacterRequest
            import requests
            import uuid
            
            access_key_id = self.config.get("aliyun", {}).get("access_key_id")
            access_key_secret = self.config.get("aliyun", {}).get("access_key_secret")
            
            if not access_key_id or not access_key_secret:
                logger.error("阿里云OCR配置缺失")
                return []
            
            # 读取图片并检查尺寸
            image = cv2.imread(image_path)
            if image is None:
                logger.error("无法读取图片文件")
                return []
            
            # 检查图片尺寸，如果太大则进行压缩
            max_size = 1920  # 最大边长限制
            height, width = image.shape[:2]
            
            if max(height, width) > max_size:
                # 计算缩放比例
                scale = max_size / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                # 调整图片尺寸
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                logger.info(f"图片尺寸从 {width}x{height} 压缩到 {new_width}x{new_height}")
            
            # 编码为JPEG格式以减小文件大小
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]  # JPEG质量85%
            _, buffer = cv2.imencode('.jpg', image, encode_param)
            image_data = buffer.tobytes()
            
            # 检查数据大小
            if len(image_data) > 5 * 1024 * 1024:  # 5MB限制
                logger.warning(f"图片数据过大 ({len(image_data)} bytes)，尝试进一步压缩")
                # 进一步降低质量
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                _, buffer = cv2.imencode('.jpg', image, encode_param)
                image_data = buffer.tobytes()
                logger.info(f"进一步压缩后数据大小: {len(image_data)} bytes")
            
            # 创建客户端
            client = AcsClient(access_key_id, access_key_secret, 'cn-shanghai')
            
            # 创建请求
            request = RecognizeCharacterRequest.RecognizeCharacterRequest()
            request.set_accept_format('json')
            
            # 使用base64数据URL（确保格式正确）
            import base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            request.set_ImageURL(f"data:image/jpeg;base64,{image_base64}")
            
            # 设置必需参数
            request.set_OutputProbability(True)  # 设置输出概率为True（必需参数）
            request.set_MinHeight(8)  # 最小高度
            
            # 发送请求
            response = client.do_action_with_exception(request)
            result = json.loads(response)
            
            # 解析结果
            results = []
            if result.get('Data', {}).get('Results'):
                for item in result['Data']['Results']:
                    results.append({
                        'text': item['Text'],
                        'confidence': item.get('Probability', 1.0),
                        'bbox': self._convert_aliyun_bbox(item.get('TextRectangles', {})),
                        'orientation': 'horizontal'
                    })
            
            logger.info(f"阿里云OCR识别到 {len(results)} 个文字")
            return results
            
        except Exception as e:
            logger.error(f"阿里云OCR识别失败: {e}")
            # 检查是否是特定的错误类型
            error_msg = str(e)
            if "InvalidApi.NotPurchase" in error_msg:
                logger.error("阿里云OCR服务未开通或未购买，请访问以下链接开通服务:")
                logger.error("https://help.aliyun.com/document_detail/465341.html")
            elif "InvalidImage.URL" in error_msg:
                logger.error("阿里云OCR不支持base64数据URL格式，请将图片上传到阿里云OSS后使用OSS链接")
                logger.error("参考文档: https://help.aliyun.com/document_detail/155645.html")
            elif "MissingOutputProbability" in error_msg:
                logger.error("阿里云OCR请求缺少必需参数 OutputProbability")
            elif "403" in error_msg:
                logger.error("阿里云OCR服务访问被拒绝，请检查AccessKey权限设置")
            elif "401" in error_msg:
                logger.error("阿里云OCR认证失败，请检查AccessKey ID和Secret是否正确")
            elif "413" in error_msg:
                logger.error("图片数据过大，请使用更小尺寸的图片")
            return []
    
    def _recognize_tencent(self, image_path: str) -> List[Dict[str, Any]]:
        """腾讯云通用印刷体识别"""
        try:
            from tencentcloud.common import credential
            from tencentcloud.ocr.v20181119 import ocr_client, models
            
            secret_id = self.config.get("tencent", {}).get("secret_id")
            secret_key = self.config.get("tencent", {}).get("secret_key")
            
            if not secret_id or not secret_key:
                logger.error("腾讯云OCR配置缺失")
                return []
            
            # 读取图片并编码
            with open(image_path, 'rb') as f:
                image_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            # 创建客户端
            cred = credential.Credential(secret_id, secret_key)
            client = ocr_client.OcrClient(cred, "ap-guangzhou")
            
            # 创建请求
            req = models.GeneralBasicOCRRequest()
            req.ImageBase64 = image_base64
            
            # 发送请求
            resp = client.GeneralBasicOCR(req)
            
            # 解析结果
            results = []
            for item in resp.TextDetections:
                results.append({
                    'text': item.DetectedText,
                    'confidence': item.Confidence / 100.0,  # 转换为0-1
                    'bbox': self._convert_tencent_bbox(item.Polygon),
                    'orientation': 'horizontal'
                })
            
            logger.info(f"腾讯云OCR识别到 {len(results)} 个文字")
            return results
            
        except Exception as e:
            logger.error(f"腾讯云OCR识别失败: {e}")
            return []
    
    def _recognize_baidu(self, image_path: str) -> List[Dict[str, Any]]:
        """百度云通用文字识别"""
        try:
            from aip import AipOcr
            
            app_id = self.config.get("baidu", {}).get("app_id")
            api_key = self.config.get("baidu", {}).get("api_key")
            secret_key = self.config.get("baidu", {}).get("secret_key")
            
            if not all([app_id, api_key, secret_key]):
                logger.error("百度云OCR配置缺失")
                return []
            
            # 创建客户端
            client = AipOcr(app_id, api_key, secret_key)
            
            # 读取图片
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # 发送请求
            result = client.basicGeneral(image_data)
            
            # 解析结果
            results = []
            if 'words_result' in result:
                for item in result['words_result']:
                    results.append({
                        'text': item['words'],
                        'confidence': item.get('probability', {}).get('average', 1.0),
                        'bbox': self._convert_baidu_bbox(item.get('location', {})),
                        'orientation': 'horizontal'
                    })
            
            logger.info(f"百度云OCR识别到 {len(results)} 个文字")
            return results
            
        except Exception as e:
            logger.error(f"百度云OCR识别失败: {e}")
            return []
    
    def _convert_aliyun_bbox(self, rect: dict) -> List[List[int]]:
        """转换阿里云bbox格式"""
        try:
            left = rect.get('Left', 0)
            top = rect.get('Top', 0)
            width = rect.get('Width', 0)
            height = rect.get('Height', 0)
            
            return [
                [left, top],
                [left + width, top],
                [left + width, top + height],
                [left, top + height]
            ]
        except:
            return [[0, 0], [0, 0], [0, 0], [0, 0]]
    
    def _convert_tencent_bbox(self, polygon: list) -> List[List[int]]:
        """转换腾讯云bbox格式"""
        try:
            return [[p.X, p.Y] for p in polygon]
        except:
            return [[0, 0], [0, 0], [0, 0], [0, 0]]
    
    def _convert_baidu_bbox(self, location: dict) -> List[List[int]]:
        """转换百度云bbox格式"""
        try:
            left = location.get('left', 0)
            top = location.get('top', 0)
            width = location.get('width', 0)
            height = location.get('height', 0)
            
            return [
                [left, top],
                [left + width, top],
                [left + width, top + height],
                [left, top + height]
            ]
        except:
            return [[0, 0], [0, 0], [0, 0], [0, 0]]


# 配置示例
"""
在 config/default_config.yaml 中添加:

recognition:
  cloud_ocr:
    enabled: false  # 是否启用云OCR
    provider: aliyun  # aliyun, tencent, baidu
    
    # 级联策略
    use_as_fallback: true  # 作为本地OCR的备用方案
    fallback_confidence_threshold: 0.7  # 低于此置信度时使用云OCR
    
    # 阿里云配置
    aliyun:
      access_key_id: YOUR_ACCESS_KEY_ID
      access_key_secret: YOUR_ACCESS_KEY_SECRET
    
    # 腾讯云配置
    tencent:
      secret_id: YOUR_SECRET_ID
      secret_key: YOUR_SECRET_KEY
    
    # 百度云配置
    baidu:
      app_id: YOUR_APP_ID
      api_key: YOUR_API_KEY
      secret_key: YOUR_SECRET_KEY
"""

# 使用示例
"""
from src.core.cloud_ocr import CloudOCREngine

# 初始化
config = {
    "enabled": True,
    "provider": "aliyun",
    "aliyun": {
        "access_key_id": "xxx",
        "access_key_secret": "xxx"
    }
}

cloud_ocr = CloudOCREngine(config)

# 识别
results = cloud_ocr.recognize("path/to/image.jpg")

for result in results:
    print(f"文字: {result['text']}, 置信度: {result['confidence']}")
"""
