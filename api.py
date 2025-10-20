"""
Web API服务
使用Flask提供RESTful接口
"""

import os
import tempfile
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from loguru import logger

from src.core.config import Config
from src.core.recognizer import WheelRecognizer
from src.core.multi_angle_fusion import MultiAngleFusion
import cv2
import numpy as np


# 创建Flask应用
app = Flask(__name__)
CORS(app)

# 配置
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB最大文件大小
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

# 全局变量
recognizer = None
config = None


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def init_app(config_path=None):
    """初始化应用"""
    global recognizer, config
    
    # 加载配置
    config = Config(config_path)
    
    # 创建识别器
    recognizer = WheelRecognizer(config)
    
    logger.info("Web API服务初始化完成")


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        "status": "healthy",
        "service": "Wheel Hub OCR API",
        "version": "1.0.0"
    })


def enhance_image_for_ocr(image_path, scale_factor=3.0):
    """增强图像用于OCR识别"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # 智能计算最佳放大倍数
    if scale_factor == 0 or scale_factor < 0:  # 0 表示自动判断
        scale_factor = calculate_optimal_scale(image)
        logger.info(f"自动判断最佳放大倍数: {scale_factor}x")
    
    # 放大图像
    if scale_factor > 1.0:
        new_width = int(image.shape[1] * scale_factor)
        new_height = int(image.shape[0] * scale_factor)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # 转灰度
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 方法1: 直接增强对比度 - 适合大多数场景
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 双边滤波去噪（保留边缘）
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # 温和锐化
    kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    enhanced = cv2.filter2D(enhanced, -1, kernel_sharpen)
    
    # 直接返回灰度图（不二倿化）- PaddleOCR更适合原始灰度图
    # 转回BGR格式
    result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    logger.info(f"图像增强完成: 放大{scale_factor}x, 最终尺寸: {result.shape}")
    return result


def calculate_optimal_scale(image):
    """
    智能计算最佳放大倍数
    
    基于图像尺寸、清晰度、对比度等因素综合判断
    
    Args:
        image: 输入图像
        
    Returns:
        最佳放大倍数
    """
    height, width = image.shape[:2]
    
    # 1. 基于尺寸判断
    min_dim = min(height, width)
    max_dim = max(height, width)
    
    # 小图片需要更大的放大倍数
    if min_dim < 800:
        base_scale = 4.0
    elif min_dim < 1200:
        base_scale = 3.5
    elif min_dim < 1600:
        base_scale = 3.0
    elif min_dim < 2400:
        base_scale = 2.0
    else:
        base_scale = 1.5
    
    # 2. 基于清晰度判断(拉普拉斯方差)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 模糊图片需要更大的放大
    if laplacian_var < 100:  # 非常模糊
        blur_factor = 1.3
    elif laplacian_var < 300:  # 较模糊
        blur_factor = 1.2
    elif laplacian_var < 500:  # 轻微模糊
        blur_factor = 1.1
    else:  # 清晰
        blur_factor = 1.0
    
    # 3. 基于对比度判断
    contrast = gray.std()
    if contrast < 30:  # 低对比度
        contrast_factor = 1.2
    elif contrast < 50:
        contrast_factor = 1.1
    else:
        contrast_factor = 1.0
    
    # 综合计算
    optimal_scale = base_scale * blur_factor * contrast_factor
    
    # 限制在合理范围内(1.5-5.0)
    optimal_scale = max(1.5, min(5.0, optimal_scale))
    
    # 限制最终图像尺寸不超过8000像素(避免过大)
    final_max_dim = max_dim * optimal_scale
    if final_max_dim > 8000:
        optimal_scale = 8000 / max_dim
    
    # 四舍五入到0.5
    optimal_scale = round(optimal_scale * 2) / 2
    
    logger.debug(f"图像分析: 尺寸={width}x{height}, 清晰度={laplacian_var:.1f}, 对比度={contrast:.1f}")
    
    return optimal_scale


@app.route('/api/recognize', methods=['POST'])
def recognize():
    """识别单张图片"""
    
    # 检查是否有文件
    if 'image' not in request.files:
        return jsonify({
            "success": False,
            "error": "未找到图片文件"
        }), 400
        
    file = request.files['image']
    
    # 检查文件名
    if file.filename == '':
        return jsonify({
            "success": False,
            "error": "文件名为空"
        }), 400
        
    # 检查文件类型
    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": "不支持的文件格式"
        }), 400
        
    try:
        # 获取参数
        engine = request.form.get('engine', 'auto')
        visualize = request.form.get('visualize', 'false').lower() == 'true'
        confidence_threshold = float(request.form.get('confidence_threshold', '0.6'))
        enhance = request.form.get('enhance', 'false').lower() == 'true'  # 是否启用增强
        scale_factor = float(request.form.get('scale_factor', '3.0'))  # 放大倍数
        region_filter = request.form.get('region_filter', 'false').lower() == 'true'  # 区域过滤
        center_only = request.form.get('center_only', 'false').lower() == 'true'  # 只识别中心
        
        # 保存临时文件
        temp_dir = tempfile.gettempdir()
        temp_filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        temp_path = os.path.join(temp_dir, temp_filename)
        file.save(temp_path)
        
        # 图像增强处理
        if enhance:
            enhanced_image = enhance_image_for_ocr(temp_path, scale_factor)
            if enhanced_image is not None:
                enhanced_path = os.path.join(temp_dir, f"enhanced_{temp_filename}")
                cv2.imwrite(enhanced_path, enhanced_image)
                temp_path = enhanced_path
                logger.info(f"应用图像增强处理 (放大{scale_factor}x)")
        
        # 更新配置
        config.set("recognition.engine", engine)
        config.set("postprocessing.min_confidence", confidence_threshold)
        config.set("postprocessing.enable_region_filter", region_filter)
        config.set("postprocessing.center_region_only", center_only)
        
        # 如果启用增强,使用更宽松的检测参数
        if enhance:
            config.set("detection.paddleocr.det_db_thresh", 0.05)  # 更低的阈值,提高敏感度
            config.set("detection.paddleocr.det_db_box_thresh", 0.1)  # 更宽松的框阈值
            config.set("detection.paddleocr.det_db_unclip_ratio", 2.5)  # 更大的展开比例
            config.set("preprocessing.enable", True)  # 启用额外预处理
            config.set("postprocessing.min_confidence", 0.3)  # 降低置信度阈值
        
        # 执行识别
        result = recognizer.recognize(temp_path)
        
        # 添加增强信息到结果
        if enhance:
            result["enhanced"] = True
            result["scale_factor"] = scale_factor
        
        # 可视化
        visualization_path = None
        if visualize and result.get("success", False):
            vis_filename = f"{uuid.uuid4()}_result.jpg"
            visualization_path = os.path.join(temp_dir, vis_filename)
            recognizer.visualize(temp_path, result, visualization_path)
            result["visualization_url"] = f"/api/visualization/{vis_filename}"
            
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"识别失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/recognize/batch', methods=['POST'])
def recognize_batch():
    """批量识别"""
    
    # 检查是否有文件
    if 'images' not in request.files:
        return jsonify({
            "success": False,
            "error": "未找到图片文件"
        }), 400
        
    files = request.files.getlist('images')
    
    if not files:
        return jsonify({
            "success": False,
            "error": "图片列表为空"
        }), 400
        
    try:
        # 获取参数
        engine = request.form.get('engine', 'auto')
        parallel = request.form.get('parallel', 'true').lower() == 'true'
        confidence_threshold = float(request.form.get('confidence_threshold', '0.6'))
        
        # 更新配置
        config.set("recognition.engine", engine)
        config.set("postprocessing.min_confidence", confidence_threshold)
        
        # 保存临时文件
        temp_dir = tempfile.gettempdir()
        temp_paths = []
        
        for file in files:
            if file and allowed_file(file.filename):
                temp_filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
                temp_path = os.path.join(temp_dir, temp_filename)
                file.save(temp_path)
                temp_paths.append(temp_path)
                
        if not temp_paths:
            return jsonify({
                "success": False,
                "error": "没有有效的图片文件"
            }), 400
            
        # 批量识别
        results = recognizer.recognize_batch(temp_paths, parallel=parallel)
        
        # 清理临时文件
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        return jsonify({
            "success": True,
            "total": len(results),
            "results": results
        })
        
    except Exception as e:
        logger.error(f"批量识别失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/recognize/multi-angle', methods=['POST'])
def recognize_multi_angle():
    """多角度融合识别"""
    
    # 检查是否有文件
    if 'images' not in request.files:
        return jsonify({
            "success": False,
            "error": "未找到图片文件"
        }), 400
        
    files = request.files.getlist('images')
    
    if len(files) < 2:
        return jsonify({
            "success": False,
            "error": "至少需要2张图片"
        }), 400
        
    try:
        # 获取参数
        engine = request.form.get('engine', 'auto')
        fusion_method = request.form.get('fusion_method', 'voting')
        visualize = request.form.get('visualize', 'false').lower() == 'true'
        confidence_threshold = float(request.form.get('confidence_threshold', '0.6'))
        return_alternatives = request.form.get('return_alternatives', 'true').lower() == 'true'
        enhance = request.form.get('enhance', 'false').lower() == 'true'  # 是否启用增强
        scale_factor = float(request.form.get('scale_factor', '3.0'))  # 放大倍数
        region_filter = request.form.get('region_filter', 'false').lower() == 'true'  # 区域过滤
        center_only = request.form.get('center_only', 'false').lower() == 'true'  # 只识别中心
        
        # 更新配置
        config.set("recognition.engine", engine)
        config.set("postprocessing.min_confidence", confidence_threshold)
        config.set("multi_angle.fusion_method", fusion_method)
        config.set("multi_angle.return_alternatives", return_alternatives)
        config.set("postprocessing.enable_region_filter", region_filter)
        config.set("postprocessing.center_region_only", center_only)
        
        # 如果启用增强,使用更宽松的检测参数
        if enhance:
            config.set("detection.paddleocr.det_db_thresh", 0.05)
            config.set("detection.paddleocr.det_db_box_thresh", 0.1)
            config.set("detection.paddleocr.det_db_unclip_ratio", 2.5)
            config.set("preprocessing.enable", True)
            config.set("postprocessing.min_confidence", 0.3)
        
        # 保存临时文件
        temp_dir = tempfile.gettempdir()
        temp_paths = []
        
        for file in files:
            if file and allowed_file(file.filename):
                temp_filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
                temp_path = os.path.join(temp_dir, temp_filename)
                file.save(temp_path)
                
                # 图像增强处理
                if enhance:
                    enhanced_image = enhance_image_for_ocr(temp_path, scale_factor)
                    if enhanced_image is not None:
                        enhanced_path = os.path.join(temp_dir, f"enhanced_{temp_filename}")
                        cv2.imwrite(enhanced_path, enhanced_image)
                        temp_path = enhanced_path
                
                temp_paths.append(temp_path)
        
        if enhance:
            logger.info(f"多角度识别: 应用图像增强处理 (放大{scale_factor}x, {len(temp_paths)}张图片)")
                
        if len(temp_paths) < 2:
            return jsonify({
                "success": False,
                "error": "至少需要2张有效的图片文件"
            }), 400
            
        # 逐张识别
        individual_results = []
        visualization_urls = []
        for idx, temp_path in enumerate(temp_paths):
            result = recognizer.recognize(temp_path)
            individual_results.append(result)
            
            # 生成可视化图片
            if visualize and result.get("success", False):
                vis_filename = f"{uuid.uuid4()}_img{idx+1}_result.jpg"
                vis_path = os.path.join(temp_dir, vis_filename)
                recognizer.visualize(temp_path, result, vis_path)
                visualization_urls.append(f"/api/visualization/{vis_filename}")
            
        # 创建融合器并融合结果
        fusion = MultiAngleFusion(config["multi_angle"])
        fused_result = fusion.fuse_results(individual_results)
        
        # 添加可视化URLs到结果
        if visualization_urls:
            fused_result["visualization_urls"] = visualization_urls
        
        # 清理临时文件
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        return jsonify(fused_result)
        
    except Exception as e:
        logger.error(f"多角度融合识别失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/visualization/<filename>', methods=['GET'])
def get_visualization(filename):
    """获取可视化图片"""
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)
    
    if not os.path.exists(file_path):
        return jsonify({
            "success": False,
            "error": "文件不存在"
        }), 404
        
    return send_file(file_path, mimetype='image/jpeg')


@app.route('/api/models', methods=['GET'])
def get_models():
    """获取可用模型列表"""
    return jsonify({
        "engines": [
            {
                "name": "paddleocr",
                "description": "PaddleOCR - 速度快、准确率高",
                "supported": True
            },
            {
                "name": "easyocr",
                "description": "EasyOCR - 鲁棒性强",
                "supported": True
            }
        ],
        "fusion_methods": [
            {
                "name": "voting",
                "description": "投票融合 - 选择出现频率最高的结果"
            },
            {
                "name": "weighted",
                "description": "加权融合 - 根据置信度加权"
            },
            {
                "name": "smart",
                "description": "智能融合 - 选择最高置信度的结果"
            },
            {
                "name": "merge",
                "description": "合并融合 - 合并所有不重复的文字"
            }
        ]
    })


@app.errorhandler(413)
def request_entity_too_large(error):
    """文件过大错误处理"""
    return jsonify({
        "success": False,
        "error": "文件大小超过限制(最大16MB)"
    }), 413


if __name__ == '__main__':
    # 初始化应用
    init_app()
    
    # 启动服务
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"启动Web API服务,端口: {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
