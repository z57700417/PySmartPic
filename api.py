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
        
        # 保存临时文件
        temp_dir = tempfile.gettempdir()
        temp_filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        temp_path = os.path.join(temp_dir, temp_filename)
        file.save(temp_path)
        
        # 更新配置
        config.set("recognition.engine", engine)
        config.set("postprocessing.min_confidence", confidence_threshold)
        
        # 执行识别
        result = recognizer.recognize(temp_path)
        
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
        
        # 更新配置
        config.set("recognition.engine", engine)
        config.set("postprocessing.min_confidence", confidence_threshold)
        config.set("multi_angle.fusion_method", fusion_method)
        config.set("multi_angle.return_alternatives", return_alternatives)
        
        # 保存临时文件
        temp_dir = tempfile.gettempdir()
        temp_paths = []
        
        for file in files:
            if file and allowed_file(file.filename):
                temp_filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
                temp_path = os.path.join(temp_dir, temp_filename)
                file.save(temp_path)
                temp_paths.append(temp_path)
                
        if len(temp_paths) < 2:
            return jsonify({
                "success": False,
                "error": "至少需要2张有效的图片文件"
            }), 400
            
        # 逐张识别
        individual_results = []
        for temp_path in temp_paths:
            result = recognizer.recognize(temp_path)
            individual_results.append(result)
            
        # 创建融合器并融合结果
        fusion = MultiAngleFusion(config["multi_angle"])
        fused_result = fusion.fuse_results(individual_results)
        
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
