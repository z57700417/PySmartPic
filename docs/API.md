# API 接口文档 / API Documentation

[中文](#chinese) | [English](#english)

---

<a name="chinese"></a>
## 中文文档

### 🌐 API 服务

#### 启动服务

```bash
python api.py
```

服务将运行在 `http://localhost:5000`

---

### 📡 接口列表

#### 1. 健康检查

```http
GET /api/health
```

**响应:**
```json
{
  "status": "healthy",
  "service": "Wheel Hub OCR API",
  "version": "1.0.0"
}
```

---

#### 2. 单图识别

```http
POST /api/recognize
Content-Type: multipart/form-data
```

**参数:**

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| image | file | 是 | - | 图片文件 |
| engine | string | 否 | auto | 识别引擎 (auto/paddleocr/easyocr) |
| enhance | boolean | 否 | false | 启用图像增强 |
| scale_factor | float | 否 | 3.0 | 放大倍数 (2.0-4.0) |
| confidence_threshold | float | 否 | 0.6 | 置信度阈值 (0-1) |
| visualize | boolean | 否 | false | 生成可视化图片 |

**响应:**
```json
{
  "success": true,
  "total_texts": 3,
  "total_lines": 2,
  "lines": [
    {
      "text": "AT64202",
      "confidence": 0.92,
      "item_count": 1
    },
    {
      "text": "0909 W1D",
      "confidence": 0.88,
      "item_count": 2
    }
  ],
  "results": [...],
  "processing_time": 1.23,
  "engine_used": "paddleocr"
}
```

**示例:**

```bash
# cURL
curl -X POST \
  -F "image=@wheel.jpg" \
  -F "enhance=true" \
  -F "scale_factor=3.0" \
  http://localhost:5000/api/recognize
```

```python
# Python
import requests

url = 'http://localhost:5000/api/recognize'
files = {'image': open('wheel.jpg', 'rb')}
data = {
    'enhance': 'true',
    'scale_factor': '3.0',
    'confidence_threshold': '0.3'
}

response = requests.post(url, files=files, data=data)
result = response.json()

for line in result['lines']:
    print(f"{line['text']} - {line['confidence']:.2%}")
```

```javascript
// JavaScript
const formData = new FormData();
formData.append('image', fileInput.files[0]);
formData.append('enhance', 'true');
formData.append('scale_factor', '3.0');

fetch('http://localhost:5000/api/recognize', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    data.lines.forEach(line => {
        console.log(`${line.text} - ${line.confidence}`);
    });
});
```

---

#### 3. 批量识别

```http
POST /api/recognize/batch
Content-Type: multipart/form-data
```

**参数:**

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| images | file[] | 是 | - | 多个图片文件 |
| engine | string | 否 | auto | 识别引擎 |
| parallel | boolean | 否 | true | 并行处理 |
| confidence_threshold | float | 否 | 0.6 | 置信度阈值 |

**响应:**
```json
{
  "success": true,
  "total": 3,
  "results": [
    {
      "success": true,
      "total_lines": 2,
      "lines": [...]
    },
    ...
  ]
}
```

**示例:**

```bash
curl -X POST \
  -F "images=@img1.jpg" \
  -F "images=@img2.jpg" \
  -F "images=@img3.jpg" \
  -F "parallel=true" \
  http://localhost:5000/api/recognize/batch
```

---

#### 4. 多角度融合识别

```http
POST /api/recognize/multi-angle
Content-Type: multipart/form-data
```

**参数:**

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| images | file[] | 是 | - | 多个图片文件 (至少2张) |
| engine | string | 否 | auto | 识别引擎 |
| fusion_method | string | 否 | voting | 融合方法 |
| return_alternatives | boolean | 否 | true | 返回备选结果 |
| confidence_threshold | float | 否 | 0.6 | 置信度阈值 |

**融合方法:**
- `voting` - 投票融合 (推荐)
- `weighted` - 加权融合
- `smart` - 智能融合
- `merge` - 合并融合

**响应:**
```json
{
  "success": true,
  "total_lines": 2,
  "lines": [
    {
      "text": "AT64202",
      "confidence": 0.93,
      "occurrence_count": 3
    },
    {
      "text": "0909 W1D",
      "confidence": 0.89,
      "occurrence_count": 2
    }
  ],
  "merged_text": "AT64202",
  "fusion_method": "voting",
  "source_count": 3,
  "alternatives": [...]
}
```

**示例:**

```bash
curl -X POST \
  -F "images=@angle1.jpg" \
  -F "images=@angle2.jpg" \
  -F "images=@angle3.jpg" \
  -F "fusion_method=voting" \
  http://localhost:5000/api/recognize/multi-angle
```

---

#### 5. 获取可用模型

```http
GET /api/models
```

**响应:**
```json
{
  "engines": [
    {
      "name": "paddleocr",
      "description": "PaddleOCR - 速度快、准确率高",
      "supported": true
    },
    {
      "name": "easyocr",
      "description": "EasyOCR - 鲁棒性强",
      "supported": true
    }
  ],
  "fusion_methods": [...]
}
```

---

### 📊 响应字段说明

#### 行分组结果 (lines)

| 字段 | 类型 | 说明 |
|------|------|------|
| text | string | 该行的文字 |
| confidence | float | 平均置信度 (0-1) |
| item_count | int | 该行包含的文字块数量 |
| occurrence_count | int | 出现次数 (仅多角度) |

#### 原始结果 (results)

| 字段 | 类型 | 说明 |
|------|------|------|
| text | string | 识别的文字 |
| confidence | float | 置信度 (0-1) |
| bbox | array | 边界框坐标 |

---

### ❌ 错误处理

**错误响应格式:**
```json
{
  "success": false,
  "error": "错误信息"
}
```

**常见错误码:**

| 状态码 | 说明 |
|--------|------|
| 400 | 请求参数错误 |
| 413 | 文件大小超过限制 (最大16MB) |
| 500 | 服务器内部错误 |

---

<a name="english"></a>
## English Documentation

### 🌐 API Service

#### Start Service

```bash
python api.py
```

Service runs on `http://localhost:5000`

---

### 📡 Endpoints

#### 1. Health Check

```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "Wheel Hub OCR API",
  "version": "1.0.0"
}
```

---

#### 2. Single Image Recognition

```http
POST /api/recognize
Content-Type: multipart/form-data
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| image | file | Yes | - | Image file |
| engine | string | No | auto | Recognition engine |
| enhance | boolean | No | false | Enable image enhancement |
| scale_factor | float | No | 3.0 | Scale factor (2.0-4.0) |
| confidence_threshold | float | No | 0.6 | Confidence threshold |
| visualize | boolean | No | false | Generate visualization |

**Response:**
```json
{
  "success": true,
  "total_texts": 3,
  "total_lines": 2,
  "lines": [
    {
      "text": "AT64202",
      "confidence": 0.92,
      "item_count": 1
    }
  ],
  "processing_time": 1.23
}
```

---

#### 3. Multi-Angle Fusion

```http
POST /api/recognize/multi-angle
Content-Type: multipart/form-data
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| images | file[] | Yes | - | Multiple images (min 2) |
| fusion_method | string | No | voting | Fusion method |

**Fusion Methods:**
- `voting` - Voting fusion (recommended)
- `weighted` - Weighted fusion
- `smart` - Smart fusion
- `merge` - Merge fusion

**Response:**
```json
{
  "success": true,
  "total_lines": 2,
  "lines": [
    {
      "text": "AT64202",
      "confidence": 0.93,
      "occurrence_count": 3
    }
  ]
}
```

---

### 📊 Response Fields

#### Line Grouping (lines)

| Field | Type | Description |
|-------|------|-------------|
| text | string | Line text |
| confidence | float | Average confidence (0-1) |
| item_count | int | Number of text blocks |
| occurrence_count | int | Occurrence count (multi-angle only) |

---

### ❌ Error Handling

**Error Response:**
```json
{
  "success": false,
  "error": "Error message"
}
```

**HTTP Status Codes:**

| Code | Description |
|------|-------------|
| 400 | Bad request |
| 413 | File too large (max 16MB) |
| 500 | Internal server error |
