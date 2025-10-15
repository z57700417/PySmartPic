# Wheel Hub Letter Recognition System

A deep learning-based OCR system tailored for recognizing letters and text engraved or printed on automotive wheel hubs. It supports command-line tools, a web API service, and direct Python integration.

> Project Status: Production-ready
> Last Updated: 2025-10-15

## ✨ Features

- High accuracy with dual OCR engines: PaddleOCR and EasyOCR
- Fast performance with optional GPU acceleration
- Multi-angle fusion to improve robustness and recall
- Built-in visualization: draw bounding boxes and text on images
- YAML-based flexible configuration
- Multiple deployment modes: CLI and Web API

## 📦 System Requirements

- Python 3.8+
- Windows / Linux / macOS
- Optional: NVIDIA GPU + CUDA for acceleration (PaddleOCR/EasyOCR)

## 🚀 Quick Start

1) Install dependencies

```bash
pip install -r requirements.txt
```

2) Prepare test images

```bash
mkdir test_images
# copy your wheel hub photos into test_images/
```

3) Run examples

```bash
python examples.py
```

## 🧰 CLI Usage

CLI entry is implemented in [cli.py](cli.py). Below are the main commands.

### Single image recognition

```bash
python cli.py recognize <image_path> [options]

Options:
  -c, --config PATH       Path to YAML config
  -e, --engine TEXT       OCR engine (auto/paddleocr/easyocr)
  -o, --output PATH       Output file for results
  -v, --visualize         Save visualization image
  --confidence FLOAT      Min confidence threshold (default: 0.6)
  -f, --format TEXT       Output format (json/text/table)
  -g, --gpu               Use GPU acceleration
  --verbose               Verbose logging
```

### Batch recognition

```bash
python cli.py batch <image_dir> [options]

Options:
  -c, --config PATH       Path to YAML config
  -e, --engine TEXT       OCR engine
  -o, --output PATH       Output file for batch results
  -p, --pattern TEXT      Glob pattern (default: *.jpg)
  --parallel              Enable parallel processing
  -g, --gpu               Use GPU acceleration
```

### Multi-angle fusion

```bash
python cli.py multi-angle <img1> <img2> ... [options]

Options:
  -c, --config PATH       Path to YAML config
  -m, --fusion-method     Fusion method (voting/weighted/smart/merge)
  -o, --output PATH       Output file for fused result
  -a, --show-alternatives Show alternative candidates
  -g, --gpu               Use GPU acceleration
```

## 🐍 Python API

```python
from src.core.recognizer import WheelRecognizer

# Create recognizer
recognizer = WheelRecognizer()

# Recognize a single image
result = recognizer.recognize("wheel.jpg")

# Print results
for item in result["results"]:
    print(f"text: {item['text']}, conf: {item['confidence']:.2%}")
```

Visualization is available via `WheelRecognizer.visualize(image_path, result, output_path)`.

## 🌐 Web API (Flask)

Web service entry is [api.py](api.py). Default port is 5000.

- Start server:

```bash
python api.py
```

### Endpoints

1) Health check

```
GET /api/health
```

2) Single image recognition

```
POST /api/recognize
Content-Type: multipart/form-data

Params:
- image (file, required)
- engine (optional, default: auto)
- visualize (optional, default: false)
- confidence_threshold (optional, default: 0.6)
```

3) Batch recognition

```
POST /api/recognize/batch
Content-Type: multipart/form-data

Params:
- images (multiple files, required)
- engine (optional)
- parallel (optional, default: true)
- confidence_threshold (optional)
```

4) Multi-angle fusion

```
POST /api/recognize/multi-angle
Content-Type: multipart/form-data

Params:
- images (>=2 files, required)
- engine (optional)
- fusion_method (optional, default: voting)
- visualize (optional)
- confidence_threshold (optional)
- return_alternatives (optional, default: true)
```

## ⚙️ Configuration (YAML)

Default config file: [config/default_config.yaml](config/default_config.yaml)

Key sections:
- preprocessing: brightness/contrast, denoise, edge enhancement, binarization, morphology
- detection: PaddleOCR DB detector, max_side_len
- recognition: engine selection, PaddleOCR/EasyOCR settings
- postprocessing: min_confidence, character filtering, correction rules, deduplication, min_results
- multi_angle: fusion method and thresholds
- system: GPU usage, worker count, logging
- visualization: draw bbox/text/confidence

Example snippet:

```yaml
recognition:
  engine: "auto"
  paddleocr:
    lang: "en"

postprocessing:
  min_confidence: 0.6
  enable_correction: true
  enable_deduplication: true
  min_results: 4
```

## 🔍 Multi-Angle Fusion

Implemented in [src/core/multi_angle_fusion.py](src/core/multi_angle_fusion.py). Available methods:
- voting: frequency × confidence × length weighting
- weighted: confidence-weighted aggregation
- smart: pick highest-confidence and close alternatives
- merge: merge all non-duplicate texts, sorted by confidence

## 🧪 Tests

Basic tests are under [tests/](tests/). Run with:

```bash
pytest -q
```

## 📈 Performance Targets

- Character recognition accuracy: ≥ 90%
- End-to-end accuracy: ≥ 85%
- Avg processing time (GPU): ≤ 1s/image
- Avg processing time (CPU): ≤ 3s/image

## 🧯 Troubleshooting

- Pip too old warning:
  Upgrade with `python -m pip install --upgrade pip`.

- NumPy alias error (module has no attribute 'int'):
  Use NumPy 1.23.5 for compatibility with PaddleOCR.

- PaddleOCR vs EasyOCR:
  If one engine fails to initialize, the system can fall back to the other when `engine=auto`.

- First run model downloads:
  OCR engines may download models on first run—ensure network connectivity.

- Windows path issues:
  Always use absolute paths if relative resolution fails, and verify image files exist.

## 📚 Project Structure & Docs

- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- [QUICKSTART.md](QUICKSTART.md)
- [README.md](README.md) (Chinese)

## 🤝 Contributing

PRs and issues are welcome.

## 📝 License

MIT License

## 📧 Contact

your-email@example.com
