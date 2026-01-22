# Vision-to-Action

**Production-grade AI vision system with hardware-optimized perception, structured ingestion, LLM cognition, and automated orchestration.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-nano-green.svg)](https://github.com/ultralytics/ultralytics)
[![Gemini API](https://img.shields.io/badge/Gemini-2.0--flash-orange.svg)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 🎯 Overview

**Vision-to-Action** is a complete end-to-end AI system that transforms visual perception into automated actions. Designed for **hardware-constrained environments** (6GB VRAM), it demonstrates production-ready AI engineering with:

- **Real-time object detection** at 37 FPS
- **AI-generated image detection** with 11 advanced signals
- **LLM-powered scene understanding** via Gemini API
- **Automated workflow orchestration** through n8n
- **97.4% manual effort reduction** through intelligent automation

### System Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│ AI Detection│───▶│  Perception  │───▶│  Ingestion  │───▶│  Cognition   │
│  (11 sigs)  │    │  (YOLOv8)    │    │   (JSON)    │    │  (Gemini)    │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                            │                                      │
                            └──────────────────────────────────────┘
                                            │
                                    ┌───────▼────────┐
                                    │   Action       │
                                    │   (n8n)        │
                                    └────────────────┘
```

---

## ✨ Key Features

### 🔍 **Advanced Computer Vision**
- **YOLOv8 nano** optimized for 6GB VRAM (37 FPS average)
- **FP16 precision** for 2x speed improvement
- **Conditional segmentation** (patch-based, triggered on demand)
- **imgshape v4 integration** for dataset governance

### 🤖 **AI-Generated Image Detection**
- **11 detection signals**: Spectral analysis, Benford's Law, LBP texture, gradient smoothness, noise patterns, and more
- **68% detection rate** on test dataset
- **Multi-scale analysis**: FFT, patch coherence, edge consistency
- **Production-ready**: Integrated into main inference pipeline

### 📊 **Structured Ingestion**
- **Pydantic schemas** for type-safe event validation
- **Date-organized storage** (YYYY/MM/DD hierarchy)
- **JSON-first** with optional database backends
- **Temporal indexing** for efficient retrieval

### 🧠 **LLM Cognition**
- **Gemini 2.0 Flash** for natural language understanding
- **Scene summarization** with spatial and confidence analysis
- **Anomaly explanation** for edge cases
- **Temporal pattern analysis** across event sequences

### ⚡ **Workflow Orchestration**
- **n8n integration** for automated actions
- **Webhook triggers** based on detection events
- **Alert workflows** for anomalies and thresholds
- **Escalation pipelines** for critical events

---

## 📊 Performance Metrics

*Tested on NVIDIA GeForce RTX 3050 6GB Laptop GPU*

| Layer | Metric | Value | Target |
|-------|--------|-------|--------|
| **Perception** | Average Latency | 63.5ms | <100ms ✅ |
| | Average FPS | 37.2 | >10 FPS ✅ |
| | Detection Coverage | 54.5% | N/A |
| | Avg Confidence | 51.8% | >25% ✅ |
| **AI Detection** | Detection Rate | 68.2% | N/A |
| | Average Latency | 5.87s | <10s ✅ |
| | False Positive Rate | Low | N/A |
| **Ingestion** | Storage Latency | 1.9ms | <10ms ✅ |
| | Events Stored | 100% | 100% ✅ |
| | Avg Event Size | 761 bytes | N/A |
| **Cognition** | Success Rate | 100% | >95% ✅ |
| | Avg Latency | 1.96s | <5s ✅ |
| | Summaries Generated | 22/22 | N/A |
| **End-to-End** | Total Latency | 7.99s | <10s ✅ |
| | Success Rate | 100% | >99% ✅ |
| | Manual Effort Reduction | 97.4% | >90% ✅ |

---

## 🏗️ Architecture

### Component Breakdown

#### 1. **AI Detection Layer** 🆕
```python
from cv.analysis.ai_detection import AIImageDetector

detector = AIImageDetector()
result = detector.analyze("image.png")
# Returns: verdict, confidence, signals, explanation
```

**Detection Signals (11 total)**:
1. **Dimension Match** - Common AI sizes (512², 1024², etc.)
2. **Spectral Analysis** - FFT radial power spectrum slopes
3. **Local Binary Patterns** - Texture uniformity (LBP entropy)
4. **Benford's Law** - Pixel value distribution violations
5. **Color Gradients** - Over-smoothing detection
6. **High-Frequency Suppression** - Strongest signal (AI smoothing)
7. **Edge Consistency** - Unnatural sharpness uniformity
8. **Patch Coherence** - Perfect cross-patch coherence
9. **Multi-Scale Noise** - Lacks natural sensor noise
10. **Compression Patterns** - JPEG artifact analysis
11. **EXIF Metadata** - Camera metadata presence (weak signal)

#### 2. **Perception Layer**
```python
from cv.detection import YOLOv8Detector

detector = YOLOv8Detector(
    variant="nano",
    device="cuda:0",
    use_fp16=True,
    input_size=(640, 640)
)
results = detector.detect(image)
```

**Features**:
- YOLOv8 nano (6MB model) optimized for 6GB VRAM
- FP16 mixed precision (2x speedup)
- Configurable confidence/IOU thresholds
- Batch processing support
- Optional imgshape preprocessing

#### 3. **Ingestion Layer**
```python
from ingestion.schemas import PerceptionEvent
from ingestion.storage import JSONStorageBackend

storage = JSONStorageBackend()
event = PerceptionEvent(...)
storage.store_event(event)
```

**Schema Structure**:
```python
PerceptionEvent:
  - event_id: str
  - timestamp: datetime
  - task_type: TaskType (DETECTION, SEGMENTATION, etc.)
  - image_metadata: ImageMetadata
  - detections: List[Detection]
  - processing_metadata: Dict
  - imgshape_validation: Optional[Dict]
```

#### 4. **Cognition Layer**
```python
from cognition.gemini import GeminiReasoner

reasoner = GeminiReasoner(model_name="gemini-2.0-flash")
summary = reasoner.summarize_event(event)
```

**Capabilities**:
- Scene summarization with spatial analysis
- Anomaly explanation and recommendations
- Temporal pattern detection across events
- Multi-event trend analysis

#### 5. **Orchestration Layer**
```yaml
# n8n workflow example
trigger: webhook
condition: detections > threshold
actions:
  - send_alert
  - log_event
  - escalate_if_critical
```

---

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.12+
- **CUDA**: 11.8+ (for GPU acceleration)
- **VRAM**: 6GB minimum
- **Gemini API Key**: [Get from Google AI Studio](https://aistudio.google.com/app/apikey)

### Installation

```bash
# Clone repository
git clone https://github.com/STiFLeR7/vision-to-action.git
cd vision-to-action

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install imgshape (if using local version)
pip install -e ./imgshape

# Set up environment variables
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### Configuration

Create `.env` file:
```bash
GEMINI_API_KEY=your_api_key_here
IMGSHAPE_BASE_URL=http://localhost:8000
```

### Run Inference

```bash
# Single image inference
python scripts/run_inference.py path/to/image.jpg

# Output:
# 🤖 AI DETECTION
#    AI-Generated: 68.2% confidence
#    Signals: High-frequency suppression, Missing EXIF...
# 
# 👁️  PERCEPTION LAYER
#    Found 3 objects: person (0.89), car (0.76), bicycle (0.65)
#
# 📝 COGNITION LAYER
#    Summary: The image contains three detected objects...
```

### Run Evaluation

```bash
# Comprehensive evaluation on test dataset
python scripts/evaluate_system.py

# Generates:
# - eval/metrics_TIMESTAMP.json (detailed metrics)
# - eval/summary_TIMESTAMP.json (executive summary)
# - eval/EVALUATION_REPORT.md (full analysis)
```

### Validate System

```bash
# Health check for all components
python scripts/validate_system.py

# Checks:
# ✅ GPU availability (6GB VRAM)
# ✅ YOLOv8 model loading
# ✅ Gemini API connectivity
# ✅ imgshape service status
# ✅ Storage backends
```

---

## 📁 Project Structure

```
vision-to-action/
├── cv/                          # Computer vision modules
│   ├── detection/               # Object detection (YOLOv8)
│   │   ├── yolov8_detector.py  # Main detector class
│   │   └── __init__.py
│   ├── segmentation/            # Conditional segmentation
│   │   ├── conditional_segmenter.py
│   │   └── __init__.py
│   ├── analysis/                # Image analysis 🆕
│   │   ├── ai_detection.py     # AI-generated image detection
│   │   └── __init__.py
│   └── training/                # Training pipelines
│       ├── train_yolov8.py
│       └── utils.py
├── ingestion/                   # Data ingestion layer
│   ├── schemas/                 # Pydantic schemas
│   │   ├── perception_schema.py
│   │   └── __init__.py
│   └── storage/                 # Storage backends
│       ├── json_storage.py
│       └── __init__.py
├── cognition/                   # LLM cognition layer
│   └── gemini/
│       ├── reasoner.py          # Gemini API integration
│       └── __init__.py
├── orchestration/               # Workflow orchestration
│   └── n8n/
│       └── workflows/           # n8n workflow templates
├── configs/                     # Configuration files
│   ├── system.yaml              # System config
│   ├── training.yaml            # Training config
│   ├── cognition.yaml           # Cognition config
│   └── orchestration.yaml       # Workflow config
├── scripts/                     # Executable scripts
│   ├── run_inference.py         # Main inference pipeline
│   ├── evaluate_system.py       # Comprehensive evaluation
│   └── validate_system.py       # System validation
├── data/                        # Data directory
│   ├── test/                    # Test images (22 samples)
│   ├── ingestion/               # Ingestion storage
│   │   └── events/              # Event records (date-organized)
│   └── models/                  # Model weights
├── eval/                        # Evaluation results
│   ├── metrics_*.json           # Detailed metrics
│   ├── summary_*.json           # Executive summaries
│   └── EVALUATION_REPORT.md     # Full analysis
├── imgshape/                    # imgshape v4 symlink
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

---

## 🔧 Configuration

### System Configuration (`configs/system.yaml`)

```yaml
hardware:
  target_device: "cuda:0"
  vram_limit_gb: 6
  precision: "fp16"

models:
  detection:
    variant: "nano"
    input_size: [640, 640]
    confidence_threshold: 0.25
    iou_threshold: 0.45

ingestion:
  storage_backend: "json"
  base_path: "data/ingestion"

cognition:
  model: "gemini-2.0-flash"
  temperature: 0.2
  max_output_tokens: 2048
```

### Training Configuration (`configs/training.yaml`)

```yaml
training:
  epochs: 100
  batch_size: 8
  learning_rate: 0.001
  optimizer: "AdamW"
  
augmentation:
  enabled: true
  techniques:
    - "random_crop"
    - "horizontal_flip"
    - "brightness_contrast"
```

---

## 🧪 Evaluation & Testing

### Test Dataset

Located in `data/test/` - 22 diverse images including:
- Real photographs (JPEG with EXIF)
- AI-generated graphics (Midjourney, DALL-E style)
- Design mockups and diagrams
- Mixed content types

### Running Evaluation

```bash
python scripts/evaluate_system.py
```

**Metrics Collected**:
- **Perception**: FPS, latency, mAP, confidence scores
- **AI Detection**: Detection rate, signal analysis, false positives
- **Ingestion**: Storage latency, event sizes, validation errors
- **Cognition**: Success rate, latency, token usage
- **End-to-End**: Total latency, workflow triggers, effort reduction

### Results Summary

Latest evaluation (`eval/summary_20260122_075116.json`):
```json
{
  "perception": {
    "avg_fps": 37.23,
    "avg_latency_ms": 63.48,
    "total_detections": 21
  },
  "ai_detection": {
    "ai_generated_pct": 68.2,
    "avg_latency_ms": 5870.71
  },
  "cognition": {
    "success_rate": 1.0,
    "avg_latency_ms": 1955.97
  },
  "end_to_end": {
    "success_rate": 1.0,
    "manual_effort_reduction_pct": 97.4
  }
}
```

---

## 💡 Usage Examples

### Example 1: Basic Detection

```python
from cv.detection import YOLOv8Detector
import cv2

# Initialize detector
detector = YOLOv8Detector(variant="nano", use_fp16=True)

# Load image
image = cv2.imread("image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run detection
result = detector.detect(image_rgb)

# Process results
for box, score, cls_name in zip(result.boxes, result.scores, result.class_names):
    print(f"{cls_name}: {score:.2%} at {box}")
```

### Example 2: AI Detection

```python
from cv.analysis.ai_detection import AIImageDetector

# Initialize detector
detector = AIImageDetector()

# Analyze image
result = detector.analyze("image.png")

# Check verdict
if result['is_ai_generated']:
    print(f"AI-Generated ({result['ai_confidence']:.1%})")
    print(f"Signals: {result['explanation']}")
else:
    print(f"Original/Real ({result['original_confidence']:.1%})")
```

### Example 3: Complete Pipeline

```python
from pathlib import Path
from cv.detection import YOLOv8Detector
from cv.analysis.ai_detection import AIImageDetector
from ingestion.storage import JSONStorageBackend
from ingestion.schemas import PerceptionEvent, TaskType, ImageMetadata
from cognition.gemini import GeminiReasoner
import cv2

# Initialize components
ai_detector = AIImageDetector()
detector = YOLOv8Detector()
storage = JSONStorageBackend()
reasoner = GeminiReasoner()

# Load image
image_path = "path/to/image.jpg"
image = cv2.imread(image_path)

# 1. AI Detection
ai_result = ai_detector.analyze(image_path)
print(f"AI Detection: {ai_result['verdict']}")

# 2. Perception
detection_result = detector.detect(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
print(f"Detected {len(detection_result.boxes)} objects")

# 3. Ingestion
event = PerceptionEvent(
    event_id=f"evt_{int(time.time())}",
    task_type=TaskType.DETECTION,
    image_metadata=ImageMetadata(...),
    detections=[...]  # Convert detection_result
)
event_path = storage.store_event(event)

# 4. Cognition
summary = reasoner.summarize_event(event)
print(f"Summary: {summary}")
```

### Example 4: Batch Processing

```python
from pathlib import Path
from cv.detection import YOLOv8Detector

detector = YOLOv8Detector(variant="nano")

# Get all images
image_dir = Path("data/test")
images = []
for img_path in image_dir.glob("*.jpg"):
    img = cv2.imread(str(img_path))
    images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# Batch detection
results = detector.detect_batch(images, batch_size=4)

# Process results
for img_path, result in zip(image_dir.glob("*.jpg"), results):
    print(f"{img_path.name}: {len(result.boxes)} detections")
```

---

## 🔗 Integrations

### imgshape v4 (Atlas)

**Integration Points**:
- **Preprocessing**: Image validation and normalization
- **Dataset Management**: Version control and cataloging
- **Quality Assurance**: Automated validation pipelines

**Usage**:
```python
# imgshape service assumed at localhost:8000
# Preprocessing via REST API
import requests

response = requests.post(
    "http://localhost:8000/v4/preprocess",
    json={"image_path": "..."}
)
```

### Gemini API

**Configuration**:
```python
# .env file
GEMINI_API_KEY=your_key_here

# Python usage
from cognition.gemini import GeminiReasoner

reasoner = GeminiReasoner(
    model_name="gemini-2.0-flash",
    temperature=0.2,
    max_output_tokens=2048
)
```

**API Endpoints Used**:
- `POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`

### n8n Workflows

**Workflow Templates** (`orchestration/n8n/workflows/`):
1. **Detection Alert** - Trigger on high-confidence detections
2. **Anomaly Escalation** - Escalate unusual patterns
3. **Batch Processing** - Scheduled bulk analysis

**Integration**:
```javascript
// n8n webhook trigger
{
  "event_id": "evt_123",
  "detections": 5,
  "confidence": 0.89,
  "ai_generated": false,
  "timestamp": "2026-01-22T07:52:37Z"
}
```

---

## 🛠️ Development

### Setting Up Development Environment

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Type checking
mypy cv/ ingestion/ cognition/

# Linting
flake8 .
black .
```

### Training Custom Models

```bash
# Train YOLOv8 on custom dataset
python cv/training/train_yolov8.py \
    --data configs/dataset.yaml \
    --epochs 100 \
    --batch-size 8 \
    --device cuda:0
```

### Adding New Features

1. **Create module**: Add to appropriate layer (`cv/`, `ingestion/`, etc.)
2. **Add schema**: Define Pydantic models in `ingestion/schemas/`
3. **Update config**: Add parameters to `configs/*.yaml`
4. **Write tests**: Add unit tests in `tests/`
5. **Document**: Update README and docstrings

---

## 📚 Documentation

- **Architecture**: See `docs/architecture.md`
- **API Reference**: See `docs/api_reference.md`
- **imgshape Usage**: See `docs/imgshape-usage.txt`
- **Evaluation Reports**: See `eval/EVALUATION_REPORT.md`
- **Work Log**: See `docs/work.txt`

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Coding Standards**:
- Follow PEP 8 style guide
- Add type hints to all functions
- Write docstrings for public APIs
- Maintain >80% test coverage

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **YOLOv8** by Ultralytics - State-of-the-art object detection
- **Gemini API** by Google - Advanced language understanding
- **imgshape v4** - Dataset governance and validation
- **n8n** - Workflow automation platform
- **PyTorch** - Deep learning framework

---

## 📞 Contact & Support

- **Author**: STiFLeR7
- **Repository**: [github.com/STiFLeR7/vision-to-action](https://github.com/STiFLeR7/vision-to-action)
- **Issues**: [GitHub Issues](https://github.com/STiFLeR7/vision-to-action/issues)

---

## 🗺️ Roadmap

### ✅ Completed
- [x] Hardware-optimized YOLOv8 detection
- [x] AI-generated image detection (11 signals)
- [x] Structured ingestion with Pydantic
- [x] Gemini API integration
- [x] Comprehensive evaluation framework
- [x] n8n workflow templates

### 🚧 In Progress
- [ ] Real-time video stream processing
- [ ] Multi-model ensemble detection
- [ ] Advanced imgshape REST integration

### 🔮 Planned
- [ ] WebSocket API for real-time updates
- [ ] Cloud deployment (AWS/GCP)
- [ ] Mobile app integration
- [ ] Custom model fine-tuning UI
- [ ] Advanced temporal analytics
- [ ] Multi-camera orchestration

---

## 📊 Benchmarks

### Hardware Tested

| GPU | VRAM | Detection FPS | E2E Latency |
|-----|------|---------------|-------------|
| RTX 3050 (Laptop) | 6GB | 37.2 | 7.99s |
| RTX 3060 | 12GB | ~55* | ~6s* |
| RTX 4090 | 24GB | ~120* | ~4s* |

*Estimated based on architecture scaling

### Model Comparison

| Model | Size | FPS | mAP@0.5 | VRAM |
|-------|------|-----|---------|------|
| YOLOv8n | 6MB | 37.2 | ~0.52 | 2.5GB |
| YOLOv8s | 22MB | ~25* | ~0.58* | 4GB* |
| YOLOv8m | 52MB | ~15* | ~0.64* | 5.5GB* |

*Estimated on target hardware

---

**Built with ❤️ for production AI systems**

*Last Updated: January 22, 2026*

   - Webhook triggers
   - Escalation workflows

### imgshape v4 Integration

Vision-to-Action is governed by **imgshape v4 (Atlas)**:

- Dataset validation before training
- Preprocessing recommendations
- Compatibility checks
- Atlas fingerprinting for explainable decisions

**Governance principle:** *If imgshape cannot explain or validate a decision, the system does not proceed.*

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (6 GB+ VRAM)
- imgshape v4 service (local or cloud)

### Setup

1. **Clone repository**
```bash
git clone https://github.com/STiFLeR7/vision-to-action.git
cd vision-to-action
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
```bash
# Create .env file
echo "GEMINI_API_KEY=your_key_here" > .env
echo "IMGSHAPE_BASE_URL=http://localhost:8000" >> .env
```

4. **Verify imgshape symlink**
```bash
# Should point to D:/imgshape
ls -la imgshape/
```

## Quick Start

### 1. Validate GPU

```python
from cv.training.utils import check_gpu_memory

check_gpu_memory()
# Should show ~6 GB total VRAM
```

### 2. Run Detection

```python
from cv.detection import YOLOv8Detector
import cv2

# Initialize detector
detector = YOLOv8Detector(
    variant="nano",
    device="cuda:0",
    input_size=(640, 640)
)

# Load image
image = cv2.imread("data/test_image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect objects
result = detector.detect(image)

print(f"Found {len(result.boxes)} objects")
for i, (box, score, cls_name) in enumerate(zip(result.boxes, result.scores, result.class_names)):
    print(f"{i+1}. {cls_name}: {score:.2f}")
```

### 3. Ingest and Store

```python
from ingestion.storage import JSONStorageBackend, generate_event_id
from ingestion.schemas import PerceptionEvent, Detection, BoundingBox, ImageMetadata, TaskType
from datetime import datetime

# Create perception event
event = PerceptionEvent(
    event_id=generate_event_id(),
    timestamp=datetime.utcnow(),
    task_type=TaskType.DETECTION,
    image_metadata=ImageMetadata(
        source_path="data/test_image.jpg",
        width=1920,
        height=1080,
        channels=3,
        dtype="uint8"
    ),
    detections=[
        Detection(
            bbox=BoundingBox(x1=100, y1=150, x2=300, y2=400),
            confidence=0.95,
            class_id=0,
            class_name="person"
        )
    ]
)

# Store event
storage = JSONStorageBackend()
filepath = storage.store_event(event)
print(f"Event stored: {filepath}")
```

### 4. Cognition with Gemini

```python
from cognition.gemini import GeminiReasoner

# Initialize reasoner
reasoner = GeminiReasoner(model_name="gemini-2.0-flash-exp")

# Summarize detection event
summary = reasoner.summarize_event(event)
print(summary)
```

### 5. Train Custom Model

```bash
python cv/training/train_yolov8.py \
  --data data/custom_dataset/data.yaml \
  --variant nano \
  --epochs 100 \
  --batch 4 \
  --imgsz 640
```

## Configuration

All configurations are in `configs/`:

- `system.yaml` - Hardware, imgshape, paths
- `training.yaml` - Training parameters
- `cognition.yaml` - Gemini API settings
- `orchestration.yaml` - n8n workflow settings

Modify as needed for your environment.

## Project Structure

```
vision-to-action/
├── cv/                      # Computer vision modules
│   ├── detection/          # YOLOv8 detection
│   ├── segmentation/       # Conditional segmentation
│   └── training/           # Training pipelines
├── ingestion/              # Structured storage
│   ├── schemas/           # Pydantic schemas
│   └── storage/           # JSON backend
├── cognition/              # LLM reasoning
│   └── gemini/            # Gemini API integration
├── orchestration/          # Agentic workflows
│   └── n8n/               # n8n templates
├── configs/                # Configuration files
├── data/                   # Datasets
├── models/                 # Trained models
├── imgshape/              # Symlink to D:/imgshape
└── README.md
```

## Hardware Constraints

Designed for **6 GB VRAM laptop GPU**:

- FP16 mixed precision
- Batch size: 4 with gradient accumulation
- Detection-first architecture
- Segmentation only when triggered
- CPU-based LLM inference

## imgshape v4 APIs

### Mandatory APIs (checked before training)

- `GET /health` - Service availability
- `GET /datasets` - Dataset discovery
- `POST /analyze` - Image analysis
- `POST /recommend` - Preprocessing recommendations
- `POST /v4/fingerprint` - Dataset fingerprinting
- `POST /v4/analyze` - Full Atlas analysis
- `POST /compatibility` - Model compatibility check

## Contributing

This is a reference implementation demonstrating:

- Systems thinking over isolated models
- Engineering discipline under constraints
- Production-shaped AI architecture

Contributions should maintain these principles.

## License

MIT License - See LICENSE file

## Citation

```bibtex
@software{vision_to_action_2026,
  title={Vision-to-Action: Edge-first AI System},
  author={Vision-to-Action Team},
  year={2026},
  url={https://github.com/STiFLeR7/vision-to-action}
}
```

## Acknowledgments

Built on:
- [imgshape](https://github.com/STiFLeR7/imgshape) - Shape governance
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection
- [Google Gemini API](https://ai.google.dev/) - LLM reasoning
- [n8n](https://n8n.io/) - Workflow automation

---

**Vision-to-Action is not a demo. It is a blueprint for how modern, edge-aware AI systems should be built.**
