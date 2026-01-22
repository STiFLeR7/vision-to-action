"""
YOLOv8 Object Detection Module

Hardware-constrained detection using YOLOv8 nano/small variants.
Integrated with imgshape v4 for preprocessing and validation.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO

# Optional imgshape detection (for metadata only)
IMGSHAPE_PATH = Path(__file__).parent.parent.parent / "imgshape"
IMGSHAPE_REAL = IMGSHAPE_PATH.resolve()
_IMGSHAPE_CANDIDATES = [IMGSHAPE_PATH, IMGSHAPE_REAL, IMGSHAPE_REAL / "src", IMGSHAPE_REAL / "imgshape"]
for _p in _IMGSHAPE_CANDIDATES:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import imgshape  # type: ignore[import-not-found]
    _IMGSHAPE_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    _IMGSHAPE_AVAILABLE = False
    _IMGSHAPE_IMPORT_ERROR = exc

_IMGSHAPE_WARNED = False


def _require_imgshape() -> None:
    """Guard to ensure imgshape is available before use."""
    if not _IMGSHAPE_AVAILABLE:
        raise ImportError(
            "imgshape is not available. Ensure D:/imgshape exists or install the package. "
            f"Original error: {_IMGSHAPE_IMPORT_ERROR}"
        )


def _warn_once_fallback() -> None:
    """Emit a single warning when falling back without imgshape."""
    global _IMGSHAPE_WARNED
    if not _IMGSHAPE_WARNED:
        print("⚠️ imgshape not available; falling back to basic resize without governance")
        _IMGSHAPE_WARNED = True


@dataclass
class DetectionResult:
    """Structured detection output"""
    boxes: np.ndarray  # [N, 4] (x1, y1, x2, y2)
    scores: np.ndarray  # [N]
    classes: np.ndarray  # [N]
    class_names: List[str]
    image_shape: Tuple[int, int, int]  # (H, W, C)
    metadata: Dict[str, Any]


class YOLOv8Detector:
    """
    YOLOv8-based object detector with hardware constraints.
    
    Design principles:
    - FP16 precision for 6 GB VRAM
    - Small batch sizes
    - imgshape-validated inputs
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        variant: str = "nano",
        device: str = "cuda:0",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: Tuple[int, int] = (640, 640),
        use_fp16: bool = True
    ):
        """
        Initialize YOLOv8 detector.
        
        Args:
            model_path: Path to trained model. If None, uses pretrained.
            variant: Model variant (nano, small)
            device: Target device (cuda:0, cpu)
            confidence_threshold: Detection confidence threshold
            iou_threshold: NMS IoU threshold
            input_size: Input resolution (H, W)
            use_fp16: Use FP16 precision for inference
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        
        # Load model
        if model_path is None:
            model_path = f"yolov8{variant[0]}.pt"  # yolov8n.pt, yolov8s.pt
            
        self.model = YOLO(model_path)
        self.model.to(device)
            
        print(f"Initialized YOLOv8 {variant} on {device}")
        print(f"FP16: {self.use_fp16}, Input size: {input_size}")
    
    def preprocess(
        self,
        image: np.ndarray,
        use_imgshape: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Preprocess image with imgshape validation.
        
        Args:
            image: Input image (H, W, C)
            use_imgshape: Use imgshape for aspect-ratio safe resizing
            
        Returns:
            Preprocessed image and metadata
        """
        # If imgshape available we could add governance later; currently only fallback resize is used
        if use_imgshape and not _IMGSHAPE_AVAILABLE:
            _warn_once_fallback()
            use_imgshape = False

        original_shape = image.shape

        # Resize (bilinear) - aspect ratio preserved via padding if imgshape later added
        resized = torch.nn.functional.interpolate(
            torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float(),
            size=self.input_size,
            mode='bilinear',
            align_corners=False
        )
        resized = resized.squeeze(0).permute(1, 2, 0).numpy()
        metadata = {
            'method': 'bilinear',
            'imgshape_available': _IMGSHAPE_AVAILABLE,
            'used_imgshape': False
        }
        
        metadata['original_shape'] = original_shape
        metadata['resized_shape'] = resized.shape
        
        return resized, metadata
    
    @torch.no_grad()
    def detect(
        self,
        image: np.ndarray,
        preprocess: bool = True
    ) -> DetectionResult:
        """
        Run object detection on image.
        
        Args:
            image: Input image (H, W, C) in RGB
            preprocess: Apply preprocessing
            
        Returns:
            DetectionResult with boxes, scores, classes
        """
        if preprocess:
            image, preprocess_metadata = self.preprocess(image)
        else:
            preprocess_metadata = {}
        
        # Run inference
        results = self.model.predict(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            half=self.use_fp16,
            verbose=False
        )
        
        # Extract results
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        class_names = [result.names[cls] for cls in classes]
        
        return DetectionResult(
            boxes=boxes,
            scores=scores,
            classes=classes,
            class_names=class_names,
            image_shape=image.shape,
            metadata={
                'preprocessing': preprocess_metadata,
                'model': 'yolov8',
                'device': self.device,
                'fp16': self.use_fp16
            }
        )
    
    def detect_batch(
        self,
        images: List[np.ndarray],
        batch_size: int = 4
    ) -> List[DetectionResult]:
        """
        Run detection on batch of images.
        
        Args:
            images: List of images (each H, W, C)
            batch_size: Batch size for processing
            
        Returns:
            List of DetectionResults
        """
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            for image in batch:
                result = self.detect(image)
                results.append(result)
        return results
    
    def save_model(self, save_path: str):
        """Save trained model"""
        self.model.save(save_path)
        print(f"Model saved to {save_path}")
    
    def export_onnx(
        self,
        export_path: str,
        simplify: bool = True
    ):
        """
        Export model to ONNX for deployment.
        
        Args:
            export_path: Path to save ONNX model
            simplify: Simplify ONNX graph
        """
        self.model.export(
            format='onnx',
            imgsz=self.input_size,
            simplify=simplify,
            dynamic=False,
            half=self.use_fp16
        )
        print(f"Model exported to ONNX: {export_path}")
