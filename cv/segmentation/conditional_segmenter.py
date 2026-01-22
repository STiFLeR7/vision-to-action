"""
Conditional Feature Segmentation Module

Triggered only when:
- Detected object requires finer inspection
- Anomaly threshold is crossed
- Workflow explicitly requests it

Uses patch-based approach to preserve VRAM.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO

# imgshape integration (optional during dev)
IMGSHAPE_PATH = Path(__file__).parent.parent.parent / "imgshape"
IMGSHAPE_REAL = IMGSHAPE_PATH.resolve()
_IMGSHAPE_CANDIDATES = [
    IMGSHAPE_PATH,
    IMGSHAPE_REAL,
    IMGSHAPE_REAL / "src",
    IMGSHAPE_REAL / "imgshape",
]
for _p in _IMGSHAPE_CANDIDATES:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from imgshape.preprocessing import extract_patch, reconstruct_from_patch  # type: ignore[import-not-found]
    _IMGSHAPE_AVAILABLE = True
except Exception as exc:  # pragma: no cover - only hit when imgshape missing
    _IMGSHAPE_AVAILABLE = False
    _IMGSHAPE_IMPORT_ERROR = exc


def _require_imgshape() -> None:
    """Guard to ensure imgshape is available before use."""
    if not _IMGSHAPE_AVAILABLE:
        raise ImportError(
            "imgshape is not available. Ensure D:/imgshape exists or install the package. "
            f"Original error: {_IMGSHAPE_IMPORT_ERROR}"
        )


@dataclass
class SegmentationResult:
    """Structured segmentation output"""
    mask: np.ndarray  # (H, W) binary mask
    score: float
    bbox: np.ndarray  # [4] (x1, y1, x2, y2)
    class_id: int
    class_name: str
    patch_based: bool
    metadata: Dict[str, Any]


class ConditionalSegmenter:
    """
    Conditional feature segmentation with patch-based processing.
    
    Design principles:
    - Only runs when triggered
    - Patch-based to preserve VRAM
    - Frozen or partially frozen encoder
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        variant: str = "nano",
        device: str = "cuda:0",
        patch_size: Tuple[int, int] = (256, 256),
        confidence_threshold: float = 0.25,
        use_fp16: bool = True
    ):
        """
        Initialize conditional segmenter.
        
        Args:
            model_path: Path to trained segmentation model
            variant: Model variant (nano)
            device: Target device
            patch_size: Patch size for region-based segmentation
            confidence_threshold: Segmentation confidence threshold
            use_fp16: Use FP16 precision
        """
        self.device = device
        self.patch_size = patch_size
        self.confidence_threshold = confidence_threshold
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        
        # Load model
        if model_path is None:
            model_path = f"yolov8{variant[0]}-seg.pt"  # yolov8n-seg.pt
            
        self.model = YOLO(model_path)
        self.model.to(device)
        
        if self.use_fp16:
            self.model.model.half()
            
        print(f"Initialized conditional segmenter on {device}")
        print(f"Patch-based: {patch_size}, FP16: {self.use_fp16}")
    
    def should_segment(
        self,
        detection_confidence: float,
        trigger_threshold: float = 0.7,
        force: bool = False
    ) -> bool:
        """
        Determine if segmentation should be triggered.
        
        Args:
            detection_confidence: Detection confidence score
            trigger_threshold: Minimum confidence to trigger
            force: Force segmentation regardless of confidence
            
        Returns:
            True if segmentation should run
        """
        if force:
            return True
        return detection_confidence >= trigger_threshold
    
    def extract_region_patch(
        self,
        image: np.ndarray,
        bbox: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract patch around detected region using imgshape.
        
        Args:
            image: Full image (H, W, C)
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Patch and extraction metadata
        """
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Extract with padding
        _require_imgshape()
        patch, metadata = extract_patch(
            image,
            center=(int((x1+x2)/2), int((y1+y2)/2)),
            patch_size=self.patch_size,
            padding_mode='reflect'
        )
        
        metadata['bbox'] = bbox
        metadata['original_shape'] = image.shape
        
        return patch, metadata
    
    @torch.no_grad()
    def segment_region(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        use_patch: bool = True
    ) -> SegmentationResult:
        """
        Segment a detected region.
        
        Args:
            image: Full image (H, W, C)
            bbox: Region to segment [x1, y1, x2, y2]
            use_patch: Use patch-based processing
            
        Returns:
            SegmentationResult with mask
        """
        if use_patch:
            patch, patch_metadata = self.extract_region_patch(image, bbox)
            input_image = patch
        else:
            input_image = image
            patch_metadata = {}
        
        # Run segmentation
        results = self.model.predict(
            input_image,
            conf=self.confidence_threshold,
            device=self.device,
            half=self.use_fp16,
            verbose=False
        )
        
        result = results[0]
        
        if result.masks is None or len(result.masks) == 0:
            # No masks found, return empty
            return SegmentationResult(
                mask=np.zeros(image.shape[:2], dtype=np.uint8),
                score=0.0,
                bbox=bbox,
                class_id=-1,
                class_name="none",
                patch_based=use_patch,
                metadata={'status': 'no_mask_found'}
            )
        
        # Get best mask
        mask = result.masks.data[0].cpu().numpy()
        score = result.boxes.conf[0].cpu().item()
        class_id = int(result.boxes.cls[0].cpu().item())
        class_name = result.names[class_id]
        
        # Reconstruct mask to original image if patch-based
        if use_patch:
            _require_imgshape()
            mask = reconstruct_from_patch(
                mask,
                target_shape=image.shape[:2],
                patch_metadata=patch_metadata
            )
        
        return SegmentationResult(
            mask=mask,
            score=score,
            bbox=bbox,
            class_id=class_id,
            class_name=class_name,
            patch_based=use_patch,
            metadata={
                'patch_metadata': patch_metadata if use_patch else None,
                'device': self.device,
                'fp16': self.use_fp16
            }
        )
    
    def segment_detections(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        trigger_threshold: float = 0.7
    ) -> List[SegmentationResult]:
        """
        Conditionally segment multiple detections.
        
        Args:
            image: Full image
            detections: List of detection results
            trigger_threshold: Confidence threshold to trigger segmentation
            
        Returns:
            List of SegmentationResults (only for triggered detections)
        """
        results = []
        
        for det in detections:
            if self.should_segment(det['confidence'], trigger_threshold):
                result = self.segment_region(image, det['bbox'])
                results.append(result)
                
        return results
