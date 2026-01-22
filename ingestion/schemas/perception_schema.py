"""
Ingestion Layer Schemas

Defines structured formats for CV outputs.
JSON-first design for storage and downstream reasoning.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class TaskType(str, Enum):
    """Computer vision task type"""
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"


class BoundingBox(BaseModel):
    """Bounding box representation"""
    x1: float = Field(..., description="Top-left x coordinate")
    y1: float = Field(..., description="Top-left y coordinate")
    x2: float = Field(..., description="Bottom-right x coordinate")
    y2: float = Field(..., description="Bottom-right y coordinate")
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def center(self) -> tuple:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class Detection(BaseModel):
    """Single detection result"""
    bbox: BoundingBox
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_id: int
    class_name: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "bbox": {"x1": 100, "y1": 150, "x2": 300, "y2": 400},
                "confidence": 0.95,
                "class_id": 0,
                "class_name": "person"
            }
        }


class Segmentation(BaseModel):
    """Segmentation result for a detected region"""
    detection: Detection
    mask_rle: Optional[str] = Field(None, description="RLE encoded mask")
    mask_url: Optional[str] = Field(None, description="URL to stored mask")
    score: float = Field(..., ge=0.0, le=1.0)
    patch_based: bool = False


class ImageMetadata(BaseModel):
    """Image/frame metadata"""
    source_path: str
    width: int
    height: int
    channels: int
    dtype: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    frame_id: Optional[int] = None
    dataset_name: Optional[str] = None


class PerceptionEvent(BaseModel):
    """
    Core perception event record.
    
    This is the primary ingestion format.
    """
    event_id: str = Field(..., description="Unique event identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    task_type: TaskType
    
    # Image info
    image_metadata: ImageMetadata
    
    # Results
    detections: List[Detection] = Field(default_factory=list)
    segmentations: List[Segmentation] = Field(default_factory=list)
    
    # Processing metadata
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # imgshape governance
    imgshape_validation: Optional[Dict[str, Any]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "event_id": "evt_20260122_120000_001",
                "timestamp": "2026-01-22T12:00:00Z",
                "task_type": "detection",
                "image_metadata": {
                    "source_path": "data/images/frame_001.jpg",
                    "width": 1920,
                    "height": 1080,
                    "channels": 3,
                    "dtype": "uint8",
                    "timestamp": "2026-01-22T12:00:00Z"
                },
                "detections": [
                    {
                        "bbox": {"x1": 100, "y1": 150, "x2": 300, "y2": 400},
                        "confidence": 0.95,
                        "class_id": 0,
                        "class_name": "person"
                    }
                ],
                "processing_metadata": {
                    "model": "yolov8n",
                    "device": "cuda:0",
                    "inference_time_ms": 15.2
                }
            }
        }


class PerceptionBatch(BaseModel):
    """Batch of perception events"""
    batch_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    events: List[PerceptionEvent]
    summary: Optional[Dict[str, Any]] = None


class AnomalyRecord(BaseModel):
    """Anomaly detection record"""
    anomaly_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_id: str
    anomaly_type: str
    severity: str = Field(..., description="low, medium, high, critical")
    description: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
