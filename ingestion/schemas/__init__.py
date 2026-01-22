"""
Schema initialization
"""

from .perception_schema import (
    TaskType,
    BoundingBox,
    Detection,
    Segmentation,
    ImageMetadata,
    PerceptionEvent,
    PerceptionBatch,
    AnomalyRecord
)

__all__ = [
    'TaskType',
    'BoundingBox',
    'Detection',
    'Segmentation',
    'ImageMetadata',
    'PerceptionEvent',
    'PerceptionBatch',
    'AnomalyRecord'
]
