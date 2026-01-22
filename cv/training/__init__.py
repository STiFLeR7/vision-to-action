"""
Training module initialization
"""

from .train_yolov8 import YOLOv8Trainer, train_detection_model
from .utils import (
    check_gpu_memory,
    calculate_optimal_batch_size,
    validate_training_config,
    create_data_yaml
)

__all__ = [
    'YOLOv8Trainer',
    'train_detection_model',
    'check_gpu_memory',
    'calculate_optimal_batch_size',
    'validate_training_config',
    'create_data_yaml'
]
