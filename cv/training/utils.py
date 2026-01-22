"""
Training utilities and helpers
"""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


def check_gpu_memory():
    """Check available GPU memory"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024**3)  # GB
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            cached = torch.cuda.memory_reserved(i) / (1024**3)
            free = total_memory - cached
            
            print(f"GPU {i}: {props.name}")
            print(f"  Total: {total_memory:.2f} GB")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Cached: {cached:.2f} GB")
            print(f"  Free: {free:.2f} GB")
            
            return {
                'device': i,
                'name': props.name,
                'total_gb': total_memory,
                'free_gb': free
            }
    else:
        print("No CUDA GPU available")
        return None


def calculate_optimal_batch_size(
    img_size: int = 640,
    vram_limit_gb: float = 6.0,
    model_variant: str = "nano"
) -> int:
    """
    Calculate optimal batch size for hardware constraints.
    
    Args:
        img_size: Input image size
        vram_limit_gb: VRAM limit in GB
        model_variant: Model variant (nano, small)
        
    Returns:
        Recommended batch size
    """
    # Rough estimates (MB per image)
    memory_per_image = {
        'nano': 50,
        'small': 80
    }
    
    base_memory = memory_per_image.get(model_variant, 50)
    
    # Scale with image size
    scale_factor = (img_size / 640) ** 2
    memory_per_batch = base_memory * scale_factor
    
    # Reserve 2 GB for model and buffers
    available_memory_mb = (vram_limit_gb - 2.0) * 1024
    
    batch_size = int(available_memory_mb / memory_per_batch)
    batch_size = max(1, min(batch_size, 16))  # Clamp between 1-16
    
    print(f"Estimated batch size for {model_variant} at {img_size}px: {batch_size}")
    return batch_size


def validate_training_config(config: Dict[str, Any]) -> bool:
    """
    Validate training configuration.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        True if valid
    """
    required_keys = ['data', 'epochs', 'batch', 'device']
    
    for key in required_keys:
        if key not in config:
            print(f"Missing required key: {key}")
            return False
    
    # Check GPU availability if cuda specified
    if 'cuda' in config['device'] and not torch.cuda.is_available():
        print("CUDA device specified but not available")
        return False
    
    return True


def create_data_yaml(
    train_path: str,
    val_path: str,
    class_names: List[str],
    output_path: str
) -> str:
    """
    Create YOLOv8 data.yaml file.
    
    Args:
        train_path: Path to training images
        val_path: Path to validation images
        class_names: List of class names
        output_path: Where to save data.yaml
        
    Returns:
        Path to created data.yaml
    """
    import yaml
    
    data = {
        'path': str(Path(train_path).parent),
        'train': str(Path(train_path).name),
        'val': str(Path(val_path).name),
        'nc': len(class_names),
        'names': class_names
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"Created data.yaml at: {output_path}")
    return output_path


def monitor_training_metrics(results_csv: str) -> Dict[str, Any]:
    """
    Monitor and analyze training metrics from results.csv
    
    Args:
        results_csv: Path to results.csv from training
        
    Returns:
        Summary statistics
    """
    import pandas as pd
    
    df = pd.read_csv(results_csv)
    
    summary = {
        'final_map50': df['metrics/mAP50(B)'].iloc[-1],
        'final_map50_95': df['metrics/mAP50-95(B)'].iloc[-1],
        'best_map50': df['metrics/mAP50(B)'].max(),
        'best_map50_95': df['metrics/mAP50-95(B)'].max(),
        'final_precision': df['metrics/precision(B)'].iloc[-1],
        'final_recall': df['metrics/recall(B)'].iloc[-1],
        'total_epochs': len(df)
    }
    
    return summary
