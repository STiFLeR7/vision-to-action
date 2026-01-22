"""
YOLOv8 Training Script

Hardware-constrained training with imgshape v4 governance.
Implements the training pipeline defined in the SoW.
"""

import torch
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from ultralytics import YOLO
import sys

# Add imgshape to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'imgshape'))

from configs.config_manager import get_config_manager


class YOLOv8Trainer:
    """
    Hardware-constrained YOLOv8 trainer.
    
    Design principles:
    - 6 GB VRAM constraint
    - FP16 precision
    - Small batches with gradient accumulation
    - imgshape validation before training
    """
    
    def __init__(
        self,
        data_yaml: str,
        model_variant: str = "nano",
        pretrained: bool = True,
        device: str = "cuda:0"
    ):
        """
        Initialize trainer.
        
        Args:
            data_yaml: Path to data configuration YAML
            model_variant: Model variant (nano, small)
            pretrained: Use pretrained weights
            device: Target device
        """
        self.data_yaml = data_yaml
        self.model_variant = model_variant
        self.device = device
        
        # Load configurations
        self.config_manager = get_config_manager()
        self.system_config = self.config_manager.get_system_config()
        self.training_config = self.config_manager.get_training_config()
        
        # Initialize model
        if pretrained:
            model_name = f"yolov8{model_variant[0]}.pt"  # yolov8n.pt
        else:
            model_name = f"yolov8{model_variant[0]}.yaml"
            
        self.model = YOLO(model_name)
        
        print(f"Initialized YOLOv8 {model_variant} trainer")
        print(f"Device: {device}, Pretrained: {pretrained}")
    
    def validate_with_imgshape(self) -> bool:
        """
        Validate dataset with imgshape v4 before training.
        
        Returns:
            True if validation passes
        """
        import requests
        
        imgshape_url = self.config_manager.get_imgshape_url()
        
        # Health check
        try:
            response = requests.get(f"{imgshape_url}/health", timeout=10)
            if response.status_code != 200:
                print(f"⚠️ imgshape health check failed")
                return False
            print("✓ imgshape v4 Atlas is online")
        except Exception as e:
            print(f"⚠️ Cannot connect to imgshape: {e}")
            return False
        
        # TODO: Implement full dataset validation with imgshape APIs
        # - Dataset discovery
        # - Compatibility check
        # - Atlas fingerprinting
        
        return True
    
    def train(
        self,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        imgsz: int = 640,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train YOLOv8 model with hardware constraints.
        
        Args:
            epochs: Number of epochs (from config if None)
            batch_size: Batch size (from config if None)
            imgsz: Input image size
            save_dir: Directory to save results
            
        Returns:
            Training results dictionary
        """
        # Validate with imgshape first
        if self.system_config.imgshape['validate_before_training']:
            print("\n🔍 Validating dataset with imgshape v4...")
            if not self.validate_with_imgshape():
                raise RuntimeError("imgshape validation failed. Cannot proceed with training.")
        
        # Get training parameters
        if epochs is None:
            epochs = self.training_config['training']['epochs']
        if batch_size is None:
            batch_size = self.training_config['training']['batch_size']
        
        if save_dir is None:
            save_dir = str(Path(self.system_config.paths['models']) / 'training_runs')
        
        # Training configuration
        train_args = {
            'data': self.data_yaml,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': imgsz,
            'device': self.device,
            'patience': self.training_config['training']['patience'],
            'save': True,
            'project': save_dir,
            'name': f"yolov8{self.model_variant}_{Path(self.data_yaml).stem}",
            'exist_ok': True,
            'pretrained': True,
            'optimizer': self.training_config['training']['optimizer']['name'],
            'lr0': self.training_config['training']['optimizer']['lr'],
            'weight_decay': self.training_config['training']['optimizer']['weight_decay'],
            'val': True,
            'plots': True,
            'verbose': True
        }
        
        # Hardware constraints
        if self.system_config.hardware['enable_memory_efficient_mode']:
            train_args['amp'] = True  # FP16 training
            train_args['workers'] = 4
            train_args['cache'] = False  # Don't cache to save memory
        
        print(f"\n🚀 Starting training with hardware constraints:")
        print(f"   VRAM limit: {self.system_config.hardware['vram_limit_mb']} MB")
        print(f"   Precision: {self.system_config.hardware['precision']}")
        print(f"   Batch size: {batch_size}")
        print(f"   Epochs: {epochs}\n")
        
        # Train
        results = self.model.train(**train_args)
        
        print(f"\n✓ Training complete!")
        print(f"   Best model saved to: {save_dir}")
        
        return results
    
    def export_model(
        self,
        format: str = 'onnx',
        simplify: bool = True,
        imgsz: int = 640
    ):
        """
        Export trained model for deployment.
        
        Args:
            format: Export format (onnx, torchscript, etc.)
            simplify: Simplify ONNX graph
            imgsz: Input size for export
        """
        export_path = self.model.export(
            format=format,
            imgsz=imgsz,
            simplify=simplify,
            half=self.system_config.hardware['precision'] == 'fp16'
        )
        
        print(f"✓ Model exported to: {export_path}")
        return export_path


def train_detection_model(
    data_yaml: str,
    variant: str = "nano",
    epochs: int = 100,
    batch_size: int = 4
):
    """
    Convenience function to train a detection model.
    
    Args:
        data_yaml: Path to dataset YAML
        variant: Model variant (nano, small)
        epochs: Training epochs
        batch_size: Batch size
    """
    trainer = YOLOv8Trainer(
        data_yaml=data_yaml,
        model_variant=variant,
        pretrained=True
    )
    
    results = trainer.train(
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Export to ONNX for deployment
    trainer.export_model(format='onnx')
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLOv8 detection model")
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml')
    parser.add_argument('--variant', type=str, default='nano', choices=['nano', 'small'], 
                       help='Model variant')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    
    args = parser.parse_args()
    
    train_detection_model(
        data_yaml=args.data,
        variant=args.variant,
        epochs=args.epochs,
        batch_size=args.batch
    )
