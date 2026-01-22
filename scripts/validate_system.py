#!/usr/bin/env python3
"""
System validation script.

Checks:
- GPU availability and memory
- imgshape v4 service connection
- Configuration validity
- Environment variables
"""

import torch
import requests
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config_manager import get_config_manager
from cv.training.utils import check_gpu_memory


def validate_system():
    """Run full system validation"""
    
    print("\n" + "="*60)
    print("Vision-to-Action System Validation")
    print("="*60 + "\n")
    
    all_checks_passed = True
    
    # 1. Check GPU
    print("1. GPU Check")
    print("-" * 60)
    if torch.cuda.is_available():
        gpu_info = check_gpu_memory()
        print(f"✓ CUDA available")
        print(f"✓ GPU: {gpu_info['name']}")
        print(f"✓ Total VRAM: {gpu_info['total_gb']:.2f} GB")
        print(f"✓ Free VRAM: {gpu_info['free_gb']:.2f} GB")
        
        if gpu_info['total_gb'] < 6.0:
            print(f"⚠️  Warning: GPU has less than 6 GB VRAM")
    else:
        print("⚠️  CUDA not available - will run on CPU")
        all_checks_passed = False
    
    # 2. Configuration
    print("\n2. Configuration Check")
    print("-" * 60)
    try:
        config_manager = get_config_manager()
        system_config = config_manager.get_system_config()
        print(f"✓ System config loaded")
        print(f"  Environment: {system_config.environment}")
        print(f"  Target device: {system_config.hardware['target_device']}")
        print(f"  VRAM limit: {system_config.hardware['vram_limit_mb']} MB")
        print(f"  Precision: {system_config.hardware['precision']}")
        
        config_manager.get_training_config()
        print(f"✓ Training config loaded")
        
        config_manager.get_cognition_config()
        print(f"✓ Cognition config loaded")
        
        config_manager.get_orchestration_config()
        print(f"✓ Orchestration config loaded")
        
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        all_checks_passed = False
    
    # 3. imgshape v4 service
    print("\n3. imgshape v4 Service Check")
    print("-" * 60)
    try:
        imgshape_url = system_config.imgshape['base_url']
        print(f"Testing connection to: {imgshape_url}")
        
        response = requests.get(f"{imgshape_url}/health", timeout=5)
        if response.status_code == 200:
            print(f"✓ imgshape v4 Atlas is online")
            health_data = response.json()
            print(f"  Version: {health_data.get('version', 'unknown')}")
        else:
            print(f"⚠️  imgshape returned status {response.status_code}")
            all_checks_passed = False
    except Exception as e:
        print(f"✗ Cannot connect to imgshape: {e}")
        print(f"  Make sure imgshape v4 service is running")
        all_checks_passed = False
    
    # 4. Environment variables
    print("\n4. Environment Variables Check")
    print("-" * 60)
    import os
    
    gemini_key = os.getenv('GEMINI_API_KEY')
    if gemini_key:
        print(f"✓ GEMINI_API_KEY is set")
    else:
        print(f"⚠️  GEMINI_API_KEY not set - cognition layer will fail")
        print(f"  Set it with: export GEMINI_API_KEY=your_key")
    
    # 5. Paths
    print("\n5. Required Paths Check")
    print("-" * 60)
    
    paths_to_check = {
        'Data': Path('D:/vision-to-action/data'),
        'Models': Path('D:/vision-to-action/models'),
        'imgshape': Path('D:/vision-to-action/imgshape'),
        'Configs': Path('D:/vision-to-action/configs')
    }
    
    for name, path in paths_to_check.items():
        if path.exists():
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name}: {path} - NOT FOUND")
            all_checks_passed = False
    
    # Final summary
    print("\n" + "="*60)
    if all_checks_passed:
        print("✓ All critical checks passed!")
        print("System is ready to use.")
    else:
        print("⚠️  Some checks failed - review errors above")
        print("System may have limited functionality.")
    print("="*60 + "\n")
    
    return all_checks_passed


if __name__ == "__main__":
    success = validate_system()
    sys.exit(0 if success else 1)
