"""
Configuration Manager for Vision-to-Action System

Handles loading and validation of all configuration files.
Supports both local and production environments.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

try:
    from dotenv import load_dotenv, find_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None
    find_dotenv = None

# Load environment from .env once at import time
if load_dotenv and find_dotenv:
    dotenv_path = os.getenv("GEMINI_DOTENV_PATH") or find_dotenv()
    if dotenv_path:
        load_dotenv(dotenv_path, override=False)


@dataclass
class SystemConfig:
    """System-wide configuration"""
    name: str
    version: str
    environment: str
    hardware: Dict[str, Any]
    imgshape: Dict[str, Any]
    data: Dict[str, Any]
    models: Dict[str, Any]
    paths: Dict[str, Any]


class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Path to configuration directory. 
                       Defaults to D:/vision-to-action/configs
        """
        if config_dir is None:
            config_dir = "D:/vision-to-action/configs"
        
        self.config_dir = Path(config_dir)
        self._configs: Dict[str, Dict[str, Any]] = {}
        
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load a configuration file.
        
        Args:
            config_name: Name of config file (without .yaml extension)
            
        Returns:
            Parsed configuration dictionary
        """
        if config_name in self._configs:
            return self._configs[config_name]
            
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        self._configs[config_name] = config
        return config
    
    def get_system_config(self) -> SystemConfig:
        """Get system configuration as dataclass"""
        config = self.load_config('system')
        return SystemConfig(
            name=config['system']['name'],
            version=config['system']['version'],
            environment=config['system']['environment'],
            hardware=config['hardware'],
            imgshape=config['imgshape'],
            data=config['data'],
            models=config['models'],
            paths=config['paths']
        )
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.load_config('training')
    
    def get_cognition_config(self) -> Dict[str, Any]:
        """Get cognition (Gemini) configuration"""
        return self.load_config('cognition')
    
    def get_orchestration_config(self) -> Dict[str, Any]:
        """Get orchestration (n8n) configuration"""
        return self.load_config('orchestration')
    
    def validate_environment(self) -> bool:
        """
        Validate required environment variables and paths.
        
        Returns:
            True if environment is valid
        """
        system_config = self.get_system_config()
        
        # Check required paths exist
        for path_name, path_value in system_config.paths.items():
            path = Path(path_value)
            if not path.exists():
                print(f"Warning: Path does not exist: {path_name}={path}")
                # Create if it's a data/output directory
                if path_name in ['data', 'logs', 'outputs']:
                    path.mkdir(parents=True, exist_ok=True)
                    print(f"Created directory: {path}")
        
        # Check Gemini API key
        cognition_config = self.get_cognition_config()
        api_key_var = cognition_config['gemini']['api_key_env_var']
        if not os.getenv(api_key_var):
            print(f"Warning: {api_key_var} environment variable not set")
            
        return True
    
    def get_imgshape_url(self) -> str:
        """Get imgshape service URL"""
        system_config = self.get_system_config()
        return system_config.imgshape['base_url']
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        system_config = self.get_system_config()
        return system_config.environment == 'production'


# Global instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
