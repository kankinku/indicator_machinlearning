
import os
import importlib.util
import sys
from typing import Dict, Type
from pathlib import Path
from src.shared.logger import get_logger
from .base import CustomFeatureBase, CustomFeatureParam
from ..definitions import FeatureDefinition, FeatureParamSchema, INDICATOR_UNIVERSE

logger = get_logger("feature.custom_loader")

class CustomFeatureLoader:
    """
    Dynamically loads custom indicator classes from the src/features/custom directory
    and registers them into the global INDICATOR_UNIVERSE.
    """
    
    def __init__(self, custom_dir: str = None):
        if custom_dir is None:
            # Default to current dir
            self.custom_dir = Path(__file__).parent
        else:
            self.custom_dir = Path(custom_dir)
            
        self.loaded_features: Dict[str, Type[CustomFeatureBase]] = {}

    def load_all(self):
        """Scan directory and load all valid custom features."""
        logger.info(f"Scanning for custom indicators in: {self.custom_dir}")
        
        for file_path in self.custom_dir.glob("*.py"):
            if file_path.name == "base.py" or file_path.name == "__init__.py" or file_path.name == "loader.py":
                continue
                
            self._load_module(file_path)
            
    def _load_module(self, file_path: Path):
        module_name = f"custom_feature_{file_path.stem}"
        
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                
                # Scan for subclasses of CustomFeatureBase
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        issubclass(attr, CustomFeatureBase) and 
                        attr is not CustomFeatureBase):
                        
                        # Instantiate to get metadata
                        instance = attr()
                        self._register_feature(instance)
                        self.loaded_features[instance.id] = attr
                        logger.info(f"Loaded Custom Feature: {instance.id} ({instance.name})")

        except Exception as e:
            logger.error(f"Failed to load custom feature from {file_path.name}: {e}")

    def _register_feature(self, instance: CustomFeatureBase):
        """Register the custom feature into the global definition universe."""
        
        # Convert CustomFeatureParam to FeatureParamSchema
        schema_params = []
        for p in instance.params:
            schema_params.append(FeatureParamSchema(
                name=p.name,
                param_type=p.param_type,
                min=p.min,
                max=p.max,
                step=p.step,
                choices=p.choices,
                default=p.default
            ))
            
        definition = FeatureDefinition(
            id=instance.id,
            name=instance.name,
            category=instance.category,
            description=instance.description,
            params=schema_params
        )
        
        # Register to Global Universe
        INDICATOR_UNIVERSE[instance.id] = definition

# Global Loader Instance
loader = CustomFeatureLoader()
