
import os
import importlib
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
        self.loaded_instances: Dict[str, CustomFeatureBase] = {}

    def load_all(self):
        """Scan directory and load all valid custom features."""
        logger.debug(f"[CustomFeature] 스캔: {self.custom_dir}")
        
        for file_path in self.custom_dir.glob("*.py"):
            if file_path.name == "base.py" or file_path.name == "__init__.py" or file_path.name == "loader.py":
                continue
                
            self._load_module(file_path)
            
    def _load_module(self, file_path: Path):
        module_name = f"src.features.custom.{file_path.stem}"
        
        try:
            try:
                module = importlib.import_module(module_name)
            except Exception:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if not spec or not spec.loader:
                    raise ImportError(f"Could not load module from {file_path}")
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
                    self.loaded_instances[instance.id] = instance
                    logger.debug(f"[CustomFeature] 로드: {instance.id} ({instance.name})")

        except Exception as e:
            logger.error(f"[CustomFeature] 로드 실패: {file_path.name} ({e})")

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

    def register_into_registry(self, registry, overwrite: bool = False) -> None:
        """
        Register custom features into FeatureRegistry at runtime (no disk writes).
        """
        self.load_all()

        from src.contracts import FeatureMetadata, TunableParamSpec

        for feature_id, feature_cls in self.loaded_features.items():
            instance = self.loaded_instances.get(feature_id) or feature_cls()
            params = []
            for p in instance.params:
                params.append(TunableParamSpec(
                    name=p.name,
                    param_type=p.param_type,
                    min=p.min,
                    max=p.max,
                    step=p.step,
                    choices=p.choices,
                    default=p.default
                ))

            metadata = FeatureMetadata(
                feature_id=instance.id,
                name=instance.name,
                category=str(instance.category).upper(),
                description=instance.description,
                params=params,
                code_snippet="",
                handler_func=feature_cls.__name__,
                source="custom",
                outputs={"value": "value"}
            )
            registry.register_runtime(metadata, feature_cls, overwrite=overwrite)

# Global Loader Instance
loader = CustomFeatureLoader()
