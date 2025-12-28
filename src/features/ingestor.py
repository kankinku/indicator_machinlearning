import ast
import uuid
from typing import Optional, List, Dict, Any

from src.contracts import FeatureMetadata, TunableParamSpec
from src.shared.logger import get_logger

logger = get_logger("feature.ingestor")

class CodeIngestor:
    """
    Ingests raw Python code, validates it, and converts it into FeatureMetadata.
    This acts as the 'Digestive System' of the evolution architecture.
    """
    
    def ingest_snippet(
        self, 
        name: str, 
        code_snippet: str, 
        category: str,
        description: str = "", 
        source: str = "manual_ingestion"
    ) -> Optional[FeatureMetadata]:
        """
        Parses a python code snippet and attempts to create a FeatureMetadata object.
        """
        try:
            # 1. Syntax Validtion
            self._validate_syntax(code_snippet)
            
            # 2. Extract Key Components (Class/Function name)
            handler_name, params = self._analyze_ast(code_snippet)
            
            if not handler_name:
                raise ValueError("Could not find a valid class or function execution entry point in snippet.")
                
            # 3. Create ID
            # ID format: CATEGORY_NAME_HASH (ex: MOMENTUM_RSI_A1B2)
            short_id = str(uuid.uuid4())[:8].upper()
            feature_id = f"{category.upper()}_{name.upper().replace(' ', '_')}_{short_id}"
            
            # 4. Construct Metadata
            metadata = FeatureMetadata(
                feature_id=feature_id,
                name=name,
                category=category,
                description=description,
                code_snippet=code_snippet,
                handler_func=handler_name,
                params=params,
                complexity_score=1.0, # TODO: Calculate based on AST depth
                source=source,
                tags=[]
            )
            
            logger.info(f"[Feature] 인제스트 완료: {feature_id}")
            return metadata
            
        except Exception as e:
            logger.error(f"[Feature] 인제스트 실패: {name} ({e})")
            return None

    def _validate_syntax(self, code: str):
        """Checks if the code is valid Python syntax."""
        ast.parse(code)

    def _analyze_ast(self, code: str) -> tuple[Optional[str], List[TunableParamSpec]]:
        """
        Analyzes the AST to find the main class/function and infer parameters.
        Returns: (handler_name, list_of_params)
        """
        tree = ast.parse(code)
        
        # Simple Heuristic: 
        # Look for a class definition first, then a function definition.
        # We assume the user provides a class with a 'compute' method or a standalone function.
        
        handler_name = None
        params = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                handler_name = node.name
                # TODO: Parse __init__ args to infer parameters?
                # For now, we rely on manual param definition or advanced inference later.
                break
            elif isinstance(node, ast.FunctionDef):
                if not handler_name: # Prefer class if found earlier
                    handler_name = node.name
        
        # Placeholder for parameter inference logic
        # Ideally we would inspect the signature of the handler.
        
        return handler_name, params
