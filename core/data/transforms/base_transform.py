"""
Base Transform Classes

This module provides the foundation for data transformation operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Transform(ABC):
    """
    Base class for all data transformations.
    """
    
    def __init__(self, **kwargs):
        self.params = kwargs
    
    @abstractmethod
    def __call__(self, data: Any) -> Any:
        """
        Apply the transformation to the data.
        
        Args:
            data: Input data to transform
            
        Returns:
            Transformed data
        """
        pass
    
    def set_params(self, **kwargs):
        """Update transformation parameters."""
        self.params.update(kwargs)
    
    def get_params(self) -> Dict[str, Any]:
        """Get transformation parameters."""
        return self.params.copy()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.params})"


class CompositeTransform(Transform):
    """
    Composite transformation that applies multiple transforms in sequence.
    """
    
    def __init__(self, transforms: list, **kwargs):
        super().__init__(**kwargs)
        self.transforms = transforms
    
    def __call__(self, data: Any) -> Any:
        """Apply all transforms in sequence."""
        for transform in self.transforms:
            data = transform(data)
        return data
    
    def add_transform(self, transform: Transform):
        """Add a transform to the sequence."""
        self.transforms.append(transform)
    
    def __repr__(self) -> str:
        transform_names = [transform.__class__.__name__ for transform in self.transforms]
        return f"CompositeTransform([{', '.join(transform_names)}])" 