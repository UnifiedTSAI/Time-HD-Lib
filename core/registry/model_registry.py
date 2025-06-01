"""
Model Registry System

This module provides comprehensive model registration and retrieval functionality
for time series forecasting models in the framework.
"""

import torch.nn as nn
from typing import Type, Dict, Any, Optional, Callable
from .base_registry import BaseRegistry


class ModelRegistry(BaseRegistry):
    """
    Registry for time series forecasting models.
    
    Manages the registration, discovery, and instantiation of model classes.
    Supports automatic model discovery through decorators and provides
    metadata tracking for research attribution.
    """
    
    def __init__(self):
        """Initialize the model registry."""
        super().__init__("Model Registry")
    
    def register_model(self, name: str, model_class: Type[nn.Module], **metadata) -> Type[nn.Module]:
        """
        Register a model class with the registry.
        
        Args:
            name: Unique name to register the model under
            model_class: PyTorch model class to register
            **metadata: Additional metadata (e.g., paper, year, description, author)
            
        Returns:
            The model class (enables use as decorator)
            
        Raises:
            ValueError: If model_class is not a subclass of torch.nn.Module
        """
        if not issubclass(model_class, nn.Module):
            raise ValueError(f"Model '{name}' must be a subclass of torch.nn.Module")
        
        return self.register(name, model_class, **metadata)
    
    def get_model(self, name: str) -> Optional[Type[nn.Module]]:
        """
        Retrieve a model class by name.
        
        Args:
            name: Name of the model to retrieve
            
        Returns:
            The model class or None if not found
        """
        return self.get(name)
    
    def create_model(self, name: str, configs: Any) -> nn.Module:
        """
        Create a model instance with the given configuration.
        
        Args:
            name: Name of the registered model to instantiate
            configs: Configuration object containing model parameters
            
        Returns:
            An initialized instance of the requested model
            
        Raises:
            ValueError: If model is not registered in the registry
        """
        model_class = self.get_model(name)
        if model_class is None:
            available_models = ", ".join(self.list_registered())
            raise ValueError(f"Model '{name}' is not registered. Available models: [{available_models}]")
        
        return model_class(configs)
    
    def list_models(self) -> list:
        """
        Get a list of all registered model names.
        
        Returns:
            List of registered model names sorted alphabetically
        """
        return sorted(self.list_registered())


# Global model registry instance
model_registry = ModelRegistry()


def register_model(name: str, **metadata) -> Callable:
    """
    Decorator for registering time series forecasting models.
    
    This decorator automatically registers model classes with the global
    model registry, enabling automatic discovery and instantiation.
    
    Args:
        name: Unique name to register the model under
        **metadata: Additional metadata about the model (paper, year, description, etc.)
        
    Returns:
        Decorator function that registers the model class
        
    Example:
        @register_model("UCast", paper="U-Cast: Learning Latent Hierarchical...", year=2024)
        class UCastModel(nn.Module):
            def __init__(self, configs):
                super().__init__()
                # Model implementation
    """
    def decorator(model_class: Type[nn.Module]) -> Type[nn.Module]:
        model_registry.register_model(name, model_class, **metadata)
        return model_class
    return decorator 