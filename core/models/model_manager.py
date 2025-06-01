"""
Model Manager

This module provides high-level model management functionality with automatic
model discovery, registration, and instantiation capabilities.
"""

import torch.nn as nn
from typing import Optional, Dict, Any
from core.registry import model_registry
from core.config import BaseConfig


class ModelManager:
    """
    High-level model management interface.
    
    Provides centralized access to model creation and management through the registry system.
    Handles automatic model discovery, registration verification, and metadata tracking.
    """
    
    def __init__(self):
        """Initialize the model manager with the global model registry."""
        self.registry = model_registry
        self._ensure_models_loaded()
    
    def _ensure_models_loaded(self):
        """
        Ensure all available models are loaded and registered.
        
        Imports all model modules to trigger their @register_model decorators,
        making them available through the registry system.
        """
        try:
            # Import legacy models to register them
            import models.UCast  # Triggers @register_model decorator
            import models.DLinear
            import models.TimesNet
            
            # Import Transformer-based models (2017-2021)
            import models.Autoformer
            import models.Informer
            import models.FEDformer
            import models.PatchTST
            import models.iTransformer
            import models.Transformer
            import models.Nonstationary_Transformer
            import models.ETSformer
            import models.Crossformer
            import models.Pyraformer
            
            # Import CNN-based models
            import models.ModernTCN
            import models.MICN
            
            # Import RNN-based and specialized models
            import models.TiDE
            import models.SegRNN
            import models.LightTS
            
            # Import MLP-based and frequency-domain models
            import models.TSMixer
            import models.FreTS
            
            # Future model imports can be added here
            
        except ImportError as e:
            print(f"Warning: Failed to import some models - they may not be available: {e}")
    
    def create_model(self, model_name: str, config: BaseConfig) -> nn.Module:
        """
        Create and initialize a model instance.
        
        Args:
            model_name: Name of the registered model to create
            config: Configuration object containing model parameters
            
        Returns:
            Initialized PyTorch model instance
            
        Raises:
            ValueError: If the model is not registered or configuration is invalid
        """
        return self.registry.create_model(model_name, config)
    
    def get_model_class(self, model_name: str) -> Optional[type]:
        """
        Retrieve a model class by name without instantiation.
        
        Args:
            model_name: Name of the registered model
            
        Returns:
            Model class or None if not found
        """
        return self.registry.get_model(model_name)
    
    def list_available_models(self) -> list:
        """
        Get a sorted list of all available model names.
        
        Returns:
            Alphabetically sorted list of registered model names
        """
        return sorted(self.registry.list_models())
    
    def is_model_available(self, model_name: str) -> bool:
        """
        Check if a model is available in the registry.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model is registered, False otherwise
        """
        return model_name in self.registry
    
    def get_model_metadata(self, model_name: str) -> Dict[str, Any]:
        """
        Retrieve metadata for a registered model.
        
        Metadata may include paper information, year, description,
        and other relevant details for research attribution.
        
        Args:
            model_name: Name of the registered model
            
        Returns:
            Dictionary containing model metadata
        """
        return self.registry.get_metadata(model_name)
    
    def register_model(self, name: str, model_class: type, **metadata) -> type:
        """
        Register a new model with the manager.
        
        This method is typically used for dynamic model registration
        or when integrating custom models into the framework.
        
        Args:
            name: Unique name to register the model under
            model_class: PyTorch model class to register
            **metadata: Additional metadata for research attribution
            
        Returns:
            The registered model class
            
        Raises:
            ValueError: If model_class is not a valid PyTorch module
        """
        return self.registry.register_model(name, model_class, **metadata)
    
    def get_model_count(self) -> int:
        """
        Get the total number of registered models.
        
        Returns:
            Number of models currently registered
        """
        return len(self.registry)
    
    def print_model_summary(self):
        """Print a summary of all registered models with their metadata."""
        models = self.list_available_models()
        print(f"\n=== High-Dimensional Time Series Framework - Registered Models ({len(models)}) ===")
        
        for model_name in models:
            metadata = self.get_model_metadata(model_name)
            year = metadata.get('year', 'Unknown')
            paper = metadata.get('paper', 'No paper information')
            print(f"  • {model_name:<20} ({year}) - {paper}")
        
        print(f"\nTotal: {len(models)} models available")


# Global model manager instance
model_manager = ModelManager() 