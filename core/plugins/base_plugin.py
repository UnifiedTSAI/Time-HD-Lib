"""
Base Plugin System

This module provides the foundation for the plugin system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List


class BasePlugin(ABC):
    """
    Base class for all plugins.
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.enabled = True
        self.dependencies = []
    
    @abstractmethod
    def initialize(self, **kwargs):
        """Initialize the plugin with configuration."""
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the plugin's main functionality."""
        pass
    
    def cleanup(self):
        """Clean up resources when plugin is disabled."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        return {
            'name': self.name,
            'version': self.version,
            'enabled': self.enabled,
            'dependencies': self.dependencies
        }


class ModelPlugin(BasePlugin):
    """
    Base class for model plugins.
    """
    
    def __init__(self, name: str, model_class, version: str = "1.0.0"):
        super().__init__(name, version)
        self.model_class = model_class
    
    @abstractmethod
    def create_model(self, config: Any) -> Any:
        """Create model instance with given configuration."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model-specific information."""
        info = self.get_info()
        info.update({
            'model_class': self.model_class.__name__ if self.model_class else None,
            'type': 'model'
        })
        return info


class DataPlugin(BasePlugin):
    """
    Base class for data processing plugins.
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        super().__init__(name, version)
    
    @abstractmethod
    def process_data(self, data: Any, **kwargs) -> Any:
        """Process data according to plugin logic."""
        pass
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get data-specific information."""
        info = self.get_info()
        info['type'] = 'data'
        return info


class MetricsPlugin(BasePlugin):
    """
    Base class for metrics plugins.
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        super().__init__(name, version)
    
    @abstractmethod
    def calculate_metrics(self, predictions: Any, targets: Any, **kwargs) -> Dict[str, float]:
        """Calculate metrics for predictions and targets."""
        pass
    
    def get_metrics_info(self) -> Dict[str, Any]:
        """Get metrics-specific information."""
        info = self.get_info()
        info['type'] = 'metrics'
        return info


class VisualizationPlugin(BasePlugin):
    """
    Base class for visualization plugins.
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        super().__init__(name, version)
    
    @abstractmethod
    def create_visualization(self, data: Any, **kwargs) -> Any:
        """Create visualization from data."""
        pass
    
    def get_visualization_info(self) -> Dict[str, Any]:
        """Get visualization-specific information."""
        info = self.get_info()
        info['type'] = 'visualization'
        return info


class HookPlugin(BasePlugin):
    """
    Base class for hook plugins that can be attached to experiment lifecycle.
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        super().__init__(name, version)
        self.hook_points = []
    
    def on_experiment_start(self, experiment, **kwargs):
        """Called when experiment starts."""
        pass
    
    def on_experiment_end(self, experiment, results, **kwargs):
        """Called when experiment ends."""
        pass
    
    def on_epoch_start(self, experiment, epoch, **kwargs):
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(self, experiment, epoch, metrics, **kwargs):
        """Called at the end of each epoch."""
        pass
    
    def on_batch_start(self, experiment, batch_idx, **kwargs):
        """Called at the start of each batch."""
        pass
    
    def on_batch_end(self, experiment, batch_idx, loss, **kwargs):
        """Called at the end of each batch."""
        pass
    
    def get_hook_info(self) -> Dict[str, Any]:
        """Get hook-specific information."""
        info = self.get_info()
        info.update({
            'type': 'hook',
            'hook_points': self.hook_points
        })
        return info 