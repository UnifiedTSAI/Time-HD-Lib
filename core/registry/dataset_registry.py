"""
Dataset Registry System

This module provides dataset registration and retrieval functionality.
"""

from torch.utils.data import Dataset
from typing import Type, Dict, Any, Optional, Callable
from .base_registry import BaseRegistry


class DatasetRegistry(BaseRegistry):
    """
    Registry for time series datasets.
    
    Manages registration and retrieval of dataset classes.
    """
    
    def __init__(self):
        super().__init__("Dataset Registry")
    
    def register_dataset(self, name: str, dataset_class: Type[Dataset], **metadata) -> Type[Dataset]:
        """
        Register a dataset class.
        
        Args:
            name: Name to register the dataset under
            dataset_class: The dataset class to register
            **metadata: Additional metadata (e.g., description, source, frequency)
            
        Returns:
            The dataset class (for use as decorator)
        """
        if not issubclass(dataset_class, Dataset):
            raise ValueError(f"Dataset {name} must be a subclass of torch.utils.data.Dataset")
        
        return self.register(name, dataset_class, **metadata)
    
    def get_dataset(self, name: str) -> Optional[Type[Dataset]]:
        """
        Get a dataset class by name.
        
        Args:
            name: Name of the dataset to retrieve
            
        Returns:
            The dataset class or None if not found
        """
        return self.get(name)
    
    def create_dataset(self, name: str, *args, **kwargs) -> Dataset:
        """
        Create a dataset instance with the given arguments.
        
        Args:
            name: Name of the dataset to create
            *args: Positional arguments for dataset constructor
            **kwargs: Keyword arguments for dataset constructor
            
        Returns:
            An instance of the dataset
            
        Raises:
            ValueError: If dataset is not registered
        """
        dataset_class = self.get_dataset(name)
        if dataset_class is None:
            raise ValueError(f"Dataset '{name}' is not registered. Available datasets: {self.list_registered()}")
        
        return dataset_class(*args, **kwargs)
    
    def list_datasets(self) -> list:
        """Get a list of all registered dataset names."""
        return self.list_registered()


# Global dataset registry instance
dataset_registry = DatasetRegistry()


def register_dataset(name: str, **metadata) -> Callable:
    """
    Decorator for registering datasets.
    
    Args:
        name: Name to register the dataset under
        **metadata: Additional metadata about the dataset
        
    Returns:
        Decorator function
        
    Example:
        @register_dataset("MyDataset", frequency="hourly", source="custom")
        class MyDataset(Dataset):
            pass
    """
    def decorator(dataset_class: Type[Dataset]) -> Type[Dataset]:
        dataset_registry.register_dataset(name, dataset_class, **metadata)
        return dataset_class
    return decorator 