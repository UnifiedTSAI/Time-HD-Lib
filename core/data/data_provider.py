"""
Data Provider Interface

This module provides a unified interface for data loading and preprocessing,
integrating with the existing data provider infrastructure while offering
extensibility for future enhancements.
"""

from typing import Tuple, Any, Optional
from torch.utils.data import Dataset, DataLoader
from core.config import BaseConfig
from .data_factory import data_provider as core_data_provider


def data_provider(config: BaseConfig, flag: str, accelerator=None) -> Tuple[Dataset, DataLoader]:
    """
    Unified data provider interface for time series datasets.
    
    This function serves as the main entry point for data loading across
    the framework, providing a consistent interface while maintaining
    backward compatibility with the existing data infrastructure.
    
    Args:
        config: Configuration object containing dataset and preprocessing parameters
        flag: Data split identifier ('train', 'val', 'test', 'pred')
        accelerator: Optional accelerator for distributed training and logging
        
    Returns:
        Tuple containing:
            - Dataset: PyTorch dataset instance with preprocessed data
            - DataLoader: Configured data loader for batch iteration
    """
    # Use the core data provider implementation
    return core_data_provider(config, flag, accelerator) 