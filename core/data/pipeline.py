"""
Data Processing Pipeline

This module provides a flexible and extensible data processing pipeline.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

from core.config import BaseConfig
from .transforms import Transform


class DataPipeline(ABC):
    """
    Abstract base class for data processing pipelines.
    """
    
    def __init__(self, config: BaseConfig):
        self.config = config
        self.transforms: List[Transform] = []
    
    def add_transform(self, transform: Transform):
        """Add a transform to the pipeline."""
        self.transforms.append(transform)
    
    def apply_transforms(self, data: Any) -> Any:
        """Apply all transforms to the data."""
        for transform in self.transforms:
            data = transform(data)
        return data
    
    @abstractmethod
    def load_data(self, flag: str) -> Tuple[Dataset, DataLoader]:
        """Load data for the specified flag."""
        pass


class TimeSeriesPipeline(DataPipeline):
    """
    Pipeline for time series data processing.
    """
    
    def __init__(self, config: BaseConfig):
        super().__init__(config)
        self._setup_default_transforms()
    
    def _setup_default_transforms(self):
        """Setup default transforms for time series data."""
        # We'll add default transforms here
        pass
    
    def load_data(self, flag: str) -> Tuple[Dataset, DataLoader]:
        """
        Load time series data for the specified flag.
        
        Args:
            flag: Data split flag ('train', 'val', 'test')
            
        Returns:
            Tuple of (dataset, dataloader)
        """
        # For now, delegate to the legacy data provider
        from data_provider.data_factory import data_provider as legacy_data_provider
        
        dataset, dataloader = legacy_data_provider(self.config, flag)
        
        # Apply transforms if any
        if self.transforms:
            # Create a wrapper dataset that applies transforms
            dataset = TransformDataset(dataset, self.transforms)
            
            # Recreate dataloader with transformed dataset
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=(flag == 'train'),
                num_workers=self.config.num_workers,
                drop_last=False,
                pin_memory=True
            )
        
        return dataset, dataloader


class TransformDataset(Dataset):
    """
    Wrapper dataset that applies transforms to data.
    """
    
    def __init__(self, dataset: Dataset, transforms: List[Transform]):
        self.dataset = dataset
        self.transforms = transforms
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        # Apply transforms
        for transform in self.transforms:
            data = transform(data)
        
        return data


class HighDimensionalPipeline(TimeSeriesPipeline):
    """
    Specialized pipeline for high-dimensional time series data.
    """
    
    def __init__(self, config: BaseConfig):
        super().__init__(config)
        self._setup_high_dimensional_transforms()
    
    def _setup_high_dimensional_transforms(self):
        """Setup transforms specific to high-dimensional data."""
        # Add specialized transforms for high-dimensional data
        pass 