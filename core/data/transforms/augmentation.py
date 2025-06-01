"""
Data Augmentation Transforms

This module provides data augmentation techniques for time series data.
"""

import torch
import numpy as np
import random
from typing import Any, Optional

from .base_transform import Transform


class TimeSeriesAugmentation(Transform):
    """
    Basic time series augmentation.
    """
    
    def __init__(self, probability: float = 0.5, **kwargs):
        super().__init__(probability=probability, **kwargs)
    
    def __call__(self, data: Any) -> Any:
        """Apply augmentation with given probability."""
        if random.random() < self.params['probability']:
            return self._apply_augmentation(data)
        return data
    
    def _apply_augmentation(self, data: Any) -> Any:
        """Override this method in subclasses."""
        return data


class GaussianNoise(TimeSeriesAugmentation):
    """
    Add Gaussian noise to time series data.
    """
    
    def __init__(self, noise_level: float = 0.01, probability: float = 0.5, **kwargs):
        super().__init__(probability=probability, noise_level=noise_level, **kwargs)
    
    def _apply_augmentation(self, data: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise."""
        if isinstance(data, torch.Tensor):
            noise = torch.randn_like(data) * self.params['noise_level']
            return data + noise
        return data


class TimeWarping(TimeSeriesAugmentation):
    """
    Apply time warping to time series data.
    """
    
    def __init__(self, sigma: float = 0.2, probability: float = 0.5, **kwargs):
        super().__init__(probability=probability, sigma=sigma, **kwargs)
    
    def _apply_augmentation(self, data: torch.Tensor) -> torch.Tensor:
        """Apply time warping."""
        if not isinstance(data, torch.Tensor) or len(data.shape) < 2:
            return data
        
        seq_len = data.shape[-2]  # Assume time dimension is second to last
        
        # Create random time warping
        time_steps = torch.arange(seq_len, dtype=torch.float32)
        warp = torch.randn(seq_len) * self.params['sigma']
        warped_time = time_steps + warp
        
        # Normalize to [0, seq_len-1]
        warped_time = (warped_time - warped_time.min()) / (warped_time.max() - warped_time.min()) * (seq_len - 1)
        
        # Interpolate
        warped_data = torch.zeros_like(data)
        for i in range(data.shape[-1]):  # For each feature
            warped_data[..., i] = torch.nn.functional.interpolate(
                data[..., i].unsqueeze(0).unsqueeze(0),
                size=seq_len,
                mode='linear',
                align_corners=True
            ).squeeze()
        
        return warped_data


class Scaling(TimeSeriesAugmentation):
    """
    Apply random scaling to time series data.
    """
    
    def __init__(self, scale_range: tuple = (0.8, 1.2), probability: float = 0.5, **kwargs):
        super().__init__(probability=probability, scale_range=scale_range, **kwargs)
    
    def _apply_augmentation(self, data: torch.Tensor) -> torch.Tensor:
        """Apply random scaling."""
        if isinstance(data, torch.Tensor):
            scale_min, scale_max = self.params['scale_range']
            scale = random.uniform(scale_min, scale_max)
            return data * scale
        return data


class Jittering(TimeSeriesAugmentation):
    """
    Apply jittering (small random variations) to time series data.
    """
    
    def __init__(self, jitter_strength: float = 0.01, probability: float = 0.5, **kwargs):
        super().__init__(probability=probability, jitter_strength=jitter_strength, **kwargs)
    
    def _apply_augmentation(self, data: torch.Tensor) -> torch.Tensor:
        """Apply jittering."""
        if isinstance(data, torch.Tensor):
            jitter = torch.randn_like(data) * self.params['jitter_strength'] * data.std()
            return data + jitter
        return data


class WindowSlicing(TimeSeriesAugmentation):
    """
    Apply window slicing augmentation.
    """
    
    def __init__(self, slice_ratio: float = 0.1, probability: float = 0.5, **kwargs):
        super().__init__(probability=probability, slice_ratio=slice_ratio, **kwargs)
    
    def _apply_augmentation(self, data: torch.Tensor) -> torch.Tensor:
        """Apply window slicing."""
        if not isinstance(data, torch.Tensor) or len(data.shape) < 2:
            return data
        
        seq_len = data.shape[-2]
        slice_len = int(seq_len * self.params['slice_ratio'])
        
        if slice_len > 0:
            start_idx = random.randint(0, seq_len - slice_len)
            # Zero out the selected slice
            data_copy = data.clone()
            data_copy[..., start_idx:start_idx + slice_len, :] = 0
            return data_copy
        
        return data


class MagnitudeWarping(TimeSeriesAugmentation):
    """
    Apply magnitude warping to time series data.
    """
    
    def __init__(self, sigma: float = 0.2, probability: float = 0.5, **kwargs):
        super().__init__(probability=probability, sigma=sigma, **kwargs)
    
    def _apply_augmentation(self, data: torch.Tensor) -> torch.Tensor:
        """Apply magnitude warping."""
        if not isinstance(data, torch.Tensor):
            return data
        
        # Create smooth random multipliers
        seq_len = data.shape[-2] if len(data.shape) >= 2 else data.shape[0]
        multipliers = torch.randn(seq_len) * self.params['sigma']
        
        # Smooth the multipliers
        multipliers = torch.nn.functional.conv1d(
            multipliers.unsqueeze(0).unsqueeze(0),
            torch.ones(1, 1, 3) / 3,
            padding=1
        ).squeeze()
        
        # Apply exponential to ensure positive multipliers
        multipliers = torch.exp(multipliers)
        
        # Apply to data
        if len(data.shape) >= 2:
            return data * multipliers.unsqueeze(-1)
        else:
            return data * multipliers 