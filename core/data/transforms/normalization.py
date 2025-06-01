"""
Normalization Transforms

This module provides various normalization techniques for time series data.
"""

import torch
import numpy as np
from typing import Any, Tuple, Optional

from .base_transform import Transform


class StandardNormalization(Transform):
    """
    Standard normalization (z-score normalization).
    """
    
    def __init__(self, axis: int = 1, eps: float = 1e-8, **kwargs):
        super().__init__(axis=axis, eps=eps, **kwargs)
        self.mean = None
        self.std = None
        self.fitted = False
    
    def fit(self, data: torch.Tensor):
        """Fit normalization parameters."""
        self.mean = data.mean(dim=self.params['axis'], keepdim=True)
        self.std = data.std(dim=self.params['axis'], keepdim=True)
        self.fitted = True
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply standard normalization."""
        if isinstance(data, (list, tuple)):
            # Handle multiple inputs (e.g., from DataLoader)
            normalized_data = []
            for item in data:
                if isinstance(item, torch.Tensor):
                    normalized_data.append(self._normalize_tensor(item))
                else:
                    normalized_data.append(item)
            return type(data)(normalized_data)
        elif isinstance(data, torch.Tensor):
            return self._normalize_tensor(data)
        else:
            return data
    
    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize a single tensor."""
        if not self.fitted:
            self.fit(tensor)
        
        normalized = (tensor - self.mean) / (self.std + self.params['eps'])
        return normalized
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Inverse normalization."""
        if not self.fitted:
            raise ValueError("Normalization must be fitted before inverse transform")
        
        return data * (self.std + self.params['eps']) + self.mean


class MinMaxNormalization(Transform):
    """
    Min-Max normalization to [0, 1] range.
    """
    
    def __init__(self, axis: int = 1, feature_range: Tuple[float, float] = (0, 1), **kwargs):
        super().__init__(axis=axis, feature_range=feature_range, **kwargs)
        self.min_val = None
        self.max_val = None
        self.fitted = False
    
    def fit(self, data: torch.Tensor):
        """Fit normalization parameters."""
        self.min_val = data.min(dim=self.params['axis'], keepdim=True)[0]
        self.max_val = data.max(dim=self.params['axis'], keepdim=True)[0]
        self.fitted = True
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply min-max normalization."""
        if isinstance(data, (list, tuple)):
            # Handle multiple inputs (e.g., from DataLoader)
            normalized_data = []
            for item in data:
                if isinstance(item, torch.Tensor):
                    normalized_data.append(self._normalize_tensor(item))
                else:
                    normalized_data.append(item)
            return type(data)(normalized_data)
        elif isinstance(data, torch.Tensor):
            return self._normalize_tensor(data)
        else:
            return data
    
    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize a single tensor."""
        if not self.fitted:
            self.fit(tensor)
        
        min_range, max_range = self.params['feature_range']
        scale = max_range - min_range
        
        normalized = (tensor - self.min_val) / (self.max_val - self.min_val + 1e-8)
        normalized = normalized * scale + min_range
        
        return normalized
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Inverse normalization."""
        if not self.fitted:
            raise ValueError("Normalization must be fitted before inverse transform")
        
        min_range, max_range = self.params['feature_range']
        scale = max_range - min_range
        
        # Reverse scaling
        data = (data - min_range) / scale
        data = data * (self.max_val - self.min_val) + self.min_val
        
        return data


class RobustNormalization(Transform):
    """
    Robust normalization using median and IQR.
    """
    
    def __init__(self, axis: int = 1, **kwargs):
        super().__init__(axis=axis, **kwargs)
        self.median = None
        self.iqr = None
        self.fitted = False
    
    def fit(self, data: torch.Tensor):
        """Fit normalization parameters."""
        self.median = data.median(dim=self.params['axis'], keepdim=True)[0]
        q75 = data.quantile(0.75, dim=self.params['axis'], keepdim=True)
        q25 = data.quantile(0.25, dim=self.params['axis'], keepdim=True)
        self.iqr = q75 - q25
        self.fitted = True
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply robust normalization."""
        if isinstance(data, (list, tuple)):
            # Handle multiple inputs (e.g., from DataLoader)
            normalized_data = []
            for item in data:
                if isinstance(item, torch.Tensor):
                    normalized_data.append(self._normalize_tensor(item))
                else:
                    normalized_data.append(item)
            return type(data)(normalized_data)
        elif isinstance(data, torch.Tensor):
            return self._normalize_tensor(data)
        else:
            return data
    
    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize a single tensor."""
        if not self.fitted:
            self.fit(tensor)
        
        normalized = (tensor - self.median) / (self.iqr + 1e-8)
        return normalized
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Inverse normalization."""
        if not self.fitted:
            raise ValueError("Normalization must be fitted before inverse transform")
        
        return data * (self.iqr + 1e-8) + self.median 