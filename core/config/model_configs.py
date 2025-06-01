"""
Model Configuration Classes

This module defines configuration classes for different types of models.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union
from .base_config import BaseConfig


@dataclass
class ModelConfig(BaseConfig):
    """Base configuration for all models."""
    
    # Model architecture
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 2048
    dropout: float = 0.1
    activation: str = 'gelu'
    
    # Input/Output dimensions
    enc_in: int = 7
    dec_in: int = 7
    c_out: int = 7
    
    # Training parameters
    learning_rate: float = 0.0001
    batch_size: int = 32
    train_epochs: int = 10
    patience: int = 3
    
    # Task specific
    seq_len: int = 96
    label_len: int = 0
    pred_len: int = 96
    task_name: str = 'long_term_forecast'
    
    # Time features
    freq: str = 'h'  # frequency: h for hourly, d for daily, etc.
    
    # Classification specific (when task_name is 'classification')
    num_class: int = 10
    
    def validate(self):
        """Validate model configuration."""
        super().validate()
        
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError("dropout must be between 0 and 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.task_name not in ['long_term_forecast', 'short_term_forecast', 'classification', 'imputation', 'anomaly_detection']:
            raise ValueError("task_name must be one of: long_term_forecast, short_term_forecast, classification, imputation, anomaly_detection")


@dataclass
class TransformerConfig(ModelConfig):
    """Configuration for Transformer-based models."""
    
    # Transformer specific
    factor: int = 1
    embed: str = 'timeF'
    
    # Attention specific
    use_norm: bool = True
    
    def validate(self):
        super().validate()
        
        if self.factor <= 0:
            raise ValueError("factor must be positive")
        if self.embed not in ['timeF', 'fixed', 'learned']:
            raise ValueError("embed must be one of: timeF, fixed, learned")


@dataclass
class UCastConfig(ModelConfig):
    """Configuration for U-Cast model."""
    
    # U-Cast specific parameters
    expand: int = 2
    d_conv: int = 4
    alpha: float = 0.0
    channel_reduction_ratio: float = 16
    
    # Moving average for decomposition
    moving_avg: int = 25
    
    def validate(self):
        super().validate()
        
        if self.expand <= 0:
            raise ValueError("expand must be positive")
        if self.d_conv <= 0:
            raise ValueError("d_conv must be positive")
        if self.channel_reduction_ratio <= 0:
            raise ValueError("channel_reduction_ratio must be positive")
        if self.moving_avg <= 0:
            raise ValueError("moving_avg must be positive")


@dataclass
class TimesNetConfig(ModelConfig):
    """Configuration for TimesNet model."""
    
    # TimesNet specific
    top_k: int = 5
    num_kernels: int = 6
    
    def validate(self):
        super().validate()
        
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.num_kernels <= 0:
            raise ValueError("num_kernels must be positive")


@dataclass
class PatchTSTConfig(ModelConfig):
    """Configuration for PatchTST model."""
    
    # Patch specific
    patch_len: int = 16
    stride: int = 8
    
    def validate(self):
        super().validate()
        
        if self.patch_len <= 0:
            raise ValueError("patch_len must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")


@dataclass 
class DLinearConfig(ModelConfig):
    """Configuration for DLinear model."""
    
    # DLinear specific
    individual: bool = False
    moving_avg: int = 25
    
    def validate(self):
        super().validate()
        # DLinear doesn't need many parameters from base ModelConfig
        # Override some validations if needed


@dataclass
class AutoformerConfig(ModelConfig):
    """Configuration for Autoformer model."""
    
    # Autoformer specific
    factor: int = 1
    moving_avg: int = 25
    embed: str = 'timeF'
    
    def validate(self):
        super().validate()
        if self.factor <= 0:
            raise ValueError("factor must be positive")
        if self.moving_avg <= 0:
            raise ValueError("moving_avg must be positive")


@dataclass
class InformerConfig(ModelConfig):
    """Configuration for Informer model."""
    
    # Informer specific
    factor: int = 1
    distil: bool = True
    embed: str = 'timeF'
    
    def validate(self):
        super().validate()
        if self.factor <= 0:
            raise ValueError("factor must be positive")


@dataclass
class FEDformerConfig(ModelConfig):
    """Configuration for FEDformer model."""
    
    # FEDformer specific
    factor: int = 1
    moving_avg: int = 25
    embed: str = 'timeF'
    version: str = 'Wavelets'  # or 'Fourier'
    mode_select: str = 'random'  # or 'low'
    modes: int = 32
    
    def validate(self):
        super().validate()
        if self.factor <= 0:
            raise ValueError("factor must be positive")
        if self.moving_avg <= 0:
            raise ValueError("moving_avg must be positive")
        if self.version not in ['Wavelets', 'Fourier']:
            raise ValueError("version must be 'Wavelets' or 'Fourier'")
        if self.mode_select not in ['random', 'low']:
            raise ValueError("mode_select must be 'random' or 'low'")


@dataclass
class iTransformerConfig(ModelConfig):
    """Configuration for iTransformer model."""
    
    # iTransformer specific
    factor: int = 1
    embed: str = 'timeF'
    
    def validate(self):
        super().validate()
        if self.factor <= 0:
            raise ValueError("factor must be positive")


@dataclass
class NonstationaryTransformerConfig(ModelConfig):
    """Configuration for Nonstationary Transformer model."""
    
    # Nonstationary Transformer specific
    factor: int = 1
    embed: str = 'timeF'
    p_hidden_dims: List[int] = field(default_factory=lambda: [128, 128])
    p_hidden_layers: int = 2
    
    def validate(self):
        super().validate()
        if self.factor <= 0:
            raise ValueError("factor must be positive")
        if self.p_hidden_layers <= 0:
            raise ValueError("p_hidden_layers must be positive")


@dataclass
class ETSformerConfig(ModelConfig):
    """Configuration for ETSformer model."""
    
    # ETSformer specific
    top_k: int = 5
    embed: str = 'timeF'
    
    def validate(self):
        super().validate()
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")


@dataclass
class CrossformerConfig(ModelConfig):
    """Configuration for Crossformer model."""
    
    # Crossformer specific
    factor: int = 1
    seg_len: int = 12
    win_size: int = 2
    
    def validate(self):
        super().validate()
        if self.factor <= 0:
            raise ValueError("factor must be positive")
        if self.seg_len <= 0:
            raise ValueError("seg_len must be positive")
        if self.win_size <= 0:
            raise ValueError("win_size must be positive")


@dataclass
class PyraformerConfig(ModelConfig):
    """Configuration for Pyraformer model."""
    
    # Pyraformer specific
    window_size: List[int] = field(default_factory=lambda: [4, 4])
    inner_size: int = 5
    
    def validate(self):
        super().validate()
        if self.inner_size <= 0:
            raise ValueError("inner_size must be positive")
        if any(w <= 0 for w in self.window_size):
            raise ValueError("all window_size values must be positive")


@dataclass
class ModernTCNConfig(ModelConfig):
    """Configuration for ModernTCN model."""
    
    # ModernTCN specific
    patch_size: int = 8
    patch_stride: int = 4
    stem_ratio: int = 6
    downsample_ratio: int = 2
    ffn_ratio: int = 8
    num_blocks: List[int] = field(default_factory=lambda: [1])
    large_size: List[int] = field(default_factory=lambda: [51])
    small_size: List[int] = field(default_factory=lambda: [5])
    dims: List[int] = field(default_factory=lambda: [16, 16, 16, 16])
    dw_dims: List[int] = field(default_factory=lambda: [64, 64, 64, 64])
    
    def validate(self):
        super().validate()
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.patch_stride <= 0:
            raise ValueError("patch_stride must be positive")


@dataclass
class MICNConfig(ModelConfig):
    """Configuration for MICN model."""
    
    # MICN specific
    conv_kernel: List[int] = field(default_factory=lambda: [12, 16])
    
    def validate(self):
        super().validate()
        if any(k <= 0 for k in self.conv_kernel):
            raise ValueError("all conv_kernel values must be positive")


@dataclass
class TiDEConfig(ModelConfig):
    """Configuration for TiDE model."""
    
    # TiDE specific
    feature_encode_dim: int = 2
    bias: bool = True
    
    def validate(self):
        super().validate()
        if self.feature_encode_dim <= 0:
            raise ValueError("feature_encode_dim must be positive")


@dataclass
class SegRNNConfig(ModelConfig):
    """Configuration for SegRNN model."""
    
    # SegRNN specific
    seg_len: int = 24
    
    def validate(self):
        super().validate()
        if self.seg_len <= 0:
            raise ValueError("seg_len must be positive")


@dataclass
class LightTSConfig(ModelConfig):
    """Configuration for LightTS model."""
    
    # LightTS specific
    chunk_size: int = 24
    
    def validate(self):
        super().validate()
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")


@dataclass
class TSMixerConfig(ModelConfig):
    """Configuration for TSMixer model."""
    
    # TSMixer is simple and uses base ModelConfig
    pass


@dataclass
class FreTSConfig(ModelConfig):
    """Configuration for FreTS model."""
    
    # FreTS specific
    embed_size: int = 128
    hidden_size: int = 256
    channel_independence: str = '0'
    sparsity_threshold: float = 0.01
    scale: float = 0.02
    
    def validate(self):
        super().validate()
        if self.embed_size <= 0:
            raise ValueError("embed_size must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.sparsity_threshold < 0:
            raise ValueError("sparsity_threshold must be non-negative")


# Model configuration factory
MODEL_CONFIGS = {
    'UCast': UCastConfig,
    'DLinear': DLinearConfig,
    'TimesNet': TimesNetConfig,
    'Autoformer': AutoformerConfig,
    'Informer': InformerConfig,
    'FEDformer': FEDformerConfig,
    'PatchTST': PatchTSTConfig,
    'iTransformer': iTransformerConfig,
    'Transformer': TransformerConfig,
    'Nonstationary_Transformer': NonstationaryTransformerConfig,
    'ETSformer': ETSformerConfig,
    'Crossformer': CrossformerConfig,
    'Pyraformer': PyraformerConfig,
    'ModernTCN': ModernTCNConfig,
    'MICN': MICNConfig,
    'TiDE': TiDEConfig,
    'SegRNN': SegRNNConfig,
    'LightTS': LightTSConfig,
    'TSMixer': TSMixerConfig,
    'FreTS': FreTSConfig,
    # Add more model configs as needed
}


def get_model_config_class(model_name: str):
    """Get the appropriate config class for a model."""
    return MODEL_CONFIGS.get(model_name, ModelConfig) 