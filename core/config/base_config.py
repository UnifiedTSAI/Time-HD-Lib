"""
Base Configuration System

This module provides the foundation for all configuration classes.
"""

import yaml
import json
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict


@dataclass
class BaseConfig:
    """
    Base configuration class that provides common functionality.
    
    All configuration classes should inherit from this base class.
    """
    
    def __post_init__(self):
        """Post-initialization hook for validation."""
        self.validate()
    
    def validate(self):
        """Validate configuration parameters. Override in subclasses."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_yaml(self, file_path: Optional[Union[str, Path]] = None) -> str:
        """
        Convert configuration to YAML format.
        
        Args:
            file_path: Optional path to save the YAML file
            
        Returns:
            YAML string representation
        """
        yaml_str = yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        
        if file_path is not None:
            with open(file_path, 'w') as f:
                f.write(yaml_str)
        
        return yaml_str
    
    def to_json(self, file_path: Optional[Union[str, Path]] = None) -> str:
        """
        Convert configuration to JSON format.
        
        Args:
            file_path: Optional path to save the JSON file
            
        Returns:
            JSON string representation
        """
        json_str = json.dumps(self.to_dict(), indent=2)
        
        if file_path is not None:
            with open(file_path, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            Configuration instance
        """
        # Get dataclass fields
        import inspect
        from dataclasses import fields
        
        # Get the fields defined in the dataclass
        if hasattr(cls, '__dataclass_fields__'):
            valid_keys = set(cls.__dataclass_fields__.keys())
            filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
            
            # Create instance with valid fields
            instance = cls(**filtered_dict)
            
            # Add any additional fields as attributes
            for k, v in config_dict.items():
                if k not in valid_keys:
                    setattr(instance, k, v)
        else:
            # Fallback for non-dataclass
            sig = inspect.signature(cls)
            valid_keys = set(sig.parameters.keys())
            filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
            instance = cls(**filtered_dict)
            
            # Add any additional fields as attributes
            for k, v in config_dict.items():
                if k not in valid_keys:
                    setattr(instance, k, v)
        
        return instance
    
    @classmethod
    def from_yaml(cls, file_path: Union[str, Path]) -> 'BaseConfig':
        """
        Create configuration from YAML file.
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            Configuration instance
        """
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod 
    def from_json(cls, file_path: Union[str, Path]) -> 'BaseConfig':
        """
        Create configuration from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Configuration instance
        """
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def update(self, **kwargs) -> 'BaseConfig':
        """
        Update configuration with new values.
        
        Args:
            **kwargs: New configuration values
            
        Returns:
            New configuration instance with updated values
        """
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return self.__class__.from_dict(config_dict)
    
    def merge(self, other: 'BaseConfig') -> 'BaseConfig':
        """
        Merge this configuration with another configuration.
        
        Args:
            other: Another configuration to merge with
            
        Returns:
            New configuration instance with merged values
        """
        config_dict = self.to_dict()
        other_dict = other.to_dict()
        config_dict.update(other_dict)
        return self.__class__.from_dict(config_dict)
    
    def __repr__(self) -> str:
        """String representation of the configuration."""
        return f"{self.__class__.__name__}({self.to_dict()})" 