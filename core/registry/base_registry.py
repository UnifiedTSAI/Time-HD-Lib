"""
Base Registry System

This module provides the foundation for all registry systems in the framework.
Registries enable dynamic component discovery and management.
"""

import inspect
from typing import Dict, Any, Optional, Callable, Type
from abc import ABC, abstractmethod


class BaseRegistry(ABC):
    """
    Base class for all registry systems.
    
    Provides common functionality for registering and retrieving framework components
    such as models, datasets, transforms, and plugins.
    """
    
    def __init__(self, name: str):
        """
        Initialize the registry.
        
        Args:
            name: Name of the registry for identification
        """
        self.name = name
        self._registry: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    def register(self, name: str, obj: Any, **metadata) -> Any:
        """
        Register an object with the given name.
        
        Args:
            name: Unique name to register the object under
            obj: Object to register (class, function, or instance)
            **metadata: Additional metadata about the object (e.g., paper, year, author)
            
        Returns:
            The registered object (enables use as decorator)
        """
        if name in self._registry:
            print(f"Warning: '{name}' is already registered in {self.name}. Overwriting previous registration.")
        
        self._registry[name] = obj
        self._metadata[name] = metadata
        return obj
    
    def get(self, name: str, default: Any = None) -> Any:
        """
        Retrieve a registered object by name.
        
        Args:
            name: Name of the object to retrieve
            default: Default value to return if object not found
            
        Returns:
            The registered object or default value
        """
        return self._registry.get(name, default)
    
    def contains(self, name: str) -> bool:
        """
        Check if an object is registered with the given name.
        
        Args:
            name: Name to check for registration
            
        Returns:
            True if object is registered, False otherwise
        """
        return name in self._registry
    
    def list_registered(self) -> list:
        """
        Get a list of all registered names.
        
        Returns:
            List of registered object names
        """
        return list(self._registry.keys())
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """
        Get metadata for a registered object.
        
        Args:
            name: Name of the registered object
            
        Returns:
            Dictionary containing metadata for the object
        """
        return self._metadata.get(name, {})
    
    def unregister(self, name: str) -> bool:
        """
        Remove an object from the registry.
        
        Args:
            name: Name of the object to unregister
            
        Returns:
            True if object was successfully unregistered, False if not found
        """
        if name in self._registry:
            del self._registry[name]
            self._metadata.pop(name, None)
            return True
        return False
    
    def clear(self):
        """Clear all registered objects and their metadata."""
        self._registry.clear()
        self._metadata.clear()
    
    def __len__(self) -> int:
        """Get the number of registered objects."""
        return len(self._registry)
    
    def __contains__(self, name: str) -> bool:
        """Check if an object is registered (supports 'in' operator).""" 
        return self.contains(name)
    
    def __repr__(self) -> str:
        """String representation of the registry."""
        return f"{self.__class__.__name__}(name='{self.name}', registered={len(self)})" 