"""
Plugin Manager

This module manages the plugin system and plugin lifecycle.
"""

import os
import importlib
import inspect
from typing import Dict, List, Any, Type, Optional
from .base_plugin import BasePlugin, ModelPlugin, DataPlugin, MetricsPlugin, VisualizationPlugin, HookPlugin


class PluginManager:
    """
    Manager for the plugin system.
    """
    
    def __init__(self):
        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_types: Dict[str, List[str]] = {
            'model': [],
            'data': [],
            'metrics': [],
            'visualization': [],
            'hook': []
        }
        self.hooks: Dict[str, List[HookPlugin]] = {
            'experiment_start': [],
            'experiment_end': [],
            'epoch_start': [],
            'epoch_end': [],
            'batch_start': [],
            'batch_end': []
        }
    
    def register_plugin(self, plugin: BasePlugin):
        """
        Register a plugin.
        
        Args:
            plugin: Plugin instance to register
        """
        if plugin.name in self.plugins:
            print(f"Warning: Plugin '{plugin.name}' is already registered. Overwriting.")
        
        self.plugins[plugin.name] = plugin
        
        # Categorize plugin by type
        if isinstance(plugin, ModelPlugin):
            self.plugin_types['model'].append(plugin.name)
        elif isinstance(plugin, DataPlugin):
            self.plugin_types['data'].append(plugin.name)
        elif isinstance(plugin, MetricsPlugin):
            self.plugin_types['metrics'].append(plugin.name)
        elif isinstance(plugin, VisualizationPlugin):
            self.plugin_types['visualization'].append(plugin.name)
        elif isinstance(plugin, HookPlugin):
            self.plugin_types['hook'].append(plugin.name)
            self._register_hooks(plugin)
        
        print(f"Plugin '{plugin.name}' registered successfully.")
    
    def _register_hooks(self, hook_plugin: HookPlugin):
        """Register hook plugin for specific lifecycle events."""
        hook_methods = {
            'experiment_start': hook_plugin.on_experiment_start,
            'experiment_end': hook_plugin.on_experiment_end,
            'epoch_start': hook_plugin.on_epoch_start,
            'epoch_end': hook_plugin.on_epoch_end,
            'batch_start': hook_plugin.on_batch_start,
            'batch_end': hook_plugin.on_batch_end,
        }
        
        for hook_name, method in hook_methods.items():
            # Check if method is overridden (not just the base implementation)
            if method.__func__ != getattr(HookPlugin, method.__name__).__func__:
                self.hooks[hook_name].append(hook_plugin)
    
    def unregister_plugin(self, plugin_name: str):
        """
        Unregister a plugin.
        
        Args:
            plugin_name: Name of the plugin to unregister
        """
        if plugin_name not in self.plugins:
            print(f"Warning: Plugin '{plugin_name}' is not registered.")
            return
        
        plugin = self.plugins[plugin_name]
        
        # Clean up plugin
        plugin.cleanup()
        
        # Remove from plugins dict
        del self.plugins[plugin_name]
        
        # Remove from type categories
        for plugin_type, plugin_list in self.plugin_types.items():
            if plugin_name in plugin_list:
                plugin_list.remove(plugin_name)
        
        # Remove from hooks if it's a hook plugin
        if isinstance(plugin, HookPlugin):
            for hook_list in self.hooks.values():
                if plugin in hook_list:
                    hook_list.remove(plugin)
        
        print(f"Plugin '{plugin_name}' unregistered successfully.")
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get a plugin by name."""
        return self.plugins.get(plugin_name)
    
    def get_plugins_by_type(self, plugin_type: str) -> List[BasePlugin]:
        """Get all plugins of a specific type."""
        plugin_names = self.plugin_types.get(plugin_type, [])
        return [self.plugins[name] for name in plugin_names if name in self.plugins]
    
    def list_plugins(self) -> Dict[str, Any]:
        """List all registered plugins with their information."""
        plugin_info = {}
        for name, plugin in self.plugins.items():
            plugin_info[name] = plugin.get_info()
        return plugin_info
    
    def enable_plugin(self, plugin_name: str):
        """Enable a plugin."""
        if plugin_name in self.plugins:
            self.plugins[plugin_name].enabled = True
            print(f"Plugin '{plugin_name}' enabled.")
        else:
            print(f"Plugin '{plugin_name}' not found.")
    
    def disable_plugin(self, plugin_name: str):
        """Disable a plugin."""
        if plugin_name in self.plugins:
            self.plugins[plugin_name].enabled = False
            print(f"Plugin '{plugin_name}' disabled.")
        else:
            print(f"Plugin '{plugin_name}' not found.")
    
    def execute_plugin(self, plugin_name: str, *args, **kwargs) -> Any:
        """
        Execute a plugin.
        
        Args:
            plugin_name: Name of the plugin to execute
            *args, **kwargs: Arguments to pass to the plugin
            
        Returns:
            Result from plugin execution
        """
        plugin = self.get_plugin(plugin_name)
        if plugin is None:
            raise ValueError(f"Plugin '{plugin_name}' not found.")
        
        if not plugin.enabled:
            print(f"Warning: Plugin '{plugin_name}' is disabled.")
            return None
        
        return plugin.execute(*args, **kwargs)
    
    def call_hooks(self, hook_name: str, *args, **kwargs):
        """
        Call all registered hooks for a specific event.
        
        Args:
            hook_name: Name of the hook event
            *args, **kwargs: Arguments to pass to hook methods
        """
        if hook_name not in self.hooks:
            return
        
        for hook_plugin in self.hooks[hook_name]:
            if hook_plugin.enabled:
                try:
                    method = getattr(hook_plugin, f'on_{hook_name}')
                    method(*args, **kwargs)
                except Exception as e:
                    print(f"Error in hook '{hook_plugin.name}' for event '{hook_name}': {e}")
    
    def load_plugins_from_directory(self, directory_path: str):
        """
        Load plugins from a directory.
        
        Args:
            directory_path: Path to directory containing plugin files
        """
        if not os.path.exists(directory_path):
            print(f"Plugin directory '{directory_path}' does not exist.")
            return
        
        for filename in os.listdir(directory_path):
            if filename.endswith('.py') and not filename.startswith('__'):
                module_name = filename[:-3]  # Remove .py extension
                try:
                    # Import the module
                    spec = importlib.util.spec_from_file_location(
                        module_name, 
                        os.path.join(directory_path, filename)
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Look for plugin classes in the module
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, BasePlugin) and 
                            obj != BasePlugin and
                            not obj.__name__.startswith('Base')):
                            
                            # Try to instantiate the plugin
                            try:
                                plugin_instance = obj()
                                self.register_plugin(plugin_instance)
                            except Exception as e:
                                print(f"Failed to instantiate plugin '{obj.__name__}': {e}")
                
                except Exception as e:
                    print(f"Failed to load plugin from '{filename}': {e}")
    
    def save_plugin_config(self, filepath: str):
        """Save current plugin configuration to file."""
        import yaml
        
        config = {
            'plugins': {},
            'enabled_plugins': []
        }
        
        for name, plugin in self.plugins.items():
            config['plugins'][name] = plugin.get_info()
            if plugin.enabled:
                config['enabled_plugins'].append(name)
        
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Plugin configuration saved to '{filepath}'.")
    
    def load_plugin_config(self, filepath: str):
        """Load plugin configuration from file."""
        import yaml
        
        if not os.path.exists(filepath):
            print(f"Plugin configuration file '{filepath}' does not exist.")
            return
        
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        
        # Enable/disable plugins based on config
        for plugin_name in config.get('enabled_plugins', []):
            self.enable_plugin(plugin_name)
        
        # Disable plugins not in the enabled list
        for plugin_name in self.plugins:
            if plugin_name not in config.get('enabled_plugins', []):
                self.disable_plugin(plugin_name)
        
        print(f"Plugin configuration loaded from '{filepath}'.")


# Global plugin manager instance
plugin_manager = PluginManager() 