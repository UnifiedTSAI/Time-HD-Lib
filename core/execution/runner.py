"""
Experiment Runner

This module provides a clean interface for running time series experiments
with support for distributed training and batch execution.
"""

import os
import torch
from typing import Dict, Any, Tuple
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from core.config import BaseConfig, ConfigManager
from core.models import ModelManager
from core.experiments.long_term_forecasting import LongTermForecastingExperiment
# from core.experiments.short_term_forecasting import ShortTermForecastingExperiment
# from core.experiments.classification import ClassificationExperiment
# from core.experiments.imputation import ImputationExperiment
# from core.experiments.anomaly_detection import AnomalyDetectionExperiment


class ExperimentRunner:
    """
    Main runner for executing time series experiments.
    
    Handles experiment setup, execution, and result collection with support
    for distributed training via HuggingFace Accelerate.
    """
    
    def __init__(self, config: BaseConfig):
        """
        Initialize the experiment runner.
        
        Args:
            config: Configuration object containing experiment parameters
        """
        self.config = config
        self.accelerator = None
        self._setup_environment()
    
    def _setup_environment(self):
        """Setup the experiment environment including seeds, device, and accelerator."""
        # Fix random seeds for reproducibility
        random_seed = getattr(self.config, 'random_seed', 2021)
        self._fix_seed(random_seed)
        
        # Setup distributed training accelerator if available
        try:
            kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
            self.accelerator = Accelerator(kwargs_handlers=[kwargs])
        except Exception as e:
            print(f"Warning: Could not initialize accelerator: {e}")
            self.accelerator = None
        
        # Set compute device
        use_gpu = getattr(self.config, 'use_gpu', True)

        if torch.cuda.is_available() and use_gpu:
            self.device = self.accelerator.device if self.accelerator else torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # Use accelerator for consistent printing across distributed processes
        self.accelerator.print(f'Using device: {self.device}')
    
    def _fix_seed(self, seed: int):
        """
        Fix random seeds for reproducible experiments.
        
        Args:
            seed: Random seed value
        """
        import random
        import numpy as np
        
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _create_experiment(self) -> Any:
        """
        Create the appropriate experiment instance based on task type.
        
        Returns:
            Experiment instance for the specified task
            
        Raises:
            ValueError: If the task type is not supported
        """
        experiment_map = {
            'long_term_forecast': LongTermForecastingExperiment,
            # Additional experiment types can be added here
            # 'short_term_forecast': ShortTermForecastingExperiment,
            # 'classification': ClassificationExperiment,
            # 'imputation': ImputationExperiment,
            # 'anomaly_detection': AnomalyDetectionExperiment,
        }
        
        experiment_class = experiment_map.get(self.config.task_name)
        if experiment_class is None:
            available_tasks = ", ".join(experiment_map.keys())
            raise ValueError(f"Unsupported task: '{self.config.task_name}'. Available tasks: [{available_tasks}]")
        
        return experiment_class(self.config)
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the complete experiment workflow.
        
        Returns:
            Dictionary containing experiment results including training metrics,
            best model information, and test performance
        """
        # Create unique setting string for experiment identification
        setting = self._create_setting_string()
        
        # Create and configure experiment instance
        experiment = self._create_experiment()
        experiment.set_accelerator(self.accelerator)
        
        results = {}
        
        if self.config.is_training:
            self.accelerator.print(f'>>> Starting training for {setting} <<<')
            
            # Execute training phase
            all_epoch_metrics, best_metrics, best_model_path = experiment.train(setting)
            
            results['training'] = {
                'all_epoch_metrics': all_epoch_metrics,
                'best_metrics': best_metrics,
                'best_model_path': best_model_path
            }
            
            # Test with the best trained model
            self.accelerator.print(f'>>> Starting testing for {setting} <<<')
            mse, mae = experiment.test(setting, best_model_path)
            
            results['testing'] = {
                'mse': mse,
                'mae': mae
            }
            
        else:
            # Testing only mode
            self.accelerator.print(f'>>> Starting testing for {setting} <<<')
            mse, mae = experiment.test(setting)
            
            results['testing'] = {
                'mse': mse,
                'mae': mae
            }
            
            self.accelerator.print(f'>>> Testing completed for {setting} <<<')
        
        return results
    
    def _create_setting_string(self) -> str:
        """
        Create a unique setting string for experiment identification.
        
        Returns:
            Compact string representation of experiment configuration
        """
        # Generate shorter setting string to avoid path length issues
        task = getattr(self.config, 'task_name', 'forecast')
        model = getattr(self.config, 'model', 'UCast')
        data = getattr(self.config, 'data', 'data').replace(' ', '_')  # Handle spaces in dataset names
        seq_len = getattr(self.config, 'seq_len', 96)
        pred_len = getattr(self.config, 'pred_len', 96)
        
        # Use simplified format to keep paths manageable
        setting = f"{task}_{model}_{data}_sl{seq_len}_pl{pred_len}"
        
        return setting


class BatchRunner:
    """
    Runner for executing multiple experiments in batch mode.
    
    Enables systematic evaluation across different configurations,
    models, and datasets with automatic result collection.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the batch runner.
        
        Args:
            config_manager: Configuration manager for creating experiment configs
        """
        self.config_manager = config_manager
        self.experiments = []
    
    def add_experiment(self, **config_updates):
        """
        Add an experiment configuration to the batch queue.
        
        Args:
            **config_updates: Configuration parameters to override for this experiment
        """
        config = self.config_manager.get_unified_config()
        config = config.update(**config_updates)
        self.experiments.append(config)
    
    def run_batch(self) -> list:
        """
        Execute all experiments in the batch queue.
        
        Returns:
            List of dictionaries containing configuration and results for each experiment
        """
        batch_results = []
        
        for i, config in enumerate(self.experiments):
            print(f'\n=== Running Experiment {i+1}/{len(self.experiments)} ===')
            print(f'Configuration: Model={config.model}, Data={config.data}, Pred_len={config.pred_len}')
            
            # Create and run experiment
            runner = ExperimentRunner(config)
            experiment_result = runner.run()
            
            batch_results.append({
                'experiment_id': i + 1,
                'config': config.to_dict(),
                'results': experiment_result
            })
            
            print(f'Experiment {i+1} completed successfully')
        
        return batch_results


def create_runner_from_args(args) -> ExperimentRunner:
    """
    Create an experiment runner from command line arguments.
    
    Args:
        args: Parsed command line arguments from argparse
        
    Returns:
        Configured ExperimentRunner instance ready for execution
    """
    # Create configuration manager
    config_manager = ConfigManager()
    
    # Convert arguments to unified configuration
    config = config_manager.parse_args_and_create_config(args)
    
    # Create and return experiment runner
    return ExperimentRunner(config) 