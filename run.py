#!/usr/bin/env python3
"""
High-Dimensional Time Series Analysis Framework - Main Runner

This script provides the main interface for running time series experiments.
"""

import argparse
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.cli import create_argument_parser
from core.execution.runner import create_runner_from_args


def run_hyperparameter_search(accelerator, args):
    """Run hyperparameter search using the framework."""
    
    # Load dataset-specific default parameters from configs/{model}.yaml
    dataset_config_path = f'configs/{args.model}.yaml'
    if os.path.exists(dataset_config_path):
        import yaml
        with open(dataset_config_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
        if args.data in dataset_config:
            defaults = dataset_config[args.data]
            for k, v in defaults.items():
                setattr(args, k, v)
            accelerator.print(f"Loaded default parameters for {args.data} from {dataset_config_path}: {defaults}")
        else:
            accelerator.print(f"No default parameters found for {args.data} in {dataset_config_path}")
    else:
        accelerator.print(f"Dataset config file not found: {dataset_config_path}")
    
    # Import necessary modules for hyperparameter search
    import yaml
    import copy
    import itertools
    import datetime
    import json
    import numpy as np
    import torch
    from accelerate.utils import find_executable_batch_size
    
    def clear_gpu_memory():
        """Clear GPU memory to avoid out-of-memory errors."""
        # import gc
        # torch.cuda.empty_cache()
        # gc.collect()
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.ipc_collect()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    
    def save_results_to_csv(args, best_result, hp_results_dir):
        """Save hyperparameter search results to CSV file."""
        try:
            import pandas as pd
            
            # Prepare data for CSV export
            csv_data = []
            for pred_len_result in best_result['pred_len_results']:
                row = {
                    'model': args.model,
                    'dataset': args.data,
                    'pred_len': pred_len_result['pred_len'],
                    'test_mse': pred_len_result['test_loss'],
                    'test_mae': pred_len_result['test_mae_loss'],
                    **best_result['hyperparameters']
                }
                csv_data.append(row)
            
            # Save results to CSV file
            df = pd.DataFrame(csv_data)
            csv_file = os.path.join(hp_results_dir, 'results.csv')
            df.to_csv(csv_file, index=False)
        except ImportError:
            accelerator.print("pandas not available, skipping CSV export")
    
    # Create hyperparameter logs directory if it doesn't exist
    os.makedirs(args.hp_log_dir, exist_ok=True)
    
    # Load hyperparameter search configuration
    hp_config_path = f'config_hp/{args.model}.yaml'
    if not os.path.exists(hp_config_path):
        accelerator.print(f"Hyperparameter config file not found: {hp_config_path}")
        accelerator.print("Please create a hyperparameter config file first.")
        return
        
    # Load prediction length configuration
    pred_len_config_path = 'configs/pred_len_config.yaml'
    if os.path.exists(pred_len_config_path):
        with open(pred_len_config_path, 'r') as f:
            pred_len_config = yaml.safe_load(f)
            
        # Get the prediction length list for the current dataset
        if args.data in pred_len_config:
            pred_len_list = pred_len_config[args.data]
            accelerator.print(f"Using pred_len list for {args.data}: {pred_len_list}")
        else:
            # Use default prediction length list
            pred_len_list = [48]
            accelerator.print(f"No pred_len config found for {args.data}, using default: {pred_len_list}")
    else:
        accelerator.print(f"pred_len config file not found: {pred_len_config_path}")
        accelerator.print("Using default pred_len list: [48]")
        pred_len_list = [48]
        
    # Load hyperparameter configuration
    with open(hp_config_path, 'r') as f:
        hp_config = yaml.safe_load(f)
        
    if not hp_config:
        accelerator.print(f"Empty hyperparameter config file: {hp_config_path}")
        return
        
    # Build hyperparameter grid for searching
    hp_grid = {param_name: param_values for param_name, param_values in hp_config.items() 
              if isinstance(param_values, list)}
                
    if not hp_grid:
        accelerator.print("No hyperparameters to search found in config file")
        return
        
    # Generate all combinations of hyperparameters using grid search
    hp_keys = list(hp_grid.keys())
    hp_values = list(hp_grid.values())
    hp_combinations = list(itertools.product(*hp_values))
    
    accelerator.print(f"Starting hyperparameter search with {len(hp_combinations)} combinations, each on {len(pred_len_list)} pred_len values")
    
    # Create timestamp for unique logging directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    hp_results_dir = os.path.join(args.hp_log_dir, f"{args.model}_{args.data}_{timestamp}")
    
    # Create results directory and summary file on main process
    if accelerator.is_main_process:
        os.makedirs(hp_results_dir, exist_ok=True)
        summary_file = os.path.join(hp_results_dir, "hp_summary.json")
    
    # Ensure all processes wait for directory creation
    accelerator.wait_for_everyone()
    
    all_results = []
    
    # Run experiments for each hyperparameter combination
    for i, hp_combination in enumerate(hp_combinations):
        accelerator.print(f"\nHyperparameter combination {i+1}/{len(hp_combinations)}")
        
        # Create a new arguments object for this combination
        hp_args = copy.deepcopy(args)
        
        # Set hyperparameters for this combination
        hp_config_dict = {hp_keys[j]: hp_combination[j] for j in range(len(hp_keys))}
        for param_name, param_value in hp_config_dict.items():
            setattr(hp_args, param_name, param_value)
        
        accelerator.print(f"Testing hyperparameters: {hp_config_dict}")
        
        # Store results for different prediction length values
        pred_len_results = []
        
        # Train and evaluate for each prediction length
        for pred_len in pred_len_list:
            accelerator.print(f"\n  Training with pred_len = {pred_len}")
            
            # Set current prediction length and sequence length
            hp_args.pred_len = pred_len
            hp_args.seq_len = pred_len * hp_args.seq_len_factor
            accelerator.print(f"  Training with seq_len = {hp_args.seq_len}")
            
            # Create experiment runner with dynamic batch size finding
            @find_executable_batch_size(starting_batch_size=hp_args.batch_size)
            def inner_training_loop(batch_size):
                hp_args.batch_size = batch_size
                accelerator.print(f"  Trying batch size: {batch_size}")
                
                # Create experiment runner using the framework
                runner = create_runner_from_args(hp_args)
                runner.accelerator = accelerator
                
                # Force training mode
                hp_args.is_training = 1
                
                # Run the experiment
                results = runner.run()

                del runner
                
                return results
            
            try:
                results = inner_training_loop()
                
                # Extract metrics from experiment results
                if 'training' in results:
                    training_results = results['training']
                    best_metrics = training_results['best_metrics']
                    
                    # Record best results for current prediction length
                    pred_len_result = {
                        "pred_len": pred_len,
                        "best_epoch": best_metrics["epoch"],
                        "train_loss": best_metrics["train_loss"],
                        "vali_loss": best_metrics["vali_loss"],
                        "vali_mae_loss": best_metrics["vali_mae_loss"],
                        "test_loss": best_metrics["test_loss"],
                        "test_mae_loss": best_metrics["test_mae_loss"]
                    }
                else:
                    # Fallback for missing training results
                    accelerator.print(f"  Warning: No training results found for pred_len={pred_len}")
                    pred_len_result = {
                        "pred_len": pred_len,
                        "best_epoch": 0,
                        "train_loss": float('inf'),
                        "vali_loss": float('inf'),
                        "vali_mae_loss": float('inf'),
                        "test_loss": float('inf'),
                        "test_mae_loss": float('inf')
                    }
                
                pred_len_results.append(pred_len_result)
                
                accelerator.print(f"  Results: vali_loss={pred_len_result['vali_loss']:.6f}, test_loss={pred_len_result['test_loss']:.6f}")
                
            except RuntimeError as e:
                accelerator.print(f"  [RuntimeError] {e}")
                accelerator.print("  Likely OOM even at batch size 1. Skipping this configuration.")
                
                # Add failed result entry
                pred_len_result = {
                    "pred_len": pred_len,
                    "best_epoch": 0,
                    "train_loss": float('inf'),
                    "vali_loss": float('inf'),
                    "vali_mae_loss": float('inf'),
                    "test_loss": float('inf'),
                    "test_mae_loss": float('inf'),
                    "error": str(e)
                }
                pred_len_results.append(pred_len_result)
                
            except Exception as e:
                accelerator.print(f"  [Error] Unexpected error: {e}")
                
                # Add failed result entry
                pred_len_result = {
                    "pred_len": pred_len,
                    "best_epoch": 0,
                    "train_loss": float('inf'),
                    "vali_loss": float('inf'),
                    "vali_mae_loss": float('inf'),
                    "test_loss": float('inf'),
                    "test_mae_loss": float('inf'),
                    "error": str(e)
                }
                pred_len_results.append(pred_len_result)

            del results
            clear_gpu_memory()
            
        # Calculate average metrics across all prediction length values (excluding failed runs)
        valid_results = [r for r in pred_len_results if r["vali_loss"] != float('inf')]
        
        if valid_results:
            avg_vali_loss = np.mean([r["vali_loss"] for r in valid_results])
            avg_test_loss = np.mean([r["test_loss"] for r in valid_results])
            avg_test_mae_loss = np.mean([r["test_mae_loss"] for r in valid_results])
        else:
            # All runs failed for this hyperparameter combination
            avg_vali_loss = float('inf')
            avg_test_loss = float('inf')
            avg_test_mae_loss = float('inf')
        
        # Record results for this hyperparameter combination
        result = {
            "combination_id": i+1,
            "hyperparameters": hp_config_dict,
            "avg_vali_loss": float(avg_vali_loss),
            "avg_test_loss": float(avg_test_loss),
            "avg_test_mae_loss": float(avg_test_mae_loss),
            "pred_len_results": pred_len_results,
            "num_valid_runs": len(valid_results),
            "num_total_runs": len(pred_len_results)
        }
        
        # Save individual result on main process
        if accelerator.is_main_process:
            hp_result_file = os.path.join(hp_results_dir, f"result_{i+1}.json")
            with open(hp_result_file, 'w') as f:
                json.dump(result, f, indent=4)
        
        all_results.append(result)
        
        accelerator.print(f"Combination {i+1} completed: avg_vali_loss={avg_vali_loss:.6f}, valid_runs={len(valid_results)}/{len(pred_len_results)}")
        
        # Update summary file on main process
        if accelerator.is_main_process:
            with open(summary_file, 'w') as f:
                json.dump(all_results, f, indent=4)
        
        # Synchronize all distributed processes
        accelerator.wait_for_everyone()

        clear_gpu_memory()
    
    # Find best hyperparameter combination based on validation loss (excluding failed runs)
    valid_results = [r for r in all_results if r["avg_vali_loss"] != float('inf')]
    
    if not valid_results:
        accelerator.print("\n❌ No valid hyperparameter combinations found!")
        accelerator.print("All combinations resulted in errors. Please check your configuration.")
        return
    
    best_result = min(valid_results, key=lambda x: x["avg_vali_loss"])
    
    # Save best result to separate file on main process
    if accelerator.is_main_process:
        best_result_file = os.path.join(hp_results_dir, "best_result.json")
        with open(best_result_file, 'w') as f:
            json.dump(best_result, f, indent=4)
    
    # Print best hyperparameters and results
    accelerator.print("\n" + "="*80)
    accelerator.print("🎉 Hyperparameter search completed!")
    accelerator.print("="*80)
    accelerator.print(f"✨ Best hyperparameters: {best_result['hyperparameters']}")
    accelerator.print(f"📊 Best average validation loss: {best_result['avg_vali_loss']:.6f}")
    accelerator.print(f"📊 Corresponding average test loss: {best_result['avg_test_loss']:.6f}")
    accelerator.print(f"📊 Corresponding average test MAE: {best_result['avg_test_mae_loss']:.6f}")
    accelerator.print(f"✅ Valid runs: {best_result['num_valid_runs']}/{best_result['num_total_runs']}")
    
    # Print results for each prediction length
    accelerator.print("\n📈 Results for each pred_len:")
    for result in best_result['pred_len_results']:
        if 'error' not in result:
            accelerator.print(f"   pred_len={result['pred_len']}: "
                             f"test_loss={result['test_loss']:.6f}, "
                             f"test_mae={result['test_mae_loss']:.6f}")
        else:
            accelerator.print(f"   pred_len={result['pred_len']}: FAILED - {result['error']}")
        
    if accelerator.is_main_process:
        accelerator.print(f"\n💾 Results saved to: {hp_results_dir}")
        
        # Save results to CSV on main process
        save_results_to_csv(args, best_result, hp_results_dir)
        accelerator.print(f"📄 CSV results saved.")
    
    # Print summary of all hyperparameter combinations
    accelerator.print(f"\n📋 Summary of all {len(all_results)} combinations:")
    accelerator.print(f"   ✅ Successful: {len(valid_results)}")
    accelerator.print(f"   ❌ Failed: {len(all_results) - len(valid_results)}")
    
    # Ensure all processes finish together
    accelerator.wait_for_everyone()
    clear_gpu_memory()


def main():
    """Main entry point for the framework."""
    
    # Parse command line arguments first to check GPU settings
    from core.cli import create_argument_parser
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Set CUDA_VISIBLE_DEVICES based on GPU arguments
    if hasattr(args, 'gpu') and args.gpu is not None:
        # Handle GPU specification
        gpu_str = str(args.gpu).replace(' ', '')  # Remove spaces and convert to string
        if gpu_str:  # Only set if not empty
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    
    # Initialize accelerator after setting GPU environment
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    
    # Print framework header
    accelerator.print("=" * 80)
    accelerator.print("🚀 Time-HD-Lib: A Lirbrary for High-Dimensional Time Series Forecasting")
    accelerator.print("=" * 80)
    
    # Add accelerator to arguments for hyperparameter search compatibility
    args.accelerator = accelerator
    
    # Check if hyperparameter search mode is requested
    if hasattr(args, 'hyper_parameter_searching') and args.hyper_parameter_searching:
        accelerator.print("🔍 Running in hyperparameter search mode...")
        # Add required fields for legacy compatibility
        if not hasattr(args, 'model_id'):
            args.model_id = f"hp_search_{args.model}"
        if not hasattr(args, 'des'):
            args.des = "hyperparameter_search"
        
        run_hyperparameter_search(accelerator, args)
        return

    # Load pred_len from config file if not specified via command line
    if args.pred_len is None:
        import yaml
        pred_len_config_path = 'configs/pred_len_config.yaml'
        with open(pred_len_config_path, 'r') as f:
            pred_len_config = yaml.safe_load(f)
        
        args.pred_len = pred_len_config[args.data][0]

    if args.seq_len is None:
        args.seq_len = args.pred_len * args.seq_len_factor
    
    # Print configuration summary
    accelerator.print("\n📋 Configuration Summary:")
    accelerator.print(f"   Task: {args.task_name}")
    accelerator.print(f"   Model: {args.model}")
    accelerator.print(f"   Dataset: {args.data}")
    accelerator.print(f"   Input Sequence Length: {args.seq_len}")
    accelerator.print(f"   Prediction Length: {args.pred_len}")
    accelerator.print(f"   Training Mode: {'Yes' if args.is_training else 'No'}")
    accelerator.print(f"   GPU: {'Yes' if args.use_gpu else 'No'}")
    accelerator.print()
    
    # Create experiment runner from arguments
    accelerator.print("🔧 Initializing experiment runner...")
    
    runner = create_runner_from_args(args)
    
    runner.accelerator = accelerator
    
    # Run the experiment
    accelerator.print("🎯 Starting experiment execution...")
    
    results = runner.run()
    
    accelerator.print("\n" + "=" * 80)
    accelerator.print("✅ Experiment Completed Successfully!")
    accelerator.print("=" * 80)
    
    # Display training results if available
    if 'training' in results:
        training_results = results['training']
        best_metrics = training_results['best_metrics']
        accelerator.print(f"\n📊 Best Training Results (Epoch {best_metrics['epoch']}):")
        accelerator.print(f"   Train Loss: {best_metrics['train_loss']:.6f}")
        accelerator.print(f"   Validation Loss: {best_metrics['vali_loss']:.6f}")
        accelerator.print(f"   Validation MAE: {best_metrics['vali_mae_loss']:.6f}")
        accelerator.print(f"   Test Loss: {best_metrics['test_loss']:.6f}")
        accelerator.print(f"   Test MAE: {best_metrics['test_mae_loss']:.6f}")
    
    # Display testing results if available
    if 'testing' in results:
        testing_results = results['testing']
        accelerator.print(f"\n🎯 Final Test Results:")
        accelerator.print(f"   MSE: {testing_results['mse']:.6f}")
        accelerator.print(f"   MAE: {testing_results['mae']:.6f}")
    
    accelerator.print("\n🎉 All results have been saved to the experiments directory.")


def run_batch_experiments():
    """Run multiple experiments in batch mode."""
    
    print("=" * 80)
    print("🚀 Batch Experiment Runner")
    print("=" * 80)
    
    from core.config import ConfigManager
    from core.execution.runner import BatchRunner
    
    # Create configuration manager
    config_manager = ConfigManager()
    
    # Create batch experiment runner
    batch_runner = BatchRunner(config_manager)
    
    # Define experiment configurations to test
    models_to_test = ['UCast', 'DLinear', 'TimesNet']
    datasets_to_test = ['ETTh1', 'ETTh2']
    pred_lens = [96, 192, 336, 720]
    
    experiment_count = 0
    for model in models_to_test:
        for dataset in datasets_to_test:
            for pred_len in pred_lens:
                batch_runner.add_experiment(
                    model=model,
                    data=dataset,
                    pred_len=pred_len,
                    is_training=True
                )
                experiment_count += 1
    
    print(f"📊 Running {experiment_count} experiments...")
    
    # Execute all experiments in batch
    results = batch_runner.run_batch()
    
    print(f"\n✅ Completed {len(results)} experiments!")
    
    # Save batch results to file
    import yaml
    with open('batch_results.yaml', 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    print("📄 Batch results saved to 'batch_results.yaml'")


def list_available_models():
    """List all available models in the registry."""
    
    from accelerate import Accelerator
    
    # Initialize accelerator for multi-process environments
    accelerator = Accelerator()
    
    # Only print from main process to avoid duplicates
    if not accelerator.is_main_process:
        return
    
    accelerator.print("📋 Available Models:")
    accelerator.print("-" * 40)
    
    from core.registry import model_registry
    from core.models import ModelManager
    
    # Load models to trigger registration
    model_manager = ModelManager()
    model_manager._ensure_models_loaded()
    
    available_models = model_registry.list_models()
    
    if not available_models:
        accelerator.print("   No models registered yet.")
        return
    
    for model_name in available_models:
        metadata = model_registry.get_metadata(model_name)
        accelerator.print(f"   {model_name}")
        if metadata:
            accelerator.print(f"      Paper: {metadata.get('paper', 'N/A')}")
            accelerator.print(f"      Year: {metadata.get('year', 'N/A')}")
        accelerator.print()


def show_framework_info():
    """Show framework information and architecture overview."""
    
    return """
🏗️  Framework Architecture Overview
═════════════════════════════════════

📁 Core Components:
   ├── 🏗️  core/
   │   ├── 📝 config/         - Configuration management system
   │   ├── 📊 registry/       - Model & dataset registration
   │   ├── 🤖 models/         - Model management and loading
   │   ├── 📊 data/           - Data processing pipeline
   │   ├── 🧪 experiments/    - Experiment orchestration
   │   ├── ⚙️  execution/      - Execution engine
   │   ├── 🛠️  utils/         - Utility functions
   │   ├── 🔌 plugins/        - Plugin system for extensibility
   │   └── 💻 cli/            - Command-line interface

🌟 Key Features:
   ✨ Modular architecture with clear separation of concerns
   🔧 Plugin system for easy extensibility
   📋 Type-safe configuration management
   🏃‍♀️ Backward compatibility with existing models
   🎯 Unified experiment interface
   📊 Advanced metrics and visualization
   🔄 Flexible data transformation pipeline
   🚀 Distributed training support with Accelerate
   🔍 Comprehensive hyperparameter search

🚀 Usage Examples:
   # Basic training
   accelerate launch run_refactored.py --model UCast --data ETTh1 --pred_len 96
   
   # Hyperparameter search
   accelerate launch run_refactored.py --model UCast --data ETTh1 --hyper_parameter_searching
   
   # List available models
   accelerate launch --num_processes=1 run_refactored.py --list-models
   
   # Batch experiments
   python run_refactored.py --batch
    """


if __name__ == "__main__":
    
    # Handle special commands that don't require full argument parsing
    if len(sys.argv) > 1:
        if sys.argv[1] == '--batch':
            run_batch_experiments()
            sys.exit(0)
        elif sys.argv[1] == '--list-models':
            list_available_models()
            sys.exit(0)
        elif sys.argv[1] == '--info':
            print(show_framework_info())
            sys.exit(0)
        elif sys.argv[1] == '--help' or sys.argv[1] == '-h':
            # Show enhanced help with framework information
            print(show_framework_info())
            parser = create_argument_parser()
            parser.print_help()
            print("\n🔧 Additional Commands:")
            print("   --batch           Run batch experiments")
            print("   --list-models     List available models")
            print("   --info           Show framework information")
            sys.exit(0)
    
    # Run standard experiment workflow
    main() 