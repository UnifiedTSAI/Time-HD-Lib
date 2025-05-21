import os
import torch
import yaml
import numpy as np
import json
import datetime
import copy
import itertools
from accelerate.utils import find_executable_batch_size
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.tools import save_results_to_csv
import gc


def clear_gpu_memory(args):
    """Clear GPU memory based on device type."""
    if args.gpu_type == 'mps':
        torch.backends.mps.empty_cache()
    elif args.gpu_type == 'cuda':
        torch.cuda.empty_cache()


def create_experiment_setting(args, iteration=0):
    """Create a string describing the experiment settings."""
    return '{}_{}_{}_sl{}_pl{}_ft{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.task_name,
        args.data,
        args.model,
        args.seq_len,
        args.pred_len,
        args.features,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.expand,
        args.d_conv,
        args.factor,
        args.embed,
        args.distil,
        args.des, 
        iteration
    )


def run_hyperparameter_search(args):
    """Run hyperparameter search mode."""
    # Create hp_logs directory if it doesn't exist
    os.makedirs(args.hp_log_dir, exist_ok=True)
    
    # Load hyperparameter search configuration
    hp_config_path = f'config_hp/{args.model}.yaml'
    if not os.path.exists(hp_config_path):
        args.accelerator.print(f"Hyperparameter config file not found: {hp_config_path}")
        args.accelerator.print("Please create a hyperparameter config file first.")
        return
        
    # Load prediction length configuration
    pred_len_config_path = 'config_hp/pred_len_config.yaml'
    if os.path.exists(pred_len_config_path):
        with open(pred_len_config_path, 'r') as f:
            pred_len_config = yaml.safe_load(f)
            
        # Get the pred_len list for the current dataset
        if args.data in pred_len_config:
            pred_len_list = pred_len_config[args.data]
            args.accelerator.print(f"Using pred_len list for {args.data}: {pred_len_list}")
        else:
            # Use default pred_len list
            pred_len_list = [48]
            args.accelerator.print(f"No pred_len config found for {args.data}, using default: {pred_len_list}")
    else:
        args.accelerator.print(f"pred_len config file not found: {pred_len_config_path}")
        args.accelerator.print("Using default pred_len list: [48]")
        pred_len_list = [48]
        
    # Load hyperparameter configuration
    with open(hp_config_path, 'r') as f:
        hp_config = yaml.safe_load(f)
        
    if not hp_config:
        args.accelerator.print(f"Empty hyperparameter config file: {hp_config_path}")
        return
        
    # Build hyperparameter grid
    hp_grid = {param_name: param_values for param_name, param_values in hp_config.items() 
              if isinstance(param_values, list)}
                
    if not hp_grid:
        args.accelerator.print("No hyperparameters to search found in config file")
        return
        
    # Generate all combinations of hyperparameters
    hp_keys = list(hp_grid.keys())
    hp_values = list(hp_grid.values())
    hp_combinations = list(itertools.product(*hp_values))
    
    args.accelerator.print(f"Starting hyperparameter search with {len(hp_combinations)} combinations, each on {len(pred_len_list)} pred_len values")
    
    # Create timestamp for logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    hp_results_dir = os.path.join(args.hp_log_dir, f"{args.model}_{args.data}_{timestamp}")
    
    # Create directory and summary file on main process
    if args.accelerator.is_main_process:
        os.makedirs(hp_results_dir, exist_ok=True)
        summary_file = os.path.join(hp_results_dir, "hp_summary.json")
    
    # Ensure all processes wait for directory creation
    args.accelerator.wait_for_everyone()
    
    all_results = []
    
    # Run experiments for each hyperparameter combination
    for i, hp_combination in enumerate(hp_combinations):
        args.accelerator.print(f"\nHyperparameter combination {i+1}/{len(hp_combinations)}")
        
        # Create a new args object for this combination
        hp_args = copy.deepcopy(args)
        
        # Set hyperparameters for this combination
        hp_config_dict = {hp_keys[j]: hp_combination[j] for j in range(len(hp_keys))}
        for param_name, param_value in hp_config_dict.items():
            setattr(hp_args, param_name, param_value)
        
        # Store results for different pred_len values
        pred_len_results = []
        
        # Train and evaluate for each pred_len
        for pred_len in pred_len_list:
            args.accelerator.print(f"\nTraining with pred_len = {pred_len}")
            
            # Set current pred_len and seq_len
            hp_args.pred_len = pred_len
            hp_args.seq_len = pred_len * hp_args.seq_len_factor
            args.accelerator.print(f"Training with seq_len = {hp_args.seq_len}")
            
            # Create experiment setting string
            setting = create_experiment_setting(hp_args)
            
            # Train model with dynamic batch size finding
            @find_executable_batch_size(starting_batch_size=hp_args.batch_size)
            def inner_training_loop(batch_size):
                hp_args.batch_size = batch_size
                hp_args.accelerator.print(f"Trying batch size: {batch_size}")
                torch.cuda.empty_cache()
                import gc; gc.collect()
                exp = Exp_Long_Term_Forecast(hp_args)
                return exp.train(setting)
            
            try:
                model, all_epoch_metrics, best_metrics, best_model_path = inner_training_loop()
            except RuntimeError as e:
                args.accelerator.print(f"[RuntimeError] {e}")
                args.accelerator.print("Likely OOM even at batch size 1. Please reduce model size.")
                continue
            
            # Save detailed epoch metrics on main process
            if args.accelerator.is_main_process:
                epochs_metrics_file = os.path.join(hp_results_dir, f"epochs_metrics_{i+1}_predlen_{pred_len}.json")
                with open(epochs_metrics_file, 'w') as f:
                    json.dump(all_epoch_metrics, f, indent=4)
            
            # Record best results for current pred_len
            pred_len_result = {
                "pred_len": pred_len,
                "best_epoch": best_metrics["epoch"],
                "train_loss": best_metrics["train_loss"],
                "vali_loss": best_metrics["vali_loss"],
                "vali_mae_loss": best_metrics["vali_mae_loss"],
                "test_loss": best_metrics["test_loss"],
                "test_mae_loss": best_metrics["test_mae_loss"]
            }
            pred_len_results.append(pred_len_result)
            
            # Clear memory
            clear_gpu_memory(hp_args)
        
        # Calculate average metrics across all pred_len values
        avg_vali_loss = np.mean([r["vali_loss"] for r in pred_len_results])
        avg_test_loss = np.mean([r["test_loss"] for r in pred_len_results])
        avg_test_mae_loss = np.mean([r["test_mae_loss"] for r in pred_len_results])
        
        # Record results for this hyperparameter combination
        result = {
            "combination_id": i+1,
            "hyperparameters": hp_config_dict,
            "avg_vali_loss": float(avg_vali_loss),
            "avg_test_loss": float(avg_test_loss),
            "avg_test_mae_loss": float(avg_test_mae_loss),
            "pred_len_results": pred_len_results
        }
        
        # Save individual result on main process
        if args.accelerator.is_main_process:
            hp_result_file = os.path.join(hp_results_dir, f"result_{i+1}.json")
            with open(hp_result_file, 'w') as f:
                json.dump(result, f, indent=4)
        
        all_results.append(result)
        
        # Update summary file on main process
        if args.accelerator.is_main_process:
            with open(summary_file, 'w') as f:
                json.dump(all_results, f, indent=4)
        
        # Synchronize processes
        args.accelerator.wait_for_everyone()
    
    # Find best hyperparameter combination based on validation loss
    best_result = min(all_results, key=lambda x: x["avg_vali_loss"])
    
    # Save best result to separate file on main process
    if args.accelerator.is_main_process:
        best_result_file = os.path.join(hp_results_dir, "best_result.json")
        with open(best_result_file, 'w') as f:
            json.dump(best_result, f, indent=4)
    
    # Print best hyperparameters and results
    args.accelerator.print("\nHyperparameter search completed!")
    args.accelerator.print(f"Best hyperparameters: {best_result['hyperparameters']}")
    args.accelerator.print(f"Best average validation loss: {best_result['avg_vali_loss']}")
    args.accelerator.print(f"Corresponding average test loss: {best_result['avg_test_loss']}")
    args.accelerator.print(f"Corresponding average test MAE: {best_result['avg_test_mae_loss']}")
    
    # Print results for each pred_len
    args.accelerator.print("\nResults for each pred_len:")
    for result in best_result['pred_len_results']:
        args.accelerator.print(f"pred_len={result['pred_len']}: "
                             f"test_loss={result['test_loss']}, "
                             f"test_mae={result['test_mae_loss']}")
        
    if args.accelerator.is_main_process:
        args.accelerator.print(f"Results saved to: {hp_results_dir}")
        
        # Save results to CSV on main process
        save_results_to_csv(args, best_result, hp_results_dir)
        args.accelerator.print(f"CSV results saved by main process.")
    
    # Ensure all processes finish together
    args.accelerator.wait_for_everyone()
    clear_gpu_memory(args)


def run_training(args):
    """Run training mode with default hyperparameters."""
    for ii in range(args.itr):
        if args.seq_len is None:
            args.seq_len = args.pred_len * args.seq_len_factor

        # Create experiment object
        exp = Exp_Long_Term_Forecast(args)
        
        # Create experiment setting string
        setting = create_experiment_setting(args, ii)

        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = os.path.join(args.checkpoints, setting)
        if args.accelerator.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        args.accelerator.print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        
        # Train model
        model, all_epoch_metrics, best_metrics, best_model_path = exp.train(setting)

        # Print best model metrics
        args.accelerator.print('\n======== Best Model Metrics ========')
        args.accelerator.print(f"Best epoch: {best_metrics['epoch']}")
        args.accelerator.print(f"Best validation loss: {best_metrics['vali_loss']:.7f}")
        args.accelerator.print(f"Validation MAE: {best_metrics['vali_mae_loss']:.7f}")
        args.accelerator.print(f"Training loss: {best_metrics['train_loss']:.7f}")
        args.accelerator.print(f"Test loss: {best_metrics['test_loss']:.7f}")
        args.accelerator.print(f"Test MAE: {best_metrics['test_mae_loss']:.7f}")
        args.accelerator.print('====================================\n')
        
        # Save all epoch metrics for analysis if desired
        if args.accelerator.is_main_process:
            all_metrics_path = os.path.join(checkpoint_dir, 'all_epoch_metrics.json')
            with open(all_metrics_path, 'w') as f:
                json.dump(all_epoch_metrics, f, indent=4)
            args.accelerator.print(f"All epoch metrics saved to {all_metrics_path}")

        # Clear memory
        clear_gpu_memory(args)


def run_testing(args):
    """Run testing mode using trained parameters."""
    # Create experiment object
    exp = Exp_Long_Term_Forecast(args)
    
    # Create experiment setting string
    setting = create_experiment_setting(args)

    args.accelerator.print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    
    # Check if checkpoint exists before testing
    checkpoint_path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
    if not os.path.exists(checkpoint_path):
        args.accelerator.print(f"Warning: Checkpoint file not found at {checkpoint_path}")
        args.accelerator.print("Please make sure you have trained the model first or check the model_id and path.")
        return
    
    # Test model
    test_loss, test_mae_loss = exp.test(setting)
    
    # Print test results
    if test_loss is not None and test_mae_loss is not None:
        args.accelerator.print(f"Test Results - Loss: {test_loss:.7f}, MAE: {test_mae_loss:.7f}")
    
    # Clear memory
    clear_gpu_memory(args) 