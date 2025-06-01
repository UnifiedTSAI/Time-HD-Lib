#!/usr/bin/env python
import os
import argparse
import yaml

def create_default_config(model_name, output_dir="config_hp"):
    """Create a default hyperparameter search configuration file for a model."""
    
    # Common hyperparameters for most models
    default_config = {
        # Model architecture
        "d_model": [256, 512],
        "n_heads": [4, 8],
        "e_layers": [2, 3],
        "d_layers": [1, 2],
        "d_ff": [1024, 2048],
        
        # Optimization
        "learning_rate": [0.0001, 0.0005, 0.001],
        "dropout": [0.1, 0.2],
        "batch_size": [32, 64]
    }
    
    # Model-specific hyperparameters
    model_specific = {
        "TimesNet": {
            "top_k": [3, 5],
            "num_kernels": [4, 6]
        },
        "Autoformer": {
            "moving_avg": [24, 25],
            "factor": [1, 3]
        },
        "Transformer": {
            "factor": [1, 5],
            "d_ff": [1024, 2048]
        },
        "Informer": {
            "factor": [3, 5],
            "prob": [0.5, 0.8]
        },
        "Reformer": {
            "factor": [3, 5],
            "n_hashes": [4, 8]
        },
        "FEDformer": {
            "version": ["Fourier", "Wavelets"],
            "modes": [32, 64]
        },
        "DLinear": {
            "individual": [False, True]
        },
        "NLinear": {
            "individual": [False, True]
        },
        "PatchTST": {
            "patch_len": [16, 32],
            "stride": [8, 16]
        }
    }
    
    # Combine common and model-specific hyperparameters
    config = default_config.copy()
    if model_name in model_specific:
        config.update(model_specific[model_name])
    
    # Add comments to the configuration
    config_str = f"# Hyperparameter search configuration for {model_name}\n"
    config_str += "# Each parameter should be a list of values to search\n\n"
    
    # Model architecture section
    config_str += "# Model architecture\n"
    for k in ["d_model", "n_heads", "e_layers", "d_layers", "d_ff"]:
        if k in config:
            config_str += f"{k}: {config[k]}\n"
    config_str += "\n"
    
    # Optimization section
    config_str += "# Optimization\n"
    for k in ["learning_rate", "dropout", "batch_size"]:
        if k in config:
            config_str += f"{k}: {config[k]}\n"
    config_str += "\n"
    
    # Model specific section
    config_str += "# Model specific\n"
    for k, v in config.items():
        if k not in ["d_model", "n_heads", "e_layers", "d_layers", "d_ff", "learning_rate", "dropout", "batch_size"]:
            config_str += f"{k}: {v}\n"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Write configuration to file
    output_path = os.path.join(output_dir, f"{model_name}.yaml")
    with open(output_path, "w") as f:
        f.write(config_str)
    
    print(f"Created hyperparameter configuration file for {model_name} at {output_path}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create hyperparameter search configuration file')
    parser.add_argument('model_name', type=str, help='Name of the model')
    parser.add_argument('--output_dir', type=str, default="config_hp", 
                        help='Directory to save the configuration file')
    
    args = parser.parse_args()
    create_default_config(args.model_name, args.output_dir) 