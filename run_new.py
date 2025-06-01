#!/usr/bin/env python3
"""
New Run Script

This script demonstrates the new decoupled architecture of Time-HD-Lib.
"""

import random
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from core.cli import create_argument_parser
from core.config import config_manager
from core.models import model_manager


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    """Main function using the new decoupled architecture."""
    
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Set seed
    set_seed(getattr(args, 'seed', 2021))
    
    # Setup accelerator
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    
    # Add accelerator to args
    args.accelerator = accelerator
    
    # Set device
    if torch.cuda.is_available() and getattr(args, 'use_gpu', True):
        args.device = accelerator.device
        accelerator.print('Using GPU')
    
    # Create unified configuration using the new config manager
    config = config_manager.parse_args_and_create_config(args)
    
    # Print configuration info
    accelerator.print(f"\n=== Configuration Summary ===")
    accelerator.print(f"Model: {config.model}")
    accelerator.print(f"Dataset: {config.data}")
    accelerator.print(f"Sequence Length: {config.seq_len}")
    accelerator.print(f"Prediction Length: {config.pred_len}")
    accelerator.print(f"Batch Size: {config.batch_size}")
    accelerator.print("=============================\n")
    
    # List available models
    available_models = model_manager.list_available_models()
    accelerator.print(f"Available models in new registry: {available_models}")
    
    # Check if the requested model is available in the new system
    if model_manager.is_model_available(config.model):
        accelerator.print(f"✓ Model '{config.model}' found in new registry!")
        
        # Get model metadata
        metadata = model_manager.get_model_metadata(config.model)
        if metadata:
            accelerator.print(f"Model metadata: {metadata}")
        
        # Create model using new system
        model = model_manager.create_model(config.model, config)
        accelerator.print(f"✓ Model created successfully using new architecture!")
        accelerator.print(f"Model type: {type(model)}")
        
    else:
        accelerator.print(f"⚠ Model '{config.model}' not yet migrated to new registry.")
        accelerator.print("Will use legacy system as fallback.")
    
    # For now, we'll just demonstrate the new architecture
    # In a complete implementation, we would proceed with training/testing
    accelerator.print("\n🎉 New architecture demonstration completed successfully!")
    accelerator.print("Next steps: Implement full training/testing pipeline with new architecture.")


if __name__ == '__main__':
    main() 