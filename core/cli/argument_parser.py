"""
Command Line Interface - Argument Parser

This module provides comprehensive command line argument parsing functionality
for the High-Dimensional Time Series Analysis Framework.
"""

import argparse
from typing import Dict, Any


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the command line argument parser.
    
    Returns:
        Configured ArgumentParser instance with all framework options
    """
    parser = argparse.ArgumentParser(
        description='High-Dimensional Time Series Analysis Framework', 
        argument_default=argparse.SUPPRESS,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Basic configuration
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='Task type: [long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, 
                        help='Training mode: 1 for training, 0 for testing only')
    parser.add_argument('--model', type=str, required=True, default='UCast',
                        help='Model name (see --list-models for available options)')

    # Data loading configuration
    parser.add_argument('--data', type=str, required=True, default='ETTh1', 
                        help='Dataset name')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', 
                        help='Root directory path for dataset files')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', 
                        help='Specific data file name')
    parser.add_argument('--features', type=str, default='M',
                        help='Forecasting mode: [M]ultivariate, [S]ingle variable, [MS]ultivariate to single')
    parser.add_argument('--target', type=str, default='OT', 
                        help='Target feature name for S or MS tasks')
    parser.add_argument('--freq', type=str, default='h',
                        help='Time frequency: [s]econdly, [t]minutely, [h]ourly, [d]aily, [b]usiness days, [w]eekly, [m]onthly')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', 
                        help='Directory for saving model checkpoints')

    # Forecasting task parameters
    parser.add_argument('--seq_len', type=int, default=None, 
                        help='Input sequence length (lookback window)')
    parser.add_argument('--label_len', type=int, default=0, 
                        help='Start token length for decoder models')
    parser.add_argument('--pred_len', type=int, default=96, 
                        help='Prediction sequence length (forecast horizon)')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', 
                        help='Seasonal patterns for M4 dataset')
    parser.add_argument('--inverse', action='store_true', default=False,
                        help='Apply inverse transformation to denormalize outputs')
    parser.add_argument('--seq_len_factor', type=int, default=2, 
                        help='Multiplier for automatic sequence length calculation')

    # Model architecture parameters
    parser.add_argument('--expand', type=int, default=2, 
                        help='Expansion factor for Mamba-based models')
    parser.add_argument('--d_conv', type=int, default=4, 
                        help='Convolution kernel size for Mamba models')
    parser.add_argument('--top_k', type=int, default=5, 
                        help='Top-k parameter for TimesNet model')
    parser.add_argument('--num_kernels', type=int, default=6, 
                        help='Number of kernels for Inception blocks')
    parser.add_argument('--enc_in', type=int, default=7, 
                        help='Number of input channels/features')
    parser.add_argument('--dec_in', type=int, default=7, 
                        help='Number of decoder input channels')
    parser.add_argument('--c_out', type=int, default=7, 
                        help='Number of output channels/features')
    parser.add_argument('--d_model', type=int, default=512, 
                        help='Model embedding dimension')
    parser.add_argument('--n_heads', type=int, default=8, 
                        help='Number of attention heads')
    parser.add_argument('--e_layers', type=int, default=2, 
                        help='Number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, 
                        help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, 
                        help='Feedforward network dimension')
    parser.add_argument('--moving_avg', type=int, default=25, 
                        help='Moving average window size for decomposition')
    parser.add_argument('--factor', type=int, default=1, 
                        help='Attention factor for ProbSparse attention')
    parser.add_argument('--distil', action='store_false', default=True,
                        help='Disable distilling in encoder (use flag to disable)')
    parser.add_argument('--dropout', type=float, default=0.1, 
                        help='Dropout rate for regularization')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='Time features encoding: [timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', 
                        help='Activation function')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='Channel processing: 0=dependent, 1=independent (for FreTS)')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='Series decomposition method: [moving_avg, dft_decomp]')
    parser.add_argument('--use_norm', type=int, default=1, 
                        help='Apply normalization: 1=True, 0=False')
    parser.add_argument('--alpha', type=float, default=0,
                        help='Alpha parameter for specific models')
    parser.add_argument('--channel_reduction_ratio', type=float, default=16,
                        help='Channel reduction ratio for attention mechanisms')

    # Training optimization parameters
    parser.add_argument('--num_workers', type=int, default=2, 
                        help='Number of data loader worker processes')
    parser.add_argument('--itr', type=int, default=1, 
                        help='Number of experiment iterations')
    parser.add_argument('--train_epochs', type=int, default=10, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Training batch size')
    parser.add_argument('--patience', type=int, default=3, 
                        help='Early stopping patience (epochs)')
    parser.add_argument('--learning_rate', type=float, default=0.0001, 
                        help='Optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', 
                        help='Experiment description')
    parser.add_argument('--loss', type=str, default='MSE', 
                        help='Loss function')
    parser.add_argument('--lradj', type=str, default='type1', 
                        help='Learning rate adjustment strategy')
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='Enable automatic mixed precision training')

    # GPU and distributed training
    parser.add_argument('--use_gpu', type=bool, default=True, 
                        help='Enable GPU acceleration')
    parser.add_argument('--gpu', type=int, default=0, 
                        help='GPU device ID')
    parser.add_argument('--gpu_type', type=str, default='cuda', 
                        help='GPU type: [cuda, mps]')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False,
                        help='Enable multi-GPU training')
    parser.add_argument('--devices', type=str, default='0,1,2,3', 
                        help='Comma-separated GPU device IDs for multi-GPU')

    # Hyperparameter search
    parser.add_argument('--hyper_parameter_searching', action='store_true', default=False,
                        help='Enable automated hyperparameter search')
    parser.add_argument('--hp_log_dir', type=str, default='./hp_logs/', 
                        help='Directory for hyperparameter search logs')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=2, 
                        help='Random seed for reproducibility')

    return parser 