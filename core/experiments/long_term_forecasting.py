"""
Long-term Forecasting Experiment

This module provides comprehensive experiment management for long-term time series 
forecasting tasks with support for training, validation, testing, and visualization.
"""

import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Any, Tuple
import logging
from datetime import datetime

from .base_experiment import BaseExperiment
from core.config import BaseConfig
from core.data import data_provider
from core.utils.tools import EarlyStopping, adjust_learning_rate, visual
from core.utils.metrics import metric


class LongTermForecastingExperiment(BaseExperiment):
    """
    Experiment class for long-term time series forecasting.
    
    Manages the complete experiment lifecycle including model training,
    validation, testing, and performance evaluation with early stopping
    and learning rate scheduling.
    """
    
    def __init__(self, config: BaseConfig):
        """
        Initialize the long-term forecasting experiment.
        
        Args:
            config: Configuration object containing experiment parameters
        """
        super().__init__(config)
        self.early_stopping = None
        self.optimizer = None
        self.criterion = None
        self.logger = None
        self._setup_training()
    
    def _setup_training(self):
        """Setup training components including optimizer, loss function, and early stopping."""
        # Initialize optimizer with model parameters
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        # Setup loss function (MSE for forecasting tasks)
        self.criterion = nn.MSELoss()
        
        # Configure early stopping mechanism
        self.early_stopping = EarlyStopping(patience=self.config.patience, verbose=True)
    
    def _setup_logging(self, setting: str):
        """Setup logging for training process."""
        # Create logs directory
        logs_dir = './logs'
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir, exist_ok=True)
        
        # Setup logger
        log_filename = os.path.join(logs_dir, f'{setting}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        # Create logger
        self.logger = logging.getLogger(f'training_{setting}')
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create file handler
        file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
        
        return log_filename
    
    def _get_data(self, flag: str) -> Tuple[Any, Any]:
        """
        Retrieve data for the specified split.
        
        Args:
            flag: Data split identifier ('train', 'val', 'test', 'pred')
            
        Returns:
            Tuple containing (dataset, dataloader) for the specified split
        """
        return data_provider(self.config, flag, self.accelerator)
    
    def train(self, setting: str) -> Tuple[nn.Module, list, dict, str]:
        """
        Execute the complete training procedure.
        
        Performs model training with validation monitoring, early stopping,
        and comprehensive metric tracking across all epochs.
        
        Args:
            setting: Unique experiment setting string for checkpoint management
            
        Returns:
            Tuple containing:
                - all_epoch_metrics: List of metrics for each training epoch
                - best_metrics: Dictionary of best validation metrics achieved
                - best_model_path: Path to the saved best model checkpoint
        """
        # Setup logging
        log_file_path = self._setup_logging(setting)
        self.logger.info(f"Start training: {setting}")
        self.logger.info(f"Training log file: {log_file_path}")
        
        # Load data splits
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        # Setup checkpoint directory
        path = os.path.join(self.config.checkpoints, setting)
        
        # Initialize performance tracking variables
        all_epoch_metrics = []
        best_metrics = {
            "epoch": 0,
            "train_loss": float('inf'),
            "vali_loss": float('inf'),
            "vali_mae_loss": float('inf'),
            "test_loss": float('inf'),
            "test_mae_loss": float('inf')
        }
        best_model_path = ""
        
        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.config.patience, verbose=True, accelerator=self.accelerator)
        
        # Prepare components for distributed training with accelerator
        self.model, self.optimizer, train_loader, vali_loader, test_loader = self.accelerator.prepare(
            self.model, self.optimizer, train_loader, vali_loader, test_loader
        )
        
        # Main training loop
        for epoch in range(self.config.train_epochs):
            iter_count = 0
            train_loss = []
            train_loss_mse = []  # MSE component
            train_loss_additional = []  # Additional loss component
            
            self.model.train()
            epoch_time = time.time()
            batch_times = []
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                batch_start_time = time.time()
                iter_count += 1
                self.optimizer.zero_grad()
                
                # Prepare inputs
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # Decoder input preparation
                dec_inp = torch.zeros_like(batch_y[:, -self.config.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.config.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # Forward pass with automatic mixed precision
                with self.accelerator.autocast():
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # Handle models that return additional loss components
                if isinstance(outputs, tuple):
                    outputs, additional_loss = outputs
                    additional_loss_value = additional_loss.item() if torch.is_tensor(additional_loss) else additional_loss
                else:
                    additional_loss = 0
                    additional_loss_value = 0
                
                # Calculate loss (only on prediction horizon)
                batch_y = batch_y[:, -self.config.pred_len:, :].to(self.device)
                mse_loss = self.criterion(outputs, batch_y)
                total_loss = mse_loss + additional_loss
                
                # Record loss components
                mse_loss_value = mse_loss.item()
                total_loss_value = total_loss.item()
                
                train_loss.append(total_loss_value)
                train_loss_mse.append(mse_loss_value)
                train_loss_additional.append(additional_loss_value)
                
                # Log progress every 100 iterations
                if (i + 1) % 100 == 0:
                    log_msg = f"\titers: {i+1}, epoch: {epoch+1} | MSE Loss: {mse_loss_value:.7f}, Additional Loss: {additional_loss_value:.7f}, Total Loss: {total_loss_value:.7f}"
                    self.accelerator.print(log_msg)
                    self.logger.info(log_msg)
                    
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.config.train_epochs - epoch) * train_steps - i)
                    speed_msg = f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s'
                    self.accelerator.print(speed_msg)
                    self.logger.info(speed_msg)
                    iter_count = 0
                    time_now = time.time()
                
                # Backward pass and optimization
                self.accelerator.backward(total_loss)
                self.optimizer.step()
                
                batch_end_time = time.time()
                batch_times.append(batch_end_time - batch_start_time)
            
            # Calculate epoch timing statistics
            epoch_cost_time = time.time() - epoch_time
            avg_batch_time = np.mean(batch_times)
            timing_msg = f"Epoch: {epoch+1} cost time: {epoch_cost_time:.2f}s"
            self.accelerator.print(timing_msg)
            self.logger.info(timing_msg)
            
            batch_time_msg = f"Average batch training time: {avg_batch_time:.4f}s"
            self.accelerator.print(batch_time_msg)
            self.logger.info(batch_time_msg)
            
            # Evaluate model performance
            train_loss_total = np.average(train_loss)
            train_loss_mse_avg = np.average(train_loss_mse)
            train_loss_additional_avg = np.average(train_loss_additional)
            
            val_time = time.time()
            vali_loss, vali_mae_loss = self.validate(vali_loader)
            val_cost_msg = f"Val cost time: {time.time() - val_time:.2f}s"
            self.accelerator.print(val_cost_msg)
            self.logger.info(val_cost_msg)
            
            test_time = time.time()
            test_loss, test_mae_loss = self.validate(test_loader)
            test_cost_msg = f"Test cost time: {time.time() - test_time:.2f}s"
            self.accelerator.print(test_cost_msg)
            self.logger.info(test_cost_msg)
            
            # Record comprehensive epoch metrics
            epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": float(train_loss_total),
                "train_loss_mse": float(train_loss_mse_avg),
                "train_loss_additional": float(train_loss_additional_avg),
                "vali_loss": float(vali_loss),
                "vali_mae_loss": float(vali_mae_loss),
                "test_loss": float(test_loss),
                "test_mae_loss": float(test_mae_loss)
            }
            all_epoch_metrics.append(epoch_metrics)
            
            # Enhanced train loss output with components
            epoch_summary = f'Epoch: {epoch+1}, Steps: {train_steps} | Train Loss Components: [MSE: {train_loss_mse_avg:.7f}, Additional: {train_loss_additional_avg:.7f}, Total: {train_loss_total:.7f}] | Vali Loss: {vali_loss:.7f} | Test Loss: {test_loss:.7f}'
            self.accelerator.print(epoch_summary)
            self.logger.info(epoch_summary)
            
            # Early stopping check (includes saving checkpoint)
            early_stopping(vali_loss, self.model, path, metrics=epoch_metrics)

            # Update best metrics if current model is better
            if vali_loss < best_metrics["vali_loss"]:
                best_metrics.update(epoch_metrics)
                best_model_path = early_stopping.get_checkpoint_path()

            # Stop if needed
            if early_stopping.early_stop:
                early_stop_msg = "Early stopping"
                self.accelerator.print(early_stop_msg)
                self.logger.info(early_stop_msg)
                break

            # Adjust learning rate according to schedule
            adjust_learning_rate(self.optimizer, epoch + 1, self.config, self.accelerator)
        
        self.logger.info(f"Training completed. Best model path: {best_model_path}")
        return all_epoch_metrics, best_metrics, best_model_path
    
    def test(self, setting: str, best_model_path: Optional[str] = None, save_predictions: bool = False) -> Tuple[float, float]:
        """
        Evaluate the trained model on test data and optionally save predictions for visualization.

        Args:
            setting: Experiment identifier string, used for result saving and fallback checkpoint loading.
            best_model_path: Path to the model checkpoint. If None, defaults to ./checkpoints/{setting}.pth
            save_predictions: Whether to save input, predictions, and ground truth to files.

        Returns:
            Tuple of (MSE, MAE) on the test set.
        """
        # Determine checkpoint path
        if best_model_path is None:
            best_model_path = os.path.join(self.config.checkpoints, f"{setting}.pth")

        self.accelerator.print(f'Loading trained model {best_model_path} for testing')
        
        # Load model weights
        self.model = self.accelerator.unwrap_model(self.model)
        self.model.load_state_dict(torch.load(best_model_path, map_location='cpu'))

        # Get test data loader and prepare for distributed eval
        _, test_loader = self._get_data(flag='test')
        self.model, test_loader = self.accelerator.prepare(self.model, test_loader)

        if save_predictions:
            # Save detailed predictions for visualization
            mse, mae = self._test_with_saving(test_loader, setting)
        else:
            # Use original validate() to compute MSE and MAE
            mse, mae = self.validate(test_loader)

        self.accelerator.print(f'Test MSE: {mse:.6f}, Test MAE: {mae:.6f}')
        return mse, mae
    
    def _test_with_saving(self, test_loader, setting: str) -> Tuple[float, float]:
        """
        Test the model and save inputs, predictions, and ground truth for visualization.
        
        Args:
            test_loader: Test data loader
            setting: Experiment setting string for file naming
            
        Returns:
            Tuple of (MSE, MAE)
        """
        # Create results directory
        results_dir = os.path.join('./test_results', setting)
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize lists to store results
        all_inputs = []
        all_predictions = []
        all_ground_truth = []
        all_sample_indices = []  # 添加样本索引跟踪
        
        # Initialize distributed accumulators for metrics
        sum_sq_error = torch.tensor(0.0, device=self.device)
        sum_abs_error = torch.tensor(0.0, device=self.device)
        total_count = torch.tensor(0.0, device=self.device)
        
        self.model.eval()
        with torch.no_grad():
            sample_counter = 0  # 跟踪全局样本索引
            for batch_idx, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                # Move data to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Prepare decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.config.pred_len:, :])
                dec_inp = torch.cat(
                    [batch_y[:, :self.config.label_len, :], dec_inp], dim=1
                ).to(self.device)

                # Forward pass
                with self.accelerator.autocast():
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                # Slice true values
                true_slice = batch_y[:, -self.config.pred_len:, :]
                
                # Store results for saving (only on main process to avoid duplication)
                if self.accelerator.is_main_process:
                    # Convert to CPU numpy arrays
                    inputs_np = batch_x.detach().cpu().numpy()  # Shape: [batch_size, seq_len, features]
                    predictions_np = outputs.detach().cpu().numpy()  # Shape: [batch_size, pred_len, features]
                    ground_truth_np = true_slice.detach().cpu().numpy()  # Shape: [batch_size, pred_len, features]
                    
                    # 记录当前批次的样本索引
                    batch_size = inputs_np.shape[0]
                    batch_indices = np.arange(sample_counter, sample_counter + batch_size)
                    
                    all_inputs.append(inputs_np)
                    all_predictions.append(predictions_np)
                    all_ground_truth.append(ground_truth_np)
                    all_sample_indices.append(batch_indices)

                # 更新样本计数器
                sample_counter += batch_x.shape[0]

                # Compute batch errors for metrics
                error = outputs - true_slice
                sum_sq_error += error.pow(2).sum()
                sum_abs_error += error.abs().sum()
                total_count += torch.tensor(error.numel(), device=self.device)

        # Reduce metrics across all devices
        sum_sq_error = self.accelerator.reduce(sum_sq_error, reduction="sum")
        sum_abs_error = self.accelerator.reduce(sum_abs_error, reduction="sum")
        total_count = self.accelerator.reduce(total_count, reduction="sum")

        # Compute final metrics
        mse = sum_sq_error / total_count
        mae = sum_abs_error / total_count
        
        # Save results only on main process
        if self.accelerator.is_main_process and all_inputs:
            # Concatenate all batches
            inputs_all = np.concatenate(all_inputs, axis=0)  # [total_samples, seq_len, features]
            predictions_all = np.concatenate(all_predictions, axis=0)  # [total_samples, pred_len, features]
            ground_truth_all = np.concatenate(all_ground_truth, axis=0)  # [total_samples, pred_len, features]
            sample_indices_all = np.concatenate(all_sample_indices, axis=0)  # [total_samples]
            
            # Save to files
            np.save(os.path.join(results_dir, 'inputs.npy'), inputs_all)
            np.save(os.path.join(results_dir, 'predictions.npy'), predictions_all)
            np.save(os.path.join(results_dir, 'ground_truth.npy'), ground_truth_all)
            np.save(os.path.join(results_dir, 'sample_indices.npy'), sample_indices_all)
            
            # Save metadata
            metadata = {
                'model': self.config.model,
                'data': self.config.data,
                'seq_len': self.config.seq_len,
                'pred_len': self.config.pred_len,
                'features': self.config.enc_in,
                'total_samples': inputs_all.shape[0],
                'mse': mse.item(),
                'mae': mae.item(),
                'setting': setting
            }
            
            import json
            with open(os.path.join(results_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.accelerator.print(f'Test results saved to {results_dir}')
            self.accelerator.print(f'- inputs.npy: {inputs_all.shape}')
            self.accelerator.print(f'- predictions.npy: {predictions_all.shape}')
            self.accelerator.print(f'- ground_truth.npy: {ground_truth_all.shape}')
            self.accelerator.print(f'- sample_indices.npy: {sample_indices_all.shape}')
            self.accelerator.print(f'- metadata.json: experiment metadata')

        self.model.train()
        return mse.item(), mae.item()


    # def test(self, setting: str, best_model_path = None) -> Tuple[float, float]:
    #     """
    #     Evaluate the trained model on test data.

    #     Args:
    #         setting: Experiment identifier string, used for result saving and fallback checkpoint loading.
    #         best_model_path: Path to the model checkpoint. If None, will default to ./checkpoints/{setting}.pth

    #     Returns:
    #         Tuple of (MSE, MAE)
    #     """
    #     test_data, test_loader = self._get_data(flag='test')
        
    #     if best_model_path is None:
    #         best_model_path = os.path.join(self.config.checkpoints, f"{setting}.pth")

    #     self.accelerator.print(f'Loading trained model {best_model_path} for testing')
        
    #     # self.model = self._build_model()
    #     self.model = self.accelerator.unwrap_model(self.model)
    #     self.model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
    #     self.model, test_loader = self.accelerator.prepare(self.model, test_loader)
        
    #     preds = []
    #     trues = []
    #     # folder_path = './test_results/' + setting + '/'
    #     # if not os.path.exists(folder_path):
    #     #     os.makedirs(folder_path)
        
    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float().to(self.device)
    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)
                
    #             # Decoder input
    #             dec_inp = torch.zeros_like(batch_y[:, -self.config.pred_len:, :]).float()
    #             dec_inp = torch.cat([batch_y[:, :self.config.label_len, :], dec_inp], dim=1).float().to(self.device)
                
    #             # Encoder - decoder
    #             with self.accelerator.autocast():
    #                 outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
    #             # Handle different output formats
    #             if isinstance(outputs, tuple):
    #                 outputs = outputs[0]
                
    #             batch_y = batch_y[:, -self.config.pred_len:, :].to(self.device)
                
    #             # Gather for metrics in distributed training
    #             outputs, batch_y = self.accelerator.gather_for_metrics((outputs, batch_y))
                
    #             pred = outputs.detach().cpu().numpy()
    #             true = batch_y.detach().cpu().numpy()
                
    #             preds.append(pred)
    #             trues.append(true)
        
    #     preds = np.concatenate(preds, axis=0)
    #     trues = np.concatenate(trues, axis=0)
        
    #     self.accelerator.print('test shape:', preds.shape, trues.shape)
        
    #     # Result save
    #     # folder_path = './results/' + setting + '/'
    #     # if not os.path.exists(folder_path):
    #     #     os.makedirs(folder_path)
        
    #     mae, mse, rmse, mape, mspe = metric(preds, trues)
    #     self.accelerator.print('mse:{}, mae:{}'.format(mse, mae))
        
    #     f = open("result_long_term_forecast.txt", 'a')
    #     f.write(setting + "  \n")
    #     f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
    #     f.write('\n')
    #     f.write('\n')
    #     f.close()
        
    #     # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
    #     # np.save(folder_path + 'pred.npy', preds)
    #     # np.save(folder_path + 'true.npy', trues)
        
    #     return mse, mae
    
    def validate(self, vali_loader=None) -> Tuple[float, float]:
        """
        Validate the model using distributed metric aggregation to avoid GPU OOM.

        Args:
            vali_loader: Optional DataLoader for validation data. If None, it will be created internally.

        Returns:
            Tuple of (MSE, MAE)
        """
        if vali_loader is None:
            _, vali_loader = self._get_data(flag='val')

        # Initialize distributed accumulators
        sum_sq_error = torch.tensor(0.0, device=self.device)
        sum_abs_error = torch.tensor(0.0, device=self.device)
        total_count = torch.tensor(0.0, device=self.device)

        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                # Move data to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Prepare decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.config.pred_len:, :])
                dec_inp = torch.cat(
                    [batch_y[:, :self.config.label_len, :], dec_inp], dim=1
                ).to(self.device)

                # Forward pass
                with self.accelerator.autocast():
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                # Slice true values
                true_slice = batch_y[:, -self.config.pred_len:, :]

                # Compute batch errors
                error = outputs - true_slice
                sum_sq_error += error.pow(2).sum()
                sum_abs_error += error.abs().sum()
                total_count += torch.tensor(error.numel(), device=self.device)

        # Reduce metrics across all devices once
        sum_sq_error = self.accelerator.reduce(sum_sq_error, reduction="sum")
        sum_abs_error = self.accelerator.reduce(sum_abs_error, reduction="sum")
        total_count = self.accelerator.reduce(total_count, reduction="sum")

        # Compute final metrics
        mse = sum_sq_error / total_count
        mae = sum_abs_error / total_count

        self.model.train()
        return mse.item(), mae.item()