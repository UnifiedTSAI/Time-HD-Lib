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
        self._setup_training()
    
    def _setup_training(self):
        """Setup training components including optimizer, loss function, and early stopping."""
        # Initialize optimizer with model parameters
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        # Setup loss function (MSE for forecasting tasks)
        self.criterion = nn.MSELoss()
        
        # Configure early stopping mechanism
        self.early_stopping = EarlyStopping(patience=self.config.patience, verbose=True)
    
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
            
            self.model.train()
            epoch_time = time.time()
            batch_times = []  # Track training time per batch
            
            # Batch training loop
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                batch_start_time = time.time()  # Record batch start time
                
                iter_count += 1
                self.optimizer.zero_grad()
                
                # Move data to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # Prepare decoder input (for sequence-to-sequence models)
                dec_inp = torch.zeros_like(batch_y[:, -self.config.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.config.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # Forward pass with automatic mixed precision
                with self.accelerator.autocast():
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # Handle models that return additional loss components
                if isinstance(outputs, tuple):
                    outputs, additional_loss = outputs
                else:
                    additional_loss = 0
                
                # Calculate loss (only on prediction horizon)
                batch_y = batch_y[:, -self.config.pred_len:, :].to(self.device)
                loss = self.criterion(outputs, batch_y) + additional_loss
                
                train_loss.append(loss.item())
                
                # Log progress every 100 iterations
                if (i + 1) % 100 == 0:
                    self.accelerator.print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.config.train_epochs - epoch) * train_steps - i)
                    self.accelerator.print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()
                
                # Backward pass and optimization
                self.accelerator.backward(loss)
                self.optimizer.step()
                
                batch_end_time = time.time()
                batch_times.append(batch_end_time - batch_start_time)  # Record batch training time
            
            # Calculate epoch timing statistics
            epoch_cost_time = time.time() - epoch_time
            avg_batch_time = np.mean(batch_times)
            self.accelerator.print(f"Epoch: {epoch+1} cost time: {epoch_cost_time:.2f}s")
            self.accelerator.print(f"Average batch training time: {avg_batch_time:.4f}s")
            
            # Evaluate model performance
            train_loss = np.average(train_loss)
            val_time = time.time()
            vali_loss, vali_mae_loss = self.validate(vali_loader)
            self.accelerator.print(f"Val cost time: {time.time() - val_time:.2f}s")
            test_time = time.time()
            test_loss, test_mae_loss = self.validate(test_loader)
            self.accelerator.print(f"Test cost time: {time.time() - test_time:.2f}s")
            
            # Record comprehensive epoch metrics
            epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "vali_loss": float(vali_loss),
                "vali_mae_loss": float(vali_mae_loss),
                "test_loss": float(test_loss),
                "test_mae_loss": float(test_mae_loss)
            }
            all_epoch_metrics.append(epoch_metrics)
            
            self.accelerator.print(f'Epoch: {epoch+1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}')
            
            # Early stopping check (includes saving checkpoint)
            early_stopping(vali_loss, self.model, path, metrics=epoch_metrics)

            # Update best metrics if current model is better
            if vali_loss < best_metrics["vali_loss"]:
                best_metrics.update(epoch_metrics)
                best_model_path = early_stopping.get_checkpoint_path()

            # Stop if needed
            if early_stopping.early_stop:
                self.accelerator.print("Early stopping")
                break

            # Adjust learning rate according to schedule
            adjust_learning_rate(self.optimizer, epoch + 1, self.config, self.accelerator)
        
        return all_epoch_metrics, best_metrics, best_model_path
    
    def test(self, setting: str, best_model_path: Optional[str] = None) -> Tuple[float, float]:
        """
        Evaluate the trained model on test data using the same aggregation logic as in validate().

        Args:
            setting: Experiment identifier string, used for result saving and fallback checkpoint loading.
            best_model_path: Path to the model checkpoint. If None, defaults to ./checkpoints/{setting}.pth

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

        # Use validate() to compute MSE and MAE
        mse, mae = self.validate(test_loader)

        self.accelerator.print(f'Test MSE: {mse:.6f}, Test MAE: {mae:.6f}')
        return mse, mae


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
        Validate the model.
        
        Args:
            vali_loader: Validation data loader (if None, will get validation data)
            
        Returns:
            Tuple of (mse, mae)
        """
        if vali_loader is None:
            vali_data, vali_loader = self._get_data(flag='val')
        
        total_loss = []
        mae_loss = []
        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.config.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.config.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                with self.accelerator.autocast():
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # Handle different output formats
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                batch_y = batch_y[:, -self.config.pred_len:, :].to(self.device)
                
                # Gather for metrics in distributed training
                outputs, batch_y = self.accelerator.gather_for_metrics((outputs, batch_y))
                
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                
                # loss = self.criterion(pred, true)
                # mae_loss.append(nn.L1Loss()(pred, true).item())
                # total_loss.append(loss.item())
                preds.append(pred)
                trues.append(true)
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        # total_loss = np.average(total_loss)
        # mae_loss = np.average(mae_loss)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        
        self.model.train()
        return mse, mae