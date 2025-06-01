from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
from tqdm import tqdm
from torchinfo import summary
import io
import contextlib
warnings.filterwarnings('ignore')

def count_parameters(accelerator,model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        total_params += params
    accelerator.print(f"Total Trainable Params: {total_params}")
    return total_params

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            summary(model)
            model_summary_str = buf.getvalue()
        self.args.accelerator.print(model_summary_str)
        count_parameters(self.args.accelerator,model)

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # criterion = nn.MSELoss()
        # return criterion
        if self.args.data == 'PEMS':
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, test=False):
        total_loss = []
        total_mae_loss = []
        self.model.eval()
        batch_times = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_time = time.time()
                batch_x = batch_x.float().to(self.accelerator.device)
                batch_y = batch_y.float().to(self.accelerator.device)

                batch_x_mark = batch_x_mark.float().to(self.accelerator.device)
                batch_y_mark = batch_y_mark.float().to(self.accelerator.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.accelerator.device)
                # encoder-decoder
                with self.accelerator.autocast():
                    outputs, loss_aux = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs, batch_y = self.accelerator.gather_for_metrics((outputs, batch_y))

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)
                total_loss.append(loss.item())

                mae_loss = nn.L1Loss()(pred, true)
                total_mae_loss.append(mae_loss.item())
                
                batch_times.append(time.time() - batch_time)
        if test:
            self.accelerator.print("Avg batch test cost time: {}".format(sum(batch_times) / len(batch_times)))
        total_loss = np.average(total_loss)
        total_mae_loss = np.average(total_mae_loss)

        self.model.train()
        return total_loss, total_mae_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and self.args.accelerator.is_main_process:
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, accelerator=self.accelerator)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # Prepare model, optimizer, and dataloaders for distributed training
        self.model, model_optim, train_loader, vali_loader, test_loader = self.accelerator.prepare(
            self.model, model_optim, train_loader, vali_loader, test_loader)
        
        # Track all metrics during training
        all_epoch_metrics = []
        best_model_path = os.path.join(path, 'checkpoint.pth')

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            batch_times = []
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                batch_time = time.time()
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float()
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float()
                batch_y_mark = batch_y_mark.float()

                if 'PEMS' == self.args.data or 'Solar' == self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()

                # encoder-decoder forward pass
                with self.accelerator.autocast():
                    outputs, loss_aux = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                    loss = criterion(outputs, batch_y) + loss_aux
                    train_loss.append(loss.item())


                if (i + 1) % 100 == 0:
                    self.accelerator.print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    self.accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Use Accelerator's backward for proper gradient handling
                self.accelerator.backward(loss)
                model_optim.step()

                batch_times.append(time.time() - batch_time)

            self.accelerator.print("Epoch: {} train cost time: {}".format(epoch + 1, time.time() - epoch_time))
            self.accelerator.print("Avg batch train cost time: {}".format(sum(batch_times) / len(batch_times)))

            max_allocated = torch.cuda.max_memory_allocated(device=self.accelerator.device) / 1024**2
            self.accelerator.print(f"Max GPU memory allocated this epoch: {max_allocated:.2f} MB")
            torch.cuda.reset_max_memory_allocated(device=self.accelerator.device)

            train_loss = np.average(train_loss)
            vali_loss, vali_mae_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_mae_loss = self.vali(test_data, test_loader, criterion, test=True)

            self.accelerator.print(
            "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} Test MAE: {4:.7f}".format(
                epoch + 1, train_loss, vali_loss, test_loss, test_mae_loss))
            
            # Store metrics for this epoch
            epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "vali_loss": float(vali_loss),
                "vali_mae_loss": float(vali_mae_loss),
                "test_loss": float(test_loss),
                "test_mae_loss": float(test_mae_loss)
            }
            all_epoch_metrics.append(epoch_metrics)
            
            # Use EarlyStopping while tracking best validation metrics
            early_stopping(vali_loss, self.model, path, metrics=epoch_metrics)
            if early_stopping.early_stop:
                self.accelerator.print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args, accelerator=self.accelerator)

        # Get best metrics from EarlyStopping
        best_metrics = early_stopping.get_best_metrics()
        
        # Return the best model metrics along with the path to the best model
        return self.model, all_epoch_metrics, best_metrics, best_model_path

    def test(self, setting):
        """Test the model on the test dataset."""
        # Load test data
        test_data, test_loader = self._get_data(flag='test')
        
        # Load the best model from checkpoint
        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = os.path.join(path, 'checkpoint.pth')
        
        self.accelerator.print(f'Loading model from {best_model_path}')
        
        try:
            # Load model checkpoint
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Prepare model and test loader for distributed testing
            self.model, test_loader = self.accelerator.prepare(self.model, test_loader)
            
            self.accelerator.print('Model loaded successfully')
        except FileNotFoundError:
            self.accelerator.print(f'Checkpoint not found at {best_model_path}')
            return
        except Exception as e:
            self.accelerator.print(f'Error loading model: {e}')
            return
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Set criterion
        criterion = self._select_criterion()
        
        # Evaluate on test data
        test_loss, test_mae_loss = self.vali(test_data, test_loader, criterion, test=True)
        
        self.accelerator.print('>>>>>>>testing results : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        self.accelerator.print('Test Loss: {0:.7f} Test MAE: {1:.7f}'.format(test_loss, test_mae_loss))
        
        # Calculate additional metrics if needed
        if hasattr(self.args, 'use_dtw') and self.args.use_dtw:
            self.accelerator.print('Calculating DTW metrics...')
            # Implementation for DTW metrics would go here
            
        return test_loss, test_mae_loss
