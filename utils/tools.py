import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import csv

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args, accelerator=None):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # print('Updating learning rate to {}'.format(lr))
        accelerator.print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, accelerator=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.accelerator = accelerator
        self.best_metrics = None  # 存储最佳验证指标

    def __call__(self, val_loss, model, path, metrics=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, metrics)
            if metrics is not None:
                self.best_metrics = metrics
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, metrics)
            if metrics is not None:
                self.best_metrics = metrics
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, metrics=None):
        if self.verbose:
            self.accelerator.print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(model)
            torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        else:
            torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss
        
        # 如果有指标信息，保存到JSON文件
        if metrics is not None:
            metrics_file = os.path.join(path, 'best_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)

    def get_best_metrics(self):
        """返回最佳验证指标"""
        return self.best_metrics


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.plot(true, label='GroundTruth', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def save_results_to_csv(args, best_result, hp_results_dir=None):
    """
    将模型的超参数搜索结果保存到CSV文件中
    
    参数:
        args: 参数对象，包含数据集和模型信息
        best_result: 最佳超参数组合的结果，包含不同pred_len的详细指标
        hp_results_dir: 超参数搜索结果保存目录，用于打印信息
    
    返回:
        csv_file_path: 保存的CSV文件路径
    """
    # 将结果追加保存到results.csv文件中
    csv_file_path = os.path.join(args.hp_log_dir, 'results.csv')
    
    # 准备要写入的数据
    model_method = f"{args.model}"  # 模型名称
    current_dataset = args.data      # 当前数据集名称
    
    # 检查CSV文件是否存在
    csv_exists = os.path.exists(csv_file_path)
    
    # 如果CSV不存在，创建并写入表头
    if not csv_exists:
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['dataset', 'pred_len', f'{model_method}_MSE', f'{model_method}_MAE'])
        
        # 创建新文件后直接添加所有数据
        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for result in best_result['pred_len_results']:
                pred_len = result['pred_len']
                test_loss = result['test_loss']
                test_mae = result['test_mae_loss']
                writer.writerow([current_dataset, pred_len, test_loss, test_mae])
                
        # 打印信息
        if hp_results_dir:
            args.accelerator.print(f"Results created in: {csv_file_path}")
        
        return csv_file_path
    
    # 如果CSV文件存在，读取数据并追加/更新
    rows_to_write = []
    existing_rows = []
    
    # 读取现有数据
    try:
        with open(csv_file_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            existing_rows = list(reader)
            
            # 明确处理文件存在但为空的情况
            if len(existing_rows) == 0:
                if hp_results_dir:
                    args.accelerator.print(f"CSV file exists but is empty: {csv_file_path}")
                # 初始化为只有表头的文件
                existing_rows = [['dataset', 'pred_len', f'{model_method}_MSE', f'{model_method}_MAE']]
                # 重写文件添加表头
                with open(csv_file_path, 'w', newline='') as empty_file:
                    writer = csv.writer(empty_file)
                    writer.writerow(existing_rows[0])
                    
    except Exception as e:
        if hp_results_dir:
            args.accelerator.print(f"Error reading CSV file: {e}, creating new file")
        # 如果读取失败，创建新文件
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['dataset', 'pred_len', f'{model_method}_MSE', f'{model_method}_MAE'])
        existing_rows = [['dataset', 'pred_len', f'{model_method}_MSE', f'{model_method}_MAE']]
    
    # 以防万一，确保有表头 (这是一个额外的安全检查)
    if not existing_rows:
        existing_rows = [['dataset', 'pred_len', f'{model_method}_MSE', f'{model_method}_MAE']]
    
    header = existing_rows[0]
    
    # 检查是否需要添加新列
    mse_col = f'{model_method}_MSE'
    mae_col = f'{model_method}_MAE'
    
    if mse_col not in header:
        header.append(mse_col)
    if mae_col not in header:
        header.append(mae_col)
    
    # 获取列索引
    mse_index = header.index(mse_col)
    mae_index = header.index(mae_col)
    
    # 将所有现有行添加到写入列表
    rows_to_write.append(header)
    
    # 添加除表头外的所有现有行
    for row in existing_rows[1:]:
        # 扩展行以匹配表头长度
        while len(row) < len(header):
            row.append('')
        rows_to_write.append(row)
    
    # 为每个pred_len添加新行
    for result in best_result['pred_len_results']:
        pred_len = result['pred_len']
        test_loss = result['test_loss']
        test_mae = result['test_mae_loss']
        
        # 标记是否找到匹配的行
        row_found = False
        
        # 检查是否已存在相同数据集和pred_len的行
        for i in range(1, len(rows_to_write)):
            row = rows_to_write[i]
            if len(row) >= 2 and row[0] == current_dataset:
                try:
                    row_pred_len = float(row[1])
                    if int(row_pred_len) == int(pred_len):
                        # 更新这一行
                        row[mse_index] = str(test_loss)
                        row[mae_index] = str(test_mae)
                        rows_to_write[i] = row
                        row_found = True
                        break
                except (ValueError, IndexError):
                    continue
        
        # 如果没有找到匹配的行，添加新行
        if not row_found:
            new_row = [''] * len(header)
            new_row[0] = current_dataset
            new_row[1] = str(pred_len)
            new_row[mse_index] = str(test_loss)
            new_row[mae_index] = str(test_mae)
            rows_to_write.append(new_row)
    
    # 写入所有行到CSV文件
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows_to_write)
    
    # 打印信息
    if hp_results_dir:
        args.accelerator.print(f"Results appended to: {csv_file_path}")
    
    return csv_file_path