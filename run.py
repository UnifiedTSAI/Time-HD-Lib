import argparse
import os
import torch
import torch.backends
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args
import random
import numpy as np
import yaml
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from accelerate.utils import find_executable_batch_size
import itertools
import json
import datetime
import copy
from utils.tools import save_results_to_csv

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # parser = argparse.ArgumentParser(description='TimesNet')
    parser = argparse.ArgumentParser(description='TimesNet', argument_default=argparse.SUPPRESS)

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--seq_len_factor', type=int, default=2, help='input sequence length')

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='num of channels')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--channel_reduction_ratio', type=float, default=16)

    # optimization
    parser.add_argument('--num_workers', type=int, default=2, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # Hyperparameter search
    parser.add_argument('--hyper_parameter_searching', action='store_true', help='enable hyperparameter search', default=False)
    parser.add_argument('--hp_log_dir', type=str, default='./hp_logs/', help='directory to save hyperparameter search logs')

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # args = parser.parse_args()
    partial_args, _ = parser.parse_known_args()
    model_name = partial_args.model
    dataset_name = partial_args.data
    yaml_config = {}
    yaml_path = f'configs/{model_name}.yaml'
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        if dataset_name in config:
            yaml_config = config[dataset_name]
        else:
           print(f"[Warning] No config found for dataset {dataset_name} in {yaml_path}")
    else:
        print(f"[Warning] Config file not found: {yaml_path}")
    parser.set_defaults(**yaml_config)
    args = parser.parse_args()

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    args.accelerator = Accelerator(kwargs_handlers=[kwargs])
    
    if torch.cuda.is_available() and args.use_gpu:
        args.device = args.accelerator.device
        args.accelerator.print('Using GPU')

    # args.accelerator.print('Args in experiment:')
    # print_args(args)

    Exp = Exp_Long_Term_Forecast

    if args.hyper_parameter_searching:
        # Create hp_logs directory if it doesn't exist
        os.makedirs(args.hp_log_dir, exist_ok=True)
        
        # Load hyperparameter search configuration
        hp_config_path = f'config_hp/{args.model}.yaml'
        if not os.path.exists(hp_config_path):
            args.accelerator.print(f"Hyperparameter config file not found: {hp_config_path}")
            args.accelerator.print("Please create a hyperparameter config file first.")
            exit(1)
            
        # 加载pred_len配置
        pred_len_config_path = 'config_hp/pred_len_config.yaml'
        if os.path.exists(pred_len_config_path):
            with open(pred_len_config_path, 'r') as f:
                pred_len_config = yaml.safe_load(f)
                
            # 获取当前数据集的pred_len列表
            if args.data in pred_len_config:
                pred_len_list = pred_len_config[args.data]
                args.accelerator.print(f"Using pred_len list for {args.data}: {pred_len_list}")
            else:
                # 使用默认的pred_len列表
                pred_len_list = [48]
                args.accelerator.print(f"No pred_len config found for {args.data}, using default: {pred_len_list}")
        else:
            args.accelerator.print(f"pred_len config file not found: {pred_len_config_path}")
            args.accelerator.print("Using default pred_len list: [24, 48, 96, 192]")
            pred_len_list = [48]
            
        with open(hp_config_path, 'r') as f:
            hp_config = yaml.safe_load(f)
            
        if not hp_config:
            args.accelerator.print(f"Empty hyperparameter config file: {hp_config_path}")
            exit(1)
            
        # Build hyperparameter grid
        hp_grid = {}
        for param_name, param_values in hp_config.items():
            if isinstance(param_values, list):
                hp_grid[param_name] = param_values
                
        if not hp_grid:
            args.accelerator.print("No hyperparameters to search found in config file")
            exit(1)
            
        # Generate all combinations of hyperparameters
        hp_keys = list(hp_grid.keys())
        hp_values = list(hp_grid.values())
        hp_combinations = list(itertools.product(*hp_values))
        
        args.accelerator.print(f"Starting hyperparameter search with {len(hp_combinations)} combinations, each on {len(pred_len_list)} pred_len values")
        
        # Timestamp for logging
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        hp_results_dir = os.path.join(args.hp_log_dir, f"{args.model}_{args.data}_{timestamp}")
        
        # 只在主进程创建目录和文件
        if args.accelerator.is_main_process:
            os.makedirs(hp_results_dir, exist_ok=True)
            # Create a summary file to store all results
            summary_file = os.path.join(hp_results_dir, "hp_summary.json")
        
        # 确保所有进程等待目录创建完成
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
            
            # 存储不同pred_len的结果
            pred_len_results = []
            
            # 对每个pred_len进行训练和评估
            for pred_len in pred_len_list:
                args.accelerator.print(f"\nTraining with pred_len = {pred_len}")
                
                # 设置当前pred_len
                hp_args.pred_len = pred_len
                hp_args.seq_len = pred_len * hp_args.seq_len_factor
                args.accelerator.print(f"\nTraining with seq_len = {hp_args.seq_len}")
                
                # Create experiment setting string with specific pred_len
                setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                    hp_args.task_name,
                    hp_args.model_id,
                    hp_args.model,
                    hp_args.data,
                    hp_args.features,
                    hp_args.seq_len,
                    hp_args.label_len,
                    hp_args.pred_len,
                    hp_args.d_model,
                    hp_args.n_heads,
                    hp_args.e_layers,
                    hp_args.d_layers,
                    hp_args.d_ff,
                    hp_args.expand,
                    hp_args.d_conv,
                    hp_args.factor,
                    hp_args.embed,
                    hp_args.distil,
                    hp_args.des, 0)
                
                # 训练模型
                # model, all_epoch_metrics, best_metrics, best_model_path = exp.train(setting)
                @find_executable_batch_size(starting_batch_size=hp_args.batch_size)
                def inner_training_loop(batch_size):
                    hp_args.batch_size = batch_size
                    hp_args.accelerator.print(f"Trying batch size: {batch_size}")
                    torch.cuda.empty_cache()
                    import gc; gc.collect()
                    exp = Exp(hp_args)
                    return exp.train(setting)
                try:
                    model, all_epoch_metrics, best_metrics, best_model_path = inner_training_loop()
                except RuntimeError as e:
                    args.accelerator.print(f"[RuntimeError] {e}")
                    args.accelerator.print("Likely OOM even at batch size 1. Please reduce model size.")
                    continue
                
                # 只在主进程保存当前pred_len的详细epoch指标
                if args.accelerator.is_main_process:
                    epochs_metrics_file = os.path.join(hp_results_dir, f"epochs_metrics_{i+1}_predlen_{pred_len}.json")
                    with open(epochs_metrics_file, 'w') as f:
                        json.dump(all_epoch_metrics, f, indent=4)
                
                # 记录当前pred_len的最佳结果
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
                
                # 清理内存
                if hp_args.gpu_type == 'mps':
                    torch.backends.mps.empty_cache()
                elif hp_args.gpu_type == 'cuda':
                    torch.cuda.empty_cache()
            
            # 计算所有pred_len的平均指标
            avg_vali_loss = np.mean([r["vali_loss"] for r in pred_len_results])
            avg_test_loss = np.mean([r["test_loss"] for r in pred_len_results])
            avg_test_mae_loss = np.mean([r["test_mae_loss"] for r in pred_len_results])
            
            # 记录所有pred_len结果和平均结果
            result = {
                "combination_id": i+1,
                "hyperparameters": hp_config_dict,
                "avg_vali_loss": float(avg_vali_loss),
                "avg_test_loss": float(avg_test_loss),
                "avg_test_mae_loss": float(avg_test_mae_loss),
                "pred_len_results": pred_len_results
            }
            
            # 只在主进程保存每组超参数的结果
            if args.accelerator.is_main_process:
                hp_result_file = os.path.join(hp_results_dir, f"result_{i+1}.json")
                with open(hp_result_file, 'w') as f:
                    json.dump(result, f, indent=4)
            
            all_results.append(result)
            
            # 只在主进程保存当前summary
            if args.accelerator.is_main_process:
                with open(summary_file, 'w') as f:
                    json.dump(all_results, f, indent=4)
            
            # 确保所有进程同步
            args.accelerator.wait_for_everyone()
        
        # 根据平均验证损失找出最佳超参数组合
        best_result = min(all_results, key=lambda x: x["avg_vali_loss"])
        
        # 只在主进程保存最佳结果到单独的文件
        if args.accelerator.is_main_process:
            best_result_file = os.path.join(hp_results_dir, "best_result.json")
            with open(best_result_file, 'w') as f:
                json.dump(best_result, f, indent=4)
        
        # 打印最佳超参数和结果
        args.accelerator.print("\nHyperparameter search completed!")
        args.accelerator.print(f"Best hyperparameters: {best_result['hyperparameters']}")
        args.accelerator.print(f"Best average validation loss: {best_result['avg_vali_loss']}")
        args.accelerator.print(f"Corresponding average test loss: {best_result['avg_test_loss']}")
        args.accelerator.print(f"Corresponding average test MAE: {best_result['avg_test_mae_loss']}")
        
        # 打印每个pred_len的结果
        args.accelerator.print("\nResults for each pred_len:")
        for result in best_result['pred_len_results']:
            args.accelerator.print(f"pred_len={result['pred_len']}: "
                                 f"test_loss={result['test_loss']}, "
                                 f"test_mae={result['test_mae_loss']}")
            
        if args.accelerator.is_main_process:
            args.accelerator.print(f"Results saved to: {hp_results_dir}")
            
            # 将结果保存到CSV文件 - 只在主进程执行
            save_results_to_csv(args, best_result, hp_results_dir)
            args.accelerator.print(f"CSV results saved by main process.")
        
        # 确保所有进程等待主进程完成所有文件写入
        args.accelerator.wait_for_everyone()

        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()

    elif args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
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
                args.des, ii)

            args.accelerator.print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            model, all_epoch_metrics, best_metrics, _ = exp.train(setting)

            args.accelerator.print('\n======== Best Model Metrics ========')
            args.accelerator.print(f"Best epoch: {best_metrics['epoch']}")
            args.accelerator.print(f"Best validation loss: {best_metrics['vali_loss']:.7f}")
            args.accelerator.print(f"Validation MAE: {best_metrics['vali_mae_loss']:.7f}")
            args.accelerator.print(f"Training loss: {best_metrics['train_loss']:.7f}")
            args.accelerator.print(f"Test loss: {best_metrics['test_loss']:.7f}")
            args.accelerator.print(f"Test MAE: {best_metrics['test_mae_loss']:.7f}")
            args.accelerator.print('====================================\n')

            # args.accelerator.print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            # exp.test(setting)
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
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
            args.des, ii)

        args.accelerator.print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
