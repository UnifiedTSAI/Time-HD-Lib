<p align="center">
<img src="./pic/Logo.png" height = "100" alt="" align=center />
</p>

# 🚀 A Library for High-Dimensional Time Series Forecasting

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-Accelerate-yellow.svg)](https://huggingface.co/docs/accelerate)

A comprehensive, production-ready framework for high-dimensional time series forecasting with support for 20+ state-of-the-art models, distributed training, automated hyperparameter optimization.

## 🌟 Key Features

- **📊 High-Dimensional**: Optimized for datasets with thousands of dimensions
- **🤖 20+ SOTA Models**: Latest time series forecasting models (2017-2024) with unified interface
- **🚀 Distributed Training**: Built-in multi-GPU support with HuggingFace Accelerate
- **🔍 AutoML**: Automated hyperparameter search with multi-horizon evaluation


## 📋 Supported Models (20+)

### 🏛️ Transformer-Based Models
| Model | Year | Paper | Description |
|-------|------|-------|-------------|
| **Transformer** | 2017 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Original transformer architecture |
| **Informer** | 2021 | [Beyond Efficient Transformer](https://arxiv.org/abs/2012.07436) | ProbSparse attention mechanism |
| **Autoformer** | 2021 | [Decomposition Transformers](https://arxiv.org/abs/2106.13008) | Auto-correlation mechanism |
| **Pyraformer** | 2021 | [Pyramidal Attention](https://arxiv.org/abs/2110.08519) | Low-complexity attention |
| **FEDformer** | 2022 | [Frequency Enhanced Decomposed](https://arxiv.org/abs/2201.12740) | Frequency domain modeling |
| **Nonstationary Transformer** | 2022 | [Non-stationary Transformers](https://arxiv.org/abs/2205.14415) | Handles non-stationarity |
| **ETSformer** | 2022 | [Exponential Smoothing Transformers](https://arxiv.org/abs/2202.01381) | ETS-based transformers |
| **Crossformer** | 2023 | [Cross-Dimension Dependency](https://arxiv.org/abs/2108.00154) | Cross-dimensional attention |
| **PatchTST** | 2023 | [A Time Series is Worth 64 Words](https://arxiv.org/abs/2211.14730) | Patch-based transformers |
| **iTransformer** | 2024 | [Inverted Transformers](https://arxiv.org/abs/2310.06625) | Channel-attention design |

### 🧠 CNN & MLP-Based Models
| Model | Year | Paper | Description |
|-------|------|-------|-------------|
| **MICN** | 2023 | [Multi-scale Local and Global Context](https://arxiv.org/abs/2301.10956) | Isometric convolution |
| **TimesNet** | 2023 | [Temporal 2D-Variation Modeling](https://arxiv.org/abs/2210.02186) | 2D temporal modeling |
| **ModernTCN** | 2024 | [Modern Temporal Convolutional Networks](https://arxiv.org/abs/2404.00496) | Enhanced TCN architecture |
| **DLinear** | 2023 | [Are Transformers Effective?](https://arxiv.org/abs/2205.13504) | Simple linear baseline |
| **TSMixer** | 2023 | [All-MLP Architecture](https://arxiv.org/abs/2303.06053) | MLP-based mixing |
| **FreTS** | 2023 | [Simple yet Effective Approach](https://arxiv.org/abs/2302.06677) | Frequency representation |
| **TiDE** | 2023 | [Time-series Dense Encoder](https://arxiv.org/abs/2304.08424) | Dense encoder design |
| **SegRNN** | 2023 | [Segment Recurrent Neural Network](https://arxiv.org/abs/2308.11200) | Segment-based RNN |
| **LightTS** | 2023 | [Lightweight Time Series](https://arxiv.org/abs/2207.01186) | Efficient forecasting |

### 🎯 High-Dimensional Specialized
| Model | Year | Paper | Description |
|-------|------|-------|-------------|
| **UCast** | 2025 | [Learning Latent Hierarchical Channel Structure](https://arxiv.org/abs/placeholder) | High-dimensional forecasting |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/LingFengGold/Time-HD-Lib
cd Time-HD-Lib

# Method 1: Using pip
pip install -r requirements.txt

# Method 2: Using conda (recommended)
conda env create -f environment.yaml
conda activate tsf

# Install optional dependencies for full functionality
pip install pandas torchinfo einops reformer-pytorch
```

### Basic Usage

```bash
# 🖥️ Single GPU training
accelerate launch --num_processes=1 run.py --model UCast --data "Measles" --gpu 0

# 🚀 Multi-GPU training (auto-detect all GPUs)
accelerate launch run.py --model UCast --data "Measles"

# 🎯 Specific GPU selection (e.g. 4 GPUs, id: 0,2,3,7)
accelerate launch --num_processes=4 run.py --model UCast --data "Measles" --gpu 0,2,3,7

# 📋 List available models
accelerate launch run.py --list-models

# ℹ️ Show framework information
python run.py --info
```

### Hyperparameter Search

```bash
# 🔍 Automated hyperparameter search
accelerate launch run.py --model UCast --data "Measles" --hyper_parameter_searching
accelerate launch --num_processes=1 run.py --model UCast --data "Measles" --gpu 0 --hyper_parameter_searching
accelerate launch --num_processes=4 run.py --model UCast --data "Measles" --gpu 0,2,3,7 --hyper_parameter_searching
```

## 📊 Supported Datasets

### 🎯 Time-HD: High-Dimensional Benchmark

<p align="center">
<img src=".\pic\Time-HD.png" height = "200" alt="" align=center />
</p>

Our framework supports the **Time-HD** benchmark dataset through HuggingFace Datasets:

<p align="center">
<img src=".\pic\dataset.png" height = "300" alt="" align=center />
</p>

### 📈 Traditional Benchmarks
- **ETT** (ETTh1, ETTh2, ETTm1, ETTm2) - Electricity transformer temperature
- **Weather** - Multi-variate weather forecasting
- **Traffic** - Road traffic flow
- **ECL** - Electricity consuming load

## 🔧 Configuration System

### Model Configuration

Create dataset-specific configurations in `configs/`:

```yaml
# configs/UCast.yaml
Measles:
  enc_in: 1161
  train_epochs: 10
  alpha: 0.01
  seq_len_factor: 4
  learning_rate: 0.001

Air_Quality:
  enc_in: 2994
  train_epochs: 15
  alpha: 0.1
  seq_len_factor: 5
  learning_rate: 0.0001
```

### Hyperparameter Search Configuration

Define search spaces in `config_hp/`:

```yaml
# config_hp/UCast.yaml
learning_rate: [0.001, 0.0001]
seq_len_factor: [4, 5]
d_model: [256, 512]
alpha: [0.01, 0.1]
```

## 🏗️ Architecture Overview

```
📁 Time-HD-Lib Framework
├── 🚀 run.py                     # Main entry point with GPU management
├── 🏗️  core/                     # Core framework components
│   ├── 📝 config/                # Configuration management system
│   │   ├── base.py               # Base configuration classes
│   │   ├── manager.py            # Configuration manager
│   │   └── model_configs.py      # Model-specific configs
│   ├── 📊 registry/              # Model/dataset registration
│   │   ├── __init__.py           # Registry decorators
│   │   └── model_registry.py     # Model registration system
│   ├── 🤖 models/                # Model management and loading
│   │   ├── model_manager.py      # Dynamic model loading
│   │   └── __init__.py           # Model manager interface
│   ├── 📊 data/                  # Self-contained data pipeline
│   │   ├── data_provider.py      # Main data provider
│   │   ├── data_factory.py       # Dataset factory
│   │   └── data_loader.py        # Custom dataset classes
│   ├── 🧪 experiments/           # Experiment orchestration
│   │   ├── base_experiment.py    # Base experiment class
│   │   └── long_term_forecasting.py  # Forecasting experiments
│   ├── ⚙️  execution/             # Execution engine
│   │   └── runner.py             # Experiment runners
│   ├── 🛠️  utils/                # Self-contained utilities
│   │   ├── tools.py              # Training utilities
│   │   ├── metrics.py            # Evaluation metrics
│   │   ├── timefeatures.py       # Time feature extraction
│   │   ├── augmentation.py       # Data augmentation
│   │   ├── masked_attention.py   # Attention mechanisms
│   │   └── masking.py            # Masking utilities
│   ├── 🔌 plugins/               # Plugin system for extensibility
│   └── 💻 cli/                   # Command-line interface
│       └── argument_parser.py    # Comprehensive CLI parser
├── 🤖 models/                    # Model implementations with @register_model
│   ├── UCast.py                  # High-dimensional specialist
│   ├── TimesNet.py               # 2D temporal modeling
│   ├── iTransformer.py           # Inverted transformer
│   ├── ModernTCN.py              # Modern TCN
│   └── ...                       # 16+ other models
├── 🗂️ configs/                   # Model-dataset configurations
├── 🔍 config_hp/                 # Hyperparameter search configs
├── 🧱 layers/                    # Neural network building blocks
└── 📊 results/                   # Experiment outputs and logs
```

## 📈 Performance Benchmarks
<p align="center">
<img src=".\pic\benchmark.png" height = "300" alt="" align=center />
</p>

## 🎯 Best Practices

### 1. Model Hyperparameter Configuration

#### Create Model Configuration Files
Create YAML configuration files for each model in the `configs/` directory:

```yaml
# configs/YourModel.yaml
Measles:
  enc_in: 1161
  train_epochs: 10
  learning_rate: 0.001
  d_model: 512
  batch_size: 16
  seq_len_factor: 4
```

#### Prediction Length Configuration
Edit `configs/pred_len_config.yaml` to set default prediction lengths for datasets:

```yaml
# configs/pred_len_config.yaml
Measles: [7]           # Use the first value as default
Temp: [168]
```

### 2. Multi-GPU Setup and Distributed Training

#### Automatic GPU Detection
```bash
# Use all available GPUs
accelerate launch run.py --model UCast --data "Measles"
```

#### Specify Specific GPUs
```bash
# Use GPUs 0,2,3,7
accelerate launch --num_processes=4 run.py --model UCast --data "Measles" --gpu 0,2,3,7

# Single GPU training
accelerate launch --num_processes=1 run.py --model UCast --data "Measles" --gpu 0
```

#### Configure Distributed Training
```bash
# Multi-node training
accelerate launch --multi_gpu --main_process_port 29500 run.py --model UCast --data "Measles"
```

### 3. Automatic Batch Size Finding during Hyperparameter Searching

The framework automatically finds the maximum available batch size during hyperparameter searching:

```bash
# Start from batch size 64, automatically reduce to 32, 16, 8, 4, 2, 1 when encountering OOM
accelerate launch run.py --model UCast --data "Measles" --batch_size 64 --hyper_parameter_searching
```

Manual batch size control:
```yaml
# configs/UCast.yaml 
Measles:
  batch_size: 16  # Set smaller batch size for high-dimensional data
  
Wiki-20k:
  batch_size: 8   # Use even smaller batch size for ultra-high-dimensional data
```

### 4. Mixed Precision Training

#### Enable Mixed Precision
```bash
accelerate launch --mixed_precision fp16 run.py --model UCast --data "Measles"
```


### 5. Batch Training

#### Use Batch Mode
```bash
# Run predefined batch experiments
python run.py --batch
```

#### Custom Batch Experiments
```python
from core.config import ConfigManager
from core.execution.runner import BatchRunner

# Create batch experiments
config_manager = ConfigManager()
batch_runner = BatchRunner(config_manager)

# Add experiments
models = ['UCast', 'TimesNet', 'iTransformer']
datasets = ['Measles', 'SIRS', 'ETTh1']

for model in models:
    for dataset in datasets:
        batch_runner.add_experiment(
            model=model,
            data=dataset,
            is_training=True
        )

# Run batch experiments
results = batch_runner.run_batch()
```

### 6. Hyperparameter Search Configuration and Execution

#### Create Hyperparameter Search Configuration
```yaml
# config_hp/UCast.yaml
learning_rate: [0.001, 0.0001, 0.00001]
seq_len_factor: [3, 4, 5]
d_model: [256, 512, 1024]
alpha: [0.01, 0.1, 1.0]
batch_size: [8, 16, 32]
```

#### Set Prediction Length Ranges for Datasets
```yaml
# configs/pred_len_config.yaml  
Measles: [7, 14, 21]      # These 3 values will be tested during hyperparameter search
ETTh1: [96, 192, 336]     # Multiple prediction lengths for traditional datasets
"Air Quality": [28, 56]   # Suitable prediction lengths for high-dimensional data
```

#### Execute Hyperparameter Search
```bash
# Single GPU hyperparameter search
accelerate launch --num_processes=1 run.py --model UCast --data "Measles" --hyper_parameter_searching

# Multi-GPU hyperparameter search
accelerate launch --num_processes=4 run.py --model UCast --data "Measles" --gpu 0,2,3,7 --hyper_parameter_searching

# Specify log directory
accelerate launch run.py --model UCast --data "Measles" --hyper_parameter_searching --hp_log_dir ./my_hp_logs/
```

#### View Search Results
```bash
# Results are saved in hp_logs/ directory
hp_logs/
└── UCast_Measles_20241201_143022/
    ├── best_result.json     # Best configuration and results
    ├── hp_summary.json      # Summary of all configurations
    ├── results.csv          # CSV format results
    └── result_*.json        # Detailed results for each configuration
```

## 🔧 Development & Extension

### 1. Adding New Models

#### Step 1: Implement Model Class
Create a new model file in the `models/` directory:

```python
# models/YourNewModel.py
import torch
import torch.nn as nn
from core.registry import register_model

@register_model("YourNewModel", paper="Your Paper Title", year=2024)
class Model(nn.Module):  # Class name must be 'Model'
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        
        # Get parameters from configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        
        # Implement your model architecture
        self.encoder = nn.Linear(self.enc_in, self.d_model)
        self.decoder = nn.Linear(self.d_model, self.enc_in)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [batch_size, seq_len, enc_in]
        # Return: [batch_size, pred_len, enc_in]
        
        # Implement forward propagation
        encoded = self.encoder(x_enc)
        # ... Your model logic ...
        output = self.decoder(encoded)
        
        return output
```

#### Step 2: Create Model Configuration
```yaml
# configs/YourNewModel.yaml
Measles:
  enc_in: 1161
  train_epochs: 10
  learning_rate: 0.001
  d_model: 512
  batch_size: 16
  seq_len_factor: 4
  # Add model-specific parameters
  your_param: 0.1

ETTh1:
  enc_in: 7
  train_epochs: 15
  learning_rate: 0.0001
  d_model: 256
```

#### Step 3: Create Hyperparameter Search Configuration
```yaml
# config_hp/YourNewModel.yaml
learning_rate: [0.001, 0.0001]
d_model: [256, 512]
your_param: [0.1, 0.5, 1.0]
seq_len_factor: [3, 4, 5]
```

#### Step 4: Test New Model
```bash
# Test if model is correctly registered
python run.py --list-models

# Quick validation training
accelerate launch --num_processes=1 run.py --model YourNewModel --data "Measles" --train_epochs 1

# Full training
accelerate launch run.py --model YourNewModel --data "Measles"

# Hyperparameter search
accelerate launch run.py --model YourNewModel --data "Measles" --hyper_parameter_searching
```

### 2. Adding New Datasets (Upload to HuggingFace)

#### Step 1: Prepare Dataset

**📊 Standard Dataset Format**

Time-HD-Lib expects datasets to follow a standardized format:
- **📅 Date Column**: First column named `'date'` containing timestamps
- **📈 Feature Columns**: Remaining columns represent different features/dimensions  
- **⏰ Row Structure**: Each row represents one time step/timestamp
- **📋 Column Order**: `['date', 'feature_0', 'feature_1', ..., 'feature_n']`

**Example Dataset Structure:**
```
        date          feature_0    feature_1    feature_2    ...    feature_499
0    2020-01-01 00:00:00   0.234       -1.456       0.789    ...       2.341
1    2020-01-01 01:00:00  -0.567        0.891      -0.234    ...      -1.234  
2    2020-01-01 02:00:00   1.234       -0.567       1.456    ...       0.567
...               ...        ...          ...         ...    ...         ...
9999 2021-02-23 07:00:00   0.123        1.789      -0.987    ...       1.567
```

**Implementation Example:**
```python
# prepare_dataset.py
import pandas as pd
import numpy as np
from datasets import Dataset
from huggingface_hub import HfApi

# Prepare your time series data
# Data format: [time_steps, features]
data = np.random.randn(10000, 500)  # Example: 10000 time steps, 500 features
dates = pd.date_range('2020-01-01', periods=10000, freq='H')

# Create DataFrame with STANDARD FORMAT
# 🚨 IMPORTANT: First column must be 'date', followed by feature columns
df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(500)])
df['date'] = dates

# 📋 Reorder columns: date first, then all features
df = df[['date'] + [f'feature_{i}' for i in range(500)]]

# ✅ Verify format
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist()[:5], "...")  # First 5 columns
print("Sample data:")
print(df.head(3))

# Save as CSV
df.to_csv('./your_dataset.csv', index=False)
```

**🔧 Format Requirements:**
- **Time Column**: Must be named `'date'` and contain valid timestamps
- **Feature Naming**: Can use any naming convention (e.g., `feature_0`, `sensor_1`, `temperature`)
- **Data Types**: Numeric values for features, datetime for date column
- **Missing Values**: Handle NaN values before uploading (interpolate or remove)
- **Frequency**: Consistent time intervals (hourly, daily, etc.)

#### Step 2: Upload to HuggingFace (https://huggingface.co/datasets/Time-HD-Anonymous/High_Dimensional_Time_Series)


#### Step 3: Add Dataset Support in Framework or use Dataset_Custom
```python
# core/data/data_loader.py - Add new dataset class
class Dataset_YourDataset(Dataset):
    def __init__(self, args, root_path, flag='train', size=None, 
                 features='S', data_path='your_dataset.csv',
                 target='feature_0', scale=True, timeenc=0, freq='h'):
        
        # Implement data loading logic
        # Can load from HuggingFace or local CSV
        if args.use_hf_datasets:
            from datasets import load_dataset
            hf_dataset = load_dataset("your-username/your-dataset-name")
            self.data_x = hf_dataset[flag].to_pandas()
        else:
            # Load from local
            df_raw = pd.read_csv(os.path.join(root_path, data_path))
            self.data_x = df_raw
            
        # Implement the rest of data processing logic...
```

#### Step 4: Update Data Factory
```python
# core/data/data_factory.py
data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'your_dataset': Dataset_YourDataset,  # Add new dataset
}
```

#### Step 5: Add Configuration Support
```yaml
# configs/pred_len_config.yaml
your_dataset: [24, 48, 96]  # Set default prediction length

# configs/UCast.yaml (or other model configurations)
your_dataset:
  enc_in: 500  # Number of features in your dataset
  train_epochs: 10
  learning_rate: 0.001
  seq_len_factor: 4
```

#### Step 6: Test New Dataset
```bash
# Test data loading
accelerate launch --num_processes=1 run.py --model UCast --data your_dataset --train_epochs 1

# Full training
accelerate launch run.py --model UCast --data your_dataset

# Hyperparameter search
accelerate launch run.py --model UCast --data your_dataset --hyper_parameter_searching
```

## 📊 Experiment Results Management

### 📁 Output Structure
```
results/
├── UCast_ETTh1_Exp_20241201_143022/
│   ├── metrics.npy              # [mae, mse, rmse, mape, mspe]
│   ├── pred.npy                 # Model predictions
│   ├── true.npy                 # Ground truth values
test_results/
   └── UCast_ETTh1_Exp_20241201_143022/
       ├── 0.pdf                # Visualization plots
       ├── 20.pdf
       └── ...
hp_logs/                     # Hyperparameter search results
    └── UCast_ETTh1_20241201_143022/
        ├── best_result.json
        ├── hp_summary.json
        └── results.csv
```




## 📝 Citation

If you use Time-HD-Lib in your research, please cite:

```bibtex

}
@article{ucast_2024,
    title = {Are We Overlooking the Dimensions? Learning Latent Hierarchical Channel Structure for High-Dimensional Time Series Forecasting},
    author = {Juntong Ni, Shiyu Wang, Zewen Liu, Xiaoming Shi, Xinyue Zhong, Zhou Ye, Wei Jin},
    journal = {In Submission},
    year = {2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

- **Time-Series-Library** - Foundation and inspiration ([GitHub](https://github.com/thuml/Time-Series-Library))
- **HuggingFace Accelerate** - Distributed training infrastructure
- **PyTorch Ecosystem** - Deep learning framework
- **Time Series Research Community** - For advancing the field

## 🌟 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/amazing-feature`)
3. 💻 Make your changes and add tests
4. ✅ Ensure all tests pass (`python -m pytest tests/`)
5. 📝 Update documentation if needed
6. 🚀 Submit a pull request

## 📞 Support & Community

- **📧 Issues**: [GitHub Issues](https://github.com/your-org/Time-HD-Lib/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/your-org/Time-HD-Lib/discussions)  
- **📖 Documentation**: [Full Documentation](https://your-org.github.io/Time-HD-Lib)
- **🐦 Updates**: Follow us on [Twitter](https://twitter.com/your-handle)

---

**🚀 Ready to forecast the future with high-dimensional time series? Get started today!** 