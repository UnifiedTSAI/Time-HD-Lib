<p align="center">
<img src="./pic/Logo.png" height = "100" alt="" align=center />
</p>

# 🚀 Time-HD-Lib: A Library for High-Dimensional Time Series Forecasting

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-Accelerate-yellow.svg)](https://huggingface.co/docs/accelerate)

A comprehensive, production-ready framework for high-dimensional time series forecasting with support for 20+ state-of-the-art models, distributed training, automated hyperparameter optimization, and a modular plugin system.

## 🌟 Key Features

- **🤖 20+ SOTA Models**: Latest time series forecasting models (2017-2024) with unified interface
- **🚀 Distributed Training**: Built-in multi-GPU support with HuggingFace Accelerate
- **🔍 AutoML**: Automated hyperparameter search with multi-horizon evaluation
- **📊 High-Dimensional**: Optimized for datasets with thousands of dimensions
- **⚡ GPU Management**: Automatic GPU allocation and memory optimization

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

# 🎯 Specific GPU selection
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

```

## 📊 Supported Datasets

<p align="center">
<img src=".\pic\Time-HD.png" height = "200" alt="" align=center />
</p>
<p align="center">
<img src=".\pic\dataset.png" height = "300" alt="" align=center />
</p>

### 🎯 Time-HD: High-Dimensional Benchmark
Our framework supports the **Time-HD** benchmark dataset through HuggingFace Datasets:


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
├── 🗂️  configs/                  # Model-dataset configurations
├── 🔍 config_hp/                 # Hyperparameter search configs
├── 🧱 layers/                    # Neural network building blocks
└── 📊 results/                   # Experiment outputs and logs
```

## 📈 Performance Benchmarks


## 🔍 Advanced Features

### 🚀 Intelligent GPU Management

```bash
# Automatic GPU allocation
export CUDA_VISIBLE_DEVICES=""  # Framework will auto-set
accelerate launch run.py --model UCast --data "Air Quality" --gpu 0,1,3,7

# The framework automatically sets CUDA_VISIBLE_DEVICES=0,1,3,7
# before initializing the accelerator
```

### 📊 Comprehensive Hyperparameter Search

```bash
# Multi-horizon evaluation
accelerate launch run.py --model TimesNet --data ETTh1 --hyper_parameter_searching

# Results structure:
hp_logs/
└── TimesNet_ETTh1_20241201_143022/
    ├── hp_summary.json              # All combinations tested
    ├── best_result.json             # Best configuration found
    ├── results.csv                  # Tabular results for analysis
    ├── result_1.json                # Individual combination results
    ├── result_2.json
    └── ...
```

### 🔄 Batch Experiments

```python
# Run systematic model comparison
python run.py --batch

# Customizable batch experiments
from core.execution.runner import BatchRunner
from core.config import ConfigManager

batch_runner = BatchRunner(ConfigManager())
for model in ['UCast', 'TimesNet', 'iTransformer']:
    for dataset in ['Measles', 'SIRS']:
        batch_runner.add_experiment(model=model, data=dataset, pred_len=96)

results = batch_runner.run_batch()
```

### 🔌 Custom Model Registration

```python
# Easy model integration
from core.registry import register_model
import torch.nn as nn

@register_model("YourModel", paper="Your Amazing Paper", year=2024)
class YourModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        # Your implementation here
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Your forward pass
        return predictions
```

## 🎯 Best Practices

### 💾 Memory Optimization
```bash
# Automatic batch size finding (prevents OOM)
accelerate launch run.py --model UCast --data "Measles" --batch_size 64
# Framework automatically reduces to batch_size=1 if needed

# Mixed precision training
accelerate launch --mixed_precision fp8 run.py --model UCast --data "Measles"

# Gradient checkpointing for large models
# Automatically enabled for models with >100M parameters
```

### 📊 High-Dimensional Data Strategies
```yaml
# Optimal configurations for high-dimensional data
UCast:
  Air_Quality:  # 2994 dimensions
    seq_len_factor: 5        # Longer lookback for complex patterns
    alpha: 0.1               # Higher regularization
    d_model: 512             # Sufficient embedding dimension
    batch_size: 8            # Smaller batches for memory efficiency
    
  Measles:      # 1161 dimensions  
    seq_len_factor: 4        # Standard lookback
    alpha: 0.01              # Lower regularization
    d_model: 256             # Smaller model for efficiency
    batch_size: 16           # Larger batches possible
```

### 🔍 Hyperparameter Search Guidelines
```bash
# Start with coarse grids
learning_rate: [0.1, 0.01, 0.001, 0.0001]
d_model: [128, 256, 512]

# Then refine around best results
learning_rate: [0.0008, 0.001, 0.0012]
d_model: [480, 512, 544]

# Use early stopping to save compute
accelerate launch run.py --model UCast --data ETTh1 \
  --hyper_parameter_searching --train_epochs 10 --patience 3
```

## 🔧 Development & Extension

### Adding New Models

1. **Implement the model** in `models/your_model.py`:
```python
from core.registry import register_model
import torch.nn as nn

@register_model("YourModel", paper="Your Paper Title", year=2024)
class Model(nn.Module):  # Must be named "Model"
    def __init__(self, configs):
        super().__init__()
        # Implementation
```

2. **Add configuration** in `configs/YourModel.yaml`:
```yaml
ETTh1:
  enc_in: 7
  dec_in: 7
  c_out: 7
  # Other parameters
```

3. **Test the model**:
```bash
accelerate launch --num_processes=1 run.py --model YourModel --data ETTh1
```

### Adding New Datasets

1. **Extend data_loader.py**:
```python
class Dataset_YourData(Dataset):
    def __init__(self, args, root_path, flag='train', ...):
        # Implementation
```

2. **Update data_factory.py**:
```python
data_dict = {
    'your_data': Dataset_YourData,
    # existing entries
}
```

3. **Add configuration**:
```yaml
# configs/ModelName.yaml
your_data:
  enc_in: 100  # your dataset dimensions
  # other parameters
```

### 🧪 Testing & Validation

```bash
# Quick smoke test
accelerate launch --num_processes=1 run.py --model UCast --data ETTh1 \
  --train_epochs 1 --is_training 1

# Full validation with multiple seeds
for seed in 42 1337 2023; do
  accelerate launch run.py --model UCast --data "Air_Quality" \
    --seed $seed --train_epochs 20
done

# Model comparison across datasets
python scripts/benchmark_models.py --models UCast,TimesNet,iTransformer \
  --datasets ETTh1,ETTh2,Air_Quality
```

## 📊 Experiment Results Management

### 📁 Output Structure
```
results/
├── UCast_ETTh1_Exp_20241201_143022/
│   ├── metrics.npy              # [mae, mse, rmse, mape, mspe]
│   ├── pred.npy                 # Model predictions
│   ├── true.npy                 # Ground truth values
│   └── checkpoint.pth           # Best model weights
├── test_results/
│   └── UCast_ETTh1_Exp_20241201_143022/
│       ├── 0.pdf                # Visualization plots
│       ├── 20.pdf
│       └── ...
└── hp_logs/                     # Hyperparameter search results
    └── UCast_ETTh1_20241201_143022/
        ├── best_result.json
        ├── hp_summary.json
        └── results.csv
```

### 📈 Results Analysis
```python
import numpy as np
import pandas as pd
import json

# Load hyperparameter search results
with open('hp_logs/UCast_ETTh1_20241201_143022/best_result.json', 'r') as f:
    best_config = json.load(f)

print(f"Best hyperparameters: {best_config['hyperparameters']}")
print(f"Best validation loss: {best_config['avg_vali_loss']:.6f}")

# Load model predictions
pred = np.load('results/UCast_ETTh1_Exp_20241201_143022/pred.npy')
true = np.load('results/UCast_ETTh1_Exp_20241201_143022/true.npy')
metrics = np.load('results/UCast_ETTh1_Exp_20241201_143022/metrics.npy')

mae, mse, rmse, mape, mspe = metrics
print(f"Final test results: MSE={mse:.6f}, MAE={mae:.6f}")
```

## 🚨 Troubleshooting

### Common Issues

**🔥 CUDA Out of Memory**
```bash
# Use automatic batch size finding
accelerate launch run.py --model UCast --data "Air_Quality" --batch_size 64
# Framework automatically reduces to batch_size=1 if needed

# Enable gradient checkpointing
# Add to model config:
use_checkpoint: true
```

**🐌 Slow Training**
```bash
# Enable mixed precision
accelerate launch run.py --model TimesNet --data ETTh1 --use_amp

# Reduce sequence length for large datasets
--seq_len_factor 3  # instead of default 4
```

**📉 Poor Performance**
```bash
# Check data normalization
--use_norm 1  # Enable normalization

# Try different learning rates
--learning_rate 0.0001  # Start conservative

# Increase model capacity
--d_model 512 --e_layers 3
```

**🚫 Model Registration Issues**
```python
# Ensure @register_model decorator is used
from core.registry import register_model

@register_model("YourModel", paper="Title", year=2024)
class Model(nn.Module):  # Must be named "Model"
    pass
```

## 📝 Citation

If you use Time-HD-Lib in your research, please cite:

```bibtex
@software{time_hd_lib_2024,
    title = {Time-HD-Lib: A Library for High-Dimensional Time Series Forecasting},
    author = {Your Name and Contributors},
    year = {2024},
    url = {https://github.com/your-org/Time-HD-Lib},
    version = {1.0.0}
}

@article{ucast_2024,
    title = {U-Cast: Learning Latent Hierarchical Channel Structure for High-Dimensional Time Series Forecasting},
    author = {Your Name and Co-authors},
    journal = {Your Journal},
    year = {2024}
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