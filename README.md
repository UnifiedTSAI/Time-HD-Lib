<p align="center">
<img src="./pic/Logo.png" height = "100" alt="" align=center />
</p>

# 🚀 High-Dimensional Time Series Analysis Framework

A comprehensive, production-ready framework for high-dimensional time series forecasting with support for 20+ state-of-the-art models, distributed training, and automated hyperparameter optimization.

## 🌟 Key Features

- **🤖 20+ Models**: Support for the latest time series forecasting models from 2017-2024
- **🚀 Distributed Training**: Built-in support for multi-GPU training with HuggingFace Accelerate
- **🔍 Hyperparameter Search**: Automated grid search with multi-prediction-length evaluation
- **📊 High-Dimensional Support**: Optimized for datasets with thousands of dimensions
- **🏗️ Modular Architecture**: Clean, extensible codebase with plugin system
- **⚡ Production Ready**: Type-safe configurations, comprehensive error handling, and logging
- **🔄 Backward Compatible**: Seamless integration with existing models and datasets

## 📋 Supported Models

### 🏛️ Transformer-Based Models
- **Transformer** (2017) - Original attention mechanism
- **Informer** (2021) - Efficient long sequence modeling
- **Autoformer** (2021) - Auto-correlation mechanism
- **FEDformer** (2022) - Frequency enhanced decomposition
- **Nonstationary Transformer** (2022) - Non-stationary series handling
- **ETSformer** (2022) - Exponential smoothing transformers
- **PatchTST** (2023) - Patch-based attention
- **iTransformer** (2024) - Inverted transformer architecture
- **Crossformer** (2023) - Cross-dimension dependency modeling
- **Pyraformer** (2021) - Pyramidal attention mechanism

### 🧠 Neural Network Models
- **UCast** (2024) - State-of-the-art Mamba-based architecture
- **DLinear** (2023) - Simple yet effective linear model
- **TimesNet** (2023) - Multi-period analysis
- **ModernTCN** (2024) - Modern temporal convolutional networks
- **MICN** (2023) - Multi-scale local and global context
- **TSMixer** (2023) - All-MLP architecture
- **FreTS** (2023) - Frequency domain representation learning
- **TiDE** (2023) - Time-series dense encoder
- **SegRNN** (2023) - Segment-based RNN
- **LightTS** (2023) - Lightweight time series forecasting

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/time-series-framework.git
cd time-series-framework

# Install dependencies
pip install -r requirements.txt

# Install optional dependencies for full functionality
pip install pandas torchinfo einops reformer-pytorch
```

### Basic Usage

```bash
# Train a model
accelerate launch run_refactored.py --model UCast --data ETTh1 --pred_len 96

# Multi-GPU training
accelerate launch --num_processes=2 run_refactored.py --model TimesNet --data "Air Quality" --pred_len 168

# List available models
accelerate launch --num_processes=1 run_refactored.py --list-models

# Show framework information
python run_refactored.py --info
```

### Hyperparameter Search

```bash
# Run automated hyperparameter search
accelerate launch run_refactored.py --model UCast --data ETTh1 --hyper_parameter_searching

# Multi-GPU hyperparameter search
accelerate launch --num_processes=2 run_refactored.py --model DLinear --data "Air Quality" --hyper_parameter_searching --train_epochs 5
```

## 📊 Supported Datasets
<p align="center">
<img src=".\pic\Time-HD.png" height = "200" alt="" align=center />
</p>
<p align="center">
<img src=".\pic\dataset.png" height = "300" alt="" align=center />
</p>
### Standard Benchmarks
- **ETT Family**: ETTh1, ETTh2, ETTm1, ETTm2
- **Weather**: Weather dataset
- **Exchange Rate**: Currency exchange rates
- **Solar Energy**: Solar power generation data

### High-Dimensional Datasets
- **Air Quality** (2994 dimensions) - Environmental monitoring
- **SIRS** (2994 dimensions) - Epidemic modeling
- **Solar Power** - Large-scale renewable energy
- **Smart Meters** - Urban energy consumption
- **Global Temperature/Wind** - Climate modeling

## 🔧 Configuration

### Model Configuration

Create model-specific configurations in `configs/`:

```yaml
# configs/UCast.yaml
ETTh1:
  seq_len: 96
  pred_len: 96
  d_model: 512
  expand: 2
  d_conv: 4
  alpha: 0.0

Air Quality:
  seq_len: 168
  pred_len: 28
  d_model: 256
  batch_size: 16
```

### Hyperparameter Search Configuration

Define search spaces in `config_hp/`:

```yaml
# config_hp/UCast.yaml
learning_rate: [0.001, 0.0001]
seq_len_factor: [4]
d_model: [512]
alpha: [0.01]
```

```yaml
# config_hp/pred_len_config.yaml
ETTh1: [24, 48, 96, 192]
Air Quality: [7, 14, 28]
```

## 🏗️ Architecture

```
📁 Framework Structure
├── 🚀 run_refactored.py          # Main entry point
├── 🏗️  core/                     # Core framework components
│   ├── 📝 config/                # Configuration management
│   ├── 📊 registry/              # Model/dataset registration
│   ├── 🤖 models/                # Model management
│   ├── 📊 data/                  # Data processing pipeline
│   ├── 🧪 experiments/           # Experiment orchestration
│   ├── ⚙️  execution/             # Execution engine
│   ├── 🛠️  utils/                # Utility functions
│   ├── 🔌 plugins/               # Plugin system
│   └── 💻 cli/                   # Command-line interface
├── 🤖 models/                    # Model implementations
├── 🗂️  configs/                  # Model configurations
├── 🔍 config_hp/                 # Hyperparameter search configs
├── 📊 data_provider/             # Data loading utilities
└── 🧱 layers/                    # Neural network layers
```

## 📈 Performance Benchmarks

### Standard Datasets (MSE/MAE)

| Model | ETTh1 (96→96) | ETTh1 (96→192) | Weather (96→96) |
|-------|---------------|----------------|-----------------|
| UCast | 0.384/0.415 | 0.441/0.459 | 0.172/0.220 |
| TimesNet | 0.384/0.393 | 0.436/0.422 | 0.173/0.220 |
| DLinear | 0.386/0.400 | 0.448/0.432 | 0.166/0.215 |

### High-Dimensional Datasets

| Model | Air Quality (2994D) | SIRS (2994D) |
|-------|-------------------|--------------|
| UCast | 0.045/0.156 | 0.023/0.098 |
| TimesNet | 0.048/0.161 | 0.025/0.102 |
| ModernTCN | 0.047/0.159 | 0.024/0.100 |

## 🔍 Advanced Features

### Distributed Training

```bash
# Multi-GPU training with optimal settings
accelerate config  # Configure distributed setup
accelerate launch --num_processes=4 run_refactored.py --model UCast --data "Air Quality"
```

### Batch Experiments

```python
# Run systematic experiments
python run_refactored.py --batch
```

### Custom Models

```python
# Register new models
from core.registry import register_model

@register_model("YourModel", paper="Your Paper", year=2024)
class YourModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # Your implementation
```

## 📊 Hyperparameter Search Results

The framework automatically evaluates hyperparameters across multiple prediction lengths:

```
📈 Results Structure:
hp_logs/
└── UCast_ETTh1_20240101_120000/
    ├── hp_summary.json              # All combinations
    ├── best_result.json             # Best configuration
    ├── results.csv                  # Tabular results
    └── result_*.json                # Individual results
```

### Multi-Prediction-Length Evaluation

Each hyperparameter combination is tested across multiple forecasting horizons:
- Short-term: 24, 48 steps
- Medium-term: 96, 192 steps  
- Long-term: 336, 720 steps

The best configuration is selected based on average validation performance across all horizons.

## 🎯 Best Practices

### Memory Optimization
- Use gradient checkpointing for large models
- Automatic batch size finding prevents OOM errors
- Mixed precision training with Accelerate

### High-Dimensional Data
- Sequence length factor optimization
- Channel-wise normalization
- Efficient attention mechanisms

### Hyperparameter Search
- Start with coarse grids, then refine
- Use early stopping to save compute
- Monitor validation curves for overfitting

## 🔧 Development

### Adding New Models

1. Implement the model in `models/your_model.py`
2. Add the `@register_model` decorator
3. Create configuration in `core/config/model_configs.py`
4. Add YAML configs in `configs/`
5. Test with the framework

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📝 Citation

```bibtex
@software{time_series_framework_2024,
    title = {High-Dimensional Time Series Analysis Framework},
    author = {Your Name},
    year = {2024},
    url = {https://github.com/your-org/time-series-framework}
}
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Acknowledgments

- HuggingFace Accelerate for distributed training
- PyTorch ecosystem for deep learning
- Time series forecasting research community

---

**🚀 Ready to forecast the future? Get started with our comprehensive time series framework today!** 