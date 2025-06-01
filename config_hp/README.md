# Hyperparameter Search Configuration

This folder contains hyperparameter search configurations for different models. Each file defines the hyperparameters and their possible values to be searched for a specific model.

## File Format

Each configuration file should be named after the model it's for (e.g., `TimesNet.yaml`, `Autoformer.yaml`) and should contain hyperparameters as keys with their possible values as lists.

Example:
```yaml
# Model architecture
d_model: [256, 512]
n_heads: [4, 8]

# Optimization
learning_rate: [0.0001, 0.001]
```

## Multiple Prediction Lengths Evaluation

The hyperparameter search now supports evaluating each hyperparameter combination across multiple prediction lengths (pred_len). This provides a more robust evaluation of model performance across different forecasting horizons.

### Prediction Length Configuration

The file `pred_len_config.yaml` defines the prediction lengths to use for each dataset:

```yaml
# Standard datasets example
ETTh1: [24, 48, 168, 336]
# Epidemic datasets example
measles_england: [1, 7, 14, 30]
```

For each hyperparameter combination, the system will:
1. Train and evaluate the model with each pred_len value
2. Calculate the average validation loss across all pred_len values
3. Use this average to determine the best hyperparameter combination
4. The final results include both the average metrics and the individual metrics for each pred_len

If a dataset is not defined in `pred_len_config.yaml`, default values of `[24, 48, 96, 192]` will be used.

## Running Hyperparameter Search

To run hyperparameter search, use the following command:

```bash
python run.py --task_name long_term_forecast --is_training 1 --model_id search --model TimesNet --data ETTh1 --features M --seq_len 96 --pred_len 96 --hyper_parameter_searching
```

The `--hyper_parameter_searching` flag enables the hyperparameter search functionality. The results will be saved in the `hp_logs` directory. Note that the `--pred_len` argument is used only for initialization - the actual prediction lengths used during the search are defined in `pred_len_config.yaml`.

## Results Structure

Results will be saved in the following structure:
```
hp_logs/
└── ModelName_DatasetName_Timestamp/
    ├── hp_summary.json                  # All results with averaged metrics for each combination
    ├── best_result.json                 # Best hyperparameters based on average validation loss
    ├── result_1.json                    # Results for combination 1, including all pred_len results
    ├── epochs_metrics_1_predlen_24.json # Detailed metrics for combination 1, pred_len 24
    ├── epochs_metrics_1_predlen_48.json # Detailed metrics for combination 1, pred_len 48
    ├── result_2.json                    # Results for combination 2, including all pred_len results
    └── ...
```

The `best_result.json` file contains:
1. The best hyperparameter combination
2. The average metrics across all prediction lengths
3. Individual results for each prediction length

## Implementation Details

For each hyperparameter combination:
1. The model is trained separately for each prediction length
2. For each prediction length, the best validation metrics are tracked
3. The average validation loss across all prediction lengths determines the best hyperparameter combination
4. This approach provides more robust hyperparameter selection that works well across different forecasting horizons

## Creating New Configuration Files

To create a new configuration file for a model:

1. Create a YAML file with the model name (e.g., `YourModel.yaml`)
2. Define the hyperparameters and their possible values as lists
3. Run the hyperparameter search with `--model YourModel`

Alternatively, you can use the utility script to generate a template:

```bash
python utils/create_hp_config.py YourModel
```

This will create a default hyperparameter configuration file for your model in the `config_hp` directory.

Note: The more hyperparameters and values you include, the longer the search will take as it performs a grid search over all combinations multiplied by the number of prediction lengths.
