# Time-HD-Lib: A Lirbrary for High-Dimensional Time Series Forecasting

<p align="center">
<img src="./pic/Logo.png" height = "100" alt="" align=center />
</p>

## Time-HD: High-Dimensional Time Series Forecasting Benchmark
<p align="center">
<img src=".\pic\Time-HD.png" height = "300" alt="" align=center />
</p>
<p align="center">
<img src=".\pic\dataset.png" height = "300" alt="" align=center />
</p>

## U-Cast: Learning Latent Hierarchical Channel Structure for High-Dimensional Time Series Forecasting

<p align="center">
<img src=".\pic\U-Cast.png" height = "200" alt="" align=center />
</p>


## Usage
1. Install Python 3.10. For convenience, execute the following command.

```
pip install -r requirements.txt
```
or
```
conda env create -f environment.yaml
```

2. Prepare Data. Our code automatically downloads datasets from Hugging Face. Alternatively, you can manually download datasets by running: ```python download_dataset.py```. The summary of the supported high-dimensional time series datasets is shown in Table 2 above. Besides these, we also support datasets such as ECL, ETTh1, ETTh2, ETTm1, ETTm2, Weather, and Traffic.

3. Train and evaluate model. We provide the hyperparameters config for all datasets and benchmarks under the folder `./configs/`. You can run the experiment results as the following examples:

```
accelerate launch run.py --model DLinear --data sirs --pred_len 96
```

4. Develop your own model.

- Add the model file to the folder `./models`. You can follow the `./models/Transformer.py`.
- Include the newly added model in the `Exp_Basic.model_dict` of  `./exp/exp_basic.py`.
- Create the corresponding configs under the folder `./configs`.