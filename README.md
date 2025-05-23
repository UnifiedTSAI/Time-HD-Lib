<p align="center">
<img src="./pic/Logo.png" height = "100" alt="" align=center />
</p>

# Time-HD-Lib: A Lirbrary for High-Dimensional Time Series Forecasting

To cultivate a collaborative research ecosystem, we release Time-HD-Lib. This open-source library provides an end-to-end framework supporting the Time-HD benchmark, including standardized data preprocessing, seamless integration of datasets, unified evaluation protocols, and facilities for hyperparameter optimization, thus promoting reproducibility and future extensions.

## Time-HD: High-Dimensional Time Series Forecasting Benchmark
<p align="center">
<img src=".\pic\Time-HD.png" height = "200" alt="" align=center />
</p>
Time-HD is the first comprehensive benchmark suite specifically focused on high-dimensional time series, offering large-scale, domain-diverse dataset.

<p align="center">
<img src=".\pic\dataset.png" height = "200" alt="" align=center />
</p>
The goal of Time-HD is to support research in time series forecasting, a rapidly growing field. The statistics of the datasets are shown in the table above. These datasets exhibit several key characteristics: High Dimensionality, Diverse Sources, Varied Scales, Different Sampling Frequencies, and Broad Domain Coverage.


## U-Cast: Learning Latent Hierarchical Channel Structure for High-Dimensional Time Series Forecasting

<p align="center">
<img src=".\pic\U-Cast.png" height = "200" alt="" align=center />
</p>
We propose U-Cast, a new CD forecasting model that employs an innovative query-based attention mechanism to efficiently learn latent hierarchical channel structures and enables scalable modeling of inter-channel correlations. U-Cast combines strong performance across Time-HD with the best efficiency among baselines, making it a reference point for future models.


## Usage
**1. Install Python 3.10. For convenience, execute the following command.**

```
pip install -r requirements.txt
```
or
```
conda env create -f environment.yaml
```

**2. Prepare Data**

To access the dataset, follow these steps:

a. Create a **anonymous** Hugging Face account, if you do not already have one.

b. Visit the anonymous dataset page:  
   [https://huggingface.co/datasets/Time-HD-Anonymous/High_Dimensional_Time_Series](https://huggingface.co/datasets/Time-HD-Anonymous/High_Dimensional_Time_Series)

c. Click **“Agree and access repository”**. You must be logged in to complete this step.

d. Create new Access Token. Token type should be "write".

d. Authenticate on your local machine by running:

   ```bash
   huggingface-cli login
   ```
   and enter your generated token above.

e. Then, you can manually download all the dataset by running:

   ```bash
   python download_dataset.py
   ```

The summary of the supported high-dimensional time series datasets is shown in Table 2 above. Besides these, we also support datasets such as ECL, ETTh1, ETTh2, ETTm1, ETTm2, Weather, and Traffic.

**3. Train and evaluate model.** 
We provide the hyperparameters config for all datasets and benchmarks under the folder `./configs/`. You can run the experiment as the following examples:

```
accelerate launch run.py --model UCast --data "Air Quality" --pred_len 28
```

**4. Hyperparameter searching.** 
a. First set the hyperparameter searching space in "./config_hp" for the specfic model.

b. Conduct hyperparameter searching by running:
```
accelerate launch run.py --model UCast --data "Air Quality" --pred_len 28 --hyper_parameter_searching
```