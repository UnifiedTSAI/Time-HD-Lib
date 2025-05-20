#!/bin/bash

declare -A datasets

datasets['neurolib']="336"
datasets['nrel_solar_power']="336"
datasets['atec']="336"
datasets['smart_meters_in_london']="336"

datasets['global_temp']="168"
datasets['global_wind']="168"
datasets['largest_ca']="168"
datasets['largest_gla']="168"
datasets['largest_gba']="168"

datasets['china_air_quality']="28"

datasets['sirs']="7"
datasets['sp500']="7"
datasets['m5']="7"
datasets['measles_england']="7"
datasets['wikipedia_web_traffic']="7"
datasets['google_community_mobility']="7"
datasets['rebound']="7"

# datasets['electricity']="96"
# datasets['etth1']="96"
# datasets['etth2']="96"
# datasets['ettm1']="96"
# datasets['ettm2']="96"
# datasets['weather']="96"
# datasets['traffic']="96"

model=TimeMixer
for data in "${!datasets[@]}"; do
    pred_len="${datasets[$data]}"
    seq_len=$((pred_len * 2))
    echo "Submitting job for method: $model, dataset: $data with seq_len: $seq_len and pred_len: $pred_len"
    sbatch yangji_run_slurm.sh ${pred_len} ${seq_len} ${data} ${model}
done