#!/bin/bash

# 模型列表
# MODELS=("DLinear" "TSMixer" "Autoformer" "CCM" "Crossformer" "FEDformer" "FreTS" "MICN" "PAttn" "PatchTST" "TimeMixer" "TimesNet" "iTransformer")

MODELS=("DLinear" "TSMixer" "FreTS")

# 数据集列表
DATASETS=(
  "atec"
  "china_air_quality"
  "global_temp"
  "global_wind"
  "google_community_mobility"
  "largest_ca"
  "largest_gba"
  "largest_gla"
  "m5"
  "measles_england"
  "neurolib"
  "nrel_solar_power"
  "sirs"
  "smart_meters_in_london"
  "sp500"
  "wikipedia_web_traffic"
)

# 日志目录
LOG_DIR="hpo_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p $LOG_DIR

# 提交每个模型-数据集组合
for MODEL in "${MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    echo "Submitting job for Model: $MODEL, Dataset: $DATASET"
    sbatch run_hpo_single.sh $MODEL $DATASET $LOG_DIR $TIMESTAMP
    sleep 1  # 避免提交过快
  done
done
