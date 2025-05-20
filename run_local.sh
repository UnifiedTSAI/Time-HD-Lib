#!/bin/bash

MODELS=("DLinear" "PAttn" "PatchTST" "TimeMixer" "PDF" "iTransformer" "Crossformer" "TSMixer" "FreTS" "Autoformer" "FEDformer" "TimesNet" "CCM" "DUET" "UCast")
DATASETS=(
  "atec"
  "china_air_quality"
  "global_temp"
  "global_wind"
  "google_community_mobility"
  "m5"
  "measles_england"
  "sirs"
  "sp500"
  "wikipedia_web_traffic_20000"
  "largest_ca"
  "largest_gba"
  "largest_gla"
  "smart_meters_in_london"
  "neurolib"
  "nrel_solar_power"
)

# Default parameters
gpu="all"
pred_len=7
seq_len=14
batch_size=1
features="M"
task_name="long_term_forecast"
is_training=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu)
      gpu="$2"
      shift 2
      ;;
    --pred_len)
      pred_len="$2"
      shift 2
      ;;
    --seq_len)
      seq_len="$2"
      shift 2
      ;;
    --batch_size)
      batch_size="$2"
      shift 2
      ;;
    --features)
      features="$2"
      shift 2
      ;;
    --task_name)
      task_name="$2"
      shift 2
      ;;
    --is_training)
      is_training="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Set GPU
if [[ "$gpu" == "all" ]]; then
  unset CUDA_VISIBLE_DEVICES
  echo "Using all available GPUs"
else
  export CUDA_VISIBLE_DEVICES=$gpu
  echo "Using GPU(s): $gpu"
fi

# Loop over datasets and models
for data in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "=============================================="
    echo "Running model $model on dataset $data"
    echo "GPU: $gpu"
    echo "Prediction Length: $pred_len"
    echo "Sequence Length: $seq_len"
    echo "Dataset: $data"
    echo "Model: $model"
    echo "Batch Size: $batch_size"
    echo "Task Name: $task_name"
    echo "Training Mode: $is_training"
    echo "Starting training..."

    accelerate launch --mixed_precision=bf16 run.py \
      --task_name $task_name \
      --is_training $is_training \
      --model_id ${data}'_'${seq_len}'_'${pred_len} \
      --model $model \
      --data $data \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --batch_size $batch_size \
      --train_epochs 1

    echo "Training completed for model $model on dataset $data"
    echo "=============================================="
  done
done
