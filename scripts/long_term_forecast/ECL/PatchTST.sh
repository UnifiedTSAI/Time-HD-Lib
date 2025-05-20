#!/bin/bash

export CUDA_VISIBLE_DEVICES=${1}

seq_len=${3}
model_name=PatchTST

root_path_name=./dataset/electricity/
data_path_name=electricity.csv
model_id_name=ECL

pred_len=${2}

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id ${model_id_name}_${seq_len}_${pred_len} \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --n_heads 16 \
  --d_model 128 \
  --d_ff 256 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --batch_size 4 \
  --itr 1