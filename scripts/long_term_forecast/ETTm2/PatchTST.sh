#!/bin/bash

export CUDA_VISIBLE_DEVICES=${1}

seq_len=${3}
model_name=PatchTST
root_path_name=./dataset/ETT-small/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2
label_len=48
factor=3
enc_in=7
dec_in=7
c_out=7
des='Exp'
itr=1

# pred_len is passed as the second argument
pred_len=${2}

# Set specific parameters for different pred_len values
e_layers=3
d_layers=1
n_heads=4
batch_size=32

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id ${model_id_name}_${seq_len}_${pred_len} \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --factor $factor \
  --enc_in $enc_in \
  --dec_in $dec_in \
  --c_out $c_out \
  --des $des \
  --n_heads $n_heads \
  --batch_size $batch_size \
  --itr $itr
