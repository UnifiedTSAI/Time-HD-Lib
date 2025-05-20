#!/bin/bash

export CUDA_VISIBLE_DEVICES=${1}

seq_len=${3}
model_name=PatchTST
root_path_name=./dataset/ETT-small/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
label_len=48
e_layers=1
d_layers=1
factor=3
enc_in=7
dec_in=7
c_out=7
des='Exp'
itr=1

# pred_len is passed as the second argument
pred_len=${2}

# Set n_heads based on pred_len
n_heads=16

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
  --itr $itr
