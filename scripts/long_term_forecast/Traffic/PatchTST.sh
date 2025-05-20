#!/bin/bash

export CUDA_VISIBLE_DEVICES=${1}

seq_len=${3}
label_len=48
model_name=PatchTST
root_path_name=./dataset/traffic/
data_path_name=traffic.csv
model_id_name=Traffic
data_name=custom
factor=3
enc_in=862
dec_in=862
c_out=862
d_model=512
d_ff=512
top_k=5
des='Exp'
batch_size=4
itr=1
e_layers=2
d_layers=1

# pred_len is passed as the second argument
pred_len=${2}

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
  --d_model $d_model \
  --d_ff $d_ff \
  --top_k $top_k \
  --des $des \
  --batch_size $batch_size \
  --itr $itr
