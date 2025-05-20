export CUDA_VISIBLE_DEVICES=${1}

seq_len=96
model_name=SCINet

root_path_name=./dataset/ETT-small/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=custom
pred_len=${2}

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --learning_rate 1e-5 \
  --batch_size 128 \
  --dropout 0.5 \
  --itr 1