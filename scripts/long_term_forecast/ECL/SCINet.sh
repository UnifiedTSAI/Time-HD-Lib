export CUDA_VISIBLE_DEVICES=${1}

seq_len=96
model_name=SCINet

root_path_name=./dataset/electricity/
data_path_name=electricity.csv
model_id_name=ECL
pred_len=${2}

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --d_layers 2 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --batch_size 32 \
  --learning_rate 9e-4 \
  --dropout 0 \
  --itr 1
