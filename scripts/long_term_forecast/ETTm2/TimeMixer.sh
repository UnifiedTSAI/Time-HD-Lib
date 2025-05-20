export CUDA_VISIBLE_DEVICES=${1}

model_name=TimeMixer

seq_len=${3}
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.001
d_model=32
d_ff=32
batch_size=16
root_path_name=./dataset/ETT-small/
data_path_name=ETTm2.csv
pred_len=${2}

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id ETTm2_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size 128 \
  --learning_rate $learning_rate \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window
