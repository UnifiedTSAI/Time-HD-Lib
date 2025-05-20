export CUDA_VISIBLE_DEVICES=${1}

model_name=TimeMixer

seq_len=${3}
pred_len=${2}
e_layers=3
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.001
d_model=32
d_ff=64
batch_size=4
root_path=./dataset/traffic/
data_path=traffic.csv

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id Traffic_${seq_len}_${pred_len} \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window
