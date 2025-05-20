export CUDA_VISIBLE_DEVICES=${1}

model_name=TimeMixer
seq_len=${3}
pred_len=${2}
down_sampling_layers=2
down_sampling_window=2
learning_rate=0.001
batch_size=32
train_epochs=10
patience=3

root_path=./dataset/Solar/
data_path=solar_AL.txt

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id Solar_${seq_len}_$pred_len \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 1 \
  --use_norm 0 \
  --d_model 512 \
  --d_ff 2048 \
  --channel_independence 0 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window
