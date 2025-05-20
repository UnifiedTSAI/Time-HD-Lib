export CUDA_VISIBLE_DEVICES=${1}
model_name=TSMixer
dataset=Solar
model_id_name=Solar
pred_len=${2}
seq_len=${3}

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len $seq_len \
  --label_len 96 \
  --pred_len $pred_len \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --d_model 64 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0001 \
  --lradj type3 \