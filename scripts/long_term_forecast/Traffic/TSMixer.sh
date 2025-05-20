export CUDA_VISIBLE_DEVICES=${1}
model_name=TSMixer
dataset=Traffic
model_id_name=Traffic
pred_len=${2}
seq_len=${3}

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 96 \
  --pred_len $pred_len \
  --e_layers 8 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --d_model 64 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0001 \
  --lradj type3 \