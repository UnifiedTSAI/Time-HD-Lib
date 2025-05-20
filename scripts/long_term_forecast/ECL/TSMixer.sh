export CUDA_VISIBLE_DEVICES=${1}
model_name=TSMixer
dataset=ECL
model_id_name=ECL
pred_len=${2}
seq_len=${3}

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 96 \
  --pred_len $pred_len \
  --e_layers 6 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --d_model 64 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.0001 \
  --lradj type3 \