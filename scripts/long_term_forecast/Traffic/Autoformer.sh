export CUDA_VISIBLE_DEVICES=${1}

pred_len=${2}
seq_len=${3}
model_name=Autoformer

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_${seq_len}_${pred_len} \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len ${seq_len} \
  --label_len 48 \
  --pred_len ${pred_len} \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 3