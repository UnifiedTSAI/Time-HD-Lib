export CUDA_VISIBLE_DEVICES=${1}

model_name=Autoformer
pred_len=${2}
seq_len=${3}

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_${seq_len}_${pred_len} \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len ${seq_len} \
  --label_len 48 \
  --pred_len ${pred_len} \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1