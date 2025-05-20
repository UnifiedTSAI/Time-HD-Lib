export CUDA_VISIBLE_DEVICES=${1}

pred_len=${2}
seq_len=${3}
model_name=ModernTCN

root_path_name=./dataset/electricity/
data_path_name=electricity.csv
model_id_name=ECL

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
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --dropout 0.9 \
  --train_epochs 10 \
  --patience 10 \
  --learning_rate 0.0001 \
  --ffn_ratio 8 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 \
  --large_size 51 \
  --small_size 5 \
  --dims 64 64 64 64 \
  --head_dropout 0.0 