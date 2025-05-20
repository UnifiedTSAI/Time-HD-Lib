export CUDA_VISIBLE_DEVICES=${1}

pred_len=${2}
seq_len=${3}
model_name=ModernTCN

root_path_name=./dataset/ETT-small/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=custom

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --ffn_ratio 1 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 \
  --large_size 51 \
  --small_size 5 \
  --dims 64 64 64 64 \
  --head_dropout 0.0 \
  --dropout 0.3 \
  --train_epochs 10 \
  --patience 20 \
  --learning_rate 0.0001