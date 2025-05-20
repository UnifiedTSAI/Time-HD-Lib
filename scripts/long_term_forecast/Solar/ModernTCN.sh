export CUDA_VISIBLE_DEVICES=${1}

pred_len=${2}
seq_len=${3}
model_name=ModernTCN

root_path_name=./dataset/Solar/
data_path_name=solar_AL.txt
model_id_name=Solar

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len $seq_len \
  --pred_len ${pred_len} \
  --e_layers 2 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1 \
  --ffn_ratio 1 \
  --patch_size 8 \
  --patch_stride 4 \
  --num_blocks 1 \
  --large_size 51 \
  --small_size 5 \
  --dims 64 64 64 64 \
  --head_dropout 0.0 \
  --dropout 0.1 \
  --train_epochs 10 \
  --patience 5 \
  --learning_rate 0.0025