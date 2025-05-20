export CUDA_VISIBLE_DEVICES=${1}

seq_len=${3}
model_name=Autoformer

root_path_name=./dataset/Solar/
data_path_name=solar_AL.txt
model_id_name=Solar
pred_len=${2}

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --itr 1