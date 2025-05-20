export CUDA_VISIBLE_DEVICES=${1}

seq_len=${3}
model_name=FEDformer

root_path_name=./dataset/traffic/
data_path_name=traffic.csv
model_id_name=Traffic
data_name=custom
pred_len=${2}

python -u run.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 4 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1