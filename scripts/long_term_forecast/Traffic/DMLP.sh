export CUDA_VISIBLE_DEVICES=${1}

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir -p ./logs/LongForecasting
fi

model_name=DMLP
root_path_name=./dataset/traffic/
data_path_name=traffic.csv
dataset=Traffic
seq_len=${8}
pred_len=${2}

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $dataset'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate ${3} \
  --lradj ${4} \
  --d_model ${5} \
  --train_epochs ${6} \
  --norm ${7} \

  # 2>&1 | tee logs/LongForecasting/$model_name'_'$dataset'_'96_$pred_len'.log'