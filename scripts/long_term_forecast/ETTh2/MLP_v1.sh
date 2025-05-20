export CUDA_VISIBLE_DEVICES=${1}

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir -p ./logs/LongForecasting
fi

model_name=MLP_v1
root_path_name=./dataset/ETT-small/
data_path_name=ETTh2.csv
dataset=ETTh2
seq_len=96
pred_len=${2}

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $dataset'_'$seq'_'$pred_len \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate ${3} \
  --lradj ${4} \
  --d_model ${5} \
  --train_epochs ${6}

  # 2>&1 | tee logs/LongForecasting/$model_name'_'$dataset'_'96_$pred_len'.log'