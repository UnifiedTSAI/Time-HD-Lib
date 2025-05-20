export CUDA_VISIBLE_DEVICES=3

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir -p ./logs/LongForecasting
fi

model_name=Linear
dataset=weather
pred_len=96
dataset=weather

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id $dataset'_'96_$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.1 \
  2>&1 | tee logs/LongForecasting/$model_name'_'$dataset'_'96_$pred_len'.log'