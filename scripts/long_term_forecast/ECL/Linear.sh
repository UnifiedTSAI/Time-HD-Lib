if [ ! -d "./logs/LongForecasting" ]; then
    mkdir -p ./logs/LongForecasting
fi

model_name=Linear
model_t=iTransformer
dataset=ECL
pred_len=720

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
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
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.01 \
  2>&1 | tee logs/LongForecasting/$model_name'_'ECL_96_$pred_len'.log'