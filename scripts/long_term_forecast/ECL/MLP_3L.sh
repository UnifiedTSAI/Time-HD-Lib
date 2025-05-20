export CUDA_VISIBLE_DEVICES=${1}
if [ ! -d "./logs/LongForecasting" ]; then
    mkdir -p ./logs/LongForecasting
fi

model_name=MLP_3L
model_t=iTransformer
pred_len=96
dataset=ECL
dmodel=512

python -u run.py \
  --batch_size 32 \
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
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --d_model $dmodel \
  --train_epochs 100 \
  --lradj 'type3' \
  # 2>&1 | tee logs/LongForecasting/$model_name'_'$model_t'_'$dataset'_'$dmodel'_'96_$pred_len'_'$alpha'_'$beta'.log'
