# if [ ! -d "./logs" ]; then
#     mkdir ./logs
# fi

# if [ ! -d "./logs/LongForecasting" ]; then
#     mkdir ./logs/LongForecasting
# fi

model_name=Pathformer

root_path_name=./dataset/electricity
data_path_name=electricity.csv
model_id_name=ECL
data_name=custom
seq_len=96

# for pred_len in 96 192 336 720
for pred_len in 96
do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --enc_in 321 \
      --layer_nums 3 \
      --residual_connection 1\
      --k 2\
      --d_model 16 \
      --d_ff 128 \
      --patch_size_list 16 12 8 32 12 8 6 4 8 6 4 2 \
      --train_epochs 100\
      --patience 3 \
      --pct_start 0.2 \
      --itr 1 \
      --batch_size 16 \
      --learning_rate 0.001 \
      --lradj 'TST' \
      --gpu 0 \
      --save_cp True
    #   >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log

done