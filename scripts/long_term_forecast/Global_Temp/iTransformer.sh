export CUDA_VISIBLE_DEVICES=${1}

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id ${4}'_'$seq_len'_'$pred_len \
  --model ${5} \
  --data ${4} \
  --features M \
  --seq_len ${3} \
  --pred_len ${2} \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1
