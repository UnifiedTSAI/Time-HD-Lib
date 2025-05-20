#!/bin/bash
# export NCCL_DEBUG=INFO        # 启用 NCCL 调试信息
# export NCCL_IB_DISABLE=1      # 禁用 InfiniBand 通信
# export NCCL_P2P_DISABLE=1     # 禁用点对点通信
# export NCCL_SHM_DISABLE=1     # 禁用共享内存通道
# export NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo.json  # 输出 NCCL 拓扑信息

# export CUDA_VISIBLE_DEVICES=4,5,6,7

# 定义模型列表
MODELS=("UCast")

# 定义数据集列表
DATASETS=(
  "atec"
  "china_air_quality"
  "global_temp"
  "global_wind"
  "google_community_mobility"
  "m5"
  "measles_england"
  "sirs"
  "sp500"
  "wikipedia_web_traffic_20000"
  "largest_ca"
  "largest_gba"
  "largest_gla"
  "smart_meters_in_london"
  "nrel_solar_power"
  "neurolib"
)

# 常用参数
TASK="long_term_forecast"
IS_TRAINING=1
MODEL_ID="search"
FEATURES="M"
TRAIN_EPOCHS=100
BATCH_SIZE=32

# 定义日志目录
LOG_DIR="hpo_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p $LOG_DIR

# 创建主日志文件
MAIN_LOG="${LOG_DIR}/hpo_run_${TIMESTAMP}.log"
echo "Starting HPO runs at $(date)" | tee $MAIN_LOG
echo "Models: ${MODELS[*]}" | tee -a $MAIN_LOG
echo "Datasets: ${DATASETS[*]}" | tee -a $MAIN_LOG
echo "" | tee -a $MAIN_LOG

# 循环遍历每个模型和数据集
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        # 创建针对此次运行的日志文件
        RUN_LOG="${LOG_DIR}/${MODEL}_${DATASET}_${TIMESTAMP}.log"
        
        echo "=====================================" | tee -a $MAIN_LOG
        echo "Running hyperparameter search for model: $MODEL, dataset: $DATASET" | tee -a $MAIN_LOG
        echo "Log file: $RUN_LOG" | tee -a $MAIN_LOG
        echo "=====================================" | tee -a $MAIN_LOG
        
        # 执行超参数搜索并将输出重定向到日志文件
        {
            echo "Run started at: $(date)"
            echo "Command: accelerate launch run.py --task_name $TASK --is_training $IS_TRAINING --model_id $MODEL_ID --model $MODEL --data $DATASET --features $FEATURES --train_epochs $TRAIN_EPOCHS --batch_size $BATCH_SIZE --hyper_parameter_searching"
            echo ""
            
            # 执行命令并捕获输出
            accelerate launch run.py \
                --task_name $TASK \
                --is_training $IS_TRAINING \
                --model_id $MODEL_ID \
                --model $MODEL \
                --data $DATASET \
                --features $FEATURES \
                --train_epochs $TRAIN_EPOCHS \
                --batch_size $BATCH_SIZE \
                --hyper_parameter_searching
                
            echo ""
            echo "Run completed at: $(date)"
        } 2>&1 | tee $RUN_LOG
        
        # 在主日志中记录完成信息
        echo "Completed $MODEL on $DATASET at $(date). See $RUN_LOG for details." | tee -a $MAIN_LOG
        echo "Waiting 5 seconds before next run..." | tee -a $MAIN_LOG
        sleep 5
    done
done

echo "All hyperparameter searches completed at $(date)!" | tee -a $MAIN_LOG
echo "Main log file: $MAIN_LOG"
