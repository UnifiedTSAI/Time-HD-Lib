#!/bin/bash --login

#SBATCH --exclude=lac-142,lac-343,lac-199
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --gres=gpu:k80:1
#SBATCH --mem=64G

# Load Anaconda environment
source /mnt/home/songyu5/anaconda3/etc/profile.d/conda.sh
conda activate tsf
# conda activate tsf-a100
export PATH="/mnt/home/songyu5/anaconda3/envs/tsf/bin:$PATH"

# Navigate to the project directory
cd /mnt/scratch/songyu5/High-Dimensional-Time-Series

# Check if the correct number of arguments is provided
if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <pred_len> <seq_len> <data> <model>"
  exit 1
fi

# Assign arguments to variables
pred_len="$1"
seq_len="$2"
data="$3"
model="$4"

# Define the results folder using current date (mmdd format) and model name
current_date=$(date +"%m%d")
results_folder="${current_date}_${model}_results"
mkdir -p "${results_folder}"
echo "Results will be saved to: ${results_folder}"

# Run the training script with the specified parameters
accelerate launch run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id "${data}_${seq_len}_${pred_len}" \
  --model "${model}" \
  --data "${data}" \
  --seq_len "${seq_len}" \
  --pred_len "${pred_len}" \
  --batch_size 1 \
  --train_epochs 1 >> "${results_folder}/${data}_${seq_len}_${pred_len}_${model}.out" 2>&1