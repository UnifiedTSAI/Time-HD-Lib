#!/bin/bash --login

#SBATCH --exclude=lac-142,lac-343,lac-199
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=100G

# Load Anaconda environment
source /mnt/scratch/yangji73/anaconda3/etc/profile.d/conda.sh
conda activate tsf
export PATH="/mnt/scratch/yangji73/anaconda3/envs/tsf/bin:$PATH"

# Navigate to the project directory
cd /mnt/scratch/yangji73/juntong/Time-HD-Lib

# Assign arguments to variables
model="$1"
data="$2"

accelerate launch --num_processes=4 run.py --model "${model}" --data "${data}" --gpu 0,1,2,3 --hyper_parameter_searching
