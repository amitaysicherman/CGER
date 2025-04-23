#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH -c 8
#SBATCH --gres=gpu:L40:1
#SBATCH --array=0-2

level=$1
export PYTHONPATH=$PYTHONPATH:$(pwd)

split_index=$(($SLURM_ARRAY_TASK_ID))
configs=(
"python train.py --path_weights_normalize 1"
"python train.py --entropy_normalize 1"
"python train.py --entropy_normalize 1 --path_weights_normalize 1"
)

# Get the config for the current index
config=${configs[$split_index]}
# Print the config to be used
echo "Running config: $config"
# Run the config
eval $config
