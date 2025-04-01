#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH -c 8
#SBATCH --gres=gpu:L40:1
#SBATCH --array=0-3

export PYTHONPATH=$PYTHONPATH:$(pwd)

split_index=$(($SLURM_ARRAY_TASK_ID))
if [ $split_index -eq 0 ]; then
  python train.py --size "m"
elif [ $split_index -eq 1 ]; then
  python train.py --size "l" --dropout 0.05
elif [ $split_index -eq 2 ]; then
  python train.py --size "l" --dropout 0.2
elif [ $split_index -eq 3 ]; then
  python train.py --size "l" --dropout 0.3
fi