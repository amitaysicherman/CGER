#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH -c 8
#SBATCH --gres=gpu:L40:1
#SBATCH --array=0-7

export PYTHONPATH=$PYTHONPATH:$(pwd)

split_index=$(($SLURM_ARRAY_TASK_ID))
if [ $split_index -eq 0 ]; then
  python train.py --pooling 0 --bottleneck_dim 0 --size "m" --dropout 0.3
elif [ $split_index -eq 1 ]; then
  python train.py --pooling 1 --bottleneck_dim 0 --size "m" --dropout 0.3
elif [ $split_index -eq 2 ]; then
  python train.py --pooling 0 --bottleneck_dim 64 --size "m" --dropout 0.3
elif [ $split_index -eq 3 ]; then
  python train.py --pooling 1 --bottleneck_dim 64 --size "m" --dropout 0.3
elif [ $split_index -eq 4 ]; then
  python train.py --pooling 0 --bottleneck_dim 0 --size "m" --dropout 0.5
elif [ $split_index -eq 5 ]; then
  python train.py --pooling 0 --bottleneck_dim 128 --size "m" --dropout 0.5
elif [ $split_index -eq 6 ]; then
  python train.py --pooling 1 --bottleneck_dim 0 --size "m" --dropout 0.5
elif [ $split_index -eq 7 ]; then
  python train.py --pooling 1 --bottleneck_dim 128 --size "m" --dropout 0.5
fi
