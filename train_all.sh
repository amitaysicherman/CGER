#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH -c 8
#SBATCH --gres=gpu:L40:1
#SBATCH --array=0-15

export PYTHONPATH=$PYTHONPATH:$(pwd)

split_index=$(($SLURM_ARRAY_TASK_ID))
if [ $split_index -eq 0 ]; then
  python train.py --size "xs"
elif [ $split_index -eq 1 ]; then
  python train.py --size "s"
elif [ $split_index -eq 2 ]; then
  python train.py --size "m"
elif [ $split_index -eq 3 ]; then
  python train.py --size "l"
elif [ $split_index -eq 4 ]; then
  python train.py --size "xl"
elif [ $split_index -eq 5 ]; then
  python train.py --size "xs" --level "medium"
elif [ $split_index -eq 6 ]; then
  python train.py --size "s" --level "medium"
elif [ $split_index -eq 7 ]; then
  python train.py --size "m" --level "medium"
elif [ $split_index -eq 8 ]; then
  python train.py --size "l" --level "medium"
elif [ $split_index -eq 9 ]; then
  python train.py --size "xl" --level "medium"
elif [ $split_index -eq 10 ]; then
  python train.py --size "xs" --level "hard"
elif [ $split_index -eq 11 ]; then
  python train.py --size "s" --level "hard"
elif [ $split_index -eq 12 ]; then
  python train.py --size "m" --level "hard"
elif [ $split_index -eq 13 ]; then
  python train.py --size "l" --level "hard"
elif [ $split_index -eq 14 ]; then
  python train.py --size "xl" --level "hard"
fi