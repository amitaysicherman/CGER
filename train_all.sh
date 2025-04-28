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
#configs=(
#"python train.py --size s --bottleneck_dim 0 --pooling 0 --train_encoder 1 --pretrained_encoder 0 --quantize 0 --level biosnap"
#"python train.py --size m --bottleneck_dim 0 --pooling 0 --train_encoder 1 --pretrained_encoder 0 --quantize 0 --level biosnap"
#"python train.py --size l --bottleneck_dim 0 --pooling 0 --train_encoder 1 --pretrained_encoder 0 --quantize 0 --level biosnap"
#"python train.py --size s --bottleneck_dim 0 --pooling 0 --train_encoder 1 --pretrained_encoder 0 --quantize 1 --level biosnap"
#"python train.py --size m --bottleneck_dim 0 --pooling 0 --train_encoder 1 --pretrained_encoder 0 --quantize 1 --level biosnap"
#"python train.py --size l --bottleneck_dim 0 --pooling 0 --train_encoder 1 --pretrained_encoder 0 --quantize 1 --level biosnap"
#"python train.py --size s --bottleneck_dim 128 --pooling 1 --train_encoder 1 --pretrained_encoder 0 --quantize 0 --level biosnap"
#"python train.py --size m --bottleneck_dim 128 --pooling 1 --train_encoder 1 --pretrained_encoder 0 --quantize 0 --level biosnap"
#"python train.py --size l --bottleneck_dim 128 --pooling 1 --train_encoder 1 --pretrained_encoder 0 --quantize 0 --level biosnap"
#"python train.py --size s --bottleneck_dim 128 --pooling 1 --train_encoder 1 --pretrained_encoder 0 --quantize 1 --level biosnap"
#"python train.py --size m --bottleneck_dim 128 --pooling 1 --train_encoder 1 --pretrained_encoder 0 --quantize 1 --level biosnap"
#"python train.py --size l --bottleneck_dim 128 --pooling 1 --train_encoder 1 --pretrained_encoder 0 --quantize 1 --level biosnap"
#)
configs=(
"python train.py --auto_pretrained 1 --gen_mol 1"
"python train.py --auto_pretrained 1 --gen_mol 0"
)
# Get the config for the current index
config=${configs[$split_index]}
# Print the config to be used
echo "Running config: $config"
# Run the config
eval $config
