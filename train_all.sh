#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH -c 8
#SBATCH --gres=gpu:L40:1
#SBATCH --array=0-7

export PYTHONPATH=$PYTHONPATH:$(pwd)

split_index=$(($SLURM_ARRAY_TASK_ID))
configs=(
"python train.py --pooling 1 --bottleneck_dim 128 --size 'l' --dropout 0.0 --epochs 100 --cold_fasta 1 --gen_mol 1 --quantize 1"
"python train.py --pooling 1 --bottleneck_dim 128 --size 'l' --dropout 0.0 --epochs 100 --cold_smiles 1 --quantize 1"
"python train.py --pooling 1 --bottleneck_dim 128 --size 'l' --dropout 0.0 --epochs 100 --gen_mol 1 --quantize 1"
"python train.py --pooling 1 --bottleneck_dim 128 --size 'l' --dropout 0.0 --epochs 100 --quantize 1"
"python train.py --pooling 1 --bottleneck_dim 128 --size 'l' --dropout 0.2 --epochs 100 --cold_fasta 1 --gen_mol 1 --quantize 1"
"python train.py --pooling 1 --bottleneck_dim 128 --size 'l' --dropout 0.2 --epochs 100 --cold_smiles 1 --quantize 1"
"python train.py --pooling 1 --bottleneck_dim 128 --size 'l' --dropout 0.2 --epochs 100 --gen_mol 1 --quantize 1"
"python train.py --pooling 1 --bottleneck_dim 128 --size 'l' --dropout 0.2 --epochs 100 --quantize 1"
)

# Get the config for the current index
config=${configs[$split_index]}
# Print the config to be used
echo "Running config: $config"
# Run the config
eval $config
