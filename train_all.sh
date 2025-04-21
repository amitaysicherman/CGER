#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH -c 8
#SBATCH --gres=gpu:L40:1
#SBATCH --array=0-7

level=$1
export PYTHONPATH=$PYTHONPATH:$(pwd)

split_index=$(($SLURM_ARRAY_TASK_ID))
configs=(
"python train.py --level $level --pooling 1 --bottleneck_dim 128 --size 'l' --dropout 0.2 --cold_fasta 1 --gen_mol 1 --quantize 1 "
"python train.py --level $level --pooling 1 --bottleneck_dim 128 --size 'l' --dropout 0.2 --cold_smiles 1 --quantize 1"
"python train.py --level $level --pooling 1 --bottleneck_dim 128 --size 'l' --dropout 0.2 --gen_mol 1 --quantize 1"
"python train.py --level $level --pooling 1 --bottleneck_dim 128 --size 'l' --dropout 0.2 --quantize 1"
"python train.py --level $level --pooling 1 --bottleneck_dim 128 --size 'l' --dropout 0.2 --cold_fasta 1 --gen_mol 1"
"python train.py --level $level --pooling 1 --bottleneck_dim 128 --size 'l' --dropout 0.2 --cold_smiles 1"
"python train.py --level $level --pooling 1 --bottleneck_dim 128 --size 'l' --dropout 0.2 --gen_mol 1"
"python train.py --level $level --pooling 1 --bottleneck_dim 128 --size 'l' --dropout 0.2"
)

# Get the config for the current index
config=${configs[$split_index]}
# Print the config to be used
echo "Running config: $config"
# Run the config
eval $config
