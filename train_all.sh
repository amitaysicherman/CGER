#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH -c 8
#SBATCH --gres=gpu:L40:1
#SBATCH --array=0-8

level=$1
export PYTHONPATH=$PYTHONPATH:$(pwd)

split_index=$(($SLURM_ARRAY_TASK_ID))
configs=(
"python train.py --level davis"
"python train.py --gen_mol 1  --level davis"
"python train.py --cold_fasta 1 --gen_mol 1  --level davis"
"python train.py --cold_smiles 1  --level davis"
"python train.py --level kiba"
"python train.py --gen_mol 1  --level kiba"
"python train.py --cold_fasta 1 --gen_mol 1  --level kiba"
"python train.py --cold_smiles 1  --level kiba"
)
# Get the config for the current index
config=${configs[$split_index]}
# Print the config to be used
echo "Running config: $config"
# Run the config
eval $config
