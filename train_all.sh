#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH -c 8
#SBATCH --gres=gpu:L40:1
#SBATCH --array=0-3

level=$1
export PYTHONPATH=$PYTHONPATH:$(pwd)

split_index=$(($SLURM_ARRAY_TASK_ID))
configs=(
"python train.py --constraint 1 --quantize 0"
"python train.py --gen_mol 1 --constraint 1 --quantize 0"
"python train.py --cold_fasta 1 --gen_mol 1 --constraint 1 --quantize 0"
"python train.py --cold_smiles 1 --constraint 1 --quantize 0"
"python train.py"
"python train.py --gen_mol 1"
"python train.py --cold_fasta 1 --gen_mol 1"
"python train.py --cold_smiles 1"
"python train.py --quantize 0"
"python train.py --gen_mol 1 --quantize 0"
"python train.py --cold_fasta 1 --gen_mol 1 --quantize 0"
"python train.py --cold_smiles 1 --quantize 0"
"python train.py --entropy_normalize 0"
"python train.py --gen_mol 1 --entropy_normalize 0"
"python train.py --cold_fasta 1 --gen_mol 1 --entropy_normalize 0"
"python train.py --cold_smiles 1 --entropy_normalize 0"
"python train.py --train_encoder 1"
"python train.py --gen_mol 1 --train_encoder 1"
"python train.py --cold_fasta 1 --gen_mol 1 --train_encoder 1"
"python train.py --cold_smiles 1 --train_encoder 1"
"python train.py --pretrained_encoder 0"
"python train.py --gen_mol 1 --pretrained_encoder 0"
"python train.py --cold_fasta 1 --gen_mol 1 --pretrained_encoder 0"
"python train.py --cold_smiles 1 --pretrained_encoder 0"
"python train.py --random_tgt 1"
"python train.py --gen_mol 1 --random_tgt 1"
"python train.py --cold_fasta 1 --gen_mol 1 --random_tgt 1"
"python train.py --cold_smiles 1 --random_tgt 1"
"python train.py --constraint 1"
"python train.py --constraint 2"
"python train.py --constraint 1 --gen_mol 1"
"python train.py --constraint 2 --gen_mol 1"
"python train.py --constraint 1"
"python train.py --gen_mol 1 --constraint 1"
"python train.py --cold_fasta 1 --gen_mol 1 --constraint 1"
"python train.py --cold_smiles 1 --constraint 1"
)
# Get the config for the current index
config=${configs[$split_index]}
# Print the config to be used
echo "Running config: $config"
# Run the config
eval $config
