
# first argument is the dataset name
ds=$1

python prep_drug_bank.py --ds $ds --cold_smiles 0 --cold_fasta 0
python prep_drug_bank.py --ds $ds --cold_smiles 0 --cold_fasta 1
python prep_drug_bank.py --ds $ds --cold_smiles 1 --cold_fasta 0

python prep_rvq.py --ds $ds
python prep_rvq.py --ds $ds --is_molecules