#!/usr/bin/env python3
"""
Validates the DrugBank dataset splits according to the splitting rules:

1. Regular mode: All SMILES and FASTA in test/valid appear in train
2. Cold SMILES mode: SMILES in test/valid don't appear in train, but all FASTA do
3. Cold FASTA mode: FASTA in test/valid don't appear in train, but all SMILES do
"""

import os


def load_file(filepath):
    """Load data from a file into a set."""
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return set()

    with open(filepath, 'r') as f:
        return {line.strip() for line in f if line.strip()}


def validate_regular_split():
    """
    Validate regular split:
    - All SMILES and FASTA in test/valid (both pos and neg) appear in train pos
    """
    print("\n=== Testing Regular Split ===")
    base_dir = "data/drugbank"

    # Load training positive samples
    train_pos_smiles = load_file(f"{base_dir}/train_reaction.txt")
    train_pos_fasta = load_file(f"{base_dir}/train_enzyme.txt")

    # Check validation and test sets (both positive and negative)
    for split in ['valid', 'test']:
        for dataset_type in ['reaction', 'reaction_neg']:
            # Check SMILES
            split_smiles = load_file(f"{base_dir}/{split}_{dataset_type}.txt")
            missing_smiles = split_smiles - train_pos_smiles

            if missing_smiles:
                print(
                    f"❌ Found {len(missing_smiles)} SMILES in {split} {dataset_type} that don't appear in train_reaction.txt")
                print(f"    First few examples: {list(missing_smiles)[:3]}")
            else:
                print(f"✅ All SMILES in {split}_{dataset_type}.txt appear in train_reaction.txt")

        for dataset_type in ['enzyme', 'enzyme_neg']:
            # Check FASTA
            split_fasta = load_file(f"{base_dir}/{split}_{dataset_type}.txt")
            missing_fasta = split_fasta - train_pos_fasta

            if missing_fasta:
                print(
                    f"❌ Found {len(missing_fasta)} FASTA in {split} {dataset_type} that don't appear in train_enzyme.txt")
                print(f"    First few examples: {list(missing_fasta)[:3]}")
            else:
                print(f"✅ All FASTA in {split}_{dataset_type}.txt appear in train_enzyme.txt")


def validate_cold_smiles_split():
    """
    Validate cold SMILES split:
    - SMILES in test/valid don't appear in train
    - All FASTA in test/valid appear in train
    """
    print("\n=== Testing Cold SMILES Split ===")
    base_dir = "data/drugbank_cs"

    # Load training data
    train_smiles = load_file(f"{base_dir}/train_reaction.txt")
    train_fasta = load_file(f"{base_dir}/train_enzyme.txt")

    # Check for cold SMILES
    for split in ['valid', 'test']:
        # Check for positive samples
        split_smiles = load_file(f"{base_dir}/{split}_reaction.txt")
        common_smiles = split_smiles.intersection(train_smiles)

        if common_smiles:
            print(f"❌ Found {len(common_smiles)} SMILES in {split}_reaction.txt that also appear in train_reaction.txt")
            print(f"    First few examples: {list(common_smiles)[:3]}")
        else:
            print(f"✅ No SMILES in {split}_reaction.txt appear in train_reaction.txt (correctly cold)")

        # Check for negative samples
        split_smiles_neg = load_file(f"{base_dir}/{split}_reaction_neg.txt")
        common_smiles_neg = split_smiles_neg.intersection(train_smiles)

        if common_smiles_neg:
            print(
                f"❌ Found {len(common_smiles_neg)} SMILES in {split}_reaction_neg.txt that also appear in train_reaction.txt")
            print(f"    First few examples: {list(common_smiles_neg)[:3]}")
        else:
            print(f"✅ No SMILES in {split}_reaction_neg.txt appear in train_reaction.txt (correctly cold)")

    # Check that all FASTA in test/valid appear in train
    for split in ['valid', 'test']:
        for dataset_type in ['enzyme', 'enzyme_neg']:
            split_fasta = load_file(f"{base_dir}/{split}_{dataset_type}.txt")
            missing_fasta = split_fasta - train_fasta

            if missing_fasta:
                print(
                    f"❌ Found {len(missing_fasta)} FASTA in {split}_{dataset_type}.txt that don't appear in train_enzyme.txt")
                print(f"    First few examples: {list(missing_fasta)[:3]}")
            else:
                print(f"✅ All FASTA in {split}_{dataset_type}.txt appear in train_enzyme.txt (correctly non-cold)")


def validate_cold_fasta_split():
    """
    Validate cold FASTA split:
    - FASTA in test/valid don't appear in train
    - All SMILES in test/valid appear in train
    """
    print("\n=== Testing Cold FASTA Split ===")
    base_dir = "data/drugbank_cf"

    # Load training data
    train_smiles = load_file(f"{base_dir}/train_reaction.txt")
    train_fasta = load_file(f"{base_dir}/train_enzyme.txt")

    # Check for cold FASTA
    for split in ['valid', 'test']:
        # Check for positive samples
        split_fasta = load_file(f"{base_dir}/{split}_enzyme.txt")
        common_fasta = split_fasta.intersection(train_fasta)

        if common_fasta:
            print(f"❌ Found {len(common_fasta)} FASTA in {split}_enzyme.txt that also appear in train_enzyme.txt")
            print(f"    First few examples: {list(common_fasta)[:3]}")
        else:
            print(f"✅ No FASTA in {split}_enzyme.txt appear in train_enzyme.txt (correctly cold)")

        # Check for negative samples
        split_fasta_neg = load_file(f"{base_dir}/{split}_enzyme_neg.txt")
        common_fasta_neg = split_fasta_neg.intersection(train_fasta)

        if common_fasta_neg:
            print(
                f"❌ Found {len(common_fasta_neg)} FASTA in {split}_enzyme_neg.txt that also appear in train_enzyme.txt")
            print(f"    First few examples: {list(common_fasta_neg)[:3]}")
        else:
            print(f"✅ No FASTA in {split}_enzyme_neg.txt appear in train_enzyme.txt (correctly cold)")

    # Check that all SMILES in test/valid appear in train
    for split in ['valid', 'test']:
        for dataset_type in ['reaction', 'reaction_neg']:
            split_smiles = load_file(f"{base_dir}/{split}_{dataset_type}.txt")
            missing_smiles = split_smiles - train_smiles

            if missing_smiles:
                print(
                    f"❌ Found {len(missing_smiles)} SMILES in {split}_{dataset_type}.txt that don't appear in train_reaction.txt")
                print(f"    First few examples: {list(missing_smiles)[:3]}")
            else:
                print(f"✅ All SMILES in {split}_{dataset_type}.txt appear in train_reaction.txt (correctly non-cold)")


def count_dataset_samples():
    """Print sample count statistics for all datasets."""
    print("\n=== Dataset Sample Counts ===")
    for name, base_dir in [
        ("Regular", "data/drugbank"),
        ("Cold SMILES", "data/drugbank_cs"),
        ("Cold FASTA", "data/drugbank_cf")
    ]:
        print(f"\n{name} Split:")
        for split in ['train', 'valid', 'test']:
            pos_smiles = load_file(f"{base_dir}/{split}_reaction.txt")
            pos_fasta = load_file(f"{base_dir}/{split}_enzyme.txt")
            neg_smiles = load_file(f"{base_dir}/{split}_reaction_neg.txt")
            neg_fasta = load_file(f"{base_dir}/{split}_enzyme_neg.txt")

            unique_pos_smiles = len(set(pos_smiles))
            unique_pos_fasta = len(set(pos_fasta))

            print(f"  {split.capitalize()}: {len(pos_smiles)} positive samples, {len(neg_smiles)} negative samples")
            print(f"    Unique SMILES: {unique_pos_smiles}, Unique FASTA: {unique_pos_fasta}")


if __name__ == "__main__":
    print("Starting DrugBank split validation...")
    validate_regular_split()
    validate_cold_smiles_split()
    validate_cold_fasta_split()
    count_dataset_samples()
    print("\nValidation complete!")