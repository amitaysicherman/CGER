import re, sys, os
from utils import utils
from utils.preprocess import PREPROCESS
import numpy as np
import pandas as pd
import torch
import warnings

warnings.filterwarnings("ignore", category=Warning, module="torchvision")
warnings.filterwarnings('ignore')
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Any, List, Optional, Tuple
from pytorch_lightning.loggers import TensorBoardLogger
import ray
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune
import yaml
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
import multiprocessing
from multiprocessing import Manager

os.environ["TOKENIZERS_PARALLELISM"] = "false"

"""
define device for featurizer that's outside the lightning module
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# device = 'cpu'
def train(cfg, dataset, shared_metrics, tune=False):
    tb_logger = TensorBoardLogger('../logger_DTI/tb_logs', name=cfg['logger']['name'])

    # Init datamodule
    data_module: LightningModule = hydra.utils.instantiate(
        cfg['datamodule'], cfg, dataset, _recursive_=False
    )

    # Init lightning model
    model: LightningModule = hydra.utils.instantiate(
        cfg['module'], cfg, dataset, _recursive_=False
    )

    # Init callbacks (early stopping, checkpointing)
    callbacks: List[Callback] = utils.instantiate_callbacks(
        cfg['callbacks']
    )

    if tune:
        metrics = {"auc": "val_auc", "loss": "val_loss"}
        callbacks.append(TuneReportCallback(metrics, on="validation_end"))
        trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=cfg['trainer']['max_epochs'], logger=tb_logger,
                             callbacks=callbacks, enable_progress_bar=False)
        trainer.fit(model, data_module)
    else:
        trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=cfg['trainer']['max_epochs'], logger=tb_logger,
                             callbacks=callbacks, log_every_n_steps=5)
        trainer.fit(model, data_module)
        trainer.validate(model, data_module)
        trainer.test(model, data_module)
        shared_metrics["test_auc"] += [model.test_auc.item()]
        shared_metrics["test_auprc"] += [model.test_auprc.item()]
        shared_metrics["test_f1"] += [model.test_f1.item()]


def load_numpy_data(protein_path, molecule_path, labels_path, split_paths=None):
    """
    Load dataset from numpy arrays

    Args:
        protein_path: Path to protein features numpy array
        molecule_path: Path to molecule features numpy array
        labels_path: Path to labels numpy array
        split_paths: Dictionary with paths to train/valid/test split indices (optional)

    Returns:
        X_drug: DataFrame with molecule features
        X_target: DataFrame with protein features
        DTI: DataFrame with interaction data or dictionary of DataFrames if split provided
    """
    # Load the numpy arrays
    protein_features = np.load(protein_path)
    molecule_features = np.load(molecule_path)
    labels = np.load(labels_path)

    # Create DataFrames for proteins and molecules
    protein_ids = [f"protein_{i}" for i in range(protein_features.shape[0])]
    molecule_ids = [f"molecule_{i}" for i in range(molecule_features.shape[0])]

    X_target = pd.DataFrame(protein_features, index=protein_ids)
    X_drug = pd.DataFrame(molecule_features, index=molecule_ids)

    # If we don't have predefined splits
    if split_paths is None:
        # Create DTI DataFrame with all interactions
        interactions = []
        for i, protein_id in enumerate(protein_ids):
            for j, molecule_id in enumerate(molecule_ids):
                label_idx = i * len(molecule_ids) + j
                if label_idx < len(labels):
                    interactions.append({
                        'Drug': molecule_id,
                        'Target': protein_id,
                        'Label': int(labels[label_idx])
                    })

        DTI = pd.DataFrame(interactions)
        return X_drug, X_target, DTI
    else:
        # Load the split indices
        train_indices = np.load(split_paths['train'])
        valid_indices = np.load(split_paths['valid'])
        test_indices = np.load(split_paths['test'])

        # Create separate DTI DataFrames for each split
        DTI = {}

        # Helper function to create DTI DataFrame from indices
        def create_dti_from_indices(indices):
            interactions = []
            for idx in indices:
                # Convert flattened index to protein and molecule indices
                # This assumes a specific ordering - adjust as needed for your data
                protein_idx = idx // len(molecule_ids)
                molecule_idx = idx % len(molecule_ids)

                if protein_idx < len(protein_ids) and molecule_idx < len(molecule_ids):
                    interactions.append({
                        'Drug': molecule_ids[molecule_idx],
                        'Target': protein_ids[protein_idx],
                        'Label': int(labels[idx])
                    })

            return pd.DataFrame(interactions)

        # Create DTI DataFrames for each split
        for i, (split_name, indices) in enumerate(zip(['train', 'valid', 'test'],
                                                      [train_indices, valid_indices, test_indices])):
            DTI[i] = create_dti_from_indices(indices)

        return X_drug, X_target, DTI


_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": "configs",
    "config_name": "config-name",
}


@hydra.main(**_HYDRA_PARAMS)
def main(cfg) -> Optional[float]:
    logger, logger_dir = utils.get_logger(OmegaConf.to_container(cfg))
    new_dir = logger_dir.split('run')[0]

    # Load data from numpy arrays instead of using PREPROCESS
    # Adjust paths to your numpy files
    numpy_data_paths = {
        'protein': 'path/to/protein_features.npy',
        'molecule': 'path/to/molecule_features.npy',
        'labels': 'path/to/labels.npy',
        'splits': {
            'train': 'path/to/train_indices.npy',
            'valid': 'path/to/valid_indices.npy',
            'test': 'path/to/test_indices.npy'
        }
    }

    # Load numpy data
    X_drug, X_target, DTI = load_numpy_data(
        numpy_data_paths['protein'],
        numpy_data_paths['molecule'],
        numpy_data_paths['labels'],
        numpy_data_paths['splits']
    )

    # The code below remains mostly the same
    cfg = OmegaConf.to_container(cfg)

    manager = Manager()
    shared_metrics = manager.dict()
    shared_metrics["test_auc"], shared_metrics["test_auprc"], shared_metrics["test_f1"] = [], [], []

    if cfg['tuning']['param_search']['tune']:
        optuna_search = OptunaSearch(
            metric="auc",
            mode="max",
        )

        asha_scheduler = ASHAScheduler(
            time_attr='training_iteration',
            metric='auc',
            mode='max',
            max_t=100,
            grace_period=10,
        )

        # check if DTI is a dict, it will be for other presplitted datasets
        if isinstance(DTI, dict):
            DTI = DTI[0]
        dataset = utils.get_dataset(cfg, X_drug, X_target, DTI)
        cfg = utils.setup_config_tune(cfg['tuning']['param_search']['search_space'], cfg)
        ray.init()
        trainable = tune.with_parameters(train, dataset=dataset, shared_metrics=None, tune=True)
        analysis = tune.run(
            trainable,
            local_dir="/data/tanvir/DTI",
            resources_per_trial={"gpu": 0.5},
            config=cfg,
            num_samples=100,
            search_alg=optuna_search,
            scheduler=asha_scheduler,
        )
        best_trial = analysis.get_best_trial("auc", mode="max")
        print("Best Hyperparameters:")
        print(best_trial.config)
        train(best_trial.config, dataset, shared_metrics)
        ray.shutdown()
    else:
        """
        set best parameters file for the experiment and update model configs from that file 
        """
        cfg = utils.update_best_param(cfg)
        if cfg['multiprocessing']['multiprocessing']:
            X_drug_orig, X_target_orig, DTI_orig = X_drug.copy(), X_target.copy(), DTI.copy()
            dataset = {}
            for num in range(cfg['multiprocessing']['num_process']):
                if isinstance(DTI_orig, dict):
                    X_drug, X_target, DTI = X_drug_orig.copy(), X_target_orig.copy(), DTI_orig[num].copy()
                else:
                    X_drug, X_target, DTI = X_drug_orig.copy(), X_target_orig.copy(), DTI_orig.copy()

                dataset[num] = utils.get_dataset(cfg, X_drug, X_target, DTI, ddi=None, skipped=None)
            # find tst_ind from all dataset
            test_ind = []
            for num in range(cfg['multiprocessing']['num_process']):
                test_ind += dataset[num]['test'].label.tolist()
            print(np.unique(test_ind, return_counts=True))

            processes = []
            for num in range(cfg['multiprocessing']['num_process']):
                p = multiprocessing.Process(target=train, args=(cfg, dataset[num], shared_metrics))
                processes.append(p)

                if len(processes) >= cfg['multiprocessing']['concurrent_process']:
                    for batch_process in processes:
                        batch_process.start()
                    for batch_process in processes:
                        batch_process.join()
                    processes = []

            for p in processes:
                p.start()
            for p in processes:
                p.join()

            print("All processes have finished.")

            logger.info(shared_metrics["test_auc"])
            logger.info(shared_metrics["test_auprc"])
            logger.info(shared_metrics["test_f1"])
            logger.info(f'Mean test AUC: {np.mean(shared_metrics["test_auc"]):.4f}')
            logger.info(f'Mean test AUPRC: {np.mean(shared_metrics["test_auprc"]):.4f}')
            logger.info(f'Mean test F1: {np.mean(shared_metrics["test_f1"]):.4f}')
            new_dir = f'{new_dir}/run_{np.mean(shared_metrics["test_auc"]):.4f}/'
            os.rename(logger_dir, new_dir)

        else:
            pl.seed_everything(seed=42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            dataset = utils.get_dataset(cfg, X_drug, X_target, DTI, ddi=None, skipped=None)
            train(cfg, dataset, shared_metrics)

            test_auc = [shared_metrics["test_auc"]]
            test_auprc = [shared_metrics["test_auprc"]]
            test_f1 = [shared_metrics["test_f1"]]
            logger.info(f'Test AUC: {np.mean(test_auc):.4f}')
            logger.info(f'Test AUPRC: {np.mean(test_auprc):.4f}')
            logger.info(f'Test F1: {np.mean(test_f1):.4f}')
            new_dir = f'{new_dir}/run_{np.mean(test_auc):.4f}/'
            os.rename(logger_dir, new_dir)


if __name__ == "__main__":
    main()