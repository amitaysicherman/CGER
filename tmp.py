# -*- coding: utf-8 -*-
"""
@Time:Created on 2020/7/05
@author: Qichang Zhao
Modified to include test set evaluation in each epoch and use optimal thresholds
"""
import random
import os
from model import AttentionDTI
from dataset import CustomDataSet, collate_fn
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
from hyperparameter import hyperparameter
from pytorchtools import EarlyStopping
import timeit
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, auc

DATASET = "DRUGBANK"


def show_result(DATASET, lable, Accuracy_List, F1_List, AUC_List):
    Accuracy_mean, Accuracy_var = np.mean(Accuracy_List), np.var(Accuracy_List)
    F1_mean, F1_var = np.mean(F1_List), np.var(F1_List)
    AUC_mean, AUC_var = np.mean(AUC_List), np.var(AUC_List)
    print("The {} model's results:".format(lable))
    with open("./{}/results.txt".format(DATASET), 'w') as f:
        f.write('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var) + '\n')
        f.write('F1(std):{:.4f}({:.4f})'.format(F1_mean, F1_var) + '\n')
        f.write('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var) + '\n')

    print('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var))
    print('F1(std):{:.4f}({:.4f})'.format(F1_mean, F1_var))
    print('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var))


def load_tensor(file_name, dtype):
    # return [dtype(d).to(hp.device) for d in np.load(file_name + '.npy', allow_pickle=True)]
    return [dtype(d) for d in np.load(file_name + '.npy', allow_pickle=True)]


def find_optimal_threshold(y_true, y_scores, metric='accuracy'):
    """Find the optimal threshold for either accuracy or F1 score"""
    thresholds = np.unique(y_scores)
    best_threshold = 0
    best_score = 0

    for threshold in thresholds:
        y_pred = (np.array(y_scores) >= threshold).astype(int)

        if metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        elif metric == 'f1':
            score = f1_score(y_true, y_pred)

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def evaluate_with_threshold(y_true, y_scores, acc_threshold, f1_threshold):
    """Evaluate performance using optimal thresholds for accuracy and F1"""
    # For AUC
    auc_score = roc_auc_score(y_true, y_scores)

    # For accuracy
    y_pred_acc = (np.array(y_scores) >= acc_threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred_acc)

    # For F1
    y_pred_f1 = (np.array(y_scores) >= f1_threshold).astype(int)
    f1 = f1_score(y_true, y_pred_f1)

    return auc_score, accuracy, f1


def test_precess(model, data_loader, LOSS, acc_threshold=0.5, f1_threshold=0.5):
    model.eval()
    test_losses = []
    Y, S = [], []
    with torch.no_grad():
        pbar = tqdm(enumerate(BackgroundGenerator(data_loader)), total=len(data_loader))
        for i, data in pbar:
            '''data preparation '''
            compounds, proteins, labels = data
            compounds = compounds.cuda()
            proteins = proteins.cuda()
            labels = labels.cuda()

            predicted_scores = model(compounds, proteins)
            loss = LOSS(predicted_scores, labels)
            correct_labels = labels.to('cpu').data.numpy()
            predicted_scores = F.softmax(predicted_scores, 1).to('cpu').data.numpy()
            predicted_scores = predicted_scores[:, 1]  # Get probability for positive class

            Y.extend(correct_labels)
            S.extend(predicted_scores)
            test_losses.append(loss.item())

    # Calculate metrics using optimal thresholds
    AUC = roc_auc_score(Y, S)

    # Apply optimal thresholds
    Y_pred_acc = (np.array(S) >= acc_threshold).astype(int)
    Accuracy = accuracy_score(Y, Y_pred_acc)

    Y_pred_f1 = (np.array(S) >= f1_threshold).astype(int)
    F1 = f1_score(Y, Y_pred_f1)

    test_loss = np.average(test_losses)
    return Y, S, test_loss, AUC, Accuracy, F1


def split_to_dataset(split):
    reaction_file = f"../data/drugbank/{split}_reaction.txt"
    enzyme_file = f"../data/drugbank/{split}_enzyme.txt"
    lines = []
    with open(reaction_file, "r") as f:
        reactions = f.read().splitlines()
    with open(enzyme_file, "r") as f:
        enzymes = f.read().splitlines()
    assert len(reactions) == len(enzymes)
    for i in range(len(reactions)):
        lines.append(f"- - {reactions[i]} {enzymes[i]} 1\n")
    reaction_neg_file = f"../data/drugbank/{split}_reaction_neg.txt"
    enzyme_neg_file = f"../data/drugbank/{split}_enzyme_neg.txt"
    with open(reaction_neg_file, "r") as f:
        reactions = f.read().splitlines()
    with open(enzyme_neg_file, "r") as f:
        enzymes = f.read().splitlines()
    assert len(reactions) == len(enzymes)
    for i in range(len(reactions)):
        lines.append(f"- - {reactions[i]} {enzymes[i]} 0")
    random.shuffle(lines)
    return lines


if __name__ == "__main__":
    """select seed"""
    SEED = 1234
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # torch.backends.cudnn.deterministic = True

    """init hyperparameters"""
    hp = hyperparameter()

    # random shuffle
    AUC_List_stable, Accuracy_List_stable, F1_List_stable = [], [], []

    train_dataset = split_to_dataset("train")
    valid_dataset = split_to_dataset("valid")
    test_dataset = split_to_dataset("test")
    train_dataset = CustomDataSet(train_dataset)
    valid_dataset = CustomDataSet(valid_dataset)
    test_dataset = CustomDataSet(test_dataset)

    train_dataset_load = DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=True, num_workers=0,
                                    collate_fn=collate_fn)
    valid_dataset_load = DataLoader(valid_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,
                                    collate_fn=collate_fn)
    test_dataset_load = DataLoader(test_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,
                                   collate_fn=collate_fn)

    """ create model"""
    model = AttentionDTI(hp).cuda()
    """weight initialize"""
    weight_p, bias_p = [], []
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]

    optimizer = optim.AdamW(
        [{'params': weight_p, 'weight_decay': hp.weight_decay}, {'params': bias_p, 'weight_decay': 0}],
        lr=hp.Learning_rate)

    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.Learning_rate, max_lr=hp.Learning_rate * 10,
                                            cycle_momentum=False,
                                            step_size_up=len(train_dataset))
    Loss = nn.CrossEntropyLoss()

    save_path = "./" + DATASET
    note = ''
    writer = SummaryWriter(log_dir=save_path, comment=note)

    """Output files."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_results = save_path + '/The_results_of_whole_dataset.txt'

    with open(file_results, 'w') as f:
        hp_attr = '\n'.join(['%s:%s' % item for item in hp.__dict__.items()])
        f.write(hp_attr + '\n')

    early_stopping = EarlyStopping(savepath=save_path, patience=hp.Patience, verbose=True, delta=0)

    """Start training."""
    print('Training...')
    start = timeit.default_timer()

    # Initialize best thresholds
    best_acc_threshold = 0.5
    best_f1_threshold = 0.5

    for epoch in range(1, hp.Epoch + 1):
        # Train
        train_pbar = tqdm(
            enumerate(BackgroundGenerator(train_dataset_load)),
            total=len(train_dataset_load))

        train_losses_in_epoch = []
        model.train()
        for train_i, train_data in train_pbar:
            '''data preparation '''
            train_compounds, train_proteins, train_labels = train_data
            train_compounds = train_compounds.cuda()
            train_proteins = train_proteins.cuda()
            train_labels = train_labels.cuda()

            optimizer.zero_grad()

            predicted_interaction = model(train_compounds, train_proteins)
            train_loss = Loss(predicted_interaction, train_labels)
            train_losses_in_epoch.append(train_loss.item())
            train_loss.backward()
            optimizer.step()
            scheduler.step()

        train_loss_a_epoch = np.average(train_losses_in_epoch)
        writer.add_scalar('Train Loss', train_loss_a_epoch, epoch)

        # Validate and find optimal thresholds
        model.eval()
        valid_losses_in_epoch = []
        Y_valid, S_valid = [], []

        with torch.no_grad():
            valid_pbar = tqdm(
                enumerate(BackgroundGenerator(valid_dataset_load)),
                total=len(valid_dataset_load))

            for valid_i, valid_data in valid_pbar:
                valid_compounds, valid_proteins, valid_labels = valid_data
                valid_compounds = valid_compounds.cuda()
                valid_proteins = valid_proteins.cuda()
                valid_labels = valid_labels.cuda()

                valid_scores = model(valid_compounds, valid_proteins)
                valid_loss = Loss(valid_scores, valid_labels)
                valid_labels = valid_labels.to('cpu').data.numpy()
                valid_scores = F.softmax(valid_scores, 1).to('cpu').data.numpy()
                valid_scores = valid_scores[:, 1]  # Probability for positive class

                valid_losses_in_epoch.append(valid_loss.item())
                Y_valid.extend(valid_labels)
                S_valid.extend(valid_scores)

        # Find optimal thresholds on validation set
        best_acc_threshold, best_valid_acc = find_optimal_threshold(Y_valid, S_valid, 'accuracy')
        best_f1_threshold, best_valid_f1 = find_optimal_threshold(Y_valid, S_valid, 'f1')

        # Calculate validation metrics
        valid_auc = roc_auc_score(Y_valid, S_valid)
        valid_loss_a_epoch = np.average(valid_losses_in_epoch)

        # Test on validation set with optimal thresholds
        _, _, _, valid_auc, valid_acc, valid_f1 = test_precess(
            model, valid_dataset_load, Loss, best_acc_threshold, best_f1_threshold)

        # Test on test set with optimal thresholds from validation
        _, S_test, test_loss, test_auc, test_acc, test_f1 = test_precess(
            model, test_dataset_load, Loss, best_acc_threshold, best_f1_threshold)

        # Log all to tensorboard
        writer.add_scalar('Valid Loss', valid_loss_a_epoch, epoch)
        writer.add_scalar('Valid AUC', valid_auc, epoch)
        writer.add_scalar('Valid Best Accuracy', valid_acc, epoch)
        writer.add_scalar('Valid Best F1', valid_f1, epoch)
        writer.add_scalar('Test Loss', test_loss, epoch)
        writer.add_scalar('Test AUC', test_auc, epoch)
        writer.add_scalar('Test Best Accuracy', test_acc, epoch)
        writer.add_scalar('Test Best F1', test_f1, epoch)
        writer.add_scalar('Accuracy Threshold', best_acc_threshold, epoch)
        writer.add_scalar('F1 Threshold', best_f1_threshold, epoch)
        writer.add_scalar('Learn Rate', optimizer.param_groups[0]['lr'], epoch)

        # Print progress
        epoch_len = len(str(hp.Epoch))
        print_msg = (f'[{epoch:>{epoch_len}}/{hp.Epoch:>{epoch_len}}] ' +
                     f'train_loss: {train_loss_a_epoch:.5f} ' +
                     f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                     f'valid_AUC: {valid_auc:.5f} ' +
                     f'valid_best_acc: {valid_acc:.5f} (t={best_acc_threshold:.3f}) ' +
                     f'valid_best_F1: {valid_f1:.5f} (t={best_f1_threshold:.3f})')
        print(print_msg)

        # Print test set results
        test_msg = (f'Test results: ' +
                    f'test_loss: {test_loss:.5f} ' +
                    f'test_AUC: {test_auc:.5f} ' +
                    f'test_best_acc: {test_acc:.5f} ' +
                    f'test_best_F1: {test_f1:.5f}')
        print(test_msg)

        # Early stopping based on validation loss
        early_stopping(valid_loss_a_epoch, model, epoch)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    # Load the best model for final evaluation
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(f"{save_path}/checkpoint.pt"))

    # Final evaluation on test set
    _, S_test, _, test_auc, test_acc, test_f1 = test_precess(
        model, test_dataset_load, Loss, best_acc_threshold, best_f1_threshold)

    print(f"Final evaluation on test set (with best validation thresholds):")
    print(f"AUC: {test_auc:.5f}")
    print(f"Accuracy: {test_acc:.5f} (threshold={best_acc_threshold:.3f})")
    print(f"F1 Score: {test_f1:.5f} (threshold={best_f1_threshold:.3f})")

    # Save final results
    AUC_List_stable.append(test_auc)
    Accuracy_List_stable.append(test_acc)
    F1_List_stable.append(test_f1)

    with open(save_path + "/The_results_of_whole_dataset.txt", 'a') as f:
        f.write("\nFinal Evaluation Results\n")
        f.write(f"AUC: {test_auc:.5f}\n")
        f.write(f"Best Accuracy: {test_acc:.5f} (threshold={best_acc_threshold:.3f})\n")
        f.write(f"Best F1 Score: {test_f1:.5f} (threshold={best_f1_threshold:.3f})\n")

# Show final results
show_result(DATASET, "stable", Accuracy_List_stable, F1_List_stable, AUC_List_stable)