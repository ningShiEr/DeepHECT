import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score, matthews_corrcoef
import numpy as np

def compute_metrics(y_true, y_pred_prob, y_pred_label):
    acc = accuracy_score(y_true, y_pred_label)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_label, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_pred_prob)
    except:
        auc = 0.5
    auprc = average_precision_score(y_true, y_pred_prob)
    mcc = matthews_corrcoef(y_true, y_pred_label)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'auprc': auprc,
        'mcc': mcc
    }

def get_scheduler(optimizer, config):
    if config.lr_scheduler is None:
        return None
    elif config.lr_scheduler == 'StepLR':
        return lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    elif config.lr_scheduler == 'ReduceLROnPlateau':
        return lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=config.patience,
            factor=config.gamma, min_lr=config.min_lr
        )
    elif config.lr_scheduler == 'CosineAnnealing':
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)
    elif config.lr_scheduler == 'CosineAnnealingWarmRestarts':
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=config.min_lr)
    else:
        raise ValueError(f"未知调度器: {config.lr_scheduler}")