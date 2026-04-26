import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, roc_auc_score,
                             average_precision_score)

def compute_metrics(y_true, y_prob, y_pred):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob),
        'auprc': average_precision_score(y_true, y_prob)
    }
    return metrics