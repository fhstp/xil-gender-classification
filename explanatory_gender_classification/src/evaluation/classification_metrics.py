"""Standard classification metrics (accuracy, balanced accuracy, F1, confusion matrix)."""
import torch
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from typing import Dict, List

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute standard classification metrics.
    Assume 0 is female, 1 is male
    """
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    # 0 = female, 1 = male
    # Precision, Recall, F1 for Female
    f_prec = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    f_rec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f_f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    
    # Precision, Recall, F1 for Male
    m_prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    m_rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    m_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    return {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'macro_f1': macro_f1,
        'female_precision': f_prec,
        'female_recall': f_rec,
        'female_f1': f_f1,
        'male_precision': m_prec,
        'male_recall': m_rec,
        'male_f1': m_f1,
        'confusion_matrix': cm.tolist()
    }
