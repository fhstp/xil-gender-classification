"""Fairness metrics including per-class error rates and error-rate gap."""
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Dict

def compute_fairness_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute fairness and bias metrics.
    Assume 0 is female, 1 is male
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        # tn = True Female, fp = False Male (Female misclassified as Male)
        # fn = False Female (Male misclassified as Female), tp = True Male
        
        female_total = tn + fp
        male_total = fn + tp
        
        female_error_rate = fp / female_total if female_total > 0 else 0
        male_error_rate = fn / male_total if male_total > 0 else 0
        
        female_fnr = fp / female_total if female_total > 0 else 0 # actually FNR for female means labeled male but is female
        male_fnr = fn / male_total if male_total > 0 else 0 # labeled female but is male
        
        female_fpr = fn / male_total if male_total > 0 else 0 # male labeled as female
        male_fpr = fp / female_total if female_total > 0 else 0 # female labeled as male
        
        tpr_gap = abs((tn / female_total if female_total > 0 else 0) - (tp / male_total if male_total > 0 else 0))
        fpr_gap = abs(female_fpr - male_fpr)
        
        total_errors = fp + fn
        balanced_misclassification_ratio = fp / total_errors if total_errors > 0 else 0.5
        
        return {
            'female_error_rate': female_error_rate,
            'male_error_rate': male_error_rate,
            'error_rate_gap': abs(female_error_rate - male_error_rate),
            'female_false_negative_rate': female_fnr,
            'male_false_negative_rate': male_fnr,
            'female_false_positive_rate': female_fpr,
            'male_false_positive_rate': male_fpr,
            'TPR_gap': tpr_gap,
            'FPR_gap': fpr_gap,
            'balanced_misclassification_ratio': balanced_misclassification_ratio
        }
    else:
        return {}
