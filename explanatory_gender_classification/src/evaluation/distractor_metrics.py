"""Distractor analysis metrics for measuring model sensitivity to irrelevant regions."""
import torch
from typing import Dict, Any

def compute_distractor_metrics(
    prediction_before: int,
    prediction_after: int,
    confidence_before: float,
    confidence_after: float,
    logit_female_before: float,
    logit_female_after: float,
    logit_male_before: float,
    logit_male_after: float,
    saliency_before: torch.Tensor,
    saliency_after: torch.Tensor,
    person_mask: torch.Tensor,
    inserted_object_mask: torch.Tensor,
    bsr_before: float,
    bsr_after: float
) -> Dict[str, Any]:
    
    flip_rate = 1 if prediction_before != prediction_after else 0
    delta_confidence = confidence_after - confidence_before
    
    target_logit_before = logit_female_before if prediction_before == 0 else logit_male_before
    target_logit_after = logit_female_after if prediction_before == 0 else logit_male_after
    delta_target_logit = target_logit_after - target_logit_before
    
    # Calculate saliency shifts
    total_saliency_after = saliency_after.sum().item()
    if total_saliency_after > 0:
        saliency_on_inserted = (saliency_after * inserted_object_mask).sum().item() / total_saliency_after
    else:
        saliency_on_inserted = 0.0
        
    total_saliency_before = saliency_before.sum().item()
    saliency_on_person_before = (saliency_before * person_mask).sum().item() / total_saliency_before if total_saliency_before > 0 else 0
    saliency_on_person_after = (saliency_after * person_mask).sum().item() / total_saliency_after if total_saliency_after > 0 else 0
    
    return {
        'prediction_before': prediction_before,
        'prediction_after': prediction_after,
        'confidence_before': confidence_before,
        'confidence_after': confidence_after,
        'logit_female_before': logit_female_before,
        'logit_female_after': logit_female_after,
        'logit_male_before': logit_male_before,
        'logit_male_after': logit_male_after,
        'prediction_flip': flip_rate,
        'delta_confidence': delta_confidence,
        'delta_target_logit': delta_target_logit,
        'saliency_on_inserted_object': saliency_on_inserted,
        'saliency_on_person_before': saliency_on_person_before,
        'saliency_on_person_after': saliency_on_person_after,
        'BSR_before': bsr_before,
        'BSR_after': bsr_after
    }
