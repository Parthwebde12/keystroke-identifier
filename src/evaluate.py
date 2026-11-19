import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_eer(y_true, scores):
    """Equal Error Rate: the point where false accepts equal false rejects."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2)


def compute_auc(y_true, scores):
    return float(roc_auc_score(y_true, scores))