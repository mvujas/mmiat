from sklearn.metrics import roc_curve, roc_auc_score

from typing import Any

from mmiat.evaluation.metrics import average_case_accuracy


def calculate_miametrics(membership_labels, confidences) -> dict[str, Any]:
    """
    Calculates standard metrics for membership-inference attacks.

    Args:
        membership_labels (array-like): Labels indicating whether the data point is a member (1) or non-member (0).
        confidences (array-like): Target scores, can either be probability estimates of the positive class, 
                                    confidence values, or non-thresholded measure of decisions.

    Returns:
        dict: A dictionary containing the following keys:
            - "fpr" (array): False Positive Rate.
            - "tpr" (array): True Positive Rate.
            - "auc" (float): Area Under the ROC Curve.
            - "balanced_accuracy" (float): Balanced accuracy score.
    """
    fpr, tpr, _ = roc_curve(membership_labels, confidences)
    auc = roc_auc_score(membership_labels, confidences).item()
    accuracy = average_case_accuracy(membership_labels, confidences).item()
    return {
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc,
        "accuracy": accuracy
    }