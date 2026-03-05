"""Metrics utilities for accuracy, precision, recall, F1 using scikit-learn."""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score as sklearn_f1_score

def accuracy(y_true, y_pred):
    """
    Compute accuracy score.
    Args:
        y_true (list): Ground truth (correct) labels.
        y_pred (list): Predicted labels, as returned by a classifier.
    Returns:
        float: Accuracy score.
    """
    return accuracy_score(y_true, y_pred)


def precision(y_true, y_pred, average='binary'):
    """
    Compute precision score.
    Args:
        y_true (list): Ground truth (correct) labels.
        y_pred (list): Predicted labels.
        average (str): Averaging method ('binary', 'micro', 'macro', 'weighted').
    Returns:
        float: Precision score.
    """
    return precision_score(y_true, y_pred, average=average, zero_division=0)


def recall(y_true, y_pred, average='binary'):
    """
    Compute recall score.
    Args:
        y_true (list): Ground truth (correct) labels.
        y_pred (list): Predicted labels.
        average (str): Averaging method ('binary', 'micro', 'macro', 'weighted').
    Returns:
        float: Recall score.
    """
    return recall_score(y_true, y_pred, average=average, zero_division=0)


def f1_score(y_true, y_pred, average='binary'):
    """
    Compute F1 score.
    Args:
        y_true (list): Ground truth (correct) labels.
        y_pred (list): Predicted labels.
        average (str): Averaging method ('binary', 'micro', 'macro', 'weighted').
    Returns:
        float: F1 score.
    """
    return sklearn_f1_score(y_true, y_pred, average=average, zero_division=0)
