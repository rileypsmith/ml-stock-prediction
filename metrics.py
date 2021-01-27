"""
Custom metrics for evaluating model performance.

@author: Riley Smith
Created: 1-12-2021
"""
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred, max_threshold=None):
    """
    Compute MAPE between true and predicted labels. Both should be ndarrays. If
    any of the actual labels are 0, these will not be included in the mean.

    Args:
    --y_true: The ground truth
    --y_pred: The predictions
    --max_threshold: Either None or a float. If a float, all predictions above
        this value are ignored (for handling out of sample predictions)

    Returns:
    A float. The MAPE of the given predictions and ground truth (ignoring all
    predictions above the specified max threshold).    
    """
    if max_threshold is not None:
        # Ignore points below the max threshold (helps with out of sample predictions)
        indices = np.argwhere(y_pred < max_threshold)
        y_true = y_true[indices]
        y_pred = y_pred[indices]
    divided = np.divide(np.abs(y_true - y_pred), y_true, out=np.zeros_like(y_true), where=y_true!=0)
    return divided.sum() / np.count_nonzero(y_true)

def direction_accuracy(y_true, y_pred):
    """
    Measure the direction of accuracy predictions.

    Args:
    --y_true: A 1D ndarray containing 0s (stock went down) and 1s (stock went up)
    --y_pred: A 1D ndarray containing 0s (predicted down) and 1s (predicted up)

    Returns:
    A float. The percentage of correct predictions.
    """
    return (y_true == y_pred).mean()
