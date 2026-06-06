"""
evaluate.py — Evaluation utilities.
"""
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, classification_report


def predict(model, X):
    """Return binary predictions (0/1)."""
    probs = model(tf.constant(X, dtype=tf.float32), training=False).numpy().flatten()
    return (probs >= 0.5).astype(int)


def accuracy(model, client_data_or_xy):
    """
    Compute accuracy on either:
      - a list of (X, y) client tuples  (concatenates all)
      - a single (X, y) tuple
    """
    if isinstance(client_data_or_xy, list):
        X = np.concatenate([c[0] for c in client_data_or_xy])
        y = np.concatenate([c[1] for c in client_data_or_xy])
    else:
        X, y = client_data_or_xy
    return accuracy_score(y.astype(int), predict(model, X))


def full_metrics(model, X_test, y_test):
    """
    Return dict with accuracy, f1, and print classification report.
    """
    y_pred = predict(model, X_test)
    y_true = y_test.astype(int)
    metrics = {
        'accuracy':  accuracy_score(y_true, y_pred),
        'f1':        f1_score(y_true, y_pred, average='binary', zero_division=0),
    }
    print(classification_report(y_true, y_pred, zero_division=0))
    return metrics
