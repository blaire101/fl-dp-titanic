"""
dp_sgd.py — Gaussian mechanism for (epsilon, delta)-Differential Privacy.

sigma = sqrt(2 * ln(1.25 / delta)) * C / epsilon
"""
import numpy as np
import tensorflow as tf


def compute_sigma(epsilon: float, delta: float, clip_norm: float) -> float:
    """
    Compute noise standard deviation for Gaussian mechanism.

    Args:
        epsilon:   Privacy budget (smaller = stronger privacy).
        delta:     Failure probability (e.g. 1e-5).
        clip_norm: Gradient clipping threshold C.

    Returns:
        sigma: Standard deviation of Gaussian noise to add.
    """
    sigma = (np.sqrt(2 * np.log(1.25 / delta)) * clip_norm) / epsilon
    return float(sigma)


def clip_gradients(gradients, clip_norm: float):
    """
    Clip each gradient tensor by its L2 norm.
    Limits the global sensitivity to clip_norm.
    """
    return [
        tf.clip_by_norm(g, clip_norm) if g is not None else g
        for g in gradients
    ]


def add_gaussian_noise(gradients, sigma: float):
    """
    Add Gaussian noise N(0, sigma^2) to each gradient tensor.
    """
    return [
        g + tf.random.normal(shape=tf.shape(g), stddev=sigma)
        if g is not None else g
        for g in gradients
    ]


def dp_gradients(gradients, clip_norm: float, sigma: float):
    """
    Apply full DP-SGD pipeline: clip then add noise.

    Args:
        gradients: List of gradient tensors.
        clip_norm: Clipping threshold C.
        sigma:     Noise standard deviation (from compute_sigma).

    Returns:
        Privatised gradient list.
    """
    clipped = clip_gradients(gradients, clip_norm)
    noisy   = add_gaussian_noise(clipped, sigma)
    return noisy
