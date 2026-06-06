"""
fedavg.py — Federated Averaging (FedAvg) training engine.

Supports both standard FedAvg and DP-FedAvg (with DP-SGD).
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from src.dp_sgd import compute_sigma, dp_gradients


def build_mlp(input_dim: int = 6, seed: int = 42) -> keras.Model:
    """MLP: input_dim -> 32 -> 16 -> 1 (sigmoid)."""
    tf.random.set_seed(seed)
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1,  activation='sigmoid'),
    ])
    return model


def fedavg_aggregate(weights_list, n_samples_list):
    """
    Equal-weight gradient aggregation across clients.
    (Equal weight chosen to avoid Pclass=3 dominating due to
    its larger sample size and lower survival rate.)
    """
    K = len(weights_list)
    return [
        sum(w[i] for w in weights_list) / K
        for i in range(len(weights_list[0]))
    ]


def run_fedavg(
    client_data,
    rounds:          int   = 45,
    lr:              float = 0.001,
    use_dp:          bool  = False,
    clip_norm:       float = 0.5,
    epsilon:         float = None,
    delta:           float = 1e-5,
    seed:            int   = 42,
    verbose:         bool  = True,
) -> tuple:
    """
    Run FedAvg (with optional DP-SGD) for `rounds` communication rounds.

    Args:
        client_data: List of (X_train, y_train) tuples, one per client.
        rounds:      Number of communication rounds.
        lr:          Adam learning rate.
        use_dp:      Whether to apply DP-SGD.
        clip_norm:   Gradient clipping norm C.
        epsilon:     Privacy budget ε (required if use_dp=True).
        delta:       DP failure probability δ.
        seed:        Random seed for reproducibility.
        verbose:     Print progress bar.

    Returns:
        (global_model, accuracy_history)
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)

    sigma = compute_sigma(epsilon, delta, clip_norm) if use_dp else 0.0
    if verbose:
        dp_str = f"sigma={sigma:.4f}" if use_dp else "No DP"
        print(f"  ε={epsilon}, {dp_str}")

    global_model = build_mlp(seed=seed)
    _ = global_model(client_data[0][0][:1])          # initialise weights
    optimizer    = keras.optimizers.Adam(lr)
    loss_fn      = keras.losses.BinaryCrossentropy()

    acc_history  = []
    bar_len      = 25

    for rnd in range(rounds):
        grads_all = []

        for X_c, y_c in client_data:
            X_t = tf.constant(X_c, dtype=tf.float32)
            y_t = tf.constant(y_c, dtype=tf.float32)

            with tf.GradientTape() as tape:
                pred = global_model(X_t, training=True)
                loss = loss_fn(tf.expand_dims(y_t, -1), pred)

            grads = tape.gradient(loss, global_model.trainable_variables)

            if use_dp:
                grads = dp_gradients(grads, clip_norm, sigma)
            else:
                grads = [tf.clip_by_norm(g, clip_norm) for g in grads]

            grads_all.append(grads)

        avg_grads = fedavg_aggregate(grads_all, [len(c[0]) for c in client_data])
        optimizer.apply_gradients(zip(avg_grads, global_model.trainable_variables))

        # evaluate
        from src.evaluate import accuracy
        acc = accuracy(global_model, client_data)
        acc_history.append(acc)

        if verbose:
            filled = int(bar_len * (rnd + 1) / rounds)
            bar    = '█' * filled + '░' * (bar_len - filled)
            pct    = (rnd + 1) / rounds * 100
            print(f'\rRound {rnd+1:3d}/{rounds} [{bar}] {pct:5.1f}%  '
                  f'Acc={acc:.4f}', end='', flush=True)

    if verbose:
        print()

    return global_model, acc_history
