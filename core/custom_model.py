"""
custom_model.py  —  Custom Face Recognition Model (From Scratch)
=================================================================

A lightweight Multi-Layer Perceptron (MLP) trained entirely from scratch
using NumPy. No pretrained weights. No external DNN frameworks.

Architecture:
  Input (128-D SFace embeddings)
     → PCA (reduce to 32-D)
     → Hidden Layer 1: 64 neurons  (ReLU + Dropout)
     → Hidden Layer 2: 32 neurons  (ReLU + Dropout)
     → Output Layer:  N_classes    (Softmax)

Training: Mini-batch SGD with momentum, learning rate decay, L2 regularisation.
"""

import os
import json
import time
from typing import Optional
import numpy as np

# ── Activation functions ────────────────────────────────────────────────────

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(np.float32)

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy_loss(probs, y_one_hot, weights, l2_lambda=1e-4):
    n = probs.shape[0]
    eps = 1e-9
    ce = -np.sum(y_one_hot * np.log(probs + eps)) / n
    l2 = sum(np.sum(w ** 2) for w in weights)
    return ce + l2_lambda * l2

# ── PCA (from scratch) ──────────────────────────────────────────────────────

class PCA:
    def __init__(self, n_components: int = 32):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        cov = Xc.T @ Xc / (len(X) - 1)
        vals, vecs = np.linalg.eigh(cov)
        idx = np.argsort(vals)[::-1]
        vals, vecs = vals[idx], vecs[:, idx]
        self.components_ = vecs[:, :self.n_components].T          # (n_comp, n_feat)
        self.explained_variance_ratio_ = (
            vals[:self.n_components] / vals.sum()
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

# ── Weight initialisation ───────────────────────────────────────────────────

def he_init(fan_in, fan_out, rng):
    return rng.standard_normal((fan_in, fan_out)).astype(np.float32) * np.sqrt(2.0 / fan_in)

# ── MLP Classifier ──────────────────────────────────────────────────────────

class FaceMLP:
    """
    A 2-hidden-layer MLP trained with mini-batch SGD + momentum.
    Accepts PCA-reduced face embeddings and predicts class index.
    """

    def __init__(
        self,
        input_dim: int   = 32,
        hidden1:   int   = 64,
        hidden2:   int   = 32,
        n_classes: int   = 2,
        lr:        float = 0.05,
        momentum:  float = 0.9,
        l2:        float = 1e-4,
        dropout:   float = 0.2,
        seed:      int   = 42,
    ):
        self.lr       = lr
        self.momentum = momentum
        self.l2       = l2
        self.dropout  = dropout
        rng = np.random.default_rng(seed)

        # Weights + biases
        self.W1 = he_init(input_dim, hidden1, rng)
        self.b1 = np.zeros((1, hidden1), dtype=np.float32)
        self.W2 = he_init(hidden1, hidden2, rng)
        self.b2 = np.zeros((1, hidden2), dtype=np.float32)
        self.W3 = he_init(hidden2, n_classes, rng)
        self.b3 = np.zeros((1, n_classes), dtype=np.float32)

        # Momentum buffers
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)
        self.vW3 = np.zeros_like(self.W3)
        self.vb3 = np.zeros_like(self.b3)

        self.classes_   = None
        self.input_dim  = input_dim
        self.hidden1    = hidden1
        self.hidden2    = hidden2
        self.n_classes  = n_classes

    # ── Forward pass ──────────────────────────────────────────────────────────
    def _forward(self, X, training=False, rng=None):
        Z1 = X @ self.W1 + self.b1
        A1 = relu(Z1)
        if training and rng is not None:
            mask1 = (rng.random(A1.shape) > self.dropout).astype(np.float32) / (1 - self.dropout)
            A1 *= mask1
        else:
            mask1 = None

        Z2 = A1 @ self.W2 + self.b2
        A2 = relu(Z2)
        if training and rng is not None:
            mask2 = (rng.random(A2.shape) > self.dropout).astype(np.float32) / (1 - self.dropout)
            A2 *= mask2
        else:
            mask2 = None

        Z3   = A2 @ self.W3 + self.b3
        probs = softmax(Z3)
        cache = (X, Z1, A1, mask1, Z2, A2, mask2, Z3)
        return probs, cache

    # ── Backward pass ──────────────────────────────────────────────────────────
    def _backward(self, probs, y_oh, cache):
        X, Z1, A1, mask1, Z2, A2, mask2, Z3 = cache
        n = X.shape[0]
        weights = [self.W1, self.W2, self.W3]

        dZ3 = (probs - y_oh) / n
        dW3 = A2.T @ dZ3 + self.l2 * self.W3
        db3 = dZ3.sum(axis=0, keepdims=True)

        dA2 = dZ3 @ self.W3.T
        if mask2 is not None:
            dA2 *= mask2
        dZ2 = dA2 * relu_grad(Z2)
        dW2 = A1.T @ dZ2 + self.l2 * self.W2
        db2 = dZ2.sum(axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        if mask1 is not None:
            dA1 *= mask1
        dZ1 = dA1 * relu_grad(Z1)
        dW1 = X.T @ dZ1 + self.l2 * self.W1
        db1 = dZ1.sum(axis=0, keepdims=True)

        return dW1, db1, dW2, db2, dW3, db3

    # ── SGD + momentum update ──────────────────────────────────────────────────
    def _update(self, grads, lr):
        dW1, db1, dW2, db2, dW3, db3 = grads
        m = self.momentum

        self.vW1 = m * self.vW1 - lr * dW1;  self.W1 += self.vW1
        self.vb1 = m * self.vb1 - lr * db1;  self.b1 += self.vb1
        self.vW2 = m * self.vW2 - lr * dW2;  self.W2 += self.vW2
        self.vb2 = m * self.vb2 - lr * db2;  self.b2 += self.vb2
        self.vW3 = m * self.vW3 - lr * dW3;  self.W3 += self.vW3
        self.vb3 = m * self.vb3 - lr * db3;  self.b3 += self.vb3

    # ── Training loop (yields per-epoch metrics) ───────────────────────────────
    def fit_epoch_by_epoch(self, X, y, n_epochs=60, batch_size=16):
        """
        Generator: yields dict with epoch metrics after each epoch.
        Allows Streamlit to update charts live.
        """
        rng = np.random.default_rng(42)
        n   = X.shape[0]
        self.classes_ = np.unique(y)
        n_cls = len(self.classes_)

        # One-hot encode
        label_map = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([label_map[yi] for yi in y])
        y_oh  = np.eye(n_cls, dtype=np.float32)[y_idx]

        # Re-init output layer if needed
        if self.n_classes != n_cls:
            self.n_classes = n_cls
            rng2 = np.random.default_rng(99)
            self.W3 = he_init(self.hidden2, n_cls, rng2)
            self.b3 = np.zeros((1, n_cls), dtype=np.float32)
            self.vW3 = np.zeros_like(self.W3)
            self.vb3 = np.zeros_like(self.b3)

        lr = self.lr
        for epoch in range(1, n_epochs + 1):
            # LR decay
            lr = self.lr * (0.97 ** (epoch // 10))

            # Shuffle
            perm = rng.permutation(n)
            X_sh, y_oh_sh, yi_sh = X[perm], y_oh[perm], y_idx[perm]

            ep_loss = 0.0
            n_batches = max(1, n // batch_size)

            for b in range(n_batches):
                Xb = X_sh[b * batch_size:(b + 1) * batch_size]
                yb = y_oh_sh[b * batch_size:(b + 1) * batch_size]
                if Xb.shape[0] == 0:
                    continue
                probs, cache = self._forward(Xb, training=True, rng=rng)
                loss = cross_entropy_loss(probs, yb, [self.W1, self.W2, self.W3], self.l2)
                ep_loss += loss
                grads = self._backward(probs, yb, cache)
                self._update(grads, lr)

            # Eval on full set (no dropout)
            probs_full, _ = self._forward(X, training=False)
            preds = probs_full.argmax(axis=1)
            acc = (preds == y_idx).mean()
            avg_loss = ep_loss / n_batches

            yield {
                "epoch":    epoch,
                "loss":     float(avg_loss),
                "accuracy": float(acc),
                "lr":       float(lr),
                "preds":    preds,
                "probs":    probs_full,
                "y_true":   y_idx,
                "classes":  self.classes_,
            }

    def predict(self, X):
        probs, _ = self._forward(X, training=False)
        return probs.argmax(axis=1)

    def predict_proba(self, X):
        probs, _ = self._forward(X, training=False)
        return probs

    def get_weights_info(self):
        """Return weight matrices for visualisation."""
        return {
            "W1": self.W1.copy(),
            "W2": self.W2.copy(),
            "W3": self.W3.copy(),
        }


# ── Full pipeline (PCA + MLP) ───────────────────────────────────────────────

class CustomFaceClassifier:
    """
    End-to-end pipeline:
      raw 128-D SFace embeddings  →  PCA(32)  →  MLP(64→32→N)
    """

    def __init__(self, pca_components: int = 32):
        self.pca = PCA(n_components=pca_components)
        self.mlp: Optional[FaceMLP] = None
        self.label_names: list = []

    def prepare_data(self, embeddings: list, labels: list):
        """
        embeddings: list of np arrays (128-D each)
        labels:     list of str names (same length)
        Returns (X_pca, y_labels)
        """
        X = np.array(embeddings, dtype=np.float32)
        X_pca = self.pca.fit_transform(X)
        y = np.array(labels)
        self.label_names = sorted(set(labels))
        return X_pca, y

    def build_mlp(self, n_classes: int):
        self.mlp = FaceMLP(
            input_dim=self.pca.n_components,
            hidden1=64,
            hidden2=32,
            n_classes=n_classes,
            lr=0.05,
            momentum=0.9,
            l2=1e-4,
            dropout=0.2,
        )
        return self.mlp

    def get_architecture_info(self):
        return {
            "layers": [
                {"name": "Input",   "neurons": 128,                       "activation": "—",       "color": "#6c63ff"},
                {"name": "PCA",     "neurons": self.pca.n_components,     "activation": "linear",  "color": "#00d4ff"},
                {"name": "Hidden 1","neurons": 64,                        "activation": "ReLU",    "color": "#ff6b9d"},
                {"name": "Hidden 2","neurons": 32,                        "activation": "ReLU",    "color": "#ff8e53"},
                {"name": "Output",  "neurons": len(self.label_names) or 2,"activation": "Softmax", "color": "#00e87a"},
            ]
        }
