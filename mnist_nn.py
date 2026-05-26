"""
Neural Network from Scratch — MNIST classifier
Architecture: 784 → 256 → 128 → 10
Activations:  ReLU, ReLU, Softmax
Optimizer:    SGD with momentum
"""

import numpy as np
import pickle
import os
import urllib.request


# ─────────────────────────────────────────────
#  Activation functions & their derivatives
# ─────────────────────────────────────────────

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def softmax(z):
    # Numerically stable: subtract row-max before exp
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy_loss(probs, y_onehot):
    n = probs.shape[0]
    log_p = np.log(probs + 1e-12)
    return -np.sum(y_onehot * log_p) / n


# ─────────────────────────────────────────────
#  Neural Network class
# ─────────────────────────────────────────────

class NeuralNetwork:
    """
    Fully-connected feedforward network with configurable layers.

    Parameters
    ----------
    layer_sizes : list[int]
        Number of neurons per layer, including input and output.
        e.g. [784, 256, 128, 10]
    lr : float
        Learning rate.
    momentum : float
        Momentum coefficient for SGD.
    """

    def __init__(self, layer_sizes, lr=0.01, momentum=0.9):
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.momentum = momentum
        self.num_layers = len(layer_sizes) - 1   # number of weight matrices

        # He initialisation for ReLU layers
        self.W = []
        self.b = []
        for i in range(self.num_layers):
            fan_in = layer_sizes[i]
            scale = np.sqrt(2.0 / fan_in)
            self.W.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale)
            self.b.append(np.zeros((1, layer_sizes[i + 1])))

        # Momentum buffers (same shape as W, b)
        self.vW = [np.zeros_like(w) for w in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]

    # ── Forward pass ──────────────────────────

    def forward(self, X):
        """
        Returns
        -------
        probs   : softmax output  (N, 10)
        cache   : list of (Z, A) tuples for backprop
        """
        cache = []
        A = X
        for i in range(self.num_layers - 1):        # hidden layers → ReLU
            Z = A @ self.W[i] + self.b[i]
            A = relu(Z)
            cache.append((Z, A))

        # Output layer → Softmax
        Z_out = A @ self.W[-1] + self.b[-1]
        probs = softmax(Z_out)
        cache.append((Z_out, probs))
        return probs, cache

    # ── Backward pass ─────────────────────────

    def backward(self, X, y_onehot, cache):
        """
        Compute gradients via backprop and update weights with SGD+momentum.
        """
        n = X.shape[0]
        grads_W = [None] * self.num_layers
        grads_b = [None] * self.num_layers

        # Gradient at output (softmax + cross-entropy combined)
        _, probs = cache[-1][0], cache[-1][1]
        delta = (probs - y_onehot) / n           # (N, 10)

        for i in reversed(range(self.num_layers)):
            A_prev = cache[i - 1][1] if i > 0 else X
            grads_W[i] = A_prev.T @ delta
            grads_b[i] = delta.sum(axis=0, keepdims=True)

            if i > 0:
                Z_prev = cache[i - 1][0]
                delta = (delta @ self.W[i].T) * relu_deriv(Z_prev)

        # SGD + momentum update
        for i in range(self.num_layers):
            self.vW[i] = self.momentum * self.vW[i] - self.lr * grads_W[i]
            self.vb[i] = self.momentum * self.vb[i] - self.lr * grads_b[i]
            self.W[i] += self.vW[i]
            self.b[i] += self.vb[i]

    # ── Training ──────────────────────────────

    def train(self, X_train, y_train, X_val, y_val,
              epochs=20, batch_size=128):
        n = X_train.shape[0]
        y_train_oh = self._onehot(y_train)

        for epoch in range(1, epochs + 1):
            # Shuffle
            idx = np.random.permutation(n)
            X_s, y_s = X_train[idx], y_train_oh[idx]

            # Mini-batch SGD
            for start in range(0, n, batch_size):
                Xb = X_s[start:start + batch_size]
                yb = y_s[start:start + batch_size]
                probs, cache = self.forward(Xb)
                self.backward(Xb, yb, cache)

            # Metrics
            train_loss, train_acc = self.evaluate(X_train, y_train)
            val_loss,   val_acc   = self.evaluate(X_val,   y_val)
            print(f"Epoch {epoch:>2}/{epochs}  "
                  f"train loss={train_loss:.4f}  acc={train_acc*100:.2f}%  │  "
                  f"val loss={val_loss:.4f}  acc={val_acc*100:.2f}%")

    # ── Inference ─────────────────────────────

    def predict(self, X):
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        probs, _ = self.forward(X)
        return probs

    def evaluate(self, X, y):
        probs, _ = self.forward(X)
        loss = cross_entropy_loss(probs, self._onehot(y))
        acc  = np.mean(np.argmax(probs, axis=1) == y)
        return loss, acc

    # ── Persistence ───────────────────────────

    def save(self, path):
        data = {"layer_sizes": self.layer_sizes,
                "lr": self.lr, "momentum": self.momentum,
                "W": self.W, "b": self.b}
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Model saved → {path}")

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        net = cls(data["layer_sizes"], data["lr"], data["momentum"])
        net.W = data["W"]
        net.b = data["b"]
        return net

    # ── Helpers ───────────────────────────────

    @staticmethod
    def _onehot(y, num_classes=10):
        n = len(y)
        oh = np.zeros((n, num_classes))
        oh[np.arange(n), y] = 1
        return oh


# ─────────────────────────────────────────────
#  MNIST loader  — downloads real MNIST data
# ─────────────────────────────────────────────

def load_mnist(cache_path="mnist_data/mnist.npz"):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path):
        print(f"Loading MNIST from cache: {cache_path} …")
        d = np.load(cache_path)
        return (d["X_train"], d["y_train"].astype(np.int32),
                d["X_test"],  d["y_test"].astype(np.int32))

    # ── Try keras first (fastest) ─────────────
    try:
        print("Downloading MNIST via Keras …")
        from tensorflow.keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(-1, 784).astype(np.float32) / 255.0
        X_test  = X_test.reshape(-1, 784).astype(np.float32) / 255.0
        np.savez(cache_path,
                 X_train=X_train, y_train=y_train,
                 X_test=X_test,   y_test=y_test)
        print(f"Saved to {cache_path}")
        return (X_train, y_train.astype(np.int32),
                X_test,  y_test.astype(np.int32))
    except Exception:
        pass

    # ── Fallback: download raw IDX files from mirror ──
    print("Downloading MNIST from mirror (raw IDX files) …")

    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images":  "t10k-images-idx3-ubyte.gz",
        "test_labels":  "t10k-labels-idx1-ubyte.gz",
    }

    raw_dir = "mnist_data/raw"
    os.makedirs(raw_dir, exist_ok=True)

    def download(fname):
        dest = os.path.join(raw_dir, fname)
        if not os.path.exists(dest):
            url = base_url + fname
            print(f"  Fetching {url} …")
            urllib.request.urlretrieve(url, dest)
        return dest

    import gzip, struct

    def read_images(path):
        with gzip.open(path, "rb") as f:
            magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, rows * cols).astype(np.float32) / 255.0

    def read_labels(path):
        with gzip.open(path, "rb") as f:
            magic, n = struct.unpack(">II", f.read(8))
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.astype(np.int32)

    X_train = read_images(download(files["train_images"]))
    y_train = read_labels(download(files["train_labels"]))
    X_test  = read_images(download(files["test_images"]))
    y_test  = read_labels(download(files["test_labels"]))

    np.savez(cache_path,
             X_train=X_train, y_train=y_train,
             X_test=X_test,   y_test=y_test)
    print(f"Saved to {cache_path}")
    return X_train, y_train, X_test, y_test


# ─────────────────────────────────────────────
#  Main — train and save
# ─────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("  Neural Network from Scratch — MNIST")
    print("=" * 60)

    # Load real MNIST data
    print("\nLoading MNIST …")
    X_train, y_train, X_test, y_test = load_mnist()

    # Split 5 000 samples from training set as validation
    X_val, y_val     = X_train[:5000],  y_train[:5000]
    X_train, y_train = X_train[5000:],  y_train[5000:]
    print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

    # Build model: 784 → 256 → 128 → 10
    net = NeuralNetwork(
        layer_sizes=[784, 256, 128, 10],
        lr=0.01,
        momentum=0.9,
    )

    # Train
    print("\nTraining …\n")
    net.train(X_train, y_train, X_val, y_val, epochs=20, batch_size=128)

    # Final test accuracy
    test_loss, test_acc = net.evaluate(X_test, y_test)
    print(f"\n{'='*60}")
    print(f"  Test accuracy : {test_acc*100:.2f}%")
    print(f"  Test loss     : {test_loss:.4f}")
    print(f"{'='*60}")

    # Save weights
    net.save("mnist_model.pkl")
    print("\nDone! Run `python draw.py` to try the drawing interface.")