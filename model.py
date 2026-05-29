import numpy as np
from config import INPUT_DIM, HIDDEN_DIM, MODEL_PATH

class DenoisingAutoencoder:

    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = HIDDEN_DIM):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W1 = np.random.randn(hidden_dim, input_dim).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((hidden_dim, 1), dtype=np.float32)

        self.W2 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * np.sqrt(1.0 / hidden_dim)
        self.b2 = np.zeros((input_dim, 1), dtype=np.float32)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def _relu_d(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1.0, 0.0)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.A0 = X.T
        self.Z1 = self.W1 @ self.A0 + self.b1
        self.A1 = self._relu(self.Z1)
        self.Z2 = self.W2 @ self.A1 + self.b2
        self.A2 = self.Z2
        return self.A2.T

    def backward(self, X: np.ndarray, Y: np.ndarray) -> None:
        m = X.shape[0]
        dZ2 = self.A2 - Y.T
        self.dW2 = (dZ2 @ self.A1.T) / m
        self.db2 = dZ2.sum(axis=1, keepdims=True) / m
        dZ1 = (self.W2.T @ dZ2) * self._relu_d(self.Z1)
        self.dW1 = (dZ1 @ self.A0.T) / m
        self.db1 = dZ1.sum(axis=1, keepdims=True) / m

    def step(self, lr: float) -> None:
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2

    def sacuvaj(self, putanja: str = MODEL_PATH, norm_faktor: float = 1.0) -> None:
            np.savez(putanja, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                    input_dim=self.input_dim, hidden_dim=self.hidden_dim,
                    norm_faktor=norm_faktor)
            print(f"  Model sačuvan: {putanja}")

    @classmethod
    def ucitaj(cls, putanja: str = MODEL_PATH) -> tuple["DenoisingAutoencoder", float]:
        d = np.load(putanja)
        mdl = cls(int(d['input_dim']), int(d['hidden_dim']))
        mdl.W1, mdl.b1, mdl.W2, mdl.b2 = d['W1'], d['b1'], d['W2'], d['b2']
        norm_faktor = float(d['norm_faktor']) if 'norm_faktor' in d else 1.0
        print(f"  Model učitan: {putanja}")
        return mdl, norm_faktor             