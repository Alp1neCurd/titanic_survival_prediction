import numpy as np
import torch
import torch.nn as nn

from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import DataLoader, TensorDataset


def build_dnn(parameters, input_size):
    """Build a two-layer neural network for binary classification."""
    activation_classes = {"ReLU": nn.ReLU,
                          "LeakyReLU": nn.LeakyReLU,
                          "ELU": nn.ELU}

    activation_class = activation_classes[parameters["activation"]]
    normalization = (nn.BatchNorm1d(parameters["hidden_size_1"]) if parameters["batch_norm"] else nn.Identity())

    return nn.Sequential(nn.Linear(input_size, parameters["hidden_size_1"]),

                         activation_class(),
                         normalization,

                         nn.Dropout(parameters["dropout_1"]),
                         nn.Linear(parameters["hidden_size_1"], parameters["hidden_size_2"]),

                         activation_class(),

                         nn.Dropout(parameters["dropout_2"]),
                         nn.Linear(parameters["hidden_size_2"], 1),)


def train_dnn(parameters, features, targets, seed):
    """Train the neural network with a fixed random seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    feature_tensor = torch.tensor(np.asarray(features), dtype=torch.float32,)
    target_tensor = torch.tensor(np.asarray(targets), dtype=torch.float32,).reshape(-1, 1)

    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(TensorDataset(feature_tensor, target_tensor),batch_size=parameters["batch_size"], shuffle=True, generator=generator,)

    model = build_dnn(parameters, feature_tensor.shape[1])
    criterion = nn.BCEWithLogitsLoss()

    optimizer_class = (torch.optim.AdamW if parameters["optimizer"] == "AdamW" else torch.optim.Adam)
    optimizer = optimizer_class(model.parameters(), lr=parameters["learning_rate"], weight_decay=parameters["weight_decay"],)

    scheduler = None

    if parameters["scheduler"] == "Cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=parameters["epochs"], eta_min=1e-5,)

    for _ in range(parameters["epochs"]):
        model.train()

        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_features), batch_targets)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

    model.eval()

    return model


class TorchDNNClassifier(ClassifierMixin, BaseEstimator):
    """Use the PyTorch DNN through the sklearn estimator interface."""

    def __init__(self, parameters=None, random_state=42):
        self.parameters = parameters
        self.random_state = random_state

    def fit(self, X, y):
        if self.parameters is None:
            raise ValueError("DNN parameters are required")

        features = np.asarray(X)
        self.model_ = train_dnn(self.parameters, features, y, self.random_state,)
        self.classes_ = np.unique(y)
        self.n_features_in_ = features.shape[1]

        return self

    def predict_proba(self, X):
        features = torch.tensor(np.asarray(X), dtype=torch.float32,)

        with torch.no_grad():
            probabilities = torch.sigmoid(self.model_(features)).numpy().ravel()

        return np.column_stack((1 - probabilities, probabilities))

    def predict(self, X):
        probabilities = self.predict_proba(X)[:, 1]

        return (probabilities >= 0.5).astype(int)