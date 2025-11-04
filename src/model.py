import logging

import torch.nn as nn

logger = logging.getLogger(__name__)


class CashbackMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        input_dim = config["model"]["input_dim"]
        hidden_layers = config["model"]["hidden_layers"]
        dropout_rates = config["model"]["dropout_rates"]

        layers = []
        prev_dim = input_dim

        if config["model"]["use_batch_norm"]:
            layers.append(nn.BatchNorm1d(input_dim))

        for i, (hidden_dim, dropout_rate) in enumerate(
            zip(hidden_layers, dropout_rates)
        ):
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = hidden_dim

        layers.extend([nn.Linear(prev_dim, 1), nn.Sigmoid()])

        self.network = nn.Sequential(*layers)
        self._init_weights()

        logger.info(
            f"Model initialized with input_dim={input_dim}, hidden_layers={hidden_layers}"
        )

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.network(x)
