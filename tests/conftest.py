"""Pytest configuration and shared fixtures."""

import pytest
import pandas as pd
import numpy as np
import torch


@pytest.fixture
def sample_config():
    """Sample configuration for tests."""
    return {
        "data": {
            "train_path": "dummy_train.parquet",
            "test_path": "dummy_test.parquet",
            "target_column": "active_essence",
            "exclude_columns": ["party_rk", "essence_id", "utilized"],
            "validation_size": 0.2,
            "random_seed": 42,
        },
        "model": {
            "input_dim": 10,
            "hidden_layers": [64, 32, 16],
            "dropout_rates": [0.3, 0.2, 0.1],
            "use_batch_norm": True,
            "activation": "relu",
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "num_epochs": 5,
            "early_stopping_patience": 3,
        },
    }


@pytest.fixture
def sample_dataframe():
    """Create test DataFrame with different data types."""
    np.random.seed(42)
    n_samples = 100

    data = {
        "party_rk": range(n_samples),
        "essence_id": range(1000, 1000 + n_samples),
        "active_essence": np.random.randint(0, 2, n_samples),
        "utilized": np.random.randint(0, 2, n_samples),
        "feature_1": np.random.normal(0, 1, n_samples),
        "feature_2": np.random.normal(10, 5, n_samples),
        "feature_3": np.random.exponential(2, n_samples),
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_model_config():
    """Sample model configuration for tests."""
    return {
        "model": {
            "input_dim": 10,
            "hidden_layers": [64, 32, 16],
            "dropout_rates": [0.3, 0.2, 0.1],
            "use_batch_norm": True,
            "activation": "relu",
        }
    }
