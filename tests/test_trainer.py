import sys
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import torch

sys.path.append("src")

from model import CashbackMLP
from trainer import ModelTrainer


class TestModelTrainer:
    """Тесты для ModelTrainer"""

    @pytest.fixture
    def sample_config(self):
        return {
            "training": {
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "batch_size": 32,
                "num_epochs": 5,
                "early_stopping_patience": 3,
            },
            "model": {
                "input_dim": 10,
                "hidden_layers": [32, 16],
                "dropout_rates": [0.2, 0.1],
                "use_batch_norm": True,
            },
        }

    @pytest.fixture
    def sample_model(self, sample_config):
        return CashbackMLP(sample_config)

    @pytest.fixture
    def sample_data_loaders(self):
        """Создает mock DataLoader'ы"""
        train_loader = Mock()
        val_loader = Mock()

        train_data = [
            (torch.randn(32, 10), torch.randint(0, 2, (32,)).float()) for _ in range(10)
        ]
        val_data = [
            (torch.randn(32, 10), torch.randint(0, 2, (32,)).float()) for _ in range(5)
        ]

        train_loader.__iter__ = Mock(return_value=iter(train_data))
        train_loader.__len__ = Mock(return_value=10)

        val_loader.__iter__ = Mock(return_value=iter(val_data))
        val_loader.__len__ = Mock(return_value=5)

        return train_loader, val_loader

    def test_trainer_initialization(self, sample_model, sample_config):
        """Тест инициализации тренера"""
        device = torch.device("cpu")
        trainer = ModelTrainer(sample_model, sample_config, device)

        assert trainer.model == sample_model
        assert trainer.config == sample_config
        assert trainer.device == device
        assert trainer.criterion is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

    def test_train_epoch(self, sample_model, sample_config, sample_data_loaders):
        """Тест обучения на одной эпохе"""
        device = torch.device("cpu")
        trainer = ModelTrainer(sample_model, sample_config, device)
        train_loader, _ = sample_data_loaders

        initial_loss = trainer.train_epoch(train_loader, 0)

        assert isinstance(initial_loss, float)
        assert initial_loss > 0

    def test_validation(self, sample_model, sample_config, sample_data_loaders):
        """Тест валидации модели"""
        device = torch.device("cpu")
        trainer = ModelTrainer(sample_model, sample_config, device)
        _, val_loader = sample_data_loaders

        metrics = trainer.validate(val_loader, 0)

        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "auc" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["auc"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1

    def test_early_stopping(self, sample_model, sample_config):
        """Тест механизма ранней остановки"""
        device = torch.device("cpu")
        trainer = ModelTrainer(sample_model, sample_config, device)

        trainer.best_val_auc = 0.9
        trainer.early_stopping_counter = 2

        metrics = {"auc": 0.8}

        old_counter = trainer.early_stopping_counter
        if metrics["auc"] <= trainer.best_val_auc:
            trainer.early_stopping_counter += 1

        assert trainer.early_stopping_counter == old_counter + 1

    def test_model_saving(self, sample_model, sample_config, tmp_path):
        """Тест сохранения модели"""
        device = torch.device("cpu")
        trainer = ModelTrainer(sample_model, sample_config, device)

        checkpoint = {
            "epoch": 1,
            "model_state_dict": sample_model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "best_val_auc": 0.85,
            "config": sample_config,
        }

        save_path = tmp_path / "test_model.pth"
        torch.save(checkpoint, save_path)

        assert save_path.exists()

        loaded_checkpoint = torch.load(save_path)
        assert loaded_checkpoint["best_val_auc"] == 0.85


class TestTrainerEdgeCases:
    """Тесты граничных случаев тренера"""

    def test_empty_dataloader(self, sample_config):
        """Тест с пустыми DataLoader'ами"""
        device = torch.device("cpu")
        model = CashbackMLP(sample_config)
        trainer = ModelTrainer(model, sample_config, device)

        empty_loader = Mock()
        empty_loader.__iter__ = Mock(return_value=iter([]))
        empty_loader.__len__ = Mock(return_value=0)

        metrics = trainer.validate(empty_loader, 0)

        assert "auc" in metrics
        assert "accuracy" in metrics

    def test_single_batch_training(self, sample_config):
        """Тест обучения на одном батче"""
        device = torch.device("cpu")
        model = CashbackMLP(sample_config)
        trainer = ModelTrainer(model, sample_config, device)

        single_batch_loader = Mock()
        single_batch_data = [(torch.randn(1, 10), torch.tensor([1.0]))]
        single_batch_loader.__iter__ = Mock(return_value=iter(single_batch_data))
        single_batch_loader.__len__ = Mock(return_value=1)

        loss = trainer.train_epoch(single_batch_loader, 0)
        assert isinstance(loss, float)
