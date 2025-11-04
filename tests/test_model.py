import pytest
import torch
import sys

sys.path.append("src")

from model import CashbackMLP


class TestCashbackMLP:
    """Тесты для модели CashbackMLP"""

    @pytest.fixture
    def sample_config(self):
        return {
            "model": {
                "input_dim": 10,
                "hidden_layers": [64, 32, 16],
                "dropout_rates": [0.3, 0.2, 0.1],
                "use_batch_norm": True,
                "activation": "relu",
            }
        }

    def test_model_initialization(self, sample_config):
        """Тест инициализации модели"""
        model = CashbackMLP(sample_config)

        assert model is not None
        assert hasattr(model, "network")
        assert isinstance(model.network, torch.nn.Sequential)

    def test_forward_pass(self, sample_config):
        """Тест прямого прохода модели"""
        model = CashbackMLP(sample_config)
        batch_size = 32

        # Создаем тестовый вход
        x = torch.randn(batch_size, sample_config["model"]["input_dim"])

        # Прямой проход
        output = model(x)

        # Проверяем выход
        assert output.shape == (batch_size, 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output

    def test_model_parameters(self, sample_config):
        """Тест наличия обучаемых параметров"""
        model = CashbackMLP(sample_config)

        parameters = list(model.parameters())
        assert len(parameters) > 0

        # Проверяем, что параметры требуют градиенты
        for param in parameters:
            assert param.requires_grad

    def test_different_batch_sizes(self, sample_config):
        """Тест работы с разными размерами батчей"""
        model = CashbackMLP(sample_config)

        # Используем размеры батчей которые работают с BatchNorm
        batch_sizes = [2, 16, 32]  # Минимум 2 для BatchNorm

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, sample_config["model"]["input_dim"])
            output = model(x)

            assert output.shape == (batch_size, 1)

    def test_model_eval_mode(self, sample_config):
        """Тест переключения режимов обучения/инференса"""
        model = CashbackMLP(sample_config)

        # Проверяем, что в режиме обучения dropout активен
        model.train()
        x = torch.randn(16, sample_config["model"]["input_dim"])
        output_train = model(x)

        # Проверяем, что в режиме инференса вывод стабилен
        model.eval()
        output_eval = model(x)

        # Выводы могут немного отличаться из-за dropout
        assert not torch.allclose(output_train, output_eval)

    def test_gradient_flow(self, sample_config):
        """Тест потока градиентов"""
        model = CashbackMLP(sample_config)
        batch_size = 16

        x = torch.randn(
            batch_size, sample_config["model"]["input_dim"], requires_grad=True
        )
        y = torch.randint(0, 2, (batch_size, 1)).float()

        # Прямой проход
        output = model(x)

        # Вычисляем loss
        criterion = torch.nn.BCELoss()
        loss = criterion(output, y)

        # Обратное распространение
        loss.backward()

        # Проверяем, что градиенты вычислены
        assert x.grad is not None
        assert x.grad.shape == x.shape
