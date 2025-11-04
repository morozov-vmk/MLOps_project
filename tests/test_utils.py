import sys

import pytest
import torch
import yaml

sys.path.append("src")

from utils import get_device, load_config, set_seed, setup_logging


class TestUtils:
    """Тесты утилитарных функций"""

    def test_load_config(self):
        """Тест загрузки конфигурации"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config_data = {
                "model": {"input_dim": 10},
                "training": {"learning_rate": 0.001},
            }
            yaml.dump(config_data, f)
            temp_config_path = f.name

        try:
            config = load_config(temp_config_path)

            assert config["model"]["input_dim"] == 10
            assert config["training"]["learning_rate"] == 0.001
        finally:
            os.unlink(temp_config_path)

    def test_load_config_file_not_found(self):
        """Тест загрузки несуществующего конфига"""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")

    def test_set_seed(self):
        """Тест установки random seed"""
        set_seed(42)

        import random

        import numpy as np
        import torch

        py_num = random.random()
        np_num = np.random.random()
        torch_num = torch.rand(1).item()

        set_seed(42)
        py_num2 = random.random()
        np_num2 = np.random.random()
        torch_num2 = torch.rand(1).item()

        assert py_num == py_num2
        assert np_num == np_num2
        assert torch_num == torch_num2

    def test_get_device(self):
        """Тест определения устройства"""
        device = get_device()

        assert device in [torch.device("cpu"), torch.device("cuda")]

    def test_setup_logging(self):
        """Тест настройки логирования"""
        import logging

        config = {"logging": {"level": "INFO", "log_file": "test.log"}}

        setup_logging(config)

        logger = logging.getLogger("test")
        logger.info("Test message")

        if os.path.exists("test.log"):
            os.remove("test.log")
