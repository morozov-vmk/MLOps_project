import yaml
import logging
import torch
import random
import numpy as np
import json
import os

def setup_logging(config):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"]),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config["logging"]["log_file"]),
            logging.StreamHandler()
        ]
    )

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    """Get available device"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pretrained_hf(model: torch.nn.Module, cfg: dict, out_dir: str):
    """
    Сохраняет PyTorch модель в структуре, похожей на Hugging Face:
    - pytorch_model.bin  (state_dict)
    - config.json        (архитектура/параметры)
    """
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "pytorch_model.bin")
    torch.save(model.state_dict(), model_path)

    cfg_path = os.path.join(out_dir, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

def load_pretrained_hf(model: torch.nn.Module, out_dir: str, map_location=None):
    """Загружает state_dict из out_dir/pytorch_model.bin в переданную модель"""
    model_path = os.path.join(out_dir, "pytorch_model.bin")
    state = torch.load(model_path, map_location=map_location)
    model.load_state_dict(state)
    return model