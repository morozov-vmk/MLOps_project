#!/usr/bin/env python3
# scripts/train.py
import argparse
import logging
import os
import json
#import mlflow
#import mlflow.pytorch
import torch
import numpy as np

from src.data_loader import DataProcessor
from src.model import CashbackMLP
from src.trainer import ModelTrainer
from src.utils import load_config, setup_logging, set_seed, get_device

def flatten_config(cfg, parent_key="", sep="."):
    items = {}
    for k, v in cfg.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_config(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def main():
    parser = argparse.ArgumentParser(description="Train model (uses prepared .npz file)")
    parser.add_argument("--config", type=str, default="config/model_config.yaml")
    parser.add_argument("--processed", type=str, default="data/processed/processed_data.npz")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)
    logger = logging.getLogger(__name__)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    set_seed(config["data"].get("random_seed", 42))
    device = get_device()
    logger.info("Using device: %s", device)

    os.makedirs(args.output_dir, exist_ok=True)
    best_model_path = os.path.join(args.output_dir, "best_model.pth")
    model_weights_path = os.path.join(args.output_dir, "model_weights.pth")

    # Configure MLflow: default local mlruns dir or override by MLFLOW_TRACKING_URI env var
    #mlflow.pytorch.autolog()  # автологирование PyTorch (параметры, модель, метрики) — полезно вместе с ручным логированием
    if 1:#with mlflow.start_run():
        try:
            logger.info("Loading processed data from %s", args.processed)
            if not os.path.exists(args.processed):
                raise FileNotFoundError(f"Processed file not found: {args.processed}")

            arr = np.load(args.processed, allow_pickle=True)
            X_train = arr["X_train"]
            y_train = arr["y_train"]
            X_val = arr["X_val"]
            y_val = arr["y_val"]
            X_test = arr["X_test"]
            y_test = arr["y_test"]

            # Log config flattened as params
            params = flatten_config(config)
            for k, v in params.items():
                try:
                    #mlflow.log_param(k, v)
                    pass
                except Exception:
                    pass

            processor = DataProcessor(config)
            train_loader, val_loader, test_loader = processor.create_data_loaders(
                X_train, y_train, X_val, y_val, X_test, y_test
            )

            actual_input_dim = X_train.shape[1]
            config["model"]["input_dim"] = actual_input_dim
            logger.info("Updated input_dim to: %s", actual_input_dim)

            model = CashbackMLP(config).to(device)
            logger.info("Model init done. Hidden layers: %s", config["model"].get("hidden_layers"))

            trainer = ModelTrainer(model, config, device)
            best_auc = trainer.train(train_loader, val_loader)

            # load checkpoint if trainer saved 'best_model.pth' in cwd, else use trainer return
            if os.path.exists("best_model.pth"):
                checkpoint = torch.load("best_model.pth", map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                torch.save(checkpoint, best_model_path)
            else:
                # save current model state
                torch.save({"model_state_dict": model.state_dict()}, best_model_path)

            # final model save
            torch.save(model.state_dict(), model_weights_path)
            #mlflow.log_artifact(best_model_path, artifact_path="models")
            #mlflow.log_artifact(model_weights_path, artifact_path="models")

            # Log best val metric
            try:
                #mlflow.log_metric("best_val_auc", float(best_auc))
                pass
            except Exception:
                pass

            # Save config used
            cfg_path = os.path.join(args.output_dir, "used_config.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            #mlflow.log_artifact(cfg_path, artifact_path="config")

            # Also try to include dvc.lock if exists
            if os.path.exists("dvc.lock"):
                pass
                #mlflow.log_artifact("dvc.lock", artifact_path="dvc")

            logger.info("Training finished. Best val AUC: %s", best_auc)
        except Exception as e:
            logger.exception("Training failed: %s", e)
            raise

if __name__ == "__main__":
    main()
