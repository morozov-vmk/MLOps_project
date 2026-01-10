
import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from typing import Any, Dict

import mlflow
import mlflow.pytorch
import numpy as np
import torch

from src.data_loader import DataProcessor
from src.model import CashbackMLP
from src.trainer import ModelTrainer
from src.utils import load_config, setup_logging, set_seed, get_device


def flatten_config(cfg: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    for k, v in cfg.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_config(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def safe_str(v: Any) -> str:
    try:
        return json.dumps(v)
    except Exception:
        try:
            return str(v)
        except Exception:
            return "<unserializable>"


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "<no-git>"


def log_config_as_params(cfg: Dict[str, Any]):
    flat = flatten_config(cfg)
    for k, v in flat.items():
        try:
            mlflow.log_param(k, safe_str(v))
        except Exception:
            logging.getLogger(__name__).debug("Failed to log param %s", k, exc_info=True)


def main():
    parser = argparse.ArgumentParser(description="Train model (uses prepared .npz file)")
    parser.add_argument("--config", type=str, default="config/model_config.yaml")
    parser.add_argument("--processed", type=str, default="data/processed/processed_data.npz")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--mlflow-experiment", type=str, default="default")
    parser.add_argument("--run-name", type=str, default=None)
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

    mlflow.pytorch.autolog()

    mlflow.set_experiment(args.mlflow_experiment)

    with mlflow.start_run(run_name=args.run_name) as run:
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

            try:
                log_config_as_params(config)
            except Exception:
                logger.exception("Failed to log params")

            try:
                git_hash = get_git_commit_hash()
                mlflow.set_tag("git_commit", git_hash)
            except Exception:
                logger.debug("Failed to set git tag", exc_info=True)

            try:
                data_hash = sha256_of_file(args.processed)
                mlflow.set_tag("processed_data_sha256", data_hash)
            except Exception:
                logger.debug("Failed to compute data hash", exc_info=True)

            try:
                dvc_hash = sha256_of_file("dvc.lock")
                mlflow.set_tag("dvc_lock_sha256", dvc_hash)
                mlflow.log_artifact("dvc.lock", artifact_path="dvc")
            except Exception:
                logger.debug("Failed to log dvc.lock", exc_info=True)

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

            result = trainer.train(train_loader, val_loader)
            best_auc = None
            try:
                if isinstance(result, dict):
                    best_auc = (
                        result.get("best_val_auc")
                        or result.get("best_auc")
                        or result.get("best")
                    )
                    for k, v in result.items():
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(k, float(v))
                elif isinstance(result, (int, float)):
                    best_auc = float(result)
                else:
                    logger.debug("trainer.train returned %s", type(result))
            except Exception:
                logger.exception("Failed to interpret trainer.train result")

            if os.path.exists("best_model.pth"):
                checkpoint = torch.load("best_model.pth", map_location=device)
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                torch.save(checkpoint, best_model_path)
            else:
                torch.save({"model_state_dict": model.state_dict()}, best_model_path)

            torch.save(model.state_dict(), model_weights_path)

            try:
                mlflow.log_artifacts(args.output_dir, artifact_path="artifacts")
            except Exception:
                logger.exception("Failed to log artifacts dir %s", args.output_dir)

            try:
                if best_auc is not None:
                    mlflow.log_metric("best_val_auc", float(best_auc))
            except Exception:
                logger.debug("Failed to log best_val_auc", exc_info=True)

            cfg_path = os.path.join(args.output_dir, "used_config.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            try:
                mlflow.log_artifact(cfg_path, artifact_path="config")
            except Exception:
                logger.debug("Failed to log used_config.json", exc_info=True)

            logger.info("Training finished. Best val AUC: %s", best_auc)
        except Exception as e:
            logger.exception("Training failed: %s", e)
            raise


if __name__ == "__main__":
    main()
