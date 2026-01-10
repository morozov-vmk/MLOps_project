#!/usr/bin/env python3
# scripts/evaluate.py

import argparse
import logging
import os
import json

import numpy as np
import torch
#import mlflow
#import mlflow.pytorch
import matplotlib.pyplot as plt

from src.model import CashbackMLP
from src.validator import ModelValidator
from src.data_loader import DataProcessor
from src.utils import load_config, setup_logging, get_device


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set")
    parser.add_argument(
        "--config",
        type=str,
        default="config/model_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--processed",
        type=str,
        default="data/processed/processed_data.npz",
        help="Path to processed data (.npz)",
    )
    parser.add_argument(
        "--model-weights",
        type=str,
        default="artifacts/model_weights.pth",
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/eval",
        help="Directory to save evaluation artifacts",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # ---------- setup ----------
    config = load_config(args.config)
    setup_logging(config)
    logger = logging.getLogger(__name__)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    device = get_device()
    logger.info("Using device: %s", device)

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------- MLflow ----------
    # по умолчанию пишет в ./mlruns
    #mlflow.pytorch.autolog(log_models=False)

    if 1: #with mlflow.start_run():
        try:
            # ---------- load data ----------
            logger.info("Loading processed data from %s", args.processed)
            data = np.load(args.processed, allow_pickle=True)
            X_test = data["X_test"]
            y_test = data["y_test"]

            # ---------- init model ----------
            logger.info("Initializing model...")
            model = CashbackMLP(config).to(device)

            logger.info("Loading model weights from %s", args.model_weights)
            state = torch.load(args.model_weights, map_location=device)
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                model.load_state_dict(state)

            model.eval()

            # ---------- create test loader ----------
            logger.info("Creating test DataLoader...")
            processor = DataProcessor(config)
            test_loader = processor.create_test_loader(X_test, y_test)

            # ---------- evaluation ----------
            logger.info("Running evaluation...")
            validator = ModelValidator(model, device)
            test_metrics, cm, report, probabilities, targets = validator.evaluate(
                test_loader
            )

            logger.info("Test metrics: %s", test_metrics)

            # ---------- save metrics ----------
            metrics_path = os.path.join(args.output_dir, "test_metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(test_metrics, f, ensure_ascii=False, indent=2)
            #mlflow.log_artifact(metrics_path, artifact_path="evaluation")

            report_path = os.path.join(args.output_dir, "classification_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            #mlflow.log_artifact(report_path, artifact_path="evaluation")

            # ---------- plots ----------
            logger.info("Saving ROC / PR curves...")
            plot_path = os.path.join(args.output_dir, "roc_curve.png")
            validator.plot_curves(targets, probabilities)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            #mlflow.log_artifact(plot_path, artifact_path="evaluation/plots")

            # ---------- log metrics to mlflow ----------
            for name, value in test_metrics.items():
                try:
                    #mlflow.log_metric(f"test_{name}", float(value))
                    pass
                except Exception:
                    pass

            # ---------- log dvc.lock ----------
            if os.path.exists("dvc.lock"):
                #mlflow.log_artifact("dvc.lock", artifact_path="dvc")
                pass

            logger.info("Evaluation completed successfully")

        except Exception as e:
            logger.exception("Evaluation failed")
            raise


if __name__ == "__main__":
    main()
