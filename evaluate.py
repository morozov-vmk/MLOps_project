#!/usr/bin/env python3
# scripts/evaluate.py

import argparse
import logging
import os
import json

import numpy as np
import torch
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

    config = load_config(args.config)
    setup_logging(config)
    logger = logging.getLogger(__name__)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    device = get_device()
    logger.info("Using device: %s", device)

    os.makedirs(args.output_dir, exist_ok=True)

    if 1:
        try:
            logger.info("Loading processed data from %s", args.processed)
            data = np.load(args.processed, allow_pickle=True)
            X_test = data["X_test"]
            y_test = data["y_test"]

            logger.info("Initializing model...")
            model = CashbackMLP(config).to(device)

            logger.info("Loading model weights from %s", args.model_weights)
            state = torch.load(args.model_weights, map_location=device)
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                model.load_state_dict(state)

            model.eval()

            logger.info("Creating test DataLoader...")
            processor = DataProcessor(config)
            test_loader = processor.create_test_loader(X_test, y_test)

            logger.info("Running evaluation...")
            validator = ModelValidator(model, device)
            test_metrics, cm, report, probabilities, targets = validator.evaluate(
                test_loader
            )

            logger.info("Test metrics: %s", test_metrics)

            metrics_path = os.path.join(args.output_dir, "test_metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(test_metrics, f, ensure_ascii=False, indent=2)

            report_path = os.path.join(args.output_dir, "classification_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            logger.info("Saving ROC / PR curves...")
            plot_path = os.path.join(args.output_dir, "roc_curve.png")
            validator.plot_curves(targets, probabilities)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

            logger.info("Evaluation completed successfully")

        except Exception as e:
            logger.exception("Evaluation failed")
            raise


if __name__ == "__main__":
    main()
