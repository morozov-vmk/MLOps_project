#!/usr/bin/env python3
# scripts/prepare.py
import argparse
import logging
import os
import numpy as np

from src.data_loader import DataProcessor
from src.utils import load_config, setup_logging, set_seed, get_device

def main():
    parser = argparse.ArgumentParser(description="Prepare data (preprocessing) and save numpy arrays")
    parser.add_argument("--config", type=str, default="config/model_config.yaml", help="Path to config")
    parser.add_argument("--output", type=str, default="data/processed/processed_data.npz", help="Output .npz path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)
    logger = logging.getLogger(__name__)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    set_seed(config["data"].get("random_seed", 42))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    try:
        logger.info("Starting data preparation...")
        processor = DataProcessor(config)
        X_train, y_train, X_val, y_val, X_test, y_test = processor.load_and_preprocess_data()
        logger.info("Saving processed arrays to %s", args.output)
        np.savez_compressed(
            args.output,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
        )
        logger.info("Data preparation finished successfully.")
    except Exception as e:
        logger.exception("Data preparation failed: %s", e)
        raise

if __name__ == "__main__":
    main()
