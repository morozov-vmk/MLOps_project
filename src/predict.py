import argparse
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.model import CashbackMLP
from src.utils import load_config, setup_logging, get_device

logger = logging.getLogger(__name__)


def find_model_path(provided):
    candidates = []
    if provided:
        candidates.append(provided)
    candidates.extend(
        [
            "artifacts/model_weights.pth",
            "artifacts/best_model.pth",
            "best_model.pth",
            "model_weights.pth",
        ]
    )
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def load_model(model_path, config, device):
    if "model" not in config:
        config["model"] = {}
    model = CashbackMLP(config).to(device)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and all(
        k in state for k in ["epoch", "model_state_dict"]
    ):
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and any(k.startswith("layer") for k in state.keys()):
        try:
            model.load_state_dict(state)
        except Exception:
            if "state_dict" in state:
                model.load_state_dict(state["state_dict"])
            else:
                raise
    else:
        model.load_state_dict(state)
    model.eval()
    logger.info("Model loaded from %s", model_path)
    return model


def prepare_array_from_csv(df, expected_dim=None):
    id_col = None
    for c in ["id", "Id", "index", "idx"]:
        if c in df.columns:
            id_col = c
            break

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if id_col and id_col in df.columns and id_col not in numeric_df.columns:
        ids = df[id_col]
    else:
        ids = df.index

    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.fillna(numeric_df.median(numeric_only=True))

    X = numeric_df.to_numpy(dtype=np.float32)

    if expected_dim is not None and X.shape[1] != expected_dim:
        logger.warning(
            "CSV has %d numeric columns but expected %d. "
            "If columns mismatch, inference may be incorrect.",
            X.shape[1],
            expected_dim,
        )

    return X, ids


def batch_predict(model, X, device, batch_size=1024):
    tensor = torch.from_numpy(X.astype("float32")).to(device)
    probs = []
    with torch.no_grad():
        for i in range(0, tensor.shape[0], batch_size):
            batch = tensor[i : i + batch_size]
            out = model(batch)
            out = out.squeeze().cpu().numpy()
            probs.append(out)
    probs = np.concatenate(probs, axis=0)
    probs = probs.reshape(-1)
    return probs


def main():
    parser = argparse.ArgumentParser(description="Offline inference script")
    parser.add_argument("--config", type=str, default="config/model_config.yaml")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config = {}
    if os.path.exists(args.config):
        try:
            config = load_config(args.config)
        except Exception:
            logger.warning("Failed to load config from %s, continuing with empty config", args.config)

    try:
        if config and "logging" in config:
            setup_logging(config)
        else:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    except Exception:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    device = torch.device(args.device) if args.device else get_device()
    logger.info("Using device: %s", device)

    model_path = find_model_path(args.model_path)
    if not model_path:
        raise FileNotFoundError(
            "Model not found. Provide --model_path or put weights in artifacts/model_weights.pth or artifacts/best_model.pth"
        )

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path {input_path} does not exist")

    scaler = None
    for sc in ["scaler.pkl", "artifacts/scaler.pkl"]:
        if os.path.exists(sc):
            try:
                scaler = joblib.load(sc)
                logger.info("Loaded scaler from %s", sc)
                break
            except Exception:
                logger.warning("Failed to load scaler from %s", sc)

    if input_path.suffix in [".npz", ".npz".upper()]:
        arr = np.load(str(input_path), allow_pickle=True)
        if "X_test" in arr:
            X = arr["X_test"]
        elif "X" in arr:
            X = arr["X"]
        elif "features" in arr:
            X = arr["features"]
        else:
            keys = list(arr.keys())
            X = arr[keys[0]]
        ids = np.arange(len(X))
    elif input_path.suffix in [".csv", ".txt"]:
        df = pd.read_csv(str(input_path))
        X, ids = prepare_array_from_csv(df)
    else:
        try:
            df = pd.read_parquet(str(input_path))
            X, ids = prepare_array_from_csv(df)
        except Exception:
            raise ValueError("Unsupported input file type. Use .npz or .csv (or parquet).")

    expected_dim = None
    if config and "model" in config and "input_dim" in config["model"]:
        expected_dim = int(config["model"]["input_dim"])

    if not expected_dim:
        expected_dim = X.shape[1]
        if "model" not in config:
            config["model"] = {}
        config["model"]["input_dim"] = expected_dim
        logger.info("No input_dim in config; set model.input_dim = %d", expected_dim)
    else:
        if X.shape[1] != expected_dim:
            logger.warning(
                "Input has %d features but config expects %d. Proceeding (model will be constructed with expected_dim=%d).",
                X.shape[1],
                expected_dim,
                expected_dim,
            )

    model = load_model(model_path, config, device)

    if scaler is not None:
        try:
            X = scaler.transform(X)
            logger.info("Applied scaler.transform to input")
        except Exception as e:
            logger.warning("Failed to apply scaler: %s. Proceeding without scaler.", e)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    probs = batch_predict(model, X, device, batch_size=args.batch_size)
    preds = (probs > 0.5).astype(int)

    out_df = pd.DataFrame({"index": np.arange(len(probs)), "probability": probs, "prediction": preds})
    try:
        if ids is not None and len(ids) == len(probs):
            out_df.insert(0, "input_id", ids)
    except Exception:
        pass

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(str(out_path), index=False)
    logger.info("Saved predictions to %s", out_path)


if __name__ == "__main__":
    main()
