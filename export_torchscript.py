import json
import os
import argparse
from pathlib import Path

import torch
from src.model import CashbackMLP
from src.utils import load_config

def find_weights(paths):
    for p in paths:
        if p and Path(p).exists():
            return p
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/model_config.yaml")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights (.pth)")
    parser.add_argument("--out-dir", type=str, default="model_store_prepare/mymodel", help="Output dir for model.pt and extras")
    parser.add_argument("--scaler", type=str, default="artifacts/scaler.pkl", help="Optional scaler")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    config = {}
    if Path(args.config).exists():
        try:
            config = load_config(args.config)
        except Exception:
            pass

    candidates = [args.weights, "artifacts/model_weights.pth", "artifacts/best_model.pth", "best_model.pth", "model_weights.pth"]
    weights = find_weights(candidates)
    if not weights:
        raise FileNotFoundError("Model weights not found. Provide --weights or place file in artifacts/")

    input_dim = None
    if config and "model" in config and "input_dim" in config["model"]:
        input_dim = int(config["model"]["input_dim"])
    used_cfg = Path(args.out_dir) / "used_config.json"
    if not input_dim and Path("artifacts/used_config.json").exists():
        with open("artifacts/used_config.json", "r", encoding="utf-8") as f:
            used = json.load(f)
            input_dim = int(used.get("model", {}).get("input_dim", 0)) if used else None

    if not input_dim:
        state = torch.load(weights, map_location="cpu")
        found = False
        if isinstance(state, dict):
            possible_keys = list(state.keys())
            for k in possible_keys:
                if k.endswith(".weight") and len(state[k].shape) == 2:
                    input_dim = state[k].shape[1]
                    found = True
                    break
        if not found:
            raise ValueError("Cannot infer input_dim. Provide config with model.input_dim or use --weights with known model.")

    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    config = load_config(args.config)

    if "model" not in config:
        raise KeyError("Config must contain 'model' section")

    required_keys = ["input_dim", "hidden_layers"]
    for k in required_keys:
        if k not in config["model"]:
            raise KeyError(f"config['model'] must contain '{k}'")

    with open(Path(args.out_dir) / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    model = CashbackMLP(config)
    state = torch.load(weights, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    else:
        state_dict = state
    model.load_state_dict(state_dict)
    model.eval()

    example = torch.zeros(1, input_dim)
    try:
        scripted = torch.jit.script(model)
    except Exception:
        scripted = torch.jit.trace(model, example)

    out_path = Path(args.out_dir) / "model.pt"
    scripted.save(str(out_path))
    print("Saved torchscript model to", out_path)

    if args.scaler and Path(args.scaler).exists():
        import shutil
        shutil.copy(args.scaler, Path(args.out_dir) / "scaler.pkl")
        print("Copied scaler.pkl to", args.out_dir)

    print("Prepared model files in:", args.out_dir)
    print("Files:", list(Path(args.out_dir).iterdir()))

if __name__ == "__main__":
    main()
