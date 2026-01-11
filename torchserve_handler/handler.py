import json
import os
import io
import logging
import joblib
import numpy as np
import torch

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class CashbackHandler(BaseHandler):
    def initialize(self, context):
        super().initialize(context)
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        logger.info(f"Initializing handler. model_dir={model_dir}")

        model_pt = os.path.join(model_dir, "model.pt")
        if os.path.exists(model_pt):
            self.model = torch.jit.load(model_pt, map_location="cpu")
            self.model.eval()
            logger.info("Loaded torchscript model from %s", model_pt)
        else:
            raise FileNotFoundError("model.pt not found in model archive.")

        scaler_path = os.path.join(model_dir, "scaler.pkl")
        self.scaler = None
        if os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded scaler from %s", scaler_path)
            except Exception as e:
                logger.warning("Failed to load scaler: %s", e)

        cfg_path = os.path.join(model_dir, "config.json")
        self.config = {}
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
                logger.info("Loaded config.json")
            except Exception:
                logger.warning("Failed to load config.json")

        self.device = torch.device("cpu")

    def preprocess(self, data):
        body = None
        if isinstance(data, list) and len(data) > 0:
            entry = data[0]
            if 'body' in entry:
                raw = entry['body']
                try:
                    if isinstance(raw, (bytes, bytearray)):
                        s = raw.decode('utf-8')
                        body = json.loads(s)
                    elif isinstance(raw, str):
                        body = json.loads(raw)
                    else:
                        body = raw
                except Exception:
                    body = raw
            elif 'data' in entry:
                body = entry['data']
            else:
                body = entry
        else:
            body = data

        if isinstance(body, dict) and 'instances' in body:
            arr = body['instances']
        elif isinstance(body, list):
            arr = body
        else:
            raise ValueError("Unsupported input format. Expect JSON list or {'instances': [...]}")

        X = np.array(arr, dtype=np.float32)

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if self.scaler is not None:
            try:
                X = self.scaler.transform(X)
            except Exception as e:
                logger.warning("Scaler transform failed: %s", e)

        return X

    def inference(self, X):
        if isinstance(X, np.ndarray):
            tensor = torch.from_numpy(X.astype("float32")).to(self.device)
        else:
            raise ValueError("Inference expects numpy array.")

        with torch.no_grad():
            out = self.model(tensor)
            probs = out.squeeze().cpu().numpy()

        probs = np.array(probs).reshape(-1)
        return probs

    def postprocess(self, inference_output):
        probs = np.array(inference_output).reshape(-1)
        preds = (probs > 0.5).astype(int)
        results = []
        for p, pr in zip(probs.tolist(), preds.tolist()):
            results.append({"probability": float(p), "prediction": int(pr)})
        return results
