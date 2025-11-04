import logging
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device

        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", patience=5, factor=0.5
        )

        self.best_val_auc = 0
        self.early_stopping_counter = 0
        self.train_losses = []
        self.val_metrics_history = []

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        batch_losses = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")

        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output.squeeze(), target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            batch_losses.append(loss.item())

            if batch_idx % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{np.mean(batch_losses[-10:]):.4f}"})

        return total_loss / len(train_loader)

    def validate(self, val_loader, epoch):
        """Validate model with detailed metrics"""
        self.model.eval()
        all_targets = []
        all_predictions = []
        all_probabilities = []
        val_loss = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output.squeeze(), target)
                val_loss += loss.item()

                probabilities = output.squeeze().cpu().numpy()
                predictions = (probabilities > 0.5).astype(int)

                all_probabilities.extend(probabilities)
                all_predictions.extend(predictions)
                all_targets.extend(target.cpu().numpy())

        if len(all_targets) == 0:
            return {
                "loss": 0.0,
                "accuracy": 0.0,
                "auc": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }

        all_probabilities = np.array(all_probabilities)
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        try:
            accuracy = accuracy_score(all_targets, all_predictions)
        except:
            accuracy = 0.0

        try:
            auc = roc_auc_score(all_targets, all_probabilities)
        except:
            auc = 0.0

        try:
            precision = precision_score(all_targets, all_predictions, zero_division=0)
        except:
            precision = 0.0

        try:
            recall = recall_score(all_targets, all_predictions, zero_division=0)
        except:
            recall = 0.0

        try:
            f1 = f1_score(all_targets, all_predictions, zero_division=0)
        except:
            f1 = 0.0

        metrics = {
            "loss": val_loss / max(len(val_loader), 1),
            "accuracy": accuracy,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        return metrics

    def log_metrics(self, epoch, train_loss, val_metrics):
        """Log metrics in a formatted way"""
        logger.info("┌" + "─" * 60 + "┐")
        logger.info(
            f"│ Epoch {epoch+1:3d}/{self.config['training']['num_epochs']:3d} "
            + " " * 35
            + "│"
        )
        logger.info("├" + "─" * 60 + "┤")
        logger.info(f"│ Train Loss: {train_loss:.4f}" + " " * 42 + "│")
        logger.info("│ Validation Metrics:" + " " * 38 + "│")
        logger.info(f"│   AUC:       {val_metrics['auc']:.4f}" + " " * 42 + "│")
        logger.info(f"│   Accuracy:  {val_metrics['accuracy']:.4f}" + " " * 40 + "│")
        logger.info(f"│   Precision: {val_metrics['precision']:.4f}" + " " * 40 + "│")
        logger.info(f"│   Recall:    {val_metrics['recall']:.4f}" + " " * 41 + "│")
        logger.info(f"│   F1-Score:  {val_metrics['f1']:.4f}" + " " * 41 + "│")
        logger.info(f"│   Loss:      {val_metrics['loss']:.4f}" + " " * 42 + "│")
        logger.info("└" + "─" * 60 + "┘")

    def train(self, train_loader, val_loader):
        """Full training loop with detailed logging"""
        logger.info("Starting training...")
        logger.info(
            f"Model will be evaluated on {len(val_loader.dataset)} validation samples"
        )
        logger.info(
            f"Target metric: AUC (early stopping patience: {self.config['training']['early_stopping_patience']})"
        )

        start_time = time.time()

        for epoch in range(self.config["training"]["num_epochs"]):
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)

            val_metrics = self.validate(val_loader, epoch)
            self.val_metrics_history.append(val_metrics)

            self.scheduler.step(val_metrics["auc"])

            self.log_metrics(epoch, train_loss, val_metrics)

            if val_metrics["auc"] > self.best_val_auc:
                self.best_val_auc = val_metrics["auc"]
                self.early_stopping_counter = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_val_auc": self.best_val_auc,
                        "val_metrics": val_metrics,
                        "config": self.config,
                    },
                    "best_model.pth",
                )

                logger.info(f"New best model saved! AUC: {val_metrics['auc']:.4f}")
            else:
                self.early_stopping_counter += 1
                logger.info(
                    f"Early stopping counter: {self.early_stopping_counter}/{self.config['training']['early_stopping_patience']}"
                )

            if (
                self.early_stopping_counter
                >= self.config["training"]["early_stopping_patience"]
            ):
                logger.info(f"🛑 Early stopping triggered at epoch {epoch+1}")
                break

        training_time = time.time() - start_time
        logger.info(
            f"✅ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)"
        )
        logger.info(f"Best validation AUC: {self.best_val_auc:.4f}")

        return self.best_val_auc
