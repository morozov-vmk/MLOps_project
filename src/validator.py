import torch
import numpy as np
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                           recall_score, f1_score, confusion_matrix, classification_report,
                           roc_curve, precision_recall_curve)
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

logger = logging.getLogger(__name__)

class ModelValidator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def evaluate(self, test_loader):
        """Comprehensive model evaluation with detailed logging"""
        self.model.eval()
        all_targets = []
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                probabilities = output.squeeze().cpu().numpy()
                predictions = (probabilities > 0.5).astype(int)
                
                all_probabilities.extend(probabilities)
                all_predictions.extend(predictions)
                all_targets.extend(target.cpu().numpy())
        
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        metrics = {
            'accuracy': accuracy_score(all_targets, all_predictions),
            'auc': roc_auc_score(all_targets, all_probabilities),
            'precision': precision_score(all_targets, all_predictions, zero_division=0),
            'recall': recall_score(all_targets, all_predictions, zero_division=0),
            'f1': f1_score(all_targets, all_predictions, zero_division=0)
        }
        
        cm = confusion_matrix(all_targets, all_predictions)
        
        report = classification_report(all_targets, all_predictions, output_dict=True)
        
        self._log_test_results(metrics, cm, report, all_targets, all_probabilities)
        
        return metrics, cm, report, all_probabilities, all_targets
    
    def _log_test_results(self, metrics, cm, report, targets, probabilities):
        """Log test results in formatted way"""
        logger.info("")
        logger.info("=" * 70)
        logger.info("TEST SET EVALUATION RESULTS")
        logger.info("=" * 70)
        
        logger.info("METRICS:")
        logger.info("├─ AUC:       {:.4f}".format(metrics['auc']))
        logger.info("├─ Accuracy:  {:.4f}".format(metrics['accuracy']))
        logger.info("├─ Precision: {:.4f}".format(metrics['precision']))
        logger.info("├─ Recall:    {:.4f}".format(metrics['recall']))
        logger.info("└─ F1-Score:  {:.4f}".format(metrics['f1']))
        
        logger.info("")
        logger.info("CONFUSION MATRIX:")
        logger.info("┌─────────────┬─────────────┐")
        logger.info("│             │   Predicted │")
        logger.info("│             ├──────┬──────┤")
        logger.info("│             │   0  │   1  │")
        logger.info("├─────────────┼──────┼──────┤")
        logger.info("│ Actual   0  │ {:4d} │ {:4d} │".format(cm[0, 0], cm[0, 1]))
        logger.info("│         1  │ {:4d} │ {:4d} │".format(cm[1, 0], cm[1, 1]))
        logger.info("└─────────────┴──────┴──────┘")
        
        logger.info("")
        logger.info("CLASSIFICATION REPORT:")
        for class_name in ['0', '1', 'macro avg', 'weighted avg']:
            if class_name in report:
                class_report = report[class_name]
                if class_name in ['0', '1']:
                    logger.info("Class {}: precision={:.3f}, recall={:.3f}, f1={:.3f}".format(
                        class_name, class_report['precision'], class_report['recall'], class_report['f1-score']))
                else:
                    logger.info("{}: precision={:.3f}, recall={:.3f}, f1={:.3f}".format(
                        class_name.title(), class_report['precision'], class_report['recall'], class_report['f1-score']))
    
    def plot_curves(self, targets, probabilities, save_dir='.'):
        """Plot ROC and Precision-Recall curves"""
        fpr, tpr, _ = roc_curve(targets, probabilities)
        roc_auc = roc_auc_score(targets, probabilities)
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        precision, recall, _ = precision_recall_curve(targets, probabilities)
        avg_precision = precision_score(targets, (probabilities > 0.5).astype(int))
        
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall (AP = {avg_precision:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/validation_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_pretrained(self, save_path):
        """Save model in Hugging Face compatible format"""
        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")