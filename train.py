import argparse
import os
from src.utils import setup_logging, load_config, set_seed, get_device
from src.utils import save_pretrained_hf
from src.data_loader import DataProcessor
from src.model import CashbackMLP
from src.trainer import ModelTrainer
from src.validator import ModelValidator
import logging

def main():
    parser = argparse.ArgumentParser(description='Train Cashback Prediction Model')
    parser.add_argument('--config', type=str, default='config/model_config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    set_seed(config["data"]["random_seed"])
    
    device = get_device()
    logger.info(f"Using device: {device}")
    
    try:
        logger.info("Loading and preprocessing data from parquet files...")
        processor = DataProcessor(config)
        X_train, y_train, X_val, y_val, X_test, y_test = processor.load_and_preprocess_data()
        
        actual_input_dim = X_train.shape[1]
        config["model"]["input_dim"] = actual_input_dim
        logger.info(f"Updated input_dim to: {actual_input_dim}")
        
        train_loader, val_loader, test_loader = processor.create_data_loaders(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        logger.info("Initializing model...")
        model = CashbackMLP(config).to(device)
        
        logger.info(f"Model architecture: {config['model']['hidden_layers']}")
        
        trainer = ModelTrainer(model, config, device)
        best_auc = trainer.train(train_loader, val_loader)
        
        checkpoint = torch.load('best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info("Evaluating on test set...")
        validator = ModelValidator(model, device)
        test_metrics, cm, report, probabilities, targets = validator.evaluate(test_loader)
        
        validator.plot_curves(targets, probabilities)
        
        validator.save_pretrained('model_weights.pth')
        
        logger.info("")
        logger.info("=" * 50)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)
        logger.info(f"Best Validation AUC: {best_auc:.4f}")
        logger.info(f"Test AUC: {test_metrics['auc']:.4f}")
        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test F1-Score: {test_metrics['f1']:.4f}")
        
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()