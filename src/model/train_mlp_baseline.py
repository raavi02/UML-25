import argparse
import itertools
from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.data_loading.load_data import load_gt_data, load_pred_data, prepare_mlp_data
from src.model.mlp_baseline import ResidualMLP, MLPDataModule
from src.utils.logging import logger


class MLPGridSearch:
    """Grid search for MLP hyperparameters."""
    
    def __init__(self, output_dir: str = "mlp_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Define hyperparameter grid
        self.param_grid = {
            'hidden_dims': [[512, 256], [256, 128], [512, 256, 128]],
            'dropout': [0.1, 0.2],
            'learning_rate': [1e-3, 5e-4],
            'weight_decay': [1e-4, 1e-5],
            'use_confidence': [True, False],
            'batch_size': [64, 128]
        }
    
    def generate_param_combinations(self):
        """Generate all parameter combinations."""
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        return [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    
    def run_grid_search(self, X_train, y_train, X_val, y_val, max_epochs: int = 100):
        """Run grid search over hyperparameters."""
        param_combinations = self.generate_param_combinations()
        results = []
        
        logger.info(f"Starting grid search with {len(param_combinations)} combinations")
        
        for i, params in enumerate(param_combinations):
            logger.info(f"Training combination {i+1}/{len(param_combinations)}: {params}")
            
            try:
                result = self._train_single_config(
                    X_train, y_train, X_val, y_val, params, max_epochs
                )
                results.append(result)
                
                # Save intermediate results
                self._save_results(results)
                
            except Exception as e:
                logger.error(f"Failed training with params {params}: {e}")
                continue
        
        return results
    
    def _train_single_config(self, X_train, y_train, X_val, y_val, params, max_epochs):
        """Train a single configuration."""
        # Determine input dimension
        input_dim = 90 if params['use_confidence'] else 60
        
        # Create model
        model = ResidualMLP(
            input_dim=input_dim,
            hidden_dims=params['hidden_dims'],
            dropout=params['dropout'],
            learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        # Create data module
        datamodule = MLPDataModule(
            X_train, y_train, X_val, y_val, batch_size=params['batch_size']
        )
        
        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=self.output_dir / 'checkpoints',
            filename=f"mlp-{hash(str(params))}",
            save_top_k=1,
            mode='min'
        )
        
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=15,
            mode='min'
        )
        
        # Trainer
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[checkpoint_callback, early_stop_callback],
            enable_progress_bar=True,
            log_every_n_steps=10,
            accelerator='auto'
        )
        
        # Train
        trainer.fit(model, datamodule)
        
        # Get best validation metrics
        best_val_loss = checkpoint_callback.best_model_score.item()
        
        return {
            'params': params,
            'best_val_loss': best_val_loss,
            'best_model_path': str(checkpoint_callback.best_model_path)
        }
    
    def _save_results(self, results):
        """Save results to JSON file."""
        # Convert to serializable format
        serializable_results = []
        for result in results:
            serializable_result = {
                'params': result['params'],
                'best_val_loss': float(result['best_val_loss']),
                'best_model_path': result['best_model_path']
            }
            serializable_results.append(serializable_result)
        
        # Sort by validation loss
        serializable_results.sort(key=lambda x: x['best_val_loss'])
        
        with open(self.output_dir / 'grid_search_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Train MLP baseline with grid search')
    parser.add_argument('--data_path', type=str, default='data/fly')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--output_dir', type=str, default='mlp_results')
    args = parser.parse_args()
    
    # Load data
    logger.info("Loading training data...")
    gt_train = load_gt_data(args.data_path, ood=False)
    pred_train = load_pred_data(args.data_path, ood=False)
    
    # Prepare MLP data (we'll try both with and without confidence in grid search)
    X, y = prepare_mlp_data(gt_train, pred_train, use_confidence=True)
    
    logger.info(f"Data shape - X: {X.shape}, y: {y.shape}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    
    logger.info(f"Train/val split - X_train: {X_train.shape}, X_val: {X_val.shape}")
    
    # Run grid search
    grid_search = MLPGridSearch(output_dir=args.output_dir)
    results = grid_search.run_grid_search(X_train, y_train, X_val, y_val, args.max_epochs)
    
    # Print best results
    if results:
        best_result = min(results, key=lambda x: x['best_val_loss'])
        logger.info(f"Best configuration: {best_result['params']}")
        logger.info(f"Best validation loss: {best_result['best_val_loss']:.6f}")
        
        # Evaluate on OOD data
        logger.info("Loading OOD test data...")
        gt_test = load_gt_data(args.data_path, ood=True)
        pred_test = load_pred_data(args.data_path, ood=True)
        
        X_test, y_test = prepare_mlp_data(
            gt_test, pred_test, 
            use_confidence=best_result['params']['use_confidence']
        )
        
        logger.info(f"Test data shape: X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        # TODO: Load best model and evaluate on test set
        # This would require saving the model architecture and loading it
        
    else:
        logger.error("No successful training runs!")


if __name__ == "__main__":
    main()