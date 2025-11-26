import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

from src.data_loading.load_data import (load_gt_data, load_pred_data,
                                        prepare_data)
from src.evaluation.metrics import (calculate_improvement, evaluate_model)
from src.model.gnn import create_summary_report, GNNDataModule, ResidualGNN
from src.utils.logging import logger


class GNNGridSearch:
    """Grid search for GNN hyperparameters."""
    
    def __init__(self, output_dir: str = "gnn_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Define hyperparameter grid
        self.param_grid = {
        "hidden_dims": [[16, 8], [8, 4], [16, 8, 4]],
        "hidden_heads": [[8,4], [4,2], [8,4,2]],
        "dropout": [0.1, 0.2],
        "negative_slope": [0.0, 0.2],
        "learning_rate": [1e-3, 5e-4],
        "weight_decay": [1e-4, 1e-5],
        "use_confidence": [True, False],
        "batch_size": [64, 128],
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
    
    def _train_single_config(self, X_train, y_train, X_val, y_val, skeleton, params, max_epochs):
        """Train a single configuration."""
        # Determine input dimension
        input_dim = 90 if params['use_confidence'] else 60
        
        # Create model
        model = ResidualGNN(
            input_dim=input_dim,
            hidden_dims=params['hidden_dims'],
            hidden_heads=params['hidden_heads'],
            skeleton=skeleton,
            negative_slope=params['negative_slope'],
            dropout=params['dropout'],
            learning_rate=params['learning_rate'],
            weight_decay=params['weight_decay'],
            )
        
        # Create data module
        datamodule = GNNDataModule(
            X_train, y_train, X_val, y_val, batch_size=params['batch_size']
        )
        
        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=self.output_dir / 'checkpoints',
            filename=f"gnn-{hash(str(params))}",
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

    def evaluate_best_model(self, best_result: dict, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate the best model on test data and return comprehensive metrics."""
        logger.info("Evaluating best model on test set...")
        
        try:
            # Load the best model
            model = ResidualGNN.load_from_checkpoint(best_result['best_model_path'])
            model.eval()
            
            # Move to GPU if available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            # Prepare test data
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            
            with torch.no_grad():
                y_pred = model(X_test_tensor).cpu().numpy()
            
            # Calculate metrics
            metrics = evaluate_model(model, X_test, y_test, device=str(device))
            
            # Additional metrics comparing to original predictions
            original_pred = X_test[:, :60]  # Original coordinates
            improvement_metrics = calculate_improvement(original_pred, y_pred, y_test)
            metrics.update(improvement_metrics)
            
            logger.info("Test Set Metrics:")
            logger.info(f"  MPJPE: {metrics['mpjpe']:.4f} pixels")
            logger.info(f"  PCK@5: {metrics['pck_5']:.2f}%")
            logger.info(f"  PCK@10: {metrics['pck_10']:.2f}%")
            logger.info(f"  Improvement: {metrics['relative_improvement']:.2f}%")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating best model: {e}")
            return {}


def main():
    parser = argparse.ArgumentParser(description='Train GNN baseline with grid search')
    parser.add_argument('--data_path', type=str, default='data/fly')
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--output_dir', type=str, default='gnn_results')
    parser.add_argument('--eval_on_ood', action='store_true', default=True, 
                       help='Evaluate on OOD test data')
    args = parser.parse_args()
    
    # Load data
    logger.info("Loading training data...")
    gt_train = load_gt_data(args.data_path, ood=False)
    pred_train = load_pred_data(args.data_path, ood=False)
    
    # Prepare GNN data (we'll try both with and without confidence in grid search)
    X, y = prepare_data(gt_train, pred_train, use_confidence=True)
    
    logger.info(f"Data shape - X: {X.shape}, y: {y.shape}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    
    logger.info(f"Train/val split - X_train: {X_train.shape}, X_val: {X_val.shape}")
    
    # Run grid search
    grid_search = GNNGridSearch(output_dir=args.output_dir)
    results = grid_search.run_grid_search(X_train, y_train, X_val, y_val, args.max_epochs)
    
    # Print best results and evaluate
    if results:
        best_result = min(results, key=lambda x: x['best_val_loss'])
        logger.info(f"Best configuration: {best_result['params']}")
        logger.info(f"Best validation loss: {best_result['best_val_loss']:.6f}")
        
        # Evaluate on test data
        if args.eval_on_ood:
            logger.info("Loading OOD test data...")
            gt_test = load_gt_data(args.data_path, ood=True)
            pred_test = load_pred_data(args.data_path, ood=True)
            
            X_test, y_test = prepare_data(
                gt_test, pred_test, 
                use_confidence=best_result['params']['use_confidence']
            )
            
            logger.info(f"Test data shape: X_test: {X_test.shape}, y_test: {y_test.shape}")
            
            if len(X_test) > 0:
                # Evaluate best model on test set
                test_metrics = grid_search.evaluate_best_model(best_result, X_test, y_test)
                
                # Save final results
                final_results = {
                    'best_hyperparameters': best_result['params'],
                    'best_validation_loss': float(best_result['best_val_loss']),
                    'test_metrics': test_metrics,
                    'data_shapes': {
                        'train': X_train.shape,
                        'val': X_val.shape,
                        'test': X_test.shape
                    },
                    'timestamp': pd.Timestamp.now().isoformat()
                }
                
                # Save final results
                results_path = Path(args.output_dir) / 'final_results.json'
                with open(results_path, 'w') as f:
                    json.dump(final_results, f, indent=2, default=str)
                
                logger.info(f"Final results saved to: {results_path}")
                
                # Create a summary report
                create_summary_report(final_results, args.output_dir)
            else:
                logger.error("No test data available for evaluation!")
        else:
            logger.info("Skipping OOD evaluation as requested")
        
    else:
        logger.error("No successful training runs!")


if __name__ == "__main__":
    main()