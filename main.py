"""
Entry point for SPEGNet Camouflaged Object Detection.

This module provides a command-line interface for:
1. Training the model with optional validation
2. Evaluating on test datasets
3. Making predictions on new images

Directory Structure:
    ./
    ├── configs/
    │   └── default.yaml    # Default configuration
    ├── checkpoints/
    │   └── model_best.pth  # Default model checkpoint
    └── results/            # Operation outputs

Example Usage:
    Training:
        python main.py train --config configs/default.yaml 
    
    Evaluation:
        python main.py evaluate --model checkpoints/best.pth 
        
    Prediction:
        python main.py predict --model checkpoints/best.pth --input path/to/image
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path
from typing import Dict, Optional
import json

from ptflops import get_model_complexity_info

import torch

from engine.trainer import Trainer
from engine.evaluator import Evaluator
from engine.predictor import Predictor
from utils.data_loader import get_test_loaders
from utils.run_manager import DirectoryManager
from models.spegnet import SPEGNet

# Constants for default paths
DEFAULT_CONFIG_PATH = Path('./configs/default.yaml')
DEFAULT_MODEL_PATH = Path('./checkpoints/model_best.pth')

class ConfigurationManager:
    """
    Manages configuration loading and validation.
    
    Features:
    - Hierarchical config loading
    - Path validation
    - Config merging
    - Default handling
    """
    
    @staticmethod
    def load_config(config_path: Optional[Path] = None) -> Dict:
        """
        Load configuration with fallback to defaults.
        
        Args:
            config_path: Optional path to config file
            
        Returns:
            Loaded configuration dictionary
            
        Raises:
            RuntimeError: If no valid configuration found
        """
        # Try user config first
        if config_path and config_path.exists():
            try:
                with open(config_path) as f:
                    return yaml.safe_load(f)
            except yaml.YAMLError as e:
                logging.error(f"Error in config file {config_path}: {e}")
                raise RuntimeError(f"Invalid configuration file: {config_path}")
        
        # Try default config
        if DEFAULT_CONFIG_PATH.exists():
            try:
                with open(DEFAULT_CONFIG_PATH) as f:
                    return yaml.safe_load(f)
            except yaml.YAMLError as e:
                logging.error(f"Error in default config: {e}")
                raise RuntimeError("Invalid default configuration file")
                
        raise RuntimeError(
            "No valid configuration found. Ensure either:\n"
            f"1. User config exists at specified path, or\n"
            f"2. Default config exists at {DEFAULT_CONFIG_PATH}"
        )
    
    @staticmethod
    def load_model_config(model_path: Optional[Path] = None) -> Dict:
        """
        Load model configuration from checkpoint.
        
        Args:
            model_path: Optional path to model checkpoint
            
        Returns:
            Model configuration dictionary
            
        Raises:
            RuntimeError: If no valid model checkpoint found
        """
        # Determine model path
        checkpoint_path = model_path or DEFAULT_MODEL_PATH
        
        if not checkpoint_path.exists():
            raise RuntimeError(
                "No valid model checkpoint found. Ensure either:\n"
                f"1. Model exists at specified path, or\n"
                f"2. Default model exists at {DEFAULT_MODEL_PATH}"
            )
            
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'config' not in checkpoint:
                raise ValueError("Checkpoint does not contain configuration")
            return checkpoint['config']
            
        except Exception as e:
            logging.error(f"Error loading model checkpoint: {e}")
            raise RuntimeError(f"Invalid model checkpoint: {checkpoint_path}")
def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed command line arguments
        
    Raises:
        ArgumentError: If arguments are invalid for specified mode
        
    Notes:
        - Config defaults to './configs/default.yaml'
        - Model defaults to './checkpoints/model_best.pth'
        - Input path required only for predict mode
    """
    parser = argparse.ArgumentParser(
        description='SPEGNet: Synergistic Perception-Guided Network for Camouflaged Object Detection'
    )
    
    parser.add_argument('mode', 
                       choices=['train', 'evaluate', 'predict'],
                       help='Operation mode')
    
    parser.add_argument('--config',
                       type=Path,
                       help=f'Path to config file (default: {DEFAULT_CONFIG_PATH})')
    
    parser.add_argument('--model',
                       type=Path,
                       help=f'Path to model checkpoint (default: {DEFAULT_MODEL_PATH})')
    
    parser.add_argument('--input',
                       type=Path,
                       help='Input image or directory for prediction')
    
    args = parser.parse_args()
    
    # Validate predict mode requirements
    if args.mode == 'predict' and not args.input:
        parser.error("predict mode requires --input argument")
        
    return args

def setup_logging(dir_manager: DirectoryManager) -> None:
    """
    Configure logging system.
    
    Features:
    - Timestamp-based logging
    - Both file and console output
    - Proper formatting
    
    Args:
        dir_manager: Manages output directories
    """
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(dir_manager.run_dirs.log_file),
            logging.StreamHandler()
        ]
    )


def train(config: Dict, dir_manager: DirectoryManager) -> None:
    """
    Execute model training.
    
    Features:
    - Config-based training settings
    - Automatic mixed precision
    - Checkpoint management
    - Progress tracking
    
    Args:
        config: Training configuration dictionary
        dir_manager: Output directory manager
        
    Notes:
        Uses settings from config for:
        - Training parameters
        - Dataset handling
        - Optimization settings
    """
    logging.info("Initializing training...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Training on device: {device}")
    
    
    # Setup dataset paths
    dataset_paths = config['training']['datasets']
    if not dataset_paths:
        raise ValueError("No dataset paths provided. Use --datasets or config file.")
    
    logging.info(f"Training on datasets: {dataset_paths}")
    
    
    # Initialize trainer
    trainer = Trainer(
        config=config,
        dir_manager=dir_manager,
        device=device,
    )
    
    # Start training
    trainer.train(dataset_paths)

def evaluate(config: Dict, model_path: Path, dir_manager: DirectoryManager) -> None:
    """
    Execute model evaluation.
    
    Features:
    - Multi-dataset evaluation
    - Comprehensive metrics
    - Result visualization
    
    Args:
        config: Evaluation configuration
        model_path: Path to model checkpoint
        dir_manager: Output directory manager
        
    Notes:
        Uses settings from config for:
        - Batch processing
        - Dataset paths
        - Evaluation metrics
    """
    logging.info("Starting evaluation...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Evaluating on device: {device}")
    
    
    dataset_paths = config['evaluation']['datasets']
    if not dataset_paths:
        raise ValueError("No dataset paths provided for evaluation")
    
    test_loaders = get_test_loaders(
        dataset_dirs=dataset_paths,
        model_config=config['model'],
        batch_size=config['evaluation']['batch_size'],
        num_workers=config['evaluation']['num_workers']
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = Evaluator(
        model_path=model_path,
        model_config=config['model'],
        dir_manager=dir_manager,
        device=device,
        batch_size=config['evaluation']['batch_size']
    )
    
    all_metrics = {}
    
    # Evaluate each test set
    for dataset_name, loader in test_loaders.items():
        logging.info(f"\nEvaluating on {dataset_name}:")
        metrics = evaluator.evaluate(loader, dataset_name)
        all_metrics[dataset_name] = metrics
        
        # Log metrics
        logging.info("\nEvaluation Results:")
        logging.info(f"Structure measure (Sα): {metrics['s_alpha']:.4f}")
        logging.info(f"Weighted F-measure (Fβw): {metrics['weighted_f']:.4f}")
        logging.info(f"Mean Absolute Error (M): {metrics['mae']:.4f}")
        logging.info(f"Enhanced-alignment (Eϕ): {metrics['e_phi']:.4f}")
        logging.info(f"Mean F-measure (Fβm): {metrics['mean_f']:.4f}")

    # Save overall metrics summary
    metrics_path = dir_manager.run_dirs.root / 'metrics_summary.json'
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    logging.info(f"Metrics saved to {metrics_path}")

def predict(config: Dict, model_path: Path, input_path: Path, 
           dir_manager: DirectoryManager) -> None:
    """
    Execute model prediction.
    
    Features:
    - Single image processing
    - Batch directory processing
    - Result visualization
    
    Args:
        config: Prediction configuration
        model_path: Path to model checkpoint
        input_path: Path to input image/directory
        dir_manager: Output directory manager
        
    Notes:
        Uses settings from config for:
        - Image processing
        - Output formatting
        - Batch handling
    """

    logging.info("Starting prediction...")
    
    predictor = Predictor(
        model_path=model_path,
        model_config=config['model'],
        dir_manager=dir_manager,
        batch_size=config['prediction'].get('batch_size')
    )

    # Handle directory or single file input
    if input_path.is_dir():
        logging.info(f"Processing directory: {input_path}")
        results = predictor.predict_directory(
            input_dir=str(input_path),
            output_size=config['prediction'].get('output_size')
        )
        logging.info(f"Processed {len(results)} images")
    else:
        logging.info(f"Processing single image: {input_path}")
        seg_pred, edge_pred, original_image = predictor.predict_single(
            image_path=str(input_path),
            output_size=config['prediction'].get('output_size')
        )
        # Save results
        predictor.result_manager.save_prediction(
            input_path.name,
            seg_pred,
            edge_pred,
            original_image
        )
        logging.info(f"Processing complete, results saved")

def print_model_info(config: Dict) -> None:
    """
    Print model architecture and complexity information.
    Ensures cleanup of model from memory after printing.
    
    Args:
        config: Configuration dictionary containing model settings
        
    Note:
        This function temporarily creates a model instance for information
        gathering, then ensures proper cleanup to free memory.
    """
    logging.info("Analyzing model architecture and complexity...")
    
    try:
        # Initialize on appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SPEGNet(config['model']).to(device)
        model.eval()
        
        # Print model architecture
        logging.info("\nModel Architecture:")
        logging.info("=" * 50)
        logging.info(model)
        logging.info("=" * 50)
        
        # Calculate complexity
        input_size = (3, config['model']['image_processing']['target_size'],
                     config['model']['image_processing']['target_size'])
        
        with torch.cuda.device(device):
            flops, params = get_model_complexity_info(
                model,
                input_size,
                as_strings=True,
                print_per_layer_stat=False,
                verbose=False
            )
            
        logging.info("\nModel Complexity Analysis:")
        logging.info("-" * 30)
        logging.info(f"Computational Cost: {flops}")
        logging.info(f"Number of Parameters: {params}")
        logging.info("-" * 30 + "\n")
        
    except Exception as e:
        logging.warning(f"Could not complete model analysis: {str(e)}")
        logging.warning("Continuing with execution...")
        
    finally:
        # Explicit cleanup
        try:
            del model
            torch.cuda.empty_cache()
            logging.debug("Model cleanup completed")
        except NameError:
            # Model wasn't successfully created
            pass

def main() -> None:
    """
    Main execution pipeline.
    
    Flow:
    1. Argument Parsing:
       - Mode selection
       - Optional paths
    
    2. Configuration Loading:
       - User config or default
       - Model config from checkpoint
       - Setting validation
    
    3. Directory Setup:
       - Output organization
       - Logging configuration
    
    4. Mode Execution:
       - train: Full training pipeline
       - evaluate: Dataset evaluation
       - predict: Image processing
    """
    try:
        # Parse arguments
        args = parse_args()
        
        # Setup directory manager
        dir_manager = DirectoryManager(args.mode)
        setup_logging(dir_manager)
        
        # Load appropriate configurations
        config = ConfigurationManager.load_config(args.config)

        if args.mode in ['evaluate', 'predict']:
            model_config = ConfigurationManager.load_model_config(args.model)
            # Update model settings from checkpoint
            config['model'].update(model_config['model'])

        # Log configuration
        logging.info(f"Running in {args.mode} mode")
        logging.info("Configuration:")
        logging.info(yaml.dump(config, default_flow_style=False))

        # Print model information
        print_model_info(config)

        # Execute requested mode
        if args.mode == 'train':
            train(config, dir_manager)
        elif args.mode == 'evaluate':
            evaluate(config, args.model or DEFAULT_MODEL_PATH, dir_manager)
        else:  # predict
            predict(config, args.model or DEFAULT_MODEL_PATH, 
                   args.input, dir_manager)
        
        logging.info("Process completed successfully")
        
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()