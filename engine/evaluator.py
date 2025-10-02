"""
Evaluator module for Camouflaged Object Detection.

This module provides:
1. Model evaluation on test datasets
2. Comprehensive metric computation and analysis
3. Quality-based result categorization and storage
4. Visual result organization and validation
5. Performance analysis and timing statistics

Processing Flow:
1. Model Loading and Warmup
   - Checkpoint loading
   - Device placement
   - Inference warmup runs

2. Batch Processing
   - Efficient data handling
   - Mixed precision inference
   - Multi-scale prediction resizing

3. Result Analysis
   - Metric computation
   - Quality categorization
   - Statistical aggregation

4. Result Organization
   - Category-based storage
   - Visual result generation
   - Performance summaries

Input Requirements:
    DataLoader providing batches with:
    - images: torch.Tensor[B, C, H, W]  # Normalized images
    - masks: List[torch.Tensor[1, H, W]] # Original size masks
    - names: List[str]                   # Image filenames
"""

import logging
import time
import json
from typing import Dict, List,Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from models.spegnet import SPEGNet
from utils.metrics import MetricsProcessor
from utils.visualization import ResultVisualizer
from utils.run_manager import DirectoryManager

@dataclass
class EvaluationResult:
    """
    Container for single sample evaluation results.
    
    Attributes:
        filename (str): Source image filename
        seg_pred (torch.Tensor): Final segmentation prediction
        seg_stages (List[torch.Tensor]): Multi-scale predictions
        edge_pred (torch.Tensor): Edge map prediction
        metrics (Dict[str, float]): Performance metrics
        original_image (np.ndarray): Input image for visualization
    """
    filename: str
    seg_pred: torch.Tensor  # Resized final prediction
    seg_stages: List[torch.Tensor]  # All stage predictions at original sizes
    edge_pred: torch.Tensor
    metrics: Dict[str, float]
    original_image: np.ndarray

class ResultManager:
    """
    Manages evaluation result storage and organization.
    
    Features:
    - Quality-based categorization (good/medium/bad)
    - Result visualization management
    - Metric logging and organization
    - Directory structure maintenance
    """
    
    def __init__(self, dir_manager: DirectoryManager):
        """
        Initialize result management system.
        
        Args:
            dir_manager: Provides directory structure for results
                    
        Directory Structure:
            visualizations/
            ├── good/
            │   ├── segmentation/
            │   └── edges/
            ├── medium/
            │   ├── segmentation/
            │   └── edges/
            └── bad/
                ├── segmentation/
                └── edges/
        """
        # Store directory manager reference
        self.run_dirs = dir_manager.run_dirs
        self.dataset_dirs = {}  # Store dataset-specific directories
        
        # Log initialization
        logging.info("Initialized result directory structure")
    
    def setup_dataset_directories(self, dataset_name: str):
        """Create directory structure for a specific dataset."""
        dataset_root = self.run_dirs.root / dataset_name
        
        # Create visualization directories
        viz_dir = dataset_root / 'visualizations'
        # Create metrics directory
        metrics_dir = dataset_root / 'metrics'

        for category in ['good', 'medium', 'bad']:
            (viz_dir / category / 'segmentation').mkdir(parents=True, exist_ok=True)
            (viz_dir / category / 'edges').mkdir(parents=True, exist_ok=True)
            (metrics_dir / category).mkdir(parents=True, exist_ok=True)
            
        # Store paths for this dataset
        self.dataset_dirs[dataset_name] = {
            'root': dataset_root,
            'visualizations': viz_dir,
            'metrics': metrics_dir
        }

    def determine_quality_category(self, metrics: Dict[str, float]) -> str:
        """
        Determine result quality based on both S-measure and F-measure.
        
        Args:
            metrics: Dictionary containing evaluation metrics
            
        Returns:
            Category string ('good', 'medium', or 'bad')
        """
        s_alpha = metrics['s_alpha']
        weighted_f = metrics['weighted_f']
        
        # Both measures must meet threshold for category
        if s_alpha >= 0.8 and weighted_f >= 0.8:
            return 'good'
        elif s_alpha >= 0.6 and weighted_f >= 0.6:
            return 'medium'
        return 'bad'
    
    def save_prediction(self, result: EvaluationResult, dataset_name: str) -> None:
        """
        Save prediction results in organized structure.
        
        Organization:
        1. Quality-based categorization
        2. Separate storage for segmentation and edges
        3. Multi-scale visualization generation
        4. Metric data storage
        
        Args:
            result: Evaluation data container
            dataset_name: Current dataset identifier
            
        Directory Structure:
            dataset/
            ├── visualizations/
            │   ├── good/
            │   ├── medium/
            │   └── bad/
            └── metrics/
        """

        # Ensure dataset directories exist
        if dataset_name not in self.dataset_dirs:
            self.setup_dataset_directories(dataset_name)
            
        # Determine quality category
        category = self.determine_quality_category(result.metrics)        
        # Get paths for this dataset
        viz_dir = self.dataset_dirs[dataset_name]['visualizations']
        # Save predictions in appropriate category
        category_dir = viz_dir / category
        
        try:
            # Save segmentation visualizations
            seg_vis = ResultVisualizer(category_dir / 'segmentation')
            # Convert tensors to numpy
            seg_pred_np = result.seg_pred.cpu().numpy().squeeze()
            stage_preds_np = [pred.cpu().numpy().squeeze() for pred in result.seg_stages]
        
            # Save all visualizations including stages
            seg_vis.save_all_visualizations(
                result.filename,
                seg_pred_np,
                result.original_image,
                stage_preds_np
            )
            
            # Save edge visualizations
            edge_vis = ResultVisualizer(category_dir / 'edges')
            edge_vis.save_all_visualizations(
                result.filename,
                result.edge_pred.cpu().numpy().squeeze(),
                result.original_image
            )
            
            # Save metrics
            metrics_file = self.dataset_dirs[dataset_name]['metrics'] / category / f'{result.filename}_metrics.json'
            with open(metrics_file, 'w') as f:
                json.dump(result.metrics, f, indent=4)
                
        except Exception as e:
            logging.error(f"Failed to save prediction {result.filename}: {str(e)}")
            
            
            
    def get_category_summary(self, dataset_name: str) -> Dict[str, Dict[str, int]]:
        """
        Generate summary of prediction distribution across categories for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset to summarize
            
        Returns:
            Dictionary containing:
                counts: Number of predictions in each category
                total: Total number of predictions
                
        Example:
            {
                'counts': {'good': 150, 'medium': 80, 'bad': 20},
                'total': 250
            }
        """
        if dataset_name not in self.dataset_dirs:
            raise ValueError(f"Dataset {dataset_name} not found in results")
            
        summary = {'counts': {}, 'total': 0}
        viz_dir = self.dataset_dirs[dataset_name]['visualizations']
        
        # Count predictions in each category
        for category in ['good', 'medium', 'bad']:
            # Count binary segmentation predictions as the source of truth
            seg_dir = viz_dir / category / 'segmentation' / 'binary'
            if seg_dir.exists():
                # Get the list of files 
                image_files = list(seg_dir.glob('*.png'))
                base_names = set()
                for f in image_files:
                    # Remove _stage1, _stage2 etc from name
                    base_name = f.stem.split('_stage')[0]
                    base_names.add(base_name)
                count = len(base_names)
            else:
                count = 0
                
            summary['counts'][category] = count
            summary['total'] += count
            
        return summary


class Evaluator:
    """
    SPEGNet Camouflaged Object Detection evaluator.
    
    This class handles the complete evaluation pipeline:
    1. Model loading and initialization
    2. Batch processing and inference
    3. Metric computation and tracking
    4. Result organization and storage
    
    Features:
    - Efficient batch processing
    - Comprehensive timing analysis
    - Quality-based result categorization
    - Memory-optimized prediction handling
    
    The evaluation process tracks multiple metrics:
    - Structure measure (Sα)
    - Weighted F-measure (Fβw)
    - Mean Absolute Error (M)
    - Enhanced-alignment measure (Eϕ)
    - Mean F-measure (Fβm)
    """
    
    def __init__(self, model_path: str, dir_manager: DirectoryManager, model_config: Dict, device : torch.device,batch_size : int):
        """
        Initialize evaluator components.
        
        Args:
            model_path: Path to model checkpoint (.pth file)
            dir_manager: Directory manager for result organization
            model_config: Dictionary containing model configuration
            
        Components Initialized:
        1. Model: SPEGNet loaded from checkpoint
        2. Metrics Calculator: For performance evaluation
        3. Result Manager: For output organization
        4. Timing Tracker: For performance analysis
        """
        # Setup device
        self.device = device
        self.model_config = model_config
        logging.info(f"Using device: {self.device}")
        
        try:
            # Initialize model
            self.model = self._load_model(model_path,model_config,batch_size)
            logging.info(f"Model loaded from: {model_path}")
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            raise
            
        # Initialize components
        self.result_manager = ResultManager(dir_manager)
        self.metrics_processor = MetricsProcessor()
        
        # Initialize timing statistics
        self.timing_stats = {
            'inference_times': [],    # Individual inference times
            'processing_times': [],   # Total batch processing times
            'total_time': 0          # Complete evaluation time
        }
        
    def _load_model(self, model_path: str,model_config: Dict,batch_size : int) -> torch.nn.Module:
       """
       Load SPEGNet model from checkpoint and prepare for evaluation.
       
       Args:
           model_path: Path to model checkpoint (.pth file)
           
       Returns:
           Loaded model in evaluation mode
           
       Raises:
           RuntimeError: If model loading fails
           FileNotFoundError: If checkpoint file not found
       """
       try:
            # Initialize model architecture
            model = SPEGNet(model_config)

            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])

            # Prepare for evaluation
            model = model.to(self.device)
            model.eval()

            # Proper warm-up with multiple runs and synchronization
            target_size = model_config['image_processing']['target_size']
            dummy_input = torch.randn(batch_size, 3, target_size, target_size, device=self.device)

            # Multiple warm-up runs
            with torch.inference_mode():
                for _ in range(3):  # Usually 2-3 runs is enough
                    _ = model(dummy_input)
                torch.cuda.synchronize()  # Ensure GPU completion
                
            return model
           
       except FileNotFoundError:
           logging.error(f"Model checkpoint not found: {model_path}")
           raise
       except Exception as e:
           logging.error(f"Error loading model: {str(e)}")
           raise RuntimeError(f"Failed to load model: {str(e)}")
       
    def _denormalize_image(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Denormalize image tensor back to 0-255 range.
        
        Args:
            image_tensor: Input image tensor [C, H, W] normalized with ImageNet stats
            
        Returns:
            numpy array [H, W, C] with values in [0, 255]
        """
        mean = torch.tensor(self.model_config['image_processing']['normalize_mean']).view(3, 1, 1).to(image_tensor.device)
        std = torch.tensor(self.model_config['image_processing']['normalize_std']).view(3, 1, 1).to(image_tensor.device)
        
        # Denormalize
        denorm_image = image_tensor * std + mean
        
        # Convert to numpy and correct format
        image_np = (denorm_image.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        return image_np
    
    def evaluate(self, test_loader: torch.utils.data.DataLoader, dataset_name: str) -> Dict[str, float]:
        """
        Evaluate model on test dataset.
        
        Workflow:
        1. Process batches with timing analysis
        2. Compute metrics for each prediction
        3. Save results organized by quality
        4. Track timing statistics
        5. Generate evaluation summary
        
        Args:
            dataset_name: Name of the test dataset
            
        Returns:
            Dictionary of averaged evaluation metrics:
            - s_alpha: Structure measure
            - weighted_f: Weighted F-measure
            - mae: Mean Absolute Error
            - e_phi: Enhanced-alignment measure
            - mean_f: Mean F-measure
            
        Performance Tracking:
        - Per-batch inference time
        - Total processing time
        - Memory usage monitoring
        """
        # Initialize metric tracking
        total_metrics = {
            's_alpha': 0.0,
            'weighted_f': 0.0,
            'mae': 0.0,
            'e_phi': 0.0,
            'mean_f': 0.0
        }
        
        # Setup directories for this dataset
        self.result_manager.setup_dataset_directories(dataset_name)

        # Start evaluation timing
        eval_start = time.time()
        logging.info("Starting evaluation...")
        
        total_samples_processed = 0

        try:
            # Process each batch
            progress = tqdm(test_loader, desc=f'Evaluating {dataset_name}')
            for batch_idx, batch in enumerate(progress):
                batch_metrics, batch_size = self._process_batch(dataset_name,batch)
                
                # Update total metrics
                for k, v in batch_metrics.items():
                    total_metrics[k] += v
                
                # Update total samples processed
                total_samples_processed += batch_size
                    
                # Update progress bar with running averages
                self._update_progress(progress, total_metrics, total_samples_processed)
                    
            # Compute final averaged metrics
            avg_metrics = {k: v/total_samples_processed for k, v in total_metrics.items()}
            
            # Save timing statistics
            self.timing_stats['total_time'] = time.time() - eval_start
            self.timing_stats['total_samples'] = total_samples_processed
            self._save_evaluation_summary(dataset_name,avg_metrics)
            
            return avg_metrics
            
        except Exception as e:
            logging.error(f"Evaluation failed: {str(e)}")
            raise

    def _process_batch(self, dataset_name: str, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, float], int]:
        """
        Process single batch of test samples.
        
        Processing Steps:
        1. Data Preparation
        - Device transfer
        - Input normalization
        
        2. Model Inference
        - Mixed precision prediction
        - Timing measurement
        
        3. Result Processing
        - Prediction resizing
        - Metric computation
        - Result storage
        
        Args:
            dataset_name: Name of current dataset
            batch: Data dictionary containing:
                - images: Input tensor [B, C, H, W]
                - masks: Ground truth list
                - names: Filename list
                
        Returns:
            tuple: (batch_metrics, batch_size)
                - batch_metrics: Accumulated metrics
                - batch_size: Number of processed samples
                
        Note:
            Processes samples individually to optimize memory usage
        """
        # Start batch timing
        batch_start = time.time()
        
        # Prepare inputs
        images = batch['images'].to(self.device)
        masks = [m.to(self.device) for m in batch['masks']]
        filenames = batch['names']
        batch_size = len(filenames)
        torch.cuda.synchronize()  # Ensure data transfer is complete
        
        batch_metrics = {
            's_alpha': 0.0,
            'weighted_f': 0.0,
            'mae': 0.0,
            'e_phi': 0.0,
            'mean_f': 0.0
        }
        
        # Model inference with timing
        with torch.inference_mode():
            inference_start = time.time()
            outputs = self.model(images)
            self.timing_stats['inference_times'].append(
                time.time() - inference_start
            )
            
            # Get multi-scale predictions and edge predictions
            seg_preds = outputs['predictions']  # List of predictions at different scales
            edge_pred = outputs['edge']

            for idx, filename in enumerate(filenames):
                # Get original image for visualization
                original_image = self._denormalize_image(images[idx])
                # Resize final prediction (highest resolution) to original size
                
                # Resize predictions to original mask size
                seg_resized = F.interpolate(
                    seg_preds[-1][idx:idx+1],
                    size=masks[idx].shape[-2:],
                    mode='bilinear',
                    align_corners=False
                ).sigmoid()

                # Get all stage predictions without resizing
                stage_preds = [pred[idx:idx+1].sigmoid() for pred in seg_preds]
                
                edge_resized = F.interpolate(
                    edge_pred[idx:idx+1],
                    size=masks[idx].shape[-2:],
                    mode='bilinear',
                    align_corners=False
                ).sigmoid()
                
                # Compute metrics (only segmentation metrics, no edge metrics)
                sample_metrics = self.metrics_processor.compute_metrics(
                    seg_pred=seg_resized,
                    seg_gt=[masks[idx]]
                )

                # Update batch metrics
                for k, v in sample_metrics.items():
                    batch_metrics[k] += v

                # Save results
                result = EvaluationResult(
                    filename=filename,
                    seg_pred=seg_resized,         # Resized final prediction
                    seg_stages=stage_preds,       # All stage predictions at original size
                    edge_pred=edge_resized,
                    metrics=sample_metrics,
                    original_image=original_image
                )
                self.result_manager.save_prediction(result, dataset_name)
                
        # Track batch processing time
        self.timing_stats['processing_times'].append(
            time.time() - batch_start
        )
        
        return batch_metrics, batch_size

    def _update_progress(self, progress_bar: tqdm, total_metrics: Dict[str, float], total_samples: int) -> None:
        """Update progress bar with running averages."""
        avg_inference = np.mean(self.timing_stats['inference_times'][-10:]) * 1000
        
        # Compute running averages
        running_metrics = {k: v/total_samples for k, v in total_metrics.items()}
        
        progress_bar.set_postfix({
            'S-measure': f"{running_metrics['s_alpha']:.3f}",
            'F-measure': f"{running_metrics['weighted_f']:.3f}", 
            'Inf(ms)': f"{avg_inference:.1f}"
        })

    def _save_evaluation_summary(self, dataset_name: str, metrics: Dict[str, float]):
        """
        Save comprehensive evaluation summary.
        
        Args:
            metrics: Dictionary of evaluation metrics
            
        Saves:
        1. Final averaged metrics
        2. Timing statistics
        3. Category distribution
        4. Performance analysis
        """
        # Prepare timing summary
        timing_summary = {
            'total_time': self.timing_stats['total_time'],
            'avg_inference_time': np.mean(self.timing_stats['inference_times']),
            'avg_processing_time': np.mean(self.timing_stats['processing_times']),
            'total_samples': self.timing_stats['total_samples']
        }
        
        # Get category distribution
        category_summary = self.result_manager.get_category_summary(dataset_name)
        
        # Save comprehensive summary
        summary = {
            'metrics': metrics,
            'timing': timing_summary,
            'categories': category_summary
        }
        
        # Save in dataset directory
        summary_file = self.result_manager.dataset_dirs[dataset_name]['root'] / 'evaluation_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)

        # Log final results
        self._log_final_results(dataset_name, summary)

    def _log_final_results(self, dataset_name: str, summary: Dict) -> None:
        """
        Log final evaluation results.
        
        Args:
            summary: Dictionary containing evaluation summary
        """
        metrics = summary['metrics']
        timing = summary['timing']
        categories = summary['categories']
        
        logging.info(f"\nEvaluation Results for {dataset_name}:")
        logging.info("-------------------")
        logging.info(f"Total samples: {timing['total_samples']}")
        logging.info(f"Total time: {timing['total_time']:.2f}s")
        logging.info(f"Average inference time: {timing['avg_inference_time']*1000:.2f}ms")
        
        logging.info("\nMetrics:")
        logging.info(f"S-measure (Sα): {metrics['s_alpha']:.4f}")
        logging.info(f"Weighted F-measure (Fβw): {metrics['weighted_f']:.4f}")
        logging.info(f"Mean Absolute Error (M): {metrics['mae']:.4f}")
        logging.info(f"Enhanced-alignment (Eϕ): {metrics['e_phi']:.4f}")
        logging.info(f"Mean F-measure (Fβm): {metrics['mean_f']:.4f}")
        
        logging.info("\nQuality Distribution:")
        for category, count in categories['counts'].items():
            percentage = (count / categories['total']) * 100
            logging.info(f"{category.capitalize()}: {count} ({percentage:.1f}%)")