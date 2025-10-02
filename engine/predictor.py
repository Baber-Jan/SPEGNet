"""
Predictor module for efficient Camouflaged Object Detection inference.

This module provides:
1. Optimized Single Image Prediction:
   - Efficient memory usage with torch.no_grad()
   - Multi-scale prediction handling
   - Configurable output dimensions

2. Batch Processing:
   - Config-based batch size control
   - Directory-based image processing
   - Built-in memory management

3. Result Management:
   - Structured output organization
   - Multiple visualization formats
   - Performance statistics tracking

Processing Pipeline:
1. Image Preprocessing:
   - Config-based resolution matching
   - ImageNet normalization
   - Device placement optimization

2. Model Inference:
   - Single-pass prediction
   - Multi-scale output handling
   - Edge map generation

3. Result Organization:
   - Binary and heatmap visualizations
   - Overlay generation
   - Performance logging
"""

import logging
import time
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from models.spegnet import SPEGNet
from utils.visualization import save_binary_visualization, save_heatmap_visualization, save_overlay_visualization
from utils.run_manager import DirectoryManager
from utils.image_processor import CODImageProcessor

class PredictionResultManager:
    """
    Manages prediction results storage and organization.
    
    Responsibilities:
    - Maintains directory structure for outputs
    - Handles result visualization saving
    - Tracks processing timing statistics
    - Generates performance summaries
    
    Directory Structure:
        results/
        ├── segmentation/
        │   ├── binary/      # Thresholded predictions
        │   ├── heatmap/     # Confidence visualizations
        │   └── overlay/     # Image-prediction overlays
        └── edges/
            ├── binary/      # Binary edge maps
            ├── heatmap/     # Edge confidence maps
            └── overlay/     # Edge visualization overlays
            
    Timing Tracking:
        - preprocessing: Image loading and normalization
        - inference: Model prediction time
        - postprocessing: Result generation and saving
    """
    
    def __init__(self, dir_manager: DirectoryManager, output_base_dir: Optional[str] = None):
        """
        Initialize result directory structure.
        
        Args:
            dir_manager: Directory manager instance
            output_base_dir: Optional custom base directory for results
        """
        self.run_dirs = dir_manager.run_dirs
        
        # Create visualization directories
        self.viz_root = self.run_dirs.visualizations
        
        # Segmentation directories
        self.seg_dir = self.viz_root / 'segmentation'
        self.seg_binary_dir = self.seg_dir / 'binary'
        self.seg_heatmap_dir = self.seg_dir / 'heatmap'
        self.seg_overlay_dir = self.seg_dir / 'overlay'

        # Edge directories
        self.edge_dir = self.viz_root / 'edges'
        self.edge_binary_dir = self.edge_dir / 'binary'
        self.edge_heatmap_dir = self.edge_dir / 'heatmap'
        self.edge_overlay_dir = self.edge_dir / 'overlay'
        
        # Create all directories
        for d in [self.seg_binary_dir, self.seg_heatmap_dir, self.seg_overlay_dir,
              self.edge_binary_dir, self.edge_heatmap_dir, self.edge_overlay_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Create log file
        self.log_file = self.run_dirs.log_file
        
        # Store timing information
        self.timings = {
            'preprocessing': [],
            'inference': [],
            'postprocessing': []
        }
        
    def log_message(self, message: str):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_file, 'a') as f:
            f.write(f'[{timestamp}] {message}\n')
            
    def save_prediction(self, 
                       filename: str,
                       seg_pred: np.ndarray,
                       edge_pred: np.ndarray, 
                       original_image: np.ndarray):
        """
        Save prediction results with multiple visualization types.
        
        Args:
            filename: Original image filename
            seg_pred: Segmentation prediction [H, W] with values in [0, 1]
            edge_pred: Edge prediction [H, W] with values in [0, 1]
            original_image: Original image for overlay visualization
        """
        base_name = Path(filename).stem
        
        # Save segmentation predictions
        save_binary_visualization(
            seg_pred,
            self.seg_binary_dir / f'{base_name}.png'
        )
        save_heatmap_visualization(
            seg_pred,
            self.seg_heatmap_dir / f'{base_name}.png',
            normalize=True
        )
        save_overlay_visualization(
            original_image,
            seg_pred,
            self.seg_overlay_dir / f'{base_name}.png'
        )
        
        # Save edge predictions
        save_binary_visualization(
            edge_pred,
            self.edge_binary_dir / f'{base_name}.png'
        )
        save_heatmap_visualization(
            edge_pred,
            self.edge_heatmap_dir / f'{base_name}.png',
            normalize=True
        )
        save_overlay_visualization(
            original_image,
            edge_pred,
            self.edge_overlay_dir / f'{base_name}.png'
        )
        
    def update_timing(self, phase: str, time_taken: float):
        """Update timing information."""
        self.timings[phase].append(time_taken)
        
    def summarize(self):
        """Generate summary of predictions."""
        n_predictions = len(list(self.seg_dir.glob('*.png')))
        
        # Compute average timings
        avg_timings = {
            phase: np.mean(times) if times else 0 
            for phase, times in self.timings.items()
        }
        total_time = sum(avg_timings.values())
        
        summary = {
            'total_predictions': n_predictions,
            'average_timings': avg_timings,
            'total_time_per_image': total_time,
            'total_processing_time': total_time * n_predictions
        }
        
        # Save summary
        with open(self.run_dirs.root / 'prediction_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
            
        # Log summary
        self.log_message(
            f"\nPrediction Summary:\n"
            f"Total images processed: {n_predictions}\n"
            f"Average timings (seconds):\n"
            f"  Preprocessing: {avg_timings['preprocessing']:.3f}\n"
            f"  Inference: {avg_timings['inference']:.3f}\n"
            f"  Postprocessing: {avg_timings['postprocessing']:.3f}\n"
            f"Total time per image: {total_time:.3f}s\n"
            f"Total processing time: {total_time * n_predictions:.1f}s"
        )
        
        return summary

class Predictor:
    """
    Optimized predictor for Camouflaged Object Detection.
    
    Features:
    - Efficient single/batch prediction
    - Memory-optimized processing
    - Proper result visualization and storage
    """
    
    def __init__(
        self, 
        model_path: str,
        model_config: Dict,
        dir_manager: DirectoryManager,
        output_dir: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = 1
    ):
        """
        Initialize predictor with configuration.
        
        Args:
            model_path: Path to model checkpoint
            model_config: Model configuration dictionary containing:
                - encoder: Encoder settings
                - image_processing: Image preprocessing settings
            dir_manager: Directory manager for result organization
            output_dir: Optional custom output directory
            device: Optional device specification
        """
        # Validate inputs
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
            
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Extract batch size from config
        self.batch_size = batch_size
        
        # Initialize image processor with config parameters
        img_config = model_config['image_processing']
        self.image_processor = CODImageProcessor(
            target_size=img_config['target_size'],
            normalize_mean=tuple(img_config['normalize_mean']),
            normalize_std=tuple(img_config['normalize_std'])
        )
        
        try:
            # Initialize model
            self.model = self._initialize_model(model_path, model_config)
            self.result_manager = PredictionResultManager(dir_manager, output_dir)
            self.result_manager.log_message(f"Model loaded from: {model_path}")
        except Exception as e:
            logging.error(f"Initialization failed: {str(e)}")
            raise

    def _initialize_model(self, model_path: str, model_config: Dict) -> nn.Module:
        """Initialize and warm up model."""
        model = SPEGNet(model_config)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        # Warm-up run
        target_size = model_config['image_processing']['target_size']
        dummy_input = torch.randn(self.batch_size, 3, target_size, target_size, device=self.device)
        
        with torch.inference_mode():
            _ = model(dummy_input)
            torch.cuda.synchronize()
            
        return model
        
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess image using COD image processor.
        
        Args:
            image_path: Path to input image
                
        Returns:
            Preprocessed image tensor [1, C, H, W]
        """
        start_time = time.time()
        
        # Process image using correct API
        processed = self.image_processor(image_path)
        image_tensor = processed.image.to(self.device)
        
        self.result_manager.update_timing('preprocessing', time.time() - start_time)
        return image_tensor.unsqueeze(0)  # Add batch dimension
        
    def predict_single(self, image_path: str, 
                  output_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process single image through complete prediction pipeline.
        
        Processing Steps:
        1. Image preprocessing to model input size
        2. Model inference with no_grad()
        3. Multi-scale prediction handling
        4. Optional output resizing
        
        Args:
            image_path: Path to input image
            output_size: Optional (height, width) for prediction resizing
            
        Returns:
            Tuple containing:
            - seg_pred: Segmentation map [H, W], values in [0, 1]
            - edge_pred: Edge prediction map [H, W], values in [0, 1]
            - original_image: Original RGB image array
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        
        # Inference
        with torch.no_grad():
            inference_start = time.time()
            outputs = self.model(image_tensor)
            inference_time = time.time() - inference_start
            self.result_manager.update_timing('inference', inference_time)
            # Log inference time with image name
            self.result_manager.log_message(f"Inference time for {image_path}: {inference_time:.3f}s")
            # Get predictions from all scales and edge map
            seg_preds = outputs['predictions']  # List of predictions
            edge_pred = outputs['edge']
            
            # Use final prediction (highest resolution)
            final_seg_pred = seg_preds[-1]
            
            # Resize if needed
            post_start = time.time()
            if output_size:
                final_seg_pred = F.interpolate(
                    final_seg_pred,
                    size=output_size,
                    mode='bilinear',
                    align_corners=False
                )
                edge_pred = F.interpolate(
                    edge_pred,
                    size=output_size,
                    mode='bilinear',
                    align_corners=False
                )
            
            # Convert to numpy
            seg_np = final_seg_pred.sigmoid().squeeze().cpu().numpy()
            edge_np = edge_pred.sigmoid().squeeze().cpu().numpy()
            self.result_manager.update_timing('postprocessing', time.time() - post_start)
            
            # Load original image for visualization
            original_image = np.array(Image.open(image_path).convert('RGB'))
            
            return seg_np, edge_np, original_image
        
    def predict_batch(self, image_paths: List[str],
                 output_size: Optional[Tuple[int, int]] = None) -> Dict:
        """
        Process multiple images with batch-wise handling.
        
        Processing:
        1. Batch-size based iteration
        2. Per-batch memory clearing
        3. Progress tracking
        
        Args:
            image_paths: List of image paths to process
            output_size: Optional size for prediction resizing
            
        Returns:
            Dictionary containing processing summary:
            - total_predictions: Number of processed images
            - average_timings: Processing phase timings
            - total_processing_time: Complete execution time
        """

        self.result_manager.log_message(
            f"Starting batch prediction of {len(image_paths)} images "
            f"with batch size {self.batch_size}"
        )
        
        try:
            for i in range(0, len(image_paths), self.batch_size):
                batch_paths = image_paths[i:i + self.batch_size]
                
                # Process batch
                for image_path in tqdm(batch_paths, desc=f"Batch {i//self.batch_size + 1}"):
                    seg_pred, edge_pred, original_image = self.predict_single(
                        image_path, 
                        output_size
                    )
                    
                    # Save results
                    self.result_manager.save_prediction(
                        Path(image_path).name,
                        seg_pred,
                        edge_pred,
                        original_image
                    )
                    
                # Memory cleanup after each batch
                torch.cuda.empty_cache()
                
            # Generate summary
            return self.result_manager.summarize()
            
        except Exception as e:
            self.result_manager.log_message(f"Batch processing error: {str(e)}")
            raise
        
    def predict_directory(self, input_dir: str,
                     output_size: Optional[Tuple[int, int]] = None,
                     extensions: tuple = ('.jpg', '.png', '.jpeg')) -> Dict:
        """
        Process all valid images in specified directory.
        
        Directory Processing:
        1. Recursive image finding
        2. Extension-based filtering
        3. Batch-wise processing
        
        Args:
            input_dir: Input directory path
            output_size: Optional size for predictions
            extensions: Valid image file extensions
            
        Returns:
            Processing summary dictionary
        """

        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Invalid directory: {input_dir}")
            
        image_paths = [
            str(p) for p in input_dir.glob('**/*') 
            if p.suffix.lower() in extensions
        ]
        
        if not image_paths:
            raise ValueError(f"No valid images found in {input_dir}")
            
        self.result_manager.log_message(
            f"Found {len(image_paths)} images in {input_dir}"
        )
        
        return self.predict_batch(image_paths, output_size)