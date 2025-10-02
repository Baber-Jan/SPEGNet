"""
SPEGNet Visualization Module

This module provides visualization tools for analyzing and presenting camouflaged object
detection results from SPEGNet. It generates three key visualizations as shown in
the Evaluation section (4.2) of the paper:

1. Binary Segmentation:
   - Direct threshold visualization showing detected objects
   - Preserves original prediction confidence values (0-255)
   - Useful for quantitative analysis and model comparison

2. Confidence Heatmaps:
   - Color-coded visualization of detection confidence
   - Uses COLORMAP_JET for intuitive high/low confidence regions
   - Helps analyze model's decision certainty

3. Overlay Visualizations:
   - Combines predictions with original images
   - Shows context of detections with original scene
   - Useful for qualitative analysis and presentations

Multi-scale Results:
   For SPEGNet's progressive refinement stages, we store outputs from each stage
   (as described in Section 3.3) to visualize the refinement process:
   - Stage 1 (128x128): Initial predictions
   - Stage 2 (256x256): Intermediate refinement
   - Stage 3 (512x512): Final detailed output

Directory Structure Per Run:
    results/{mode}/runs/run_{timestamp}/
    ├── good/                 # High-quality detections (Sα ≥ 0.8)
    │   ├── segmentation/     # Segmentation visualizations
    │   │   ├── binary/      # Binary prediction maps
    │   │   ├── heatmap/     # Confidence heatmaps
    │   │   └── overlay/     # Original image overlays
    │   └── edges/           # Edge detection results
    ├── medium/              # Medium-quality results (0.6 ≤ Sα < 0.8)
    │   ├── segmentation/
    │   └── edges/
    └── bad/                 # Lower-quality results (Sα < 0.6)
        ├── segmentation/
        └── edges/

Result Categories:
1. Good Quality (Sα ≥ 0.8):
   - High-confidence detections
   - Clear object boundaries
   - Accurate edge maps

2. Medium Quality (0.6 ≤ Sα < 0.8):
   - Partial detections
   - Some boundary ambiguity
   - Moderate confidence scores

3. Bad Quality (Sα < 0.6):
   - Missed detections
   - Significant boundary errors
   - Low confidence predictions

Each category contains both segmentation and edge detection results,
enabling detailed analysis of model performance across different
detection scenarios.

Example Usage:
    # Initialize result manager for a run
    result_manager = ResultVisualizer(dir_manager.run_dirs.root)
    
    # Save predictions with quality-based organization
    result_manager.save_prediction(
        filename='sample1',
        prediction=final_pred,
        metrics={'s_alpha': 0.85},
        original_image=image,
        edge_prediction=edge_map
    )

Notes:
    - Results are automatically categorized based on metrics
    - Each prediction generates multiple visualization types
    - Stage predictions show progressive refinement process
    - Directory structure matches evaluation organization
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional, List
import logging
import torch

def save_binary_visualization(
    prediction: np.ndarray,
    save_path: str,
) -> None:
    """
    Save binary visualization of prediction along with confidence.
    
    Args:
        prediction: Prediction map [H, W] with values in [0, 1]
        save_path: Path to save visualization
    """
    # Create directory if needed
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure 2D array
    if prediction.ndim > 2:
        prediction = prediction.squeeze()
    
    # Convert confidence values to 0-255 range directly
    confidence_map = (prediction * 255).astype(np.uint8)
    
    # Save image
    cv2.imwrite(str(save_path), confidence_map)

def save_heatmap_visualization(
    prediction: np.ndarray,
    save_path: str,
    alpha: float = 0.7,
    normalize: bool = True
) -> None:
    """
    Save heatmap visualization of prediction confidence.
    
    Args:
        prediction: Prediction map [H, W] with values in [0, 1]
        save_path: Path to save visualization
        alpha: Transparency for heatmap (default: 0.7)
        normalize: Whether to normalize values to [0, 1]
    """
    # Create directory if needed
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure 2D array
    if prediction.ndim > 2:
        prediction = prediction.squeeze()
        
    # Normalize if needed
    if normalize:
        prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)
    
    # Convert to uint8 for colormap
    pred_uint8 = (prediction * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(pred_uint8, cv2.COLORMAP_JET)
    
    # Save image
    cv2.imwrite(str(save_path), heatmap)

def save_overlay_visualization(
    image: np.ndarray,
    prediction: np.ndarray,
    save_path: str,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> None:
    """
    Save prediction overlay on original image.
    
    Args:
        image: Original image [H, W, 3] (RGB or BGR)
        prediction: Prediction map [H, W] with values in [0, 1]
        save_path: Path to save visualization
        alpha: Transparency for overlay (default: 0.5)
        colormap: OpenCV colormap to use (default: COLORMAP_JET)
    
    Raises:
        ValueError: If image and prediction shapes don't match
        RuntimeError: If overlay creation fails
    """
    try:
        # Create directory if needed
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure 2D prediction
        if prediction.ndim > 2:
            prediction = prediction.squeeze()
        
        # Convert RGB to BGR if needed
        if image.ndim == 3 and image.shape[-1] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image

        # Ensure matching sizes
        if image_bgr.shape[:2] != prediction.shape:
            prediction = cv2.resize(prediction, (image_bgr.shape[1], image_bgr.shape[0]))
            
        # Normalize prediction if needed
        if prediction.max() > 1.0 or prediction.min() < 0.0:
            prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)
        
        # Create heatmap
        pred_uint8 = (prediction * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(pred_uint8, colormap)
        
        # Create overlay
        overlay = cv2.addWeighted(image_bgr, 1-alpha, heatmap, alpha, 0)
        
        # Save image
        cv2.imwrite(str(save_path), overlay)
        
    except Exception as e:
        logging.error(f"Failed to create overlay visualization: {str(e)}")
        raise RuntimeError(f"Overlay creation failed: {str(e)}")

class ResultVisualizer:
    """Handles different types of result visualizations."""
    
    def __init__(self, base_dir: Path):
        """
        Initialize visualizer with directory structure.
        
        Args:
            base_dir: Base directory for saving visualizations
        """
        self.base_dir = Path(base_dir)
        
        # Create visualization directories
        self.binary_dir = self.base_dir / 'binary'
        self.heatmap_dir = self.base_dir / 'heatmap'
        self.overlay_dir = self.base_dir / 'overlay'
        
        for d in [self.binary_dir, self.heatmap_dir, self.overlay_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
    def save_all_visualizations(
        self,
        filename: str,
        prediction: np.ndarray,
        original_image: Optional[np.ndarray] = None,
        stage_predictions: Optional[List[np.ndarray]] = None
    ) -> None:
        """
        Save all visualization types for a prediction.
        
        Args:
            filename: Base filename for saving
            prediction: Prediction map [H, W] with values in [0, 1]
            original_image: Optional original image for overlay
            stage_predictions: Optional list of stage predictions at original size
        """
        # Save binary visualization
        save_binary_visualization(
            prediction,
            self.binary_dir / f"{filename}.png"
        )
        
        # Save heatmap visualization
        save_heatmap_visualization(
            prediction,
            self.heatmap_dir / f"{filename}.png"
        )
        
        # Save overlay if original image provided
        if original_image is not None:
            save_overlay_visualization(
                original_image,
                prediction,
                self.overlay_dir / f"{filename}.png",
                alpha=0.7  
            )

        # Save stage predictions (original size)
        if stage_predictions is not None:
            for stage_idx, stage_pred in enumerate(stage_predictions):
                stage_filename = f"{filename}_stage{stage_idx+1}"
                
                # Convert tensor to numpy if needed
                if torch.is_tensor(stage_pred):
                    stage_pred = stage_pred.cpu().numpy()

                # Save binary prediction for each stage
                save_binary_visualization(
                    stage_pred,
                    self.binary_dir / f"{stage_filename}.png"
                )

                # Save heatmap for each stage
                save_heatmap_visualization(
                    stage_pred,
                    self.heatmap_dir / f"{stage_filename}.png"
                )