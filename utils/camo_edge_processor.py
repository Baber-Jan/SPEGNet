"""
Ground Truth Edge Map Generator for CAMO Dataset.

This module implements efficient ground truth edge map generation for the CAMO 
camouflaged object detection dataset, supporting the edge-guided detection and
refinement components. THe edges for COD10K dataset are already given in the original dataset. 


Key Features:
    - Efficient morphological edge extraction
    - Edge continuity validation and quality assessment
    - Batch processing for complete datasets
    - Detailed progress tracking and error handling
    - Memory-optimized implementation

Example Usage:
    processor = CAMOEdgeProcessor(edge_width=1)
    
    # Process single mask
    edges, is_valid = processor.extract_edges(mask)
    
    # Process complete dataset
    stats = processor.process_dataset(
        input_path='path/to/masks',
        output_path='path/to/edges'
    )

Notes:
    - Input masks should be binary (uint8, values in [0,255])
    - Edge validation uses contour analysis for quality assessment
    - Processing statistics include success/failure tracking
    - Optional output saving with automatic directory creation

Authors:
    Baber Jan
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple
import logging
from tqdm import tqdm
import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple
import logging
from tqdm import tqdm

class CAMOEdgeProcessor:
    """
    Ground Truth Edge Map Processor for CAMO dataset.
    Extracts clean edge maps from binary ground truth masks using
    efficient morphological operations and validates edge quality.

    Features:
    - Single-pass morphological edge extraction
    - Configurable edge thickness with validation
    - Progress tracking with detailed statistics
    - Memory-efficient batch processing
    
    Attributes:
        edge_width: Edge thickness in pixels
        validation_threshold: Minimum edge continuity score (0-1)
        kernels: Pre-computed morphological operation kernels
        logger: Configured logging interface
    """
    
    def __init__(
        self,
        edge_width: int = 1,
        validation_threshold: float   = 0.5
    ):
        """
        Initialize the edge processor.

        Args:
            edge_width: Desired edge thickness in pixels (default: 1)
            validation_threshold: Minimum required edge continuity (default: 0.95)
        """
        self.edge_width = max(1, int(edge_width))
        self.validation_threshold = validation_threshold
        self.logger = self._setup_logger()
        
        # Pre-compute kernels for efficiency
        self.kernels = {
            'thin': np.ones((3, 3), np.uint8),
            'thick': np.ones((5, 5), np.uint8)
        }

    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Configure logging with standardized formatting."""
        logger = logging.getLogger('GTEdgeProcessor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger

    def extract_edges(
        self,
        mask: np.ndarray,
        validate: bool = True
    ) -> Tuple[np.ndarray, bool]:
        """
        Extract clean edge map from ground truth mask using morphological operations.
        
        Implements the edge extraction process detailed in Section 3.3 of the paper,
        using a combination of dilation and erosion to identify object boundaries.
        
        Args:
            mask: Binary ground truth mask (HxW, uint8, values in [0,255])
            validate: Whether to perform edge continuity validation
        
        Returns:
            Tuple containing:
                - edge_map: Binary edge map (HxW, uint8, 255 for edges)
                - is_valid: Whether edge map meets continuity threshold
                
        Notes:
            Edge validation uses contour analysis to ensure boundary
            continuity meets the validation_threshold requirement.
        """
        if mask.dtype != np.uint8:
            mask = (mask > 127).astype(np.uint8) * 255

        # Basic morphological edge extraction
        dilated = cv2.dilate(mask, self.kernels['thin'], iterations=self.edge_width)
        eroded = cv2.erode(mask, self.kernels['thin'], iterations=self.edge_width)
        edges = cv2.subtract(dilated, eroded)

        # Ensure edge connectivity
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, self.kernels['thin'])

        # Optional edge validation
        is_valid = True
        if validate:
            # Check edge continuity
            contours, _ = cv2.findContours(
                edges, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_NONE
            )
            
            if len(contours) > 0:
                # Calculate perimeter ratio
                actual_perimeter = sum(len(c) for c in contours)
                expected_perimeter = sum(
                    cv2.arcLength(c, True) for c in contours
                )
                
                continuity = actual_perimeter / (expected_perimeter + 1e-6)
                is_valid = continuity >= self.validation_threshold
            else:
                is_valid = False

        return edges, is_valid

    def process_dataset(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        file_pattern: str = "*.png"
    ) -> dict:
        """
        Process complete dataset of ground truth masks.
        
        Args:
            input_path: Directory containing ground truth masks
            output_path: Directory to save edge maps (created if needed)
            file_pattern: Glob pattern for mask files (default: "*.png")
            
        Returns:
            Dictionary containing:
                - total: Total number of masks found
                - processed: Successfully processed masks
                - valid: Masks meeting validation criteria
                - failed: Failed processing attempts
                
        Raises:
            FileNotFoundError: If input_path doesn't exist
            ValueError: If no mask files found matching pattern
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_path}")
        input_path = Path(input_path)
        if output_path:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

        # Initialize statistics
        stats = {
            'total': 0,
            'processed': 0,
            'valid': 0,
            'failed': 0
        }

        # Process all mask files
        mask_files = list(input_path.glob(file_pattern))
        stats['total'] = len(mask_files)

        for mask_file in tqdm(mask_files, desc="Processing GT masks"):
            try:
                # Read mask
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError(f"Failed to read mask: {mask_file}")

                # Extract and validate edges
                edges, is_valid = self.extract_edges(mask, validate=True)
                
                # Save if output path provided
                if output_path and is_valid:
                    output_file = output_path / mask_file.name
                    cv2.imwrite(str(output_file), edges)
                
                # Update statistics
                stats['processed'] += 1
                stats['valid'] += int(is_valid)
                
            except Exception as e:
                stats['failed'] += 1
                self.logger.error(f"Error processing {mask_file.name}: {str(e)}")
                continue

        # Log summary
        self.logger.info(
            f"Processing complete: "
            f"{stats['processed']}/{stats['total']} processed, "
            f"{stats['valid']} valid, "
            f"{stats['failed']} failed"
        )

        return stats