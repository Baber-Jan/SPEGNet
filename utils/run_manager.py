"""
Run Session Management for SPEGNet.

This module manages execution sessions for the SPEGNet camouflaged object detection
model, providing consistent directory structures and logging across different
operational modes (training, evaluation, prediction). Each run is organized in a
timestamped directory containing all relevant outputs and logs.

Directory Structure:
    results/
    ├── training/
    │   └── runs/
    │       └── run_YYYYMMDD_HHMMSS/
    │           ├── checkpoints/
    │           ├── metrics.json
    │           └── training_log.txt
    ├── evaluation/
    │   └── runs/
    │       └── run_YYYYMMDD_HHMMSS/
    │           └── evaluation_log.txt
    └── prediction/
        └── runs/
            └── run_YYYYMMDD_HHMMSS/
                ├── results/
                │   ├── segmentation/
                │   └── edges/
                └── prediction_log.txt

Features:
    - Automatic directory creation and management
    - Centralized logging configuration
    - Run-specific output organization
    - Mode-based directory structuring
    - Consistent path generation

Example Usage:
    # In main.py
    dir_manager = DirectoryManager("train")
    setup_logging(dir_manager)
    
    # In trainer/evaluator
    result_manager = ResultManager(dir_manager)
    result_manager.log_message("Training started")

Notes:
    - Each run gets a unique timestamp-based directory
    - All outputs from a run are contained within its directory
    - Logging is configured for both file and console output
"""

from enum import Enum
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
from dataclasses import dataclass
import logging

class RunMode(Enum):
    """
    Valid execution modes for SPEGNet.
    
    Modes:
        TRAIN: Model training with optional validation
        EVALUATE: Performance evaluation on test datasets
        PREDICT: Single image or batch prediction
    """
    TRAIN = "training"
    EVALUATE = "evaluation"
    PREDICT = "prediction"

@dataclass
@dataclass
class RunDirectories:
    """
    Directory structure for a single execution run.

    This structure maintains organization of all outputs generated during
    a single execution session of SPEGNet, whether training, evaluation,
    or prediction.
    
    Attributes:
        root: Base directory for this run (required)
            Format: results/{mode}/runs/run_{timestamp}/
        
        checkpoints: Model checkpoint storage (training only)
            Contents: Best model and periodic checkpoints
        
        visualizations: Output visualization directory (evaluation and prediction only)
            Contents: Segmentation masks and edge maps
            
        metrics: Metrics storage directory (training only)
            Contents: Training and validation metrics in JSON format
            
        log_file: Run-specific log file (all modes)
            Format: Timestamped logging output
    
    Note:
        Optional attributes are None for modes where they're not applicable
    """
    root: Path
    checkpoints: Optional[Path] = None
    visualizations: Optional[Path] = None
    metrics: Optional[Path] = None
    log_file: Optional[Path] = None

class DirectoryManager:
    """
    Manages directory structure and creation for different run modes.
    
    Features:
    - Consistent directory structure across modes
    - Automatic directory creation
    - Mode-specific organization
    - Timestamp-based run folders
    """
    
    def __init__(self, mode: str):
        """
        Initialize directory manager.
        
        Args:
            mode: Run mode ('train', 'evaluate', or 'predict')
        """
        self.mode = RunMode[mode.upper()].value
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dirs = self._setup_directories()
        
    def _setup_directories(self) -> RunDirectories:
        """
        Create directory structure based on mode.
        
        Returns:
            RunDirectories object with all necessary paths
        """
        # Create base run directory
        root = Path('results') / self.mode / 'runs' / f'run_{self.timestamp}'
        run_dirs = RunDirectories(root=root)
        
        # Setup mode-specific directory structure
        if self.mode == RunMode.TRAIN.value:
            run_dirs.checkpoints = root / 'checkpoints'
            run_dirs.metrics_file = root / 'metrics.json'
            run_dirs.log_file = root / 'training_log.txt'
            
        elif self.mode == RunMode.EVALUATE.value:
            run_dirs.log_file = root / 'evaluation_log.txt'
            
        else:  # PREDICT
            run_dirs.visualizations = root / 'results'
            (run_dirs.visualizations / 'segmentation').mkdir(parents=True)
            (run_dirs.visualizations / 'edges').mkdir(parents=True)
            run_dirs.log_file = root / 'prediction_log.txt'
            
        # Create all directories
        self._create_directories(run_dirs)
        return run_dirs
    
    def _create_directories(self, run_dirs: RunDirectories) -> None:
        """
        Create all directories in the RunDirectories structure.
        
        Args:
            run_dirs: RunDirectories object containing paths to create
        """
        for field in run_dirs.__dataclass_fields__:
            path = getattr(run_dirs, field)
            if path is not None and isinstance(path, Path) and 'file' not in field:
                path.mkdir(parents=True, exist_ok=True)
                
    def get_paths(self) -> Dict[str, Path]:
        """
        Get dictionary of all paths for current run.
        
        Returns:
            Dictionary mapping path names to Path objects
        """
        return {
            field: getattr(self.run_dirs, field)
            for field in self.run_dirs.__dataclass_fields__
            if getattr(self.run_dirs, field) is not None
        }

# Update in main.py
def setup_logging(args) -> None:
    """
    Configure logging with timestamps and levels.
    
    Args:
        args: Command line arguments containing mode
    """
    dir_manager = DirectoryManager(args.mode)
    log_file = dir_manager.run_dirs.log_file
    
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Log initial setup information
    logging.info(f"Starting {args.mode} run")
    logging.info(f"Results will be saved in: {dir_manager.run_dirs.root}")
    
    return dir_manager

# Usage in other files (e.g., trainer.py, evaluator.py, predictor.py)
class ResultManager:
    def __init__(self, dir_manager: DirectoryManager):
        """
        Initialize result manager with directory structure.
        
        Args:
            dir_manager: Directory manager for current run
        """
        self.run_dirs = dir_manager.run_dirs
        self.paths = dir_manager.get_paths()
        
    def log_message(self, message: str) -> None:
        """Log message with timestamp to run-specific log file."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.run_dirs.log_file, 'a') as f:
            f.write(f'[{timestamp}] {message}\n')