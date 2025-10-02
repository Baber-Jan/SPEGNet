#!/bin/bash

# Setup script for Camouflaged Object Detection project
# Handles dataset validation, environment creation and edge map generation
apt-get update && apt-get install -y libgl1-mesa-glx zip unzip
# Get absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Environment setup first
echo "Setting up environment..."
if ! conda env list | grep -q "spegnet"; then
    # Create Conda environment
    conda env create -f "$SCRIPT_DIR/environment.yml"
    echo "Environment created. Activate with: conda activate spegnet"
else
    echo "Environment already exists"
fi

# Install Jupyter extensions (optional, if needed)
source activate spegnet

# Function to validate and clean dataset structure
check_dataset() {
    local DATASET_NAME=$1
    local DATASET_PATH="$PROJECT_ROOT/datasets/$DATASET_NAME"
    echo "Processing dataset: $DATASET_NAME"

    # Validate dataset existence
    if [ ! -d "$DATASET_PATH" ]; then
        echo "Error: Dataset $DATASET_NAME not found in datasets/"
        return 1
    fi

    # Check for train/test splits
    local SPLITS=()
    [[ -d "$DATASET_PATH/train" ]] && SPLITS+=("train")
    [[ -d "$DATASET_PATH/test" ]] && SPLITS+=("test")

    if [ ${#SPLITS[@]} -eq 0 ]; then
        echo "Error: No train/test splits found in $DATASET_NAME"
        return 1
    fi

    # Process each split
    for SPLIT in "${SPLITS[@]}"; do
        local SPLIT_PATH="$DATASET_PATH/$SPLIT"
        local IMAGES_PATH="$SPLIT_PATH/Imgs"
        local GT_PATH="$SPLIT_PATH/GT"
        local EDGES_PATH="$SPLIT_PATH/Edges"

        # Validate required directories
        if [ ! -d "$IMAGES_PATH" ] || [ ! -d "$GT_PATH" ]; then
            echo "Error: Missing required folders in $SPLIT_PATH"
            continue
        fi

        # Track files using associative arrays
        declare -A IMAGE_FILES GT_FILES EDGE_FILES
        
        # Collect existing files
        while IFS= read -r -d '' file; do
            IMAGE_FILES["$(basename "${file%.*}")"]=1
        done < <(find "$IMAGES_PATH" -type f -print0)

        while IFS= read -r -d '' file; do
            GT_FILES["$(basename "${file%.*}")"]=1
        done < <(find "$GT_PATH" -type f -print0)

        # Cleanup GT files without corresponding images
        for gt_base in "${!GT_FILES[@]}"; do
            if [ ! "${IMAGE_FILES[$gt_base]+_}" ]; then
                echo "Removing GT without image: $gt_base"
                rm -f "$GT_PATH/$gt_base".*
            fi
        done

        # Handle edges for training splits
        if [ "$SPLIT" == "train" ]; then
            # Special handling for CAMO dataset
            if [ "$DATASET_NAME" == "CAMO" ] && [ ! -d "$EDGES_PATH" ]; then
                echo "Generating CAMO edge maps..."
                mkdir -p "$EDGES_PATH"
                
                # Generate edges with minimal logging
                python3 - <<EOF
import logging
from utils.camo_edge_processor import CAMOEdgeProcessor
import os

# Initialize processor
processor = CAMOEdgeProcessor(edge_width=1)

# Process CAMO training set
stats = processor.process_dataset(
    input_path="${GT_PATH}",
    output_path="${EDGES_PATH}",
    file_pattern="*.png"
)

# Only print errors if any
if stats['failed'] > 0:
    print(f"Failed to process {stats['failed']} edge maps")
EOF
            fi

            # Collect edge files if directory exists
            if [ -d "$EDGES_PATH" ]; then
                while IFS= read -r -d '' file; do
                    EDGE_FILES["$(basename "${file%.*}")"]=1
                done < <(find "$EDGES_PATH" -type f -print0)

                # Remove edge files without corresponding images
                for edge_base in "${!EDGE_FILES[@]}"; do
                    if [ ! "${IMAGE_FILES[$edge_base]+_}" ]; then
                        echo "Removing edge without image: $edge_base"
                        rm -f "$EDGES_PATH/$edge_base".*
                    fi
                done

                # Check for missing edges
                local MISSING_EDGES=0
                for img_base in "${!IMAGE_FILES[@]}"; do
                    if [ "${GT_FILES[$img_base]+_}" ] && [ ! "${EDGE_FILES[$img_base]+_}" ]; then
                        ((MISSING_EDGES++))
                    fi
                done

                if [ $MISSING_EDGES -gt 0 ]; then
                    echo "Warning: $MISSING_EDGES images missing edge maps in $DATASET_NAME/$SPLIT"
                fi
            elif [ "$DATASET_NAME" != "CAMO" ]; then
                echo "Warning: Missing Edges directory in $DATASET_NAME/$SPLIT"
            fi
        fi

        # Cleanup arrays
        unset IMAGE_FILES GT_FILES EDGE_FILES
    done
    
    echo "Dataset $DATASET_NAME processing completed"
    return 0
}

# Main execution
echo "Starting dataset processing..."

# Process each dataset
for DATASET in "CAMO" "COD10K" "NC4K"; do
    check_dataset "$DATASET"
done

echo "Setup completed successfully"
