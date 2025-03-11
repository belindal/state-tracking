#!/bin/bash
# Script to train a model using train.py
# Usage: ./train.sh <model_name> <data_type> <data_directory> <output_directory> [options]
#
# Required arguments:
#   <model_name>: Model to use (e.g., gpt2, gpt2-medium, EleutherAI/pythia-70M)
#   <data_type>: Type of data (e.g., permutation)
#   <data_directory>: Directory containing the training data
#   <output_directory>: Directory to save model checkpoints
#
# Example:
#   ./train.sh gpt2 permutation ./data/perm3 ./checkpoints/perm3_run1
#   ./train.sh gpt2-medium permutation ./data/perm5 ./checkpoints/perm5_run1 --num_items 5 --early_stopping

# Check if required arguments are provided
if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <model_name> <data_type> <data_directory> <output_directory> [options]"
    echo "Example: $0 gpt2 permutation ./data ./checkpoints"
    exit 1
fi

# Store the required arguments
MODEL=$1
DATA_TYPE=$2
DATA_DIR=$3
OUTPUT_DIR=$4

# Shift the first 4 arguments so that $@ contains only the optional arguments
shift 4

# Run the training script with provided arguments
python train.py \
    --model "$MODEL" \
    --num_items "$DATA_TYPE" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 256 \
    --supervision_type direct_state \
    --max_len 512 \
    --seed 42 \
    --epochs 20 \
    "$@"  # Pass any additional arguments

# Available options that can be passed as additional arguments:
# --use_bfloat16                # Use bfloat16 precision
# --save_all_checkpoints 2000   # Save checkpoints at specified interval
# --early_stopping              # Enable early stopping
# --no_pretrain                 # Don't use pre-trained model
# --from_checkpoint /path/to/checkpoint   # Load from existing checkpoint
# --max_steps 10000             # Maximum number of training steps
# --data_determinism            # Use deterministic data loading
# --full_determinism            # Enable full determinism for reproducibility
# --layerwise_supervision_type /path/to/config  # File containing layerwise supervision keys