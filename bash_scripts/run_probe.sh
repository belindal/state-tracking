#!/bin/bash

# Default values
MODEL_TYPE="pythia"
CHECKPOINT_DIR="checkpoints/pythia_S3_3/checkpoint-6000"
NUM_ITEMS=3
N_PROMPTS=1000
N_TOKENS=100

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_type)
      MODEL_TYPE="$2"
      shift 2
      ;;
    --checkpoint_dir)
      CHECKPOINT_DIR="$2"
      shift 2
      ;;
    --num_items)
      NUM_ITEMS="$2"
      shift 2
      ;;
    --n_prompts)
      N_PROMPTS="$2"
      shift 2
      ;;
    --n_tokens)
      N_TOKENS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create output directory
mkdir -p logs

# Run the probing analysis
python -m interpret.main \
  --interpret_type probe \
  --model_type ${MODEL_TYPE} \
  --checkpoint_dir ${CHECKPOINT_DIR} \
  --num_items ${NUM_ITEMS} \
  --n_prompts ${N_PROMPTS} \
  --n_tokens ${N_TOKENS} \
  2>&1 | tee logs/probe_${MODEL_TYPE}.log

echo "Probing analysis completed. Check logs/probe_${MODEL_TYPE}.log for details."