#!/bin/bash

# Default values
MODEL_TYPE="pythia"
CHECKPOINT_DIR="checkpoints/pythia_S3_3/checkpoint-6000"
NUM_ITEMS=3
N_PROMPTS=1000
N_TOKENS=100
TOKEN_INCREMENTS=1
START_IDX=5
END_IDX=""

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
    --probe_span_type)
      PROBE_SPAN_TYPE="$2"
      shift 2
      ;;
    --token_increments)
      TOKEN_INCREMENTS="$2"
      shift 2
      ;;
    --start_idx)
      START_IDX="$2"
      shift 2
      ;;
    --end_idx)
      END_IDX="--end_idx $2"
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

# Run the lengthwise probing analysis
python -m interpret.main \
  --interpret_type lengthwise_probe \
  --model_type ${MODEL_TYPE} \
  --checkpoint_dir ${CHECKPOINT_DIR} \
  --num_items ${NUM_ITEMS} \
  --n_prompts ${N_PROMPTS} \
  --n_tokens ${N_TOKENS} \
  --token_increments ${TOKEN_INCREMENTS} \
  --start_idx ${START_IDX} \
  ${END_IDX} \
  2>&1 | tee logs/lengthwise_probe_${MODEL_TYPE}.log

echo "Lengthwise probing analysis completed. Check logs/lengthwise_probe_${MODEL_TYPE}.log for details."