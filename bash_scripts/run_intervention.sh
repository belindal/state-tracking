#!/bin/bash

# Default values
MODEL_TYPE="pythia"
CHECKPOINT_DIR="checkpoints/nopt_pythia_S3_0/checkpoint-28000/"
NUM_ITEMS=3
N_PROMPTS=100
N_TOKENS=80
INTERVENE_OUTPUT_TYPE="state"
PATCHING_MODE="deletion"
TOKEN_INCREMENTS=1

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
    --intervene_output_type)
      INTERVENE_OUTPUT_TYPE="$2"
      shift 2
      ;;
    --patching_mode)
      PATCHING_MODE="$2"
      shift 2
      ;;
    --token_increments)
      TOKEN_INCREMENTS="$2"
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

# Run the intervention analysis
python -m interpret.main \
  --interpret_type intervene \
  --model_type ${MODEL_TYPE} \
  --checkpoint_dir ${CHECKPOINT_DIR} \
  --num_items ${NUM_ITEMS} \
  --n_prompts ${N_PROMPTS} \
  --n_tokens ${N_TOKENS} \
  --intervene_output_type ${INTERVENE_OUTPUT_TYPE} \
  --patching_mode ${PATCHING_MODE} \
  --token_increments ${TOKEN_INCREMENTS} \
  2>&1 | tee logs/intervene_${MODEL_TYPE}_${INTERVENE_OUTPUT_TYPE}_${PATCHING_MODE}.log

echo "Intervention analysis completed. Check logs/intervene_${MODEL_TYPE}_${INTERVENE_OUTPUT_TYPE}_${PATCHING_MODE}.log for details."
